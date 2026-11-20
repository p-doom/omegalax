"""Gate 1: per-flag numerical equivalence of the optimized copy against the original.

Why this exists next to ``compare_training_equivalence.py``
----------------------------------------------------------
That script compares *all three* switches at once against the original and reports
only ``max(abs, rel)`` over loss/grads/params.  Three things it cannot answer, and
this file is built for exactly those:

1. **Attribution.**  With ``MFU_CASCADE``, ``MFU_CONV_BTC`` and ``MFU_TRIBLOCK`` on
   together, a deviation cannot be attributed.  Here every one of the eight subsets
   of the three switches is its own arm.
2. **Where the error enters.**  Loss and parameters are the *end* of the chain.  A
   cancellation in the loss can hide a large deviation in the hidden state, and the
   ``embedding[sum]`` artefact in E26 is a recorded instance of exactly that.  Here
   the hidden representation immediately before the LM head, the recurrent GDN
   states and the raw pre-update gradients are compared elementwise.
3. **Whether the claimed code path ran.**  E31/E58 assert bit-identity of the
   default path; they do not prove that the *enabled* path dispatched instead of
   silently falling back (``cascade.is_fusible`` returns False -> production path,
   which would look like a perfect PASS).  Every arm here carries dispatch counters
   and fails if they do not match the expected signature.

Design decisions that matter for validity
-----------------------------------------
* **Import pinning.**  The copy shares the original's ``.venv``, whose editable
  install resolves ``omegalax`` to the *original*.  Each arm therefore runs in a
  fresh interpreter with ``PYTHONPATH`` pinned, and asserts ``omegalax.__file__``
  lies under the intended tree.  This is the single most dangerous failure mode of
  this setup: without it the "optimized" arm silently measures the original.
* **Switches are read at import time** (``deltanet._CONV_BTC``,
  ``pallas_triton._TRI_BLOCKED``), so the environment is set before JAX/omegalax
  are imported, never mutated in-process.
* **Instrumentation is symmetric.**  Both the original and the copy arms run under
  the *same* observation wrappers, and the noise floor is measured by two
  independent runs of the *original* through the identical instrumentation
  (``orig_a`` vs ``orig_b``).  Any effect of the extra outputs on XLA scheduling
  therefore appears in the calibration, not as a fake signal.
* **Production invariant.**  ``build_window_arrays`` sets ``reset_state_C[0] = True``
  and all other segments False; ``cascade.loss_stats`` never applies
  ``reset_state_BC`` to the recurrence and is only correct under that invariant.
  Both batch shapes here reproduce it exactly.
* **Padding is production-shaped, not synthetic.**  ``build_chunk_arrays`` pads the
  *tail* of the *last* segment with ``pad_id`` and zeros ``attention_mask``/
  ``loss_mask`` there; earlier segments are always full.  The ``padtail`` shape
  reproduces that, with a different valid length per chain.  An all-ones batch
  cannot exercise the masking in ``GatedDeltaNet.__call__``.

What is deliberately *not* claimed
----------------------------------
This runs a mock-sized model (the Pallas kernel and cuDNN attention are real, the
shapes that drive *performance* are not).  It is a correctness oracle only.  Chunk
size 64, ``linear_key/value_head_dim`` 128, GQA and a full-attention layer are held
at production values because they steer branches; ``hidden_size``, ``vocab_size``
and layer count are shrunk because they only scale cost.

Invocation goes through ``submit_gate1.sh``.  Nothing here writes a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT.parent / "omegalax"

# Held at production values because each one steers a branch or a recursion depth.
# CHUNK also fixes the block count of the blocked inverse (64 / _TRI_BLOCK=16 = 4).
CHUNK = 64
SEGMENTS = 2
# Defaults for the real run. --seq-len / --chains exist only so the pilot can shrink
# them; every reported number states which values it used.
DEFAULT_SEQ_LEN = 8 * CHUNK  # J = 8 chunks per segment; E56: few matrices under-test the inverse
DEFAULT_CHAINS = 4

# All eight subsets of the three switches. `conv_btc` alone and `conv_btc+triblock`
# are not shipping configurations (E35: -5.5 % under FSDP without the cascade) but
# they are the only way to attribute a deviation to Conv-BTC on its own, so they are
# measured here and reported as numerics-only arms.
ARM_BITS = {
    "opt_off": (0, 0, 0),
    "opt_cascade": (1, 0, 0),
    "opt_conv_btc": (0, 1, 0),
    "opt_triblock": (0, 0, 1),
    "opt_cascade_conv_btc": (1, 1, 0),
    "opt_cascade_triblock": (1, 0, 1),
    "opt_conv_btc_triblock": (0, 1, 1),
    "opt_cascade_conv_btc_triblock": (1, 1, 1),
}

PERCENTILES = (50.0, 90.0, 99.0, 99.9, 100.0)


# --------------------------------------------------------------------------- child


def _build_raw_batch(
    shape: str, seed: int, pad_id: int, vocab: int, chains: int, seq_len: int
) -> dict[str, np.ndarray]:
    """Deterministic batch in the layout ``build_window_arrays`` produces.

    ``uniform``: every segment full, mask all ones.
    ``padtail``: the last segment of every chain is short, right-padded with
    ``pad_id`` and zero-masked, with a per-chain valid length -- the only padding
    pattern the statepassing window loader can actually emit.

    Both shapes are cross-checked against the loader itself in
    ``_assert_loader_shaped``, so this is not a hand-waved "looks realistic".
    """
    rng = np.random.default_rng(1000 + seed)
    token_ids = rng.integers(1, vocab - 1, size=(chains, SEGMENTS, seq_len), dtype=np.int32)
    attention = np.ones((chains, SEGMENTS, seq_len), dtype=np.int32)
    if shape == "padtail":
        # A short tail chunk plus a stand-in EOS at the last valid position. The real
        # eos id (248046) does not fit the mock vocabulary; only the position of the
        # mask edge matters for the code path under test.
        valid = rng.integers(seq_len // 4, seq_len, size=chains)
        for chain in range(chains):
            end = int(valid[chain])
            token_ids[chain, SEGMENTS - 1, end - 1] = vocab - 1
            token_ids[chain, SEGMENTS - 1, end:] = pad_id
            attention[chain, SEGMENTS - 1, end:] = 0
    elif shape != "uniform":
        raise ValueError(f"unknown shape {shape!r}")
    reset = np.zeros((chains, SEGMENTS), dtype=np.bool_)
    reset[:, 0] = True  # build_window_arrays: reset only on the first segment
    return {
        "token_ids_BCT": token_ids,
        "attention_mask_BCT": attention,
        "loss_mask_BCT": attention.copy(),  # build_chunk_arrays: loss_mask is a copy of attention_mask
        "reset_state_BC": reset,
    }


def _assert_loader_shaped(raw: dict[str, np.ndarray], shape: str, pad_id: int) -> dict[str, object]:
    """Prove the hand-built batch is one the production loader could emit.

    Rather than asserting properties I believe the loader has, this drives the real
    ``build_window_arrays`` on a synthetic document of the matching token count and
    requires the *structure* to match: mask edges, the loss_mask == attention_mask
    identity, the pad fill, and reset_state. Only the token values differ.
    """
    from omegalax.data.pretrain_data_set import (  # noqa: PLC0415
        WindowMetadata,
        build_window_arrays,
    )

    chains, segments, seq_len = raw["token_ids_BCT"].shape
    checks: dict[str, object] = {"chains": chains, "segments": segments, "seq_len": seq_len}
    for chain in range(chains):
        valid_per_segment = raw["attention_mask_BCT"][chain].sum(axis=1)
        doc_tokens = int(valid_per_segment.sum())
        # The loader derives every mask from the document length alone.
        window = WindowMetadata(
            bucket_idx=0,
            record_idx=0,
            doc_id="0",
            window_idx=0,
            start_chunk=0,
            num_segments=segments,
            doc_token_count=doc_tokens,
            doc_num_chunks=segments,
            eos_token_idx=doc_tokens - 1,
        )
        produced = build_window_arrays(
            np.arange(1, doc_tokens + 1, dtype=np.int32),
            window,
            chunk_length=seq_len,
            pad_id=pad_id,
            eos_id=1,
        )
        if not np.array_equal(produced["attention_mask_CT"], raw["attention_mask_BCT"][chain]):
            raise AssertionError(
                f"chain {chain}: attention_mask is not loader-shaped for shape={shape!r}"
            )
        if not np.array_equal(produced["loss_mask_CT"], raw["loss_mask_BCT"][chain]):
            raise AssertionError(f"chain {chain}: loss_mask is not loader-shaped")
        if not np.array_equal(produced["reset_state_C"], raw["reset_state_BC"][chain]):
            raise AssertionError(f"chain {chain}: reset_state is not loader-shaped")
        pad_positions = produced["attention_mask_CT"] == 0
        if not np.all(raw["token_ids_BCT"][chain][pad_positions] == pad_id):
            raise AssertionError(f"chain {chain}: masked positions are not filled with pad_id")
        if np.any(raw["token_ids_BCT"][chain][~pad_positions] == pad_id):
            raise AssertionError(f"chain {chain}: pad_id appears at an unmasked position")
    checks["valid_tokens_per_chain"] = [
        int(raw["attention_mask_BCT"][c].sum()) for c in range(chains)
    ]
    checks["padded_positions"] = int((raw["attention_mask_BCT"] == 0).sum())
    checks["reset_only_first_segment"] = bool(
        raw["reset_state_BC"][:, 0].all() and not raw["reset_state_BC"][:, 1:].any()
    )
    return checks


def _child(args: argparse.Namespace) -> None:
    if args.tree is None or args.out is None:
        raise ValueError("--child requires --tree and --out")

    os.environ["JAX_PLATFORMS"] = args.platform
    os.environ["OMEGALAX_DELTANET_KERNEL"] = args.kernel

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from flax import nnx  # noqa: PLC0415
    from jax.sharding import NamedSharding, PartitionSpec  # noqa: PLC0415

    import omegalax  # noqa: PLC0415
    from omegalax.distributed.mesh import ensure_mesh, mesh_rules  # noqa: PLC0415
    from omegalax.models.qwen3_5 import deltanet as deltanet_mod  # noqa: PLC0415
    from omegalax.models.qwen3_5.config import Qwen3_5TextConfig  # noqa: PLC0415
    from omegalax.models.qwen3_5.kernels import pallas_triton  # noqa: PLC0415
    from omegalax.models.shard_config import ShardConfig  # noqa: PLC0415
    from omegalax.models.sharding_runtime import set_attn_backend  # noqa: PLC0415
    from omegalax.text import api as text_api  # noqa: PLC0415
    from omegalax.trainers import pretrain as P  # noqa: PLC0415
    from absl import flags as absl_flags  # noqa: PLC0415

    # Tokamax reads Abseil's global flags lazily on the first attention call, and
    # argparse has already consumed this script's private flags from sys.argv.
    if not absl_flags.FLAGS.is_parsed():
        absl_flags.FLAGS(["gate1_equivalence"])

    tree = args.tree.resolve()
    source = Path(omegalax.__file__).resolve()
    try:
        source.relative_to(tree / "omegalax")
    except ValueError as exc:
        raise RuntimeError(
            f"PYTHONPATH selected {source}, expected a module below {tree / 'omegalax'}"
        ) from exc

    # ---- instrumentation: observation seams and dispatch counters --------------
    # Every wrapper is a pass-through installed identically in every arm, so it
    # cannot introduce an asymmetry between the two trees.
    counters = {
        "forward_with_gdn_state": 0,
        "cascade_loss_stats": 0,
        "conv_btc": 0,
        "conv_bct": 0,
        "blocked_inverse": 0,
    }
    record: dict[str, object] = {}

    real_forward = text_api.forward_with_gdn_state

    def counting_forward(*call_args, **call_kwargs):
        counters["forward_with_gdn_state"] += 1
        result = real_forward(*call_args, **call_kwargs)
        # (hidden, aux_loss, gdn_states[, conv_states]); index 2 is the GDN tuple.
        record.setdefault("fwd_states", []).append(tuple(result[2]))
        return result

    text_api.forward_with_gdn_state = counting_forward

    real_loss_tiles = P.chunked_cross_entropy_multi_stats

    def capturing_loss_tiles(hidden_BT_D, *call_args, **call_kwargs):
        # Called once, by both the production loop and cascade.loss_stats, with the
        # hidden representation that the LM head consumes.
        record["hidden"] = hidden_BT_D
        return real_loss_tiles(hidden_BT_D, *call_args, **call_kwargs)

    P.chunked_cross_entropy_multi_stats = capturing_loss_tiles

    real_conv_bct = deltanet_mod._causal_depthwise_conv1d

    def counting_conv_bct(*call_args, **call_kwargs):
        counters["conv_bct"] += 1
        return real_conv_bct(*call_args, **call_kwargs)

    deltanet_mod._causal_depthwise_conv1d = counting_conv_bct

    # These three symbols only exist in the optimized tree; their absence is itself
    # part of the evidence that the original carries none of the optimized paths.
    optional_symbols = {}
    real_conv_btc = getattr(deltanet_mod, "_causal_depthwise_conv1d_btc", None)
    optional_symbols["_causal_depthwise_conv1d_btc"] = real_conv_btc is not None
    if real_conv_btc is not None:
        def counting_conv_btc(*call_args, **call_kwargs):
            counters["conv_btc"] += 1
            return real_conv_btc(*call_args, **call_kwargs)

        deltanet_mod._causal_depthwise_conv1d_btc = counting_conv_btc

    real_blocked = getattr(pallas_triton, "_blocked_unit_lower_inverse", None)
    optional_symbols["_blocked_unit_lower_inverse"] = real_blocked is not None
    if real_blocked is not None:
        def counting_blocked(*call_args, **call_kwargs):
            counters["blocked_inverse"] += 1
            return real_blocked(*call_args, **call_kwargs)

        pallas_triton._blocked_unit_lower_inverse = counting_blocked

    cascade_mod = getattr(P, "cascade", None)
    optional_symbols["trainers.cascade"] = cascade_mod is not None
    if cascade_mod is not None:
        real_cascade_loss = cascade_mod.loss_stats

        def counting_cascade_loss(*call_args, **call_kwargs):
            counters["cascade_loss_stats"] += 1
            return real_cascade_loss(*call_args, **call_kwargs)

        cascade_mod.loss_stats = counting_cascade_loss

    # ---- model, optimizer, batch ----------------------------------------------
    cfg = Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        layer_types=(
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ),
        rope_theta=10_000,
        partial_rotary_factor=0.25,
        mrope_section=(4, 2, 2),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_num_key_heads=1,
        linear_num_value_heads=4,
        linear_value_head_dim=128,
        intermediate_size=256,
        shd_cfg=ShardConfig.no_sharding(),
        dtype=jnp.bfloat16 if args.dtype == "bf16" else jnp.float32,
    )
    train_cfg = P.TrainConfig(
        seed=args.seed,
        batch_size=args.chains * SEGMENTS,
        seq_len=args.seq_len,
        num_steps=args.steps,
        learning_rate=1e-3,
        weight_decay=0.0,
        print_every=0,
    )
    raw_batch = _build_raw_batch(
        args.shape, args.seed, args.pad_id, cfg.vocab_size, args.chains, args.seq_len
    )
    loader_checks = _assert_loader_shaped(raw_batch, args.shape, args.pad_id)

    mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
    model_cfg = text_api.align_config_to_mesh(cfg, mesh)
    init_key = jax.device_put(
        jax.random.key(args.seed), NamedSharding(mesh, PartitionSpec())
    )
    init_key, _ = jax.random.split(init_key)
    model, model_cfg = P.init_model(model_cfg, init_key, tp_size=1, fsdp_size=1, dp_size=1)
    set_attn_backend(model, text_backend=args.attn_backend)
    with mesh_rules(mesh):
        optimizer = P.build_optimizer(model, train_cfg)
    batch, _, _ = P.prepare_pretrain_batch(
        raw_batch,
        P.PretrainMode.STATEPASSING_BPTT,
        model_cfg,
        mesh,
        global_batch_size=args.chains * SEGMENTS,  # statepassing counts *segments*
    )

    loss_kwargs = dict(
        bptt_chunks=SEGMENTS,
        pass_gdn_state=True,
        pass_rope_positions=False,
        pass_conv_state=False,
    )

    def production_loss(model_, batch_):
        return P._statepassing_loss_stats(
            model_, batch_, model_cfg, args.pad_id, P.PretrainMode.STATEPASSING_BPTT, **loss_kwargs
        )

    # Two traced functions rather than one. The observables live inside loss_fn, and
    # tracers from the inner value_and_grad trace cannot be returned from the outer
    # one; a forward-only companion trace of the *same* production function can
    # return them. donate_argnums is omitted so the model stays readable for it.
    @nnx.jit
    def observe(model_, batch_):
        record.clear()
        loss, metrics = production_loss(model_, batch_)
        return loss, metrics, record["hidden"], tuple(record["fwd_states"])

    @nnx.jit
    def graded_step(optimizer_, batch_):
        (loss, metrics), grads = nnx.value_and_grad(production_loss, has_aux=True)(
            optimizer_.model, batch_
        )
        # Returning VariableState metadata from a jitted GPU function trips an XLA
        # lowering failure; ``pure`` is also what the production optimizer sees.
        grad_state = nnx.pure(nnx.state(grads, nnx.Param))
        optimizer_.update(grads)
        return loss, metrics, grad_state, nnx.pure(nnx.state(optimizer_.model, nnx.Param))

    def flat_arrays(state, prefix):
        out = {}
        for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
            value = getattr(leaf, "value", leaf)
            name = "/".join(str(getattr(part, "key", part)) for part in path)
            array = jax.device_get(value)
            # NumPy stores JAX bfloat16 as an opaque two-byte record; cast while the
            # JAX dtype is still known. bf16 -> fp32 is exact.
            out[f"{prefix}{name}"] = np.asarray(
                array,
                dtype=np.float32 if jnp.issubdtype(array.dtype, jnp.inexact) else None,
            )
        return out

    result: dict[str, np.ndarray] = {}
    dispatch = None
    state_row_spread = None
    for step in range(1, args.steps + 1):
        for key in counters:
            counters[key] = 0
        loss, metrics, hidden, fwd_states = observe(optimizer.model, batch)
        if step == 1:
            # Counters are trace-time; only the first step actually traces.
            dispatch = dict(counters)

        result[f"hidden/{step}/full"] = np.asarray(jax.device_get(hidden), dtype=np.float32)
        result.update(flat_arrays(metrics, f"metric/{step}/"))

        # Comparable recurrent state across arms: the GDN state at the end of the
        # last segment, per chain and layer. The production loop calls the forward C
        # times (rows are chains); the cascade calls it once over B*C rows and
        # replicates each chain's final state C times, so row b*C carries chain b.
        n_calls = len(fwd_states)
        if n_calls == SEGMENTS:
            chain_final = fwd_states[-1]
            for seg_idx, states in enumerate(fwd_states):
                for layer_idx, state in enumerate(states):
                    result[f"state_all/{step}/seg{seg_idx}/layer{layer_idx}"] = np.asarray(
                        jax.device_get(state), dtype=np.float32
                    )
        elif n_calls == 1:
            # The cascade wrapper returns jnp.repeat(chain_final, C, axis=0), so every
            # row of a chain must carry the identical state. Verified, not assumed: if
            # the rows differ, the [::C] slice below is picking one of several
            # distinct values and the loop/cascade mapping would be meaningless.
            spread = 0.0
            for state in fwd_states[0]:
                host = np.asarray(jax.device_get(state), dtype=np.float64)
                rows = host.reshape((-1, SEGMENTS) + host.shape[1:])
                spread = max(spread, float(np.abs(rows - rows[:, :1]).max()))
            state_row_spread = spread
            if spread != 0.0:
                raise RuntimeError(
                    f"cascade final state differs across a chain's rows (max {spread:.3e}); "
                    "the chain-final mapping is not valid"
                )
            chain_final = tuple(state[::SEGMENTS] for state in fwd_states[0])
        else:
            raise RuntimeError(
                f"unexpected forward call count {n_calls}; expected {SEGMENTS} (loop) or 1 (cascade)"
            )
        for layer_idx, state in enumerate(chain_final):
            result[f"state_final/{step}/layer{layer_idx}"] = np.asarray(
                jax.device_get(state), dtype=np.float32
            )

        _, _, gradients, params = graded_step(optimizer, batch)
        result.update(flat_arrays(gradients, f"grad/{step}/"))
        result.update(flat_arrays(params, f"param/{step}/"))

    meta = {
        "source": str(source),
        "tree": str(tree),
        "devices": [str(device) for device in jax.devices()],
        "steps": args.steps,
        "shape": args.shape,
        "seed": args.seed,
        "chains": args.chains,
        "segments": SEGMENTS,
        "seq_len": args.seq_len,
        "chunk": CHUNK,
        "loader_checks": loader_checks,
        "dtype": args.dtype,
        "kernel": args.kernel,
        "attn_backend": args.attn_backend,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "dispatch": dispatch,
        "forward_calls": n_calls,
        "cascade_state_row_spread": state_row_spread,
        "optional_symbols": optional_symbols,
        "module_switch_state": {
            "deltanet._CONV_BTC": bool(getattr(deltanet_mod, "_CONV_BTC", False)),
            "pallas_triton._TRI_BLOCKED": bool(getattr(pallas_triton, "_TRI_BLOCKED", False)),
            "pallas_triton._TRI_INV": bool(getattr(pallas_triton, "_TRI_INV", False)),
            "cascade.enabled": bool(cascade_mod.enabled()) if cascade_mod else False,
            "cascade._INSTALLED": bool(getattr(cascade_mod, "_INSTALLED", False)),
            "cascade._SEGMENTS": getattr(cascade_mod, "_SEGMENTS", None) if cascade_mod else None,
        },
        "env": {
            name: os.environ.get(name, "")
            for name in (
                "MFU_CASCADE", "MFU_CONV_BTC", "MFU_TRIBLOCK", "MFU_TRIINV",
                "MFU_CHUNK", "MFU_REMAT", "MFU_MU_DTYPE", "MFU_FAST_CE",
                "MFU_TRI_BLOCK", "MFU_PALLAS_WARPS", "MFU_PALLAS_STAGES",
                "MFU_NUM_LOSS_TILES", "MFU_BWD_PALLAS", "MFU_TRIINV_DEBUG",
                "OMEGALAX_DELTANET_KERNEL", "XLA_FLAGS",
            )
        },
    }
    result["__meta__"] = np.asarray(json.dumps(meta, sort_keys=True))
    np.savez_compressed(args.out, **result)
    print(f"[child] omegalax -> {source}", flush=True)
    print(f"[child] dispatch={dispatch} switches={meta['module_switch_state']}", flush=True)
    print(f"[child] devices={meta['devices']} arrays={len(result) - 1} -> {args.out}", flush=True)


# -------------------------------------------------------------------------- parent


def _child_command(args: argparse.Namespace, tree: Path, out: Path, seed: int, shape: str) -> list[str]:
    return [
        str(ROOT / ".venv" / "bin" / "python"),
        str(Path(__file__).resolve()),
        "--child",
        "--tree", str(tree),
        "--out", str(out),
        "--steps", str(args.steps),
        "--seed", str(seed),
        "--shape", shape,
        "--platform", args.platform,
        "--kernel", args.kernel,
        "--attn-backend", args.attn_backend,
        "--dtype", args.dtype,
        "--pad-id", str(args.pad_id),
        "--chains", str(args.chains),
        "--seq-len", str(args.seq_len),
    ]


def _run_arm(
    args: argparse.Namespace, arm: str, tree: Path, out: Path, seed: int, shape: str
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tree)
    bits = ARM_BITS.get(arm, (0, 0, 0))
    env["MFU_CASCADE"] = str(bits[0])
    env["MFU_CONV_BTC"] = str(bits[1])
    env["MFU_TRIBLOCK"] = str(bits[2])
    # Every remaining switch is pinned to the value that reproduces the original's
    # hardcoded constant, so nothing is inherited from the surrounding environment.
    # Verified against the trees: MFU_REMAT=none -> policy=None (model.py), CHUNK=64,
    # PALLAS_WARPS/STAGES=4/2 (the original's literals), NUM_LOSS_TILES=4,
    # MU_DTYPE=fp32 -> mu_dtype=None, FAST_CE=0 -> the one-hot readout.
    env["MFU_TRIINV"] = "0"          # known broken, E49/E50
    env["MFU_BWD_PALLAS"] = "0"      # 3x slower, E43
    env["MFU_MU_DTYPE"] = "fp32"     # a bf16 moment changes the trajectory, E11/E13
    env["MFU_FAST_CE"] = "0"
    env["MFU_REMAT"] = "none"
    env["MFU_CHUNK"] = str(CHUNK)
    env["MFU_TRI_BLOCK"] = "16"      # production choice for the blocked inverse, E52
    env["MFU_PALLAS_WARPS"] = "4"
    env["MFU_PALLAS_STAGES"] = "2"
    env["MFU_NUM_LOSS_TILES"] = "4"
    env["MFU_TRIINV_DEBUG"] = "0"
    env["WANDB_MODE"] = "disabled"
    env["OMEGALAX_STARTUP_LOG"] = "0"
    print(f"\n=== arm={arm} shape={shape} seed={seed} tree={tree} flags={bits}", flush=True)
    subprocess.run(_child_command(args, tree, out, seed, shape), env=env, check=True)


def _expected_dispatch(arm: str, kernel: str) -> dict[str, object]:
    """The dispatch signature an arm must show, or its result is not evidence.

    ``MFU_TRIBLOCK`` lives in ``pallas_triton``; ``xla_reference`` does not read it
    and calls ``solve_triangular`` unconditionally. On ``--kernel=xla`` a triblock
    arm therefore cannot run the switched code at all, and must be reported as
    unproven rather than as a pass.
    """
    if kernel == "xla" and arm.startswith("opt_") and ARM_BITS[arm][2]:
        return {"unprovable": "MFU_TRIBLOCK is not reachable on the xla kernel path"}
    if arm.startswith("orig"):
        # The original tree has none of the optimized paths: it must run the segment
        # loop, the BCT conv, and the batched triangular solve.
        return {"forward_with_gdn_state": SEGMENTS, "cascade_loss_stats": 0,
                "conv_btc_positive": False, "conv_bct_positive": True,
                "blocked_inverse_positive": False}
    bits = ARM_BITS[arm]
    cascade, conv_btc, triblock = bool(bits[0]), bool(bits[1]), bool(bits[2])
    return {
        # One fused forward under the cascade, C forwards in the production loop.
        "forward_with_gdn_state": 1 if cascade else SEGMENTS,
        "cascade_loss_stats": 1 if cascade else 0,
        "conv_btc_positive": conv_btc,
        "conv_bct_positive": not conv_btc,
        "blocked_inverse_positive": triblock,
    }


def _check_dispatch(arm: str, meta: dict, kernel: str) -> list[str]:
    problems = []
    dispatch = meta.get("dispatch") or {}
    expected = _expected_dispatch(arm, kernel)
    if "unprovable" in expected:
        return [f"NOT PROVEN: {expected['unprovable']}"]
    if dispatch.get("forward_with_gdn_state") != expected["forward_with_gdn_state"]:
        problems.append(
            f"forward calls {dispatch.get('forward_with_gdn_state')} != "
            f"{expected['forward_with_gdn_state']}"
        )
    if dispatch.get("cascade_loss_stats", 0) != expected.get("cascade_loss_stats", 0):
        problems.append(
            f"cascade.loss_stats calls {dispatch.get('cascade_loss_stats')} != "
            f"{expected.get('cascade_loss_stats')}"
        )
    for key, name in (("conv_btc", "conv_btc"), ("conv_bct", "conv_bct"),
                      ("blocked_inverse", "blocked_inverse")):
        want = expected.get(f"{name}_positive")
        if want is None:
            continue
        got = dispatch.get(key, 0) > 0
        if got != want:
            problems.append(f"{name} called={got}, expected={want} (count={dispatch.get(key)})")
    switches = meta.get("module_switch_state", {})
    if not arm.startswith("orig"):
        bits = ARM_BITS[arm]
        if switches.get("deltanet._CONV_BTC") != bool(bits[1]):
            problems.append("deltanet._CONV_BTC does not match the arm")
        if switches.get("pallas_triton._TRI_BLOCKED") != bool(bits[2]):
            problems.append("pallas_triton._TRI_BLOCKED does not match the arm")
        if switches.get("cascade.enabled") != bool(bits[0]):
            problems.append("cascade.enabled does not match the arm")
        if bits[0] and switches.get("cascade._SEGMENTS") != SEGMENTS:
            problems.append(f"cascade._SEGMENTS={switches.get('cascade._SEGMENTS')} != {SEGMENTS}")
    if switches.get("pallas_triton._TRI_INV"):
        problems.append("MFU_TRIINV is active; that path is known broken (E49/E50)")
    return problems


def _tensor_stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    delta = np.abs(a - b)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    denom = max(norm_a, 1e-300)
    cos = float(np.dot(a, b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 1.0
    return {
        "n": int(a.size),
        "max_abs": float(delta.max()) if a.size else 0.0,
        "rel_l2": float(np.linalg.norm(delta) / denom),
        "cosine": cos,
        "rms_ref": float(np.sqrt(np.mean(a * a))) if a.size else 0.0,
        "finite": bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b))),
        "bitwise_equal": bool(np.array_equal(a, b)),
    }


def _family_stats(
    left: np.lib.npyio.NpzFile, right: np.lib.npyio.NpzFile, prefix: str
) -> dict[str, object] | None:
    keys = sorted(k for k in left.files if k.startswith(prefix))
    other = sorted(k for k in right.files if k.startswith(prefix))
    if keys != other:
        # Never silently omitted. ``state_all`` legitimately has no counterpart under
        # the cascade -- the segment loop it comes from does not exist there -- but
        # that is an absence of evidence and is reported as such, not as a pass.
        if not keys and not other:
            return {"status": "absent_both", "left_tensors": 0, "right_tensors": 0}
        return {
            "status": "not_comparable",
            "left_tensors": len(keys),
            "right_tensors": len(other),
            "only_left": keys[:4],
            "only_right": other[:4],
        }
    if not keys:
        return {"status": "absent_both", "left_tensors": 0, "right_tensors": 0}

    deltas, refs, dot, norm_a_sq, norm_b_sq = [], [], 0.0, 0.0, 0.0
    per_tensor = {}
    finite = True
    bitwise = True
    for key in keys:
        stats = _tensor_stats(left[key], right[key])
        per_tensor[key[len(prefix):]] = stats
        finite = finite and stats["finite"]
        bitwise = bitwise and stats["bitwise_equal"]
        a = np.asarray(left[key], dtype=np.float64).ravel()
        b = np.asarray(right[key], dtype=np.float64).ravel()
        deltas.append(np.abs(a - b))
        refs.append(a)
        dot += float(np.dot(a, b))
        norm_a_sq += float(np.dot(a, a))
        norm_b_sq += float(np.dot(b, b))

    delta = np.concatenate(deltas) if deltas else np.zeros(1)
    ref = np.concatenate(refs) if refs else np.zeros(1)
    rms = float(np.sqrt(np.mean(ref * ref)))
    scale = rms if rms > 0 else 1.0
    worst_abs = max(per_tensor, key=lambda k: per_tensor[k]["max_abs"])
    worst_rel = max(per_tensor, key=lambda k: per_tensor[k]["rel_l2"])
    worst_cos = min(per_tensor, key=lambda k: per_tensor[k]["cosine"])
    result = {
        "status": "compared",
        "tensors": len(keys),
        "elements": int(delta.size),
        "finite": finite,
        "bitwise_equal": bitwise,
        "max_abs": float(delta.max()),
        "rel_l2": float(np.linalg.norm(delta) / max(np.sqrt(norm_a_sq), 1e-300)),
        "cosine": (
            float(dot / (np.sqrt(norm_a_sq) * np.sqrt(norm_b_sq)))
            if norm_a_sq > 0 and norm_b_sq > 0
            else 1.0
        ),
        "rms_ref": rms,
        "abs_percentiles": {
            f"p{p:g}": float(np.percentile(delta, p)) for p in PERCENTILES
        },
        "rms_normalized_percentiles": {
            f"p{p:g}": float(np.percentile(delta, p) / scale) for p in PERCENTILES
        },
        "worst_tensor_max_abs": {"name": worst_abs, **per_tensor[worst_abs]},
        "worst_tensor_rel_l2": {"name": worst_rel, **per_tensor[worst_rel]},
        "worst_tensor_cosine": {"name": worst_cos, **per_tensor[worst_cos]},
    }
    # Keep the per-tensor breakdown for every family that is not the (B, C*T, D)
    # hidden blob. A single tensor hidden behind a family aggregate is exactly the
    # cancellation failure mode E26 ran into, and for `grad` -- 56 scalars-to-matrices
    # -- the breakdown is the only way to see *where* a deviation sits.
    if len(keys) <= 128:
        result["per_tensor"] = per_tensor
    return result


def _compare(
    left: np.lib.npyio.NpzFile, right: np.lib.npyio.NpzFile, steps: int
) -> dict[str, object]:
    out = {}
    for step in range(1, steps + 1):
        for family in ("hidden", "metric", "state_final", "state_all", "grad", "param"):
            out[f"step{step}/{family}"] = _family_stats(left, right, f"{family}/{step}/")
    return out


def _parent(args: argparse.Namespace) -> int:
    if args.platform == "cpu" and (args.kernel != "xla" or args.attn_backend != "xla"):
        raise ValueError("CPU sanity runs require --kernel=xla --attn-backend=xla")
    if not ORIGINAL.is_dir():
        raise RuntimeError(f"Original tree not found: {ORIGINAL}")
    out_dir = Path(args.results).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    shapes = [s.strip() for s in args.shapes.split(",")]
    selected = (
        list(ARM_BITS)
        if args.arms == "all"
        else [a.strip() for a in args.arms.split(",")]
    )
    unknown = [a for a in selected if a not in ARM_BITS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; choose from {sorted(ARM_BITS)}")
    summary: dict[str, object] = {
        "roots": {"original": str(ORIGINAL), "copy": str(ROOT)},
        "config": {
            "chains": args.chains, "segments": SEGMENTS, "seq_len": args.seq_len,
            "chunk": CHUNK,
            "steps": args.steps, "dtype": args.dtype, "kernel": args.kernel,
            "platform": args.platform,
            "attn_backend": args.attn_backend, "pad_id": args.pad_id,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "seeds": seeds, "shapes": shapes, "arms": selected,
        },
        "runs": {},
    }
    failures: list[str] = []

    for shape in shapes:
        for seed in seeds:
            # Temporary tensor archives stay inside the working copy and are removed
            # at scope exit; only the JSON summary and the job log persist.
            with tempfile.TemporaryDirectory(
                prefix="gate1_", dir=Path(__file__).resolve().parent
            ) as temp_dir:
                temp = Path(temp_dir)
                arms = {"orig_a": ORIGINAL, "orig_b": ORIGINAL}
                arms.update({arm: ROOT for arm in selected})
                paths = {}
                for arm, tree in arms.items():
                    path = temp / f"{arm}.npz"
                    _run_arm(args, arm, tree, path, seed, shape)
                    paths[arm] = path

                handles = {
                    arm: np.load(path, allow_pickle=False) for arm, path in paths.items()
                }
                try:
                    metas = {
                        arm: json.loads(str(handle["__meta__"]))
                        for arm, handle in handles.items()
                    }
                    for arm, meta in metas.items():
                        problems = _check_dispatch(arm, meta, args.kernel)
                        key = f"{shape}/seed{seed}/{arm}"
                        summary["runs"][key] = {
                            "meta": meta,
                            "dispatch_problems": problems,
                        }
                        if problems:
                            failures.append(f"{key}: dispatch {problems}")
                    reference = handles["orig_a"]
                    for arm, handle in handles.items():
                        if arm == "orig_a":
                            continue
                        comparison = _compare(reference, handle, args.steps)
                        summary["runs"][f"{shape}/seed{seed}/{arm}"]["vs_orig_a"] = comparison
                finally:
                    for handle in handles.values():
                        handle.close()

    target = out_dir / f"gate1_summary_{args.tag}.json"
    target.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nsummary -> {target}", flush=True)

    # The report is the numbers; the only hard gate here is that every arm proved
    # its own code path. A deviation is a finding to be read, not an error.
    print("\n=== dispatch verification")
    for key, run in sorted(summary["runs"].items()):
        problems = run["dispatch_problems"]
        print(f"{'FAIL' if problems else 'ok  '}  {key}  {problems if problems else ''}")
    # orig_b is the noise floor: identical code, identical instrumentation. Every
    # other row must be read against it, never against zero.
    print("\n=== step 1, vs orig_a. orig_b is the noise floor, not a result.")
    # worst_rel_l2 is the per-tensor maximum. The family aggregate can look small
    # purely because one large, exactly-equal tensor dominates the reference norm --
    # `metric` mixes supervised_tokens with nll, so its aggregate is not readable
    # on its own.
    header = (
        "arm".ljust(32) + "family".ljust(12) + "status".ljust(11)
        + "max_abs      rel_l2       worst_rel_l2  cosine         p99(abs)     p99/rms"
    )
    for shape in shapes:
        for seed in seeds:
            print(f"\n-- shape={shape} seed={seed}\n{header}")
            for arm in ["orig_b"] + selected:
                run = summary["runs"].get(f"{shape}/seed{seed}/{arm}", {})
                for family in ("hidden", "state_final", "state_all", "grad", "metric"):
                    stats = (run.get("vs_orig_a") or {}).get(f"step1/{family}")
                    if not stats:
                        continue
                    status = stats.get("status", "?")
                    if status != "compared":
                        print(f"{arm.ljust(32)}{family.ljust(12)}{status}")
                        continue
                    print(
                        f"{arm.ljust(32)}{family.ljust(12)}{status.ljust(11)}"
                        f"{stats['max_abs']:.3e}    {stats['rel_l2']:.3e}    "
                        f"{stats['worst_tensor_rel_l2']['rel_l2']:.3e}     "
                        f"{stats['cosine']:.12f}   {stats['abs_percentiles']['p99']:.3e}    "
                        f"{stats['rms_normalized_percentiles']['p99']:.3e}"
                    )
    if failures:
        print(f"\nDISPATCH FAILURES ({len(failures)}); the affected arms are not evidence:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nAll arms proved their code path. No checkpoints were written.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tree", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--shape", default="uniform", help=argparse.SUPPRESS)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--shapes", default="uniform,padtail")
    parser.add_argument("--platform", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--kernel", choices=("xla", "pallas"), default="pallas")
    parser.add_argument("--attn-backend", default="cudnn")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--pad-id", type=int, default=0)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument(
        "--arms",
        default="all",
        help="comma-separated subset of the eight copy arms, or 'all'",
    )
    parser.add_argument("--tag", default="run")
    parser.add_argument(
        "--results",
        default="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs/mfu_opt/gate1",
    )
    args = parser.parse_args()
    if args.child:
        _child(args)
        return 0
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
