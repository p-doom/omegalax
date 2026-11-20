"""Gate 4: ten-step A/B training pilot on real FineWeb-Edu data, checkpoint-free.

Gates 1-3 established that the three switches are algebraically sound and that the
residual differences sit at reduced-precision level. This pilot asks the only
question those gates cannot: does that small difference do anything *practically*
abnormal to optimisation over ten real steps?

Design
------
Stage 1 materialises the batches **once**, with the production statepassing loader
on the canonical C4 index, and writes them to a transient archive. Every arm then
trains on that identical archive, so the batches are equal by construction and the
recorded SHA-256 digests prove it rather than assert it. The loader's own
``reset_state`` invariant (reset only before segment 0) is checked explicitly.

Stage 2 runs each arm in a fresh process with ``PYTHONPATH`` pinned and
``omegalax.__file__`` asserted, hashes the initial parameters before step 0, then
drives exactly ten updates through the **production** ``make_pretrain_train_step``
and evaluates a fixed real validation set through the **production**
``make_pretrain_eval_step`` at steps 0, 5 and 10. No second training implementation
exists here; the loss, autodiff and AdamW update are production code.

Arms per seed: ``orig_a`` and ``orig_b`` are the same original code run twice --
that pair is the yardstick for ordinary GPU trajectory variability. ``opt_all`` is
the copy with ``MFU_CASCADE=1 MFU_CONV_BTC=1 MFU_TRIBLOCK=1``. Any statement about
the optimisation is ``opt_all`` vs ``orig_a`` *read against* ``orig_b`` vs ``orig_a``.

Scope
-----
The question is stability and gross training-quality equivalence over ten steps, not
bitwise identity after Adam and not throughput. Parameter relative L2 is recorded but
is explicitly not a verdict on its own: Adam's ``g/sqrt(v)`` turns a sign flip on a
near-zero gradient into a full-size update difference, which E59 measured directly.
A ten-step tiny-model pilot can rule out immediate divergence, NaN and gross loss
degradation; it cannot establish full-scale long-horizon equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORIGINAL = ROOT.parent / "omegalax"
INDEX_ROOT = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/pretrain_indexes/"
    "fineweb_edu_dedup_2048_2kto32k_2048_c4_bptt4_lr3e4_eos248046"
)
CHUNK_LENGTH = 2048     # index chunk_length; DeltaNet chunk stays 64 -> J = 32
SEGMENTS = 4            # index num_segments; C = 4 chained segments, Cascade active
GLOBAL_BATCH = 4        # segments per step, i.e. one window chain
EOS_ID = 248046
PAD_ID = 0
VOCAB = 248320
STEPS = 10
N_VAL_BATCHES = 2
EVAL_AT = (0, 5, 10)
BATCH_KEYS = ("token_ids_BCT", "attention_mask_BCT", "loss_mask_BCT", "reset_state_BC")


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:32]


# ------------------------------------------------------------------ stage 1: batches

def _child_materialize(args) -> None:
    """Pull real windows with the production loader and write them once."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    import omegalax  # noqa: PLC0415
    from omegalax.data.pretrain_statepassing import make_statepassing_iterator  # noqa: PLC0415
    from omegalax.data.pretrain_data_set import pop_pretrain_metadata  # noqa: PLC0415

    tree = args.tree.resolve()
    source = Path(omegalax.__file__).resolve()
    print(f"[materialize] omegalax -> {source}", flush=True)
    source.relative_to(tree / "omegalax")

    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, object] = {"index_root": str(INDEX_ROOT), "seed": args.seed}
    for split, count in (("train", STEPS), ("val", N_VAL_BATCHES)):
        iterator = make_statepassing_iterator(
            INDEX_ROOT / split,
            batch_size=GLOBAL_BATCH,
            chunk_length=CHUNK_LENGTH,
            pad_id=PAD_ID,
            eos_id=EOS_ID,
            shuffle=(split == "train"),
            seed=args.seed,
            dp_size=1,
            fsdp_size=1,
            grain_workers=4,
        )
        for step in range(count):
            batch = dict(next(iterator))
            pop_pretrain_metadata(batch)
            for key in BATCH_KEYS:
                value = np.asarray(batch[key])
                arrays[f"{split}/{step}/{key}"] = value
        del iterator

    # The reset invariant the Cascade silently relies on (cascade.py has no check).
    resets = [v for k, v in arrays.items() if k.endswith("reset_state_BC")]
    first_true = all(bool(r[:, 0].all()) for r in resets)
    rest_false = all(not bool(r[:, 1:].any()) for r in resets)
    if not (first_true and rest_false):
        raise RuntimeError("loader produced reset_state outside the reachable invariant")

    manifest["reset_only_first_segment"] = True
    manifest["digests"] = {k: _digest(v) for k, v in arrays.items()}
    manifest["counts"] = {
        "train_batches": STEPS, "val_batches": N_VAL_BATCHES,
        "shape_token_ids": list(arrays["train/0/token_ids_BCT"].shape),
        "real_tokens_train": int(sum(
            int(v.sum()) for k, v in arrays.items()
            if k.startswith("train/") and k.endswith("attention_mask_BCT"))),
        "supervised_tokens_train": int(sum(
            int(v.sum()) for k, v in arrays.items()
            if k.startswith("train/") and k.endswith("loss_mask_BCT"))),
        "real_tokens_val": int(sum(
            int(v.sum()) for k, v in arrays.items()
            if k.startswith("val/") and k.endswith("attention_mask_BCT"))),
        "padded_positions_train": int(sum(
            int((v == 0).sum()) for k, v in arrays.items()
            if k.startswith("train/") and k.endswith("attention_mask_BCT"))),
        "distinct_train_token_digests": len({
            _digest(v) for k, v in arrays.items()
            if k.startswith("train/") and k.endswith("token_ids_BCT")}),
        "max_token_id": int(max(
            int(v.max()) for k, v in arrays.items() if k.endswith("token_ids_BCT"))),
    }
    arrays["__manifest__"] = np.asarray(json.dumps(manifest, sort_keys=True))
    np.savez_compressed(args.out, **arrays)
    print(f"[materialize] {manifest['counts']}", flush=True)
    print(f"[materialize] -> {args.out}", flush=True)


# --------------------------------------------------------------------- stage 2: arm

def _child_arm(args) -> None:
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["OMEGALAX_DELTANET_KERNEL"] = "pallas"

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from flax import nnx  # noqa: PLC0415
    from jax.sharding import NamedSharding, PartitionSpec  # noqa: PLC0415

    import omegalax  # noqa: PLC0415
    from omegalax.distributed.mesh import ensure_mesh, mesh_rules  # noqa: PLC0415
    from omegalax.models.qwen3_5 import deltanet as dn  # noqa: PLC0415
    from omegalax.models.qwen3_5.config import Qwen3_5TextConfig  # noqa: PLC0415
    from omegalax.models.qwen3_5.kernels import pallas_triton as pt  # noqa: PLC0415
    from omegalax.models.shard_config import ShardConfig  # noqa: PLC0415
    from omegalax.models.sharding_runtime import set_attn_backend  # noqa: PLC0415
    from omegalax.text import api as text_api  # noqa: PLC0415
    from omegalax.trainers import pretrain as P  # noqa: PLC0415
    from absl import flags as absl_flags  # noqa: PLC0415

    if not absl_flags.FLAGS.is_parsed():
        absl_flags.FLAGS(["gate4_pilot"])

    tree = args.tree.resolve()
    source = Path(omegalax.__file__).resolve()
    print(f"[{args.arm}] omegalax -> {source}", flush=True)
    source.relative_to(tree / "omegalax")

    # Dispatch counters: pass-through wrappers, installed identically in every arm.
    counters = {"forward": 0, "cascade_loss": 0, "conv_btc": 0, "conv_bct": 0,
                "blocked_inverse": 0}
    real_forward = text_api.forward_with_gdn_state

    def counting_forward(*a, **k):
        counters["forward"] += 1
        return real_forward(*a, **k)

    text_api.forward_with_gdn_state = counting_forward
    real_bct = dn._causal_depthwise_conv1d

    def counting_bct(*a, **k):
        counters["conv_bct"] += 1
        return real_bct(*a, **k)

    dn._causal_depthwise_conv1d = counting_bct
    optional = {}
    for module, name, key in ((dn, "_causal_depthwise_conv1d_btc", "conv_btc"),
                              (pt, "_blocked_unit_lower_inverse", "blocked_inverse")):
        real = getattr(module, name, None)
        optional[name] = real is not None
        if real is not None:
            def make(real_fn=real, counter_key=key):
                def wrapper(*a, **k):
                    counters[counter_key] += 1
                    return real_fn(*a, **k)
                return wrapper
            setattr(module, name, make())
    cascade_mod = getattr(P, "cascade", None)
    optional["trainers.cascade"] = cascade_mod is not None
    if cascade_mod is not None:
        real_cascade = cascade_mod.loss_stats

        def counting_cascade(*a, **k):
            counters["cascade_loss"] += 1
            return real_cascade(*a, **k)

        cascade_mod.loss_stats = counting_cascade

    cfg = Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        layer_types=("linear_attention", "linear_attention", "linear_attention",
                     "full_attention"),
        rope_theta=10_000,
        partial_rotary_factor=0.25,
        mrope_section=(4, 2, 2),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,     # production
        linear_num_key_heads=1,
        linear_num_value_heads=4,
        linear_value_head_dim=128,   # production
        intermediate_size=256,
        shd_cfg=ShardConfig.no_sharding(),
        dtype=jnp.bfloat16,
    )
    train_cfg = P.TrainConfig(
        seed=args.seed, batch_size=GLOBAL_BATCH, seq_len=CHUNK_LENGTH,
        num_steps=STEPS, learning_rate=3e-4, weight_decay=0.0, print_every=0,
    )
    mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
    model_cfg = text_api.align_config_to_mesh(cfg, mesh)
    init_key = jax.device_put(jax.random.key(args.seed),
                              NamedSharding(mesh, PartitionSpec()))
    init_key, _ = jax.random.split(init_key)
    model, model_cfg = P.init_model(model_cfg, init_key, tp_size=1, fsdp_size=1, dp_size=1)
    set_attn_backend(model, text_backend="cudnn")
    with mesh_rules(mesh):
        optimizer = P.build_optimizer(model, train_cfg)

    def param_vector():
        state = nnx.pure(nnx.state(optimizer.model, nnx.Param))
        parts = []
        for path, leaf in sorted(
            jax.tree_util.tree_flatten_with_path(state)[0],
            key=lambda kv: "/".join(str(getattr(p, "key", p)) for p in kv[0]),
        ):
            value = getattr(leaf, "value", leaf)
            parts.append(np.asarray(jax.device_get(value), dtype=np.float32).ravel())
        return np.concatenate(parts)

    init_params = param_vector()
    init_digest = _digest(init_params)
    print(f"[{args.arm}] initial params sha256[:32]={init_digest} "
          f"n={init_params.size}", flush=True)

    loss_kwargs = dict(bptt_chunks=SEGMENTS, pass_gdn_state=True,
                       pass_rope_positions=False, pass_conv_state=False)
    mode = P.PretrainMode.STATEPASSING_BPTT
    train_step = P.make_pretrain_train_step(mode, model_cfg, PAD_ID, **loss_kwargs)
    eval_step = P.make_pretrain_eval_step(mode, model_cfg, PAD_ID, **loss_kwargs)

    with np.load(args.batches, allow_pickle=False) as data:
        manifest = json.loads(str(data["__manifest__"]))

        def load(split, step):
            raw = {k: np.asarray(data[f"{split}/{step}/{k}"]) for k in BATCH_KEYS}
            batch, _, _ = P.prepare_pretrain_batch(
                raw, mode, model_cfg, mesh, global_batch_size=GLOBAL_BATCH
            )
            return batch

        train_batches = [load("train", i) for i in range(STEPS)]
        val_batches = [load("val", i) for i in range(N_VAL_BATCHES)]

    def evaluate():
        nll_sum, tokens = 0.0, 0.0
        for batch in val_batches:
            metrics = eval_step(optimizer.model, batch)
            nll_sum += float(jax.device_get(metrics["nll_sum"]))
            tokens += float(jax.device_get(metrics["supervised_tokens"]))
        nll = nll_sum / max(tokens, 1.0)
        return {"val_nll": nll, "val_ppl": float(np.exp(min(nll, 50.0))),
                "val_tokens": tokens}

    trajectory = []
    evals = {}
    if 0 in EVAL_AT:
        evals["step0"] = evaluate()
        print(f"[{args.arm}] eval@0 {evals['step0']}", flush=True)
    dispatch_first = None
    previous = init_params
    for step in range(1, STEPS + 1):
        for key in counters:
            counters[key] = 0
        loss, metrics = train_step(optimizer, train_batches[step - 1])
        loss_v = float(jax.device_get(loss))
        if step == 1:
            dispatch_first = dict(counters)
        current = param_vector()
        delta = current - previous
        row = {
            "step": step,
            "loss": loss_v,
            "nll": float(jax.device_get(metrics["nll"])),
            "grad_norm": float(jax.device_get(metrics["grad_norm"])),
            "aux_loss": float(jax.device_get(metrics["aux_loss"])),
            "supervised_tokens": float(jax.device_get(metrics["supervised_tokens"])),
            "update_l2": float(np.linalg.norm(delta)),
            "param_rel_l2_vs_init": float(
                np.linalg.norm(current - init_params) / max(np.linalg.norm(init_params), 1e-30)
            ),
            "loss_finite": bool(np.isfinite(loss_v)),
            "params_finite": bool(np.all(np.isfinite(current))),
        }
        trajectory.append(row)
        print(f"[{args.arm}] step {step:>2} loss={row['loss']:.6f} "
              f"nll={row['nll']:.6f} gnorm={row['grad_norm']:.6e} "
              f"upd={row['update_l2']:.6e}", flush=True)
        previous = current
        if step in EVAL_AT:
            evals[f"step{step}"] = evaluate()
            print(f"[{args.arm}] eval@{step} {evals[f'step{step}']}", flush=True)

    final_params = previous
    result = {
        "arm": args.arm, "seed": args.seed, "source": str(source), "tree": str(tree),
        "init_params_sha256_32": init_digest, "n_params": int(init_params.size),
        "final_params_sha256_32": _digest(final_params),
        "dispatch_step1": dispatch_first,
        "optional_symbols": optional,
        "switches": {
            "deltanet._CONV_BTC": bool(getattr(dn, "_CONV_BTC", False)),
            "pallas_triton._TRI_BLOCKED": bool(getattr(pt, "_TRI_BLOCKED", False)),
            "pallas_triton._TRI_INV": bool(getattr(pt, "_TRI_INV", False)),
            "cascade.enabled": bool(cascade_mod.enabled()) if cascade_mod else False,
            "cascade._SEGMENTS": getattr(cascade_mod, "_SEGMENTS", None) if cascade_mod else None,
        },
        "env": {k: os.environ.get(k, "") for k in
                ("MFU_CASCADE", "MFU_CONV_BTC", "MFU_TRIBLOCK", "MFU_TRIINV",
                 "OMEGALAX_DELTANET_KERNEL", "XLA_FLAGS")},
        "devices": [str(d) for d in jax.devices()],
        "batch_manifest_digests": manifest["digests"],
        "batch_counts": manifest["counts"],
        "trajectory": trajectory,
        "evals": evals,
        "any_nonfinite": not all(r["loss_finite"] and r["params_finite"] for r in trajectory),
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    # Final parameters are written as a transient vector so the parent can compute
    # paired differences; the archive is deleted with the scope. Not a checkpoint:
    # no optimizer state, no step counter, no restore path.
    np.savez_compressed(Path(str(args.out) + ".params.npz"), final=final_params)
    print(f"[{args.arm}] dispatch={dispatch_first} switches={result['switches']}", flush=True)
    print(f"[{args.arm}] -> {args.out}", flush=True)


# -------------------------------------------------------------------------- parent

def _run(args, arm: str, tree: Path, mode: str, out: Path, seed: int,
         optimized: bool, batches: Path | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tree)
    env["MFU_CASCADE"] = "1" if optimized else "0"
    env["MFU_CONV_BTC"] = "1" if optimized else "0"
    env["MFU_TRIBLOCK"] = "1" if optimized else "0"
    for name, value in (("MFU_TRIINV", "0"), ("MFU_BWD_PALLAS", "0"),
                        ("MFU_MU_DTYPE", "fp32"), ("MFU_FAST_CE", "0"),
                        ("MFU_REMAT", "none"), ("MFU_CHUNK", "64"),
                        ("MFU_TRI_BLOCK", "16"), ("MFU_PALLAS_WARPS", "4"),
                        ("MFU_PALLAS_STAGES", "2"), ("MFU_NUM_LOSS_TILES", "4"),
                        ("MFU_TRIINV_DEBUG", "0"), ("WANDB_MODE", "disabled"),
                        ("OMEGALAX_STARTUP_LOG", "0")):
        env[name] = value
    cmd = [str(ROOT / ".venv" / "bin" / "python"), str(Path(__file__).resolve()),
           "--child", "--mode", mode, "--arm", arm, "--tree", str(tree),
           "--out", str(out), "--seed", str(seed)]
    if batches is not None:
        cmd += ["--batches", str(batches)]
    print(f"\n=== {mode} arm={arm} tree={tree.name} seed={seed} optimized={optimized}",
          flush=True)
    subprocess.run(cmd, env=env, check=True)


def _parent(args: argparse.Namespace) -> int:
    if not INDEX_ROOT.is_dir():
        raise RuntimeError(f"index root not found: {INDEX_ROOT}")
    out_dir = Path(args.results).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    report: dict[str, object] = {
        "index_root": str(INDEX_ROOT),
        "roots": {"original": str(ORIGINAL), "copy": str(ROOT)},
        "config": {"segments": SEGMENTS, "chunk_length": CHUNK_LENGTH,
                   "global_batch": GLOBAL_BATCH, "steps": STEPS,
                   "vocab": VOCAB, "eos_id": EOS_ID, "pad_id": PAD_ID,
                   "hidden_size": 128, "layers": 4, "linear_key_head_dim": 128,
                   "linear_value_head_dim": 128, "deltanet_chunk": 64,
                   "learning_rate": 3e-4, "kernel": "pallas",
                   "attn_backend": "cudnn", "dtype": "bf16",
                   "eval_at": list(EVAL_AT), "val_batches": N_VAL_BATCHES,
                   "seeds": seeds, "xla_flags": os.environ.get("XLA_FLAGS", "")},
        "seeds": {},
    }
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="tmparr_", dir=HERE) as temp_dir:
        temp = Path(temp_dir)
        for seed in seeds:
            batches = temp / f"s{seed}_batches.npz"
            _run(args, "materialize", ROOT, "materialize", batches, seed, False)
            arms = {}
            for arm, tree, optimized in (("orig_a", ORIGINAL, False),
                                         ("orig_b", ORIGINAL, False),
                                         ("opt_all", ROOT, True)):
                path = temp / f"s{seed}_{arm}.json"
                _run(args, arm, tree, "arm", path, seed, optimized, batches=batches)
                arms[arm] = json.loads(path.read_text())
                arms[arm]["_params_path"] = str(path) + ".params.npz"

            digests = {a: r["init_params_sha256_32"] for a, r in arms.items()}
            if len(set(digests.values())) != 1:
                problems.append(f"seed{seed}: initial parameters differ across arms {digests}")
            batch_digests = {a: json.dumps(r["batch_manifest_digests"], sort_keys=True)
                             for a, r in arms.items()}
            if len(set(batch_digests.values())) != 1:
                problems.append(f"seed{seed}: batch digests differ across arms")
            expect = {"orig_a": (0, 0, False), "orig_b": (0, 0, False),
                      "opt_all": (1, 1, True)}
            for arm, (want_cascade, _, want_blocked) in expect.items():
                d = arms[arm]["dispatch_step1"] or {}
                if (d.get("cascade_loss", 0) > 0) != bool(want_cascade):
                    problems.append(f"seed{seed} {arm}: cascade dispatch {d}")
                if (d.get("blocked_inverse", 0) > 0) != want_blocked:
                    problems.append(f"seed{seed} {arm}: blocked-inverse dispatch {d}")
                if (d.get("conv_btc", 0) > 0) != want_blocked:
                    problems.append(f"seed{seed} {arm}: conv-BTC dispatch {d}")
                if arms[arm]["any_nonfinite"]:
                    problems.append(f"seed{seed} {arm}: NON-FINITE loss or parameters")

            vectors = {}
            for arm, r in arms.items():
                with np.load(r["_params_path"], allow_pickle=False) as pd:
                    vectors[arm] = np.asarray(pd["final"], dtype=np.float64)
                r.pop("_params_path")

            def pair(left: str, right: str) -> dict:
                a, b = vectors[left], vectors[right]
                d = np.abs(a - b)
                la = arms[left]["trajectory"]
                lb = arms[right]["trajectory"]
                return {
                    "param_final_max_abs": float(d.max()),
                    "param_final_rel_l2": float(
                        np.linalg.norm(d) / max(np.linalg.norm(a), 1e-30)),
                    "loss_abs_diff_per_step": [
                        abs(x["loss"] - y["loss"]) for x, y in zip(la, lb)],
                    "loss_max_abs_diff": max(
                        abs(x["loss"] - y["loss"]) for x, y in zip(la, lb)),
                    "grad_norm_max_rel_diff": max(
                        abs(x["grad_norm"] - y["grad_norm"]) / max(abs(x["grad_norm"]), 1e-30)
                        for x, y in zip(la, lb)),
                    "val_nll_abs_diff": {
                        k: abs(arms[left]["evals"][k]["val_nll"]
                               - arms[right]["evals"][k]["val_nll"])
                        for k in arms[left]["evals"]},
                }

            report["seeds"][f"seed{seed}"] = {
                "arms": arms,
                "control_orig_b_vs_orig_a": pair("orig_b", "orig_a"),
                "test_opt_all_vs_orig_a": pair("opt_all", "orig_a"),
            }

    report["problems"] = problems
    target = out_dir / f"gate4_summary_{args.tag}.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nsummary -> {target}")

    print("\n=== per-seed loss trajectory (step: orig_a / orig_b / opt_all)")
    for skey, entry in report["seeds"].items():
        print(f"-- {skey}")
        a, b, o = (entry["arms"][x]["trajectory"] for x in ("orig_a", "orig_b", "opt_all"))
        for i in range(STEPS):
            print(f"   {i+1:>2}  {a[i]['loss']:.6f}  {b[i]['loss']:.6f}  {o[i]['loss']:.6f}"
                  f"   gnorm {a[i]['grad_norm']:.4e} {b[i]['grad_norm']:.4e} "
                  f"{o[i]['grad_norm']:.4e}")
    print("\n=== fixed real validation NLL (orig_a / orig_b / opt_all)")
    for skey, entry in report["seeds"].items():
        for ek in sorted(entry["arms"]["orig_a"]["evals"]):
            vals = [entry["arms"][x]["evals"][ek]["val_nll"] for x in
                    ("orig_a", "orig_b", "opt_all")]
            print(f"{skey} {ek:<7} {vals[0]:.6f}  {vals[1]:.6f}  {vals[2]:.6f}")
    print("\n=== paired: test (opt vs orig_a) against control (orig_b vs orig_a)")
    for skey, entry in report["seeds"].items():
        c, t = entry["control_orig_b_vs_orig_a"], entry["test_opt_all_vs_orig_a"]
        print(f"{skey} loss_max|d|   control={c['loss_max_abs_diff']:.3e}  "
              f"test={t['loss_max_abs_diff']:.3e}")
        print(f"{skey} param_rel_l2  control={c['param_final_rel_l2']:.3e}  "
              f"test={t['param_final_rel_l2']:.3e}")
        for ek in sorted(t["val_nll_abs_diff"]):
            print(f"{skey} val_nll |d| {ek:<7} control={c['val_nll_abs_diff'][ek]:.3e}  "
                  f"test={t['val_nll_abs_diff'][ek]:.3e}")
    print("\n=== checks")
    if problems:
        for p in problems:
            print("FAIL ", p)
    else:
        print("ok    identical initial parameters, identical batches, expected dispatch, "
              "all values finite")
    print("\nNo checkpoints written; temporary arrays removed with the scope.")
    return 1 if problems else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--mode", choices=("materialize", "arm"), default="arm")
    p.add_argument("--arm", default="orig_a", help=argparse.SUPPRESS)
    p.add_argument("--tree", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--batches", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--tag", default="g4")
    p.add_argument(
        "--results",
        default="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs/mfu_opt/gate4",
    )
    args = p.parse_args()
    if args.child:
        if args.mode == "materialize":
            _child_materialize(args)
        else:
            _child_arm(args)
        return 0
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
