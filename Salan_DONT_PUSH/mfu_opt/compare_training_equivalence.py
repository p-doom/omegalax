"""Compare the real tiny-training path of omegalax and omegalax_optimization.

The parent process starts fresh child interpreters with PYTHONPATH pinned to
each tree.  This is essential because the optimization copy shares the
original's virtualenv, whose editable install otherwise imports the original
tree silently.  Each child runs the actual statepassing loss, autodiff, and
AdamW update for the same deterministic ten batches.  It writes only temporary
arrays for the duration of the comparison; it never creates or restores a
training checkpoint.

The default tiny shape deliberately preserves the branches used by the target
GPU run: C=2 statepassing, 64-token DeltaNet chunks, 128-wide DeltaNet key/value
heads, GQA, three linear-attention layers, and one full-attention layer.

Typical GPU invocation (normally through submit_training_equivalence.sh):

    PYTHONPATH=/.../omegalax_optimization \
      /.../omegalax_optimization/.venv/bin/python \
      Salan_DONT_PUSH/mfu_opt/compare_training_equivalence.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT.parent / "omegalax"
CHUNK = 64
SEGMENTS = 2
SEQ_LEN = 2 * CHUNK
DEFAULT_STEPS = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tree", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--platform", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--kernel", choices=("xla", "pallas"), default="pallas")
    parser.add_argument("--attn-backend", default="cudnn")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    return parser.parse_args()


def _child(args: argparse.Namespace) -> None:
    if args.tree is None or args.out is None:
        raise ValueError("--child requires --tree and --out")
    if args.batch % SEGMENTS:
        raise ValueError(f"--batch={args.batch} must be divisible by C={SEGMENTS}")

    # All feature switches are read while omegalax modules are imported.  Set them
    # before importing JAX or omegalax, rather than changing an already-imported
    # module in process.
    os.environ["JAX_PLATFORMS"] = args.platform
    os.environ["OMEGALAX_DELTANET_KERNEL"] = args.kernel

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from flax import nnx  # noqa: PLC0415
    from jax.sharding import NamedSharding, PartitionSpec  # noqa: PLC0415

    import omegalax  # noqa: PLC0415
    from omegalax.distributed.mesh import ensure_mesh, mesh_rules  # noqa: PLC0415
    from omegalax.models.qwen3_5.config import Qwen3_5TextConfig  # noqa: PLC0415
    from omegalax.models.shard_config import ShardConfig  # noqa: PLC0415
    from omegalax.models.sharding_runtime import set_attn_backend  # noqa: PLC0415
    from omegalax.text import api as text_api  # noqa: PLC0415
    from omegalax.trainers import pretrain as pretrain_trainer  # noqa: PLC0415
    from absl import flags  # noqa: PLC0415

    # Tokamax reads Abseil's global flags lazily when attention is first called.
    # argparse has already consumed this script's private flags, so do not let
    # Tokamax re-parse ``--child`` et al. from sys.argv.
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["tiny_training_equivalence"])

    tree = args.tree.resolve()
    source = Path(omegalax.__file__).resolve()
    try:
        source.relative_to(tree / "omegalax")
    except ValueError as exc:
        raise RuntimeError(
            f"PYTHONPATH selected {source}, expected a module below {tree / 'omegalax'}"
        ) from exc

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
    train_cfg = pretrain_trainer.TrainConfig(
        seed=args.seed,
        batch_size=args.batch,
        seq_len=SEQ_LEN,
        num_steps=args.steps,
        learning_rate=1e-3,
        weight_decay=0.0,
        print_every=0,
    )
    rng = np.random.default_rng(args.seed)
    raw_batch = {
        "token_ids_BCT": rng.integers(
            1, cfg.vocab_size - 1, size=(args.batch, SEGMENTS, SEQ_LEN), dtype=np.int32
        ),
        "attention_mask_BCT": np.ones((args.batch, SEGMENTS, SEQ_LEN), dtype=np.int32),
        "loss_mask_BCT": np.ones((args.batch, SEGMENTS, SEQ_LEN), dtype=np.int32),
        "reset_state_BC": np.tile(np.array([True, False]), (args.batch, 1)),
    }

    mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
    model_cfg = text_api.align_config_to_mesh(cfg, mesh)
    init_key = jax.device_put(
        jax.random.key(args.seed), NamedSharding(mesh, PartitionSpec())
    )
    init_key, _ = jax.random.split(init_key)
    model, model_cfg = pretrain_trainer.init_model(
        model_cfg, init_key, tp_size=1, fsdp_size=1, dp_size=1
    )
    set_attn_backend(model, text_backend=args.attn_backend)
    with mesh_rules(mesh):
        optimizer = pretrain_trainer.build_optimizer(model, train_cfg)
    batch, _, _ = pretrain_trainer.prepare_pretrain_batch(
        raw_batch,
        pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
        model_cfg,
        mesh,
        global_batch_size=args.batch,
    )

    # This is make_pretrain_train_step with one additional observable: the raw
    # gradients before optimizer.update().  Thus the loss and update are the
    # production code path, while the test can distinguish FP from BP drift.
    @nnx.jit(donate_argnums=0)
    def observed_train_step(optimizer, batch):
        def loss_fn(model):
            return pretrain_trainer._statepassing_loss_stats(
                model,
                batch,
                model_cfg,
                0,
                pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
                bptt_chunks=SEGMENTS,
                pass_gdn_state=True,
                pass_rope_positions=False,
                pass_conv_state=False,
            )

        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(optimizer.model)
        # Returning VariableState metadata from a jitted GPU function triggers an
        # XLA lowering failure. The comparison needs the exact arrays only, and
        # ``pure`` is also what the production optimizer uses before Optax.
        grad_state = nnx.pure(nnx.state(grads, nnx.Param))
        optimizer.update(grads)
        param_state = nnx.pure(nnx.state(optimizer.model, nnx.Param))
        return loss, metrics, grad_state, param_state

    def state_arrays(state):
        arrays = {}
        for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
            value = getattr(leaf, "value", leaf)
            key = "/".join(str(getattr(part, "key", part)) for part in path)
            array = jax.device_get(value)
            # NumPy serializes JAX bfloat16 as an opaque two-byte record. Cast
            # while the JAX dtype is still known so the temporary .npz contains
            # comparable numeric values (the conversion is exact for bf16).
            arrays[key] = np.asarray(
                array, dtype=np.float32 if jnp.issubdtype(array.dtype, jnp.inexact) else None
            )
        return arrays

    result: dict[str, np.ndarray] = {}
    for name, value in state_arrays(nnx.pure(nnx.state(optimizer.model, nnx.Param))).items():
        result[f"params/0/{name}"] = value
    for step in range(1, args.steps + 1):
        loss, metrics, gradients, params = observed_train_step(optimizer, batch)
        result[f"forward/{step}/loss"] = np.asarray(jax.device_get(loss))
        for name in ("nll", "nll_sum", "aux_loss"):
            result[f"forward/{step}/{name}"] = np.asarray(jax.device_get(metrics[name]))
        for name, value in state_arrays(gradients).items():
            result[f"grads/{step}/{name}"] = value
        for name, value in state_arrays(params).items():
            result[f"params/{step}/{name}"] = value

    meta = {
        "source": str(source),
        "devices": [str(device) for device in jax.devices()],
        "steps": args.steps,
        "batch": args.batch,
        "env": {
            name: os.environ.get(name, "")
            for name in ("MFU_CASCADE", "MFU_CONV_BTC", "MFU_TRIBLOCK", "MFU_TRIINV")
        },
    }
    result["__meta__"] = np.asarray(json.dumps(meta, sort_keys=True))
    np.savez_compressed(args.out, **result)
    print(f"[child] omegalax -> {source}", flush=True)
    print(f"[child] devices={meta['devices']} arrays={len(result) - 1} -> {args.out}", flush=True)


def _child_command(args: argparse.Namespace, tree: Path, out: Path) -> list[str]:
    return [
        str(ROOT / ".venv" / "bin" / "python"),
        str(Path(__file__).resolve()),
        "--child",
        "--tree",
        str(tree),
        "--out",
        str(out),
        "--steps",
        str(args.steps),
        "--batch",
        str(args.batch),
        "--seed",
        str(args.seed),
        "--platform",
        args.platform,
        "--kernel",
        args.kernel,
        "--attn-backend",
        args.attn_backend,
        "--dtype",
        args.dtype,
    ]


def _run_child(args: argparse.Namespace, label: str, tree: Path, out: Path, *, optimized: bool) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tree)
    env["MFU_CASCADE"] = "1" if optimized else "0"
    env["MFU_CONV_BTC"] = "1" if optimized else "0"
    env["MFU_TRIBLOCK"] = "1" if optimized else "0"
    env["MFU_TRIINV"] = "0"
    env["MFU_BWD_PALLAS"] = "0"
    env["MFU_MU_DTYPE"] = "fp32"
    env["MFU_FAST_CE"] = "0"
    env["WANDB_MODE"] = "disabled"
    env["OMEGALAX_STARTUP_LOG"] = "0"
    print(f"\n=== {label}: tree={tree} optimized_flags={optimized}", flush=True)
    subprocess.run(_child_command(args, tree, out), env=env, check=True)


def _meta(data: np.lib.npyio.NpzFile) -> dict[str, object]:
    return json.loads(str(data["__meta__"]))


def _max_difference(left: np.ndarray, right: np.ndarray) -> tuple[float, float, bool]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return float("inf"), float("inf"), False
    absolute = float(np.max(np.abs(left - right))) if left.size else 0.0
    scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-30)
    return absolute, absolute / scale, bool(np.array_equal(left, right))


def _group_difference(
    left: np.lib.npyio.NpzFile,
    right: np.lib.npyio.NpzFile,
    prefix: str,
) -> tuple[float, float, str, bool, bool]:
    keys = sorted(key for key in left.files if key.startswith(prefix))
    if keys != sorted(key for key in right.files if key.startswith(prefix)):
        raise RuntimeError(f"{prefix}: tensor sets differ between the two trees")
    worst_abs, worst_rel, worst_name = 0.0, 0.0, ""
    all_exact = True
    finite = True
    for key in keys:
        absolute, relative, exact = _max_difference(left[key], right[key])
        if not np.isfinite(absolute):
            finite = False
        if absolute > worst_abs or relative > worst_rel:
            worst_abs, worst_rel, worst_name = absolute, relative, key
        all_exact = all_exact and exact
    return worst_abs, worst_rel, worst_name, all_exact, finite


def _update_difference(
    left: np.lib.npyio.NpzFile,
    right: np.lib.npyio.NpzFile,
    step: int,
) -> tuple[float, float, str, bool, bool]:
    prefix = f"params/{step}/"
    previous = f"params/{step - 1}/"
    keys = sorted(key for key in left.files if key.startswith(prefix))
    worst_abs, worst_rel, worst_name = 0.0, 0.0, ""
    all_exact = True
    finite = True
    for key in keys:
        name = key[len(prefix) :]
        left_update = np.asarray(left[key], dtype=np.float64) - np.asarray(
            left[f"{previous}{name}"], dtype=np.float64
        )
        right_update = np.asarray(right[key], dtype=np.float64) - np.asarray(
            right[f"{previous}{name}"], dtype=np.float64
        )
        absolute, relative, exact = _max_difference(left_update, right_update)
        if not np.isfinite(absolute):
            finite = False
        if absolute > worst_abs or relative > worst_rel:
            worst_abs, worst_rel, worst_name = absolute, relative, name
        all_exact = all_exact and exact
    return worst_abs, worst_rel, worst_name, all_exact, finite


def _print_comparison(
    label: str, left: np.lib.npyio.NpzFile, right: np.lib.npyio.NpzFile, args: argparse.Namespace
) -> bool:
    left_meta, right_meta = _meta(left), _meta(right)
    print(f"\n--- {label}")
    print(f"left : {left_meta['source']}")
    print(f"right: {right_meta['source']}")
    print("step  forward scalars max(abs,rel)  backward grad max(abs,rel)  update max(abs,rel)  params max(abs,rel)")
    passed = True
    for step in range(1, args.steps + 1):
        forward = _group_difference(left, right, f"forward/{step}/")
        backward = _group_difference(left, right, f"grads/{step}/")
        update = _update_difference(left, right, step)
        params = _group_difference(left, right, f"params/{step}/")
        groups = (forward, backward, update, params)
        step_pass = all(
            finite and (absolute <= args.atol or relative <= args.rtol)
            for absolute, relative, _, _, finite in groups
        )
        passed = passed and step_pass
        marker = "PASS" if step_pass else "FAIL"
        print(
            f"{step:>4}  {forward[0]:.3e},{forward[1]:.3e}"
            f"  {backward[0]:.3e},{backward[1]:.3e}"
            f"  {update[0]:.3e},{update[1]:.3e}"
            f"  {params[0]:.3e},{params[1]:.3e}  {marker}"
        )
    final_params = _group_difference(left, right, f"params/{args.steps}/")
    exact = all(
        _group_difference(left, right, f"forward/{step}/")[3]
        and _group_difference(left, right, f"grads/{step}/")[3]
        and _group_difference(left, right, f"params/{step}/")[3]
        for step in range(1, args.steps + 1)
    )
    print(
        f"worst final parameter tensor: {final_params[2]} "
        f"abs={final_params[0]:.3e} rel={final_params[1]:.3e}; "
        f"bitwise_identical={exact}"
    )
    return passed


def _parent(args: argparse.Namespace) -> int:
    if args.platform == "cpu" and (args.kernel != "xla" or args.attn_backend != "xla"):
        raise ValueError("CPU sanity runs require --kernel=xla --attn-backend=xla")
    if not ORIGINAL.is_dir():
        raise RuntimeError(f"Original tree not found: {ORIGINAL}")

    # Temporary files hold exact tensors for direct, elementwise comparisons; a
    # checksum alone could hide compensating errors.  The directory is deleted at
    # scope exit and is never an Orbax/training checkpoint.
    with tempfile.TemporaryDirectory(prefix="tiny_equiv_", dir=ROOT) as temp_dir:
        temp = Path(temp_dir)
        paths = {
            "orig_1": temp / "orig_1.npz",
            "orig_2": temp / "orig_2.npz",
            "optimized": temp / "optimized.npz",
        }
        _run_child(args, "original control 1", ORIGINAL, paths["orig_1"], optimized=False)
        _run_child(args, "original control 2", ORIGINAL, paths["orig_2"], optimized=False)
        _run_child(args, "optimization flags on", ROOT, paths["optimized"], optimized=True)
        with (
            np.load(paths["orig_1"], allow_pickle=False) as orig_1,
            np.load(paths["orig_2"], allow_pickle=False) as orig_2,
            np.load(paths["optimized"], allow_pickle=False) as optimized,
        ):
            control_ok = _print_comparison("determinism control: original vs original", orig_1, orig_2, args)
            optimized_ok = _print_comparison(
                "equivalence: original vs optimization flags", orig_1, optimized, args
            )
    if control_ok and optimized_ok:
        print(f"\nPASS within atol={args.atol:g}, rtol={args.rtol:g}; no checkpoints were written.")
        return 0
    print(f"\nFAIL at atol={args.atol:g}, rtol={args.rtol:g}; see the first failing step above.")
    return 1


def main() -> int:
    args = _parse_args()
    if args.child:
        _child(args)
        return 0
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
