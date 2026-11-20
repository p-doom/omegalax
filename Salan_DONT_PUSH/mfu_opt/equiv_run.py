"""Run a few deterministic training steps and dump a fingerprint of the result.

Executed once against the original codebase and once against the optimized copy
(selected via PYTHONPATH). ``equiv_check.py`` compares the two dumps.

The model is tiny, but every shape constant that drives a *branch* in the code is
held at its production value. That rule comes from a near-miss in the earlier GDN
kernel work, where the test ran chunk_size=16 against production chunk_size=64
and therefore never executed the merge levels of the blocked inversion — the one
code path the change had added.

Held at production value here:
  chunk_size             64   (delta rule chunking; seq_len is a multiple > 1)
  linear_key_head_dim   128
  linear_value_head_dim 128
  full_attention_interval 4   (needs >= 4 layers so a full-attention layer exists)
  bptt_chunks             2   (statepassing with C > 1, so state is actually carried)
  gqa factor              4   (linear_num_value_heads / linear_num_key_heads)

Deliberately shrunk (they scale cost, they do not select code paths):
  vocab_size, hidden_size, intermediate_size, num_hidden_layers, batch_size

What is compared is NOT the loss after N steps. The earlier work showed that to be
useless as an equivalence probe: run-to-run nondeterminism in XLA reduction order
gets amplified over optimizer updates, to the point where a deliberately
lower-precision bf16 variant landed closer to baseline than an exact one. We dump
per-parameter checksums of the final state instead, which pin down the whole
trajectory, plus the metrics.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402

from omegalax.models.qwen3_5.config import Qwen3_5TextConfig  # noqa: E402
from omegalax.models.shard_config import ShardConfig  # noqa: E402
from omegalax.trainers import pretrain as pretrain_trainer  # noqa: E402

CHUNK = 64  # production delta-rule chunk size
SEGMENTS = 2  # statepassing chains per sample (C)
SEQ_LEN = 2 * CHUNK  # two chunks per segment, so the scan runs more than once


def build_config(dtype: str = "fp32") -> Qwen3_5TextConfig:
    # 4 layers with full_attention_interval=4 -> layer 3 is full attention.
    return Qwen3_5TextConfig(
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
        dtype=jnp.bfloat16 if dtype == "bf16" else jnp.float32,
    )


def make_batch(rng: np.random.Generator, cfg: Qwen3_5TextConfig, batch: int):
    shape = (batch, SEGMENTS, SEQ_LEN)
    return {
        "token_ids_BCT": rng.integers(1, cfg.vocab_size - 1, size=shape, dtype=np.int32),
        "attention_mask_BCT": np.ones(shape, dtype=np.int32),
        "loss_mask_BCT": np.ones(shape, dtype=np.int32),
        "chunk_idx_BC": np.tile(np.arange(SEGMENTS, dtype=np.int32), (batch, 1)),
        "reset_state_BC": np.tile(
            np.array([True] + [False] * (SEGMENTS - 1), dtype=np.bool_), (batch, 1)
        ),
        "metadata": {"doc_ids": [f"equiv-{i}" for i in range(batch)]},
    }


def batch_iterator(batch):
    while True:
        yield {k: (v.copy() if hasattr(v, "copy") else v) for k, v in batch.items()}


def fingerprint(model) -> dict[str, list[float]]:
    """Per-parameter checksums. Order-stable, shape-independent, cheap to compare."""
    out: dict[str, list[float]] = {}
    try:
        state = nnx.state(model, nnx.Param)
    except Exception:
        return out
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        value = getattr(leaf, "value", leaf)
        arr = np.asarray(jax.device_get(value), dtype=np.float64)
        if arr.size == 0:
            continue
        key = "/".join(str(getattr(p, "key", p)) for p in path)
        out[key] = [
            float(arr.sum()),
            float((arr * arr).sum()),
            float(np.abs(arr).max()),
            int(arr.size),
        ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kernel", default="xla", choices=["pallas", "xla"])  # CPU: nur xla moeglich
    # tokamax's mosaic_gpu attention has no CPU implementation; "xla" runs the
    # same math everywhere. On GPU the production backend should be used instead.
    ap.add_argument("--attn-backend", default="xla")
    # cuDNN refuses fp32 ("Q must be fp16/bf16/fp8..."), so testing that backend
    # needs a bf16 model. Comparing two trees against each other stays valid at
    # bf16; only the absolute exactness of the reference drops.
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    # The cascade requires both of these to be False; with them True the
    # dispatch falls back to the production loop and the test would be vacuous.
    ap.add_argument("--pass-rope", default="1", choices=["0", "1"])
    ap.add_argument("--pass-conv", default="1", choices=["0", "1"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["OMEGALAX_DELTANET_KERNEL"] = args.kernel

    from absl import flags

    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["equiv_run"])

    cfg = build_config(args.dtype)
    train_cfg = pretrain_trainer.TrainConfig(
        seed=args.seed,
        batch_size=args.batch,
        seq_len=SEQ_LEN,
        num_steps=args.steps,
        learning_rate=1e-3,
        weight_decay=0.0,
        print_every=0,
    )
    batch = make_batch(np.random.default_rng(args.seed), cfg, args.batch)

    model, metrics = pretrain_trainer.run_pretrain(
        cfg,
        train_cfg,
        batch_iterator(batch),
        pass_gdn_state=True,
        pretrain_mode=pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
        log_every=0,
        tp_size=1,
        fsdp_size=1,
        dp_size=1,
        bptt_chunks=SEGMENTS,
        pass_rope_positions=(args.pass_rope == "1"),
        pass_conv_state=(args.pass_conv == "1"),
        text_attn_backend=args.attn_backend,
    )

    payload = {
        "source": os.path.dirname(os.path.dirname(pretrain_trainer.__file__)),
        "args": vars(args),
        "metrics": {k: (float(v) if np.isscalar(v) else str(v)) for k, v in metrics.items()},
        "cascade_env": os.environ.get("MFU_CASCADE", "0"),
        "params": fingerprint(model),
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[equiv_run] source={payload['source']}")
    print(f"[equiv_run] {len(payload['params'])} Parameter-Tensoren, "
          f"{len(payload['metrics'])} Metriken -> {args.out}")


if __name__ == "__main__":
    main()
