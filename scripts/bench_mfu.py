#!/usr/bin/env python3
"""MFU benchmark on 2x H100.

Runs a full forward+backward+optimizer step with a dense Qwen3-like model
using randomly-initialized weights and constant-data batches.

Usage:
    python scripts/bench_mfu.py --tp_size=2 --fsdp_size=1 --dp_size=1
    python scripts/bench_mfu.py --tp_size=1 --fsdp_size=2 --dp_size=1
"""

from __future__ import annotations

import dataclasses
import datetime
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.models.qwen3.config import Qwen3Config
from omegalax.models.shard_config import ShardConfig, axis_rules_for_mesh, shard_config_for_mesh
from omegalax.models.sharding_runtime import (
    init_model_sharded,
    shard_batch_dict as runtime_shard_batch_dict,
)
from omegalax.models.qwen3.model import Qwen3
from omegalax.text import api as text_api
from omegalax.trainers.loss import chunked_cross_entropy_loss
from omegalax.trainers.optim import MixedPrecisionOptimizer
from omegalax.trainers.perf import (
    per_device_flops_per_step,
    step_metrics,
    PEAK_TFLOPS,
)

P = PartitionSpec
FLAGS = flags.FLAGS

flags.DEFINE_integer("tp_size", 1, "Tensor parallelism size.")
flags.DEFINE_integer("fsdp_size", 1, "FSDP size.")
flags.DEFINE_integer("dp_size", 1, "Data parallelism size.")
flags.DEFINE_integer("batch_size", 4, "Global batch size.")
flags.DEFINE_integer("seq_len", 2048, "Sequence length.")
flags.DEFINE_integer("warmup_steps", 3, "Warmup steps (include JIT compile).")
flags.DEFINE_integer("measure_steps", 5, "Steps to average for MFU.")
flags.DEFINE_string("jax_cache", "/tmp/jax_cache_bench", "JAX compilation cache dir.")


# ~4B parameter dense Qwen3-style model.
# Fits in 63.8 GB/device even with fully-replicated fp32 Adam states
# (~4B * 10 bytes = 40 GB vs 63.8 GB limit).
_MODEL_CFG = dict(
    num_layers=28,
    vocab_size=151_936,
    emb_dim=3072,
    mlp_dim=9216,
    num_heads=24,        # divisible by tp=2
    head_dim=128,
    num_kv_heads=8,      # divisible by tp=2
    rope_theta=1_000_000,
    rope_scaling_factor=None,
    local_rope_theta=None,
    norm_eps=1e-6,
    tie_word_embeddings=False,
)


def make_cfg(mesh: Mesh) -> Qwen3Config:
    cfg = Qwen3Config(**_MODEL_CFG, shd_cfg=ShardConfig.default())
    return dataclasses.replace(cfg, shd_cfg=shard_config_for_mesh(cfg.shd_cfg, mesh))


def mem_gb(device) -> float:
    s = device.memory_stats()
    if s is None:
        return -1.0
    return s.get("bytes_in_use", 0) / 1e9


def peak_mem_gb(device) -> float:
    s = device.memory_stats()
    if s is None:
        return -1.0
    return s.get("peak_bytes_in_use", 0) / 1e9


def run_benchmark(tp_size: int, fsdp_size: int, dp_size: int, batch_size: int, seq_len: int,
                  warmup: int, measure: int) -> dict:
    tag = f"tp={tp_size} fsdp={fsdp_size} dp={dp_size} bs={batch_size} T={seq_len}"
    sep = "=" * 64
    print(f"\n{sep}\n  {tag}\n{sep}", flush=True)

    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    cfg = make_cfg(mesh)
    axis_rules = axis_rules_for_mesh(mesh)

    # ── model init ────────────────────────────────────────────────────────────
    print("  [1/4] init model ...", flush=True)
    t0 = time.time()
    rng = jax.device_put(jax.random.key(0), NamedSharding(mesh, P()))
    model = init_model_sharded(Qwen3, cfg, rng, mesh, axis_rules)
    jax.effects_barrier()
    mem_after_model = [mem_gb(d) for d in jax.devices()]
    n_params = sum(np.prod(v.shape) for v in jax.tree.leaves(nnx.state(model)) if hasattr(v, "shape"))
    print(f"  model init: {time.time()-t0:.1f}s  params={n_params/1e9:.2f}B  "
          f"gpu_mem={[f'{m:.1f}' for m in mem_after_model]} GB", flush=True)

    # ── optimizer init ────────────────────────────────────────────────────────
    print("  [2/4] init optimizer ...", flush=True)
    t0 = time.time()
    with mesh_rules(mesh):
        tx = optax.adamw(learning_rate=1e-4, weight_decay=0.01, mu_dtype=jnp.float32)
        optimizer = MixedPrecisionOptimizer(model, tx)
    optimizer_graphdef = nnx.graphdef(optimizer)
    optimizer_state = nnx.state(optimizer)
    replicated = NamedSharding(mesh, P())
    optimizer_state_sharding = jax.tree.map(
        lambda leaf: leaf.sharding if isinstance(leaf, jax.Array) else replicated,
        optimizer_state,
    )
    jax.effects_barrier()
    mem_after_opt = [mem_gb(d) for d in jax.devices()]
    print(f"  opt init: {time.time()-t0:.1f}s  "
          f"gpu_mem={[f'{m:.1f}' for m in mem_after_opt]} GB", flush=True)

    # ── build jit step ────────────────────────────────────────────────────────
    def _step(opt_state, batch):
        optimizer = nnx.merge(optimizer_graphdef, opt_state)
        token_ids_BT = batch["token_ids_BT"]
        loss_mask_BT = batch["loss_mask_BT"]

        def loss_fn(model):
            hidden_BTD, aux_loss = text_api.forward(model, token_ids_BT, pad_id=0, cfg=cfg)
            lm_weight = model.lm_head.kernel[...]
            loss = chunked_cross_entropy_loss(
                hidden_BTD, lm_weight, token_ids_BT, loss_mask_BT, num_tiles=_num_tiles,
                logits_out_sharding=cfg.shd_cfg.logits_btv,
            ) + aux_loss
            supervised_tokens = jnp.sum(loss_mask_BT[:, 1:].astype(jnp.float32))
            return loss, supervised_tokens

        (loss, supervised_tokens), grads = nnx.value_and_grad(loss_fn, has_aux=True)(optimizer.model)
        optimizer.update(grads)
        metrics = {"loss": loss, "grad_norm": optax.tree.norm(grads)}
        return nnx.state(optimizer), metrics

    train_step = jax.jit(
        _step,
        out_shardings=(optimizer_state_sharding, replicated),
        donate_argnums=(0,),
    )

    # Scale CE-loss tiles so per-device logits stay ≤ ~1 GiB fp32.
    # Batch is sharded across dp*fsdp combined.
    dp = max(1, int(mesh.shape.get("dp", 1)))
    fsdp = max(1, int(mesh.shape.get("fsdp", 1)))
    b_local = batch_size // (dp * fsdp)
    t1 = seq_len - 1
    max_chunk = (1 * 1024**3 // 4) // max(1, b_local * cfg.vocab_size)
    _num_tiles = max(4, -(-t1 // max(1, max_chunk)))
    print(f"  CE loss tiles={_num_tiles} (chunk≈{-(-t1//_num_tiles)} per tile, "
          f"B_local={b_local})", flush=True)

    batch_raw = {
        "token_ids_BT": np.ones((batch_size, seq_len), dtype=np.int32),
        "attention_mask_BT": np.ones((batch_size, seq_len), dtype=np.int32),
        "loss_mask_BT": np.ones((batch_size, seq_len), dtype=np.int32),
    }
    batch = runtime_shard_batch_dict(batch_raw, cfg.shd_cfg, mesh)

    per_device_flops = per_device_flops_per_step(cfg, seq_len, batch_size)
    global_tokens = seq_len * batch_size

    # ── warmup (includes JIT compile) ─────────────────────────────────────────
    print(f"  [3/4] warmup ({warmup} steps, first = compile) ...", flush=True)
    t0 = time.time()
    for step in range(warmup):
        optimizer_state, metrics = train_step(optimizer_state, batch)
    jax.tree.map(lambda x: x.block_until_ready(), (optimizer_state, metrics))
    mem_after_step = [mem_gb(d) for d in jax.devices()]
    peak_mem = [peak_mem_gb(d) for d in jax.devices()]
    print(f"  warmup done: {time.time()-t0:.1f}s  "
          f"gpu_mem={[f'{m:.1f}' for m in mem_after_step]} GB  "
          f"peak={[f'{m:.1f}' for m in peak_mem]} GB", flush=True)

    # ── timed steps ───────────────────────────────────────────────────────────
    print(f"  [4/4] measuring {measure} steps ...", flush=True)
    t_start = time.time()
    for _ in range(measure):
        optimizer_state, metrics = train_step(optimizer_state, batch)
    jax.tree.map(lambda x: x.block_until_ready(), (optimizer_state, metrics))
    elapsed = time.time() - t_start

    step_time_s = elapsed / measure
    peak_tflops = PEAK_TFLOPS["h100_sxm"]
    m = step_metrics(per_device_flops, datetime.timedelta(seconds=step_time_s), global_tokens, peak_tflops)
    loss_val = float(np.array(metrics["loss"]))

    print(f"\n  ── Results ─────────────────────────────────────────────")
    print(f"  model params:             {n_params/1e9:.2f} B")
    print(f"  step_time:                {m['step_time_s']:.3f} s")
    print(f"  tokens/s (global):        {m['global_tokens_per_sec']:.0f}")
    print(f"  tokens/s/device:          {m['tokens_per_sec_per_device']:.0f}")
    print(f"  TFLOPS/device:            {m['tflops_per_device']:.1f}")
    print(f"  MFU:                      {m['mfu']*100:.2f}%  (peak={peak_tflops} TFLOPS)")
    print(f"  loss (sanity):            {loss_val:.4f}")
    print(flush=True)
    return m


def main(_):
    jax.config.update("jax_compilation_cache_dir", FLAGS.jax_cache)
    print(f"devices: {jax.devices()}")
    for d in jax.devices():
        s = d.memory_stats()
        if s:
            print(f"  {d}: limit={s.get('bytes_limit',0)/1e9:.1f} GB")
        else:
            print(f"  {d}: (no mem stats with this allocator)")

    run_benchmark(
        tp_size=FLAGS.tp_size,
        fsdp_size=FLAGS.fsdp_size,
        dp_size=FLAGS.dp_size,
        batch_size=FLAGS.batch_size,
        seq_len=FLAGS.seq_len,
        warmup=FLAGS.warmup_steps,
        measure=FLAGS.measure_steps,
    )


if __name__ == "__main__":
    app.run(main)
