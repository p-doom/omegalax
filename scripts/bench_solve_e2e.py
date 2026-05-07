#!/usr/bin/env python3
"""End-to-end benchmark: solve_triangular vs linalg.solve in Qwen3.5 deltanet.

Runs full forward+backward+optimizer steps with a random-init Qwen3.5-9B
text model (FSDP=8) and reports step times.

Usage:
    # 1) baseline (current code with linalg.solve):
    python scripts/bench_solve_e2e.py --variant=solve

    # 2) solve_triangular:
    python scripts/bench_solve_e2e.py --variant=tri
"""

from __future__ import annotations

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
from omegalax.models.shard_config import axis_rules_for_mesh, shard_config_for_mesh
from omegalax.models.sharding_runtime import (
    init_model_sharded,
    shard_batch_dict as runtime_shard_batch_dict,
)
from omegalax.text import api as text_api
from omegalax.trainers.loss import chunked_cross_entropy_loss
from omegalax.trainers.optim import MixedPrecisionOptimizer
from omegalax.trainers.perf import per_device_flops_per_step, step_metrics, PEAK_TFLOPS

P = PartitionSpec
FLAGS = flags.FLAGS

flags.DEFINE_string("model_id", "Qwen/Qwen3.5-9B", "HF model id for config resolution.")
flags.DEFINE_enum("variant", "solve", ["solve", "tri"], "Which solver to benchmark.")
flags.DEFINE_integer("tp_size", 1, "Tensor parallelism size.")
flags.DEFINE_integer("fsdp_size", 8, "FSDP size.")
flags.DEFINE_integer("dp_size", 1, "Data parallelism size.")
flags.DEFINE_integer("batch_size", 8, "Global batch size.")
flags.DEFINE_integer("seq_len", 2048, "Sequence length.")
flags.DEFINE_integer("warmup_steps", 3, "Warmup steps (include JIT compile).")
flags.DEFINE_integer("measure_steps", 10, "Steps to average.")
flags.DEFINE_string("jax_cache", "/tmp/jax_cache_bench_solve", "JAX compilation cache dir.")


def _monkey_patch_solver(variant: str):
    """Replace the solver in chunk_gated_delta_rule before model init."""
    import omegalax.models.qwen3_5.deltanet as dn
    import types

    # Read the current source to understand the function, then replace chunk_gated_delta_rule
    original_fn = dn.chunk_gated_delta_rule

    if variant == "tri":
        import jax.scipy.linalg as jsp_linalg

        def chunk_gated_delta_rule_patched(
            q_BTHA, k_BTHA, v_BTHU, g_BTH, beta_BTH, chunk_size=64,
        ):
            q_BTHA = dn._l2norm(q_BTHA, axis=-1)
            k_BTHA = dn._l2norm(k_BTHA, axis=-1)

            q_BHTA, k_BHTA, v_BHTU = [x.transpose(0, 2, 1, 3).astype(jnp.float32) for x in (q_BTHA, k_BTHA, v_BTHU)]
            beta_BHT = beta_BTH.transpose(0, 2, 1).astype(jnp.float32)
            g_BHT = g_BTH.transpose(0, 2, 1).astype(jnp.float32)

            B, H, T, A = k_BHTA.shape
            U = v_BHTU.shape[-1]

            pad_size = (chunk_size - T % chunk_size) % chunk_size
            if pad_size > 0:
                q_BHTA = jnp.pad(q_BHTA, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
                k_BHTA = jnp.pad(k_BHTA, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
                v_BHTU = jnp.pad(v_BHTU, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
                beta_BHT = jnp.pad(beta_BHT, ((0, 0), (0, 0), (0, pad_size)))
                g_BHT = jnp.pad(g_BHT, ((0, 0), (0, 0), (0, pad_size)))
            total_T = T + pad_size

            scale = A ** -0.5
            q_BHTA = q_BHTA * scale

            vb_BHTU = v_BHTU * beta_BHT[..., None]
            kb_BHTA = k_BHTA * beta_BHT[..., None]

            J = total_T // chunk_size
            q_BHJLA = q_BHTA.reshape(B, H, J, chunk_size, A)
            k_BHJLA = k_BHTA.reshape(B, H, J, chunk_size, A)
            v_BHJLU = v_BHTU.reshape(B, H, J, chunk_size, U)
            kb_BHJLA = kb_BHTA.reshape(B, H, J, chunk_size, A)
            vb_BHJLU = vb_BHTU.reshape(B, H, J, chunk_size, U)
            g_BHJL = g_BHT.reshape(B, H, J, chunk_size)

            g_BHJL = jnp.cumsum(g_BHJL, axis=-1)

            g_row = g_BHJL[..., :, None]
            g_col = g_BHJL[..., None, :]
            diff = g_row - g_col
            tril_mask = jnp.tril(jnp.ones((chunk_size, chunk_size)))
            decay_mask_LM = jnp.exp(diff * tril_mask) * tril_mask

            upper_mask_LM = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
            attn_BHJLM = -(jnp.einsum("BHJLA,BHJMA->BHJLM", kb_BHJLA, k_BHJLA) * decay_mask_LM)
            attn_BHJLM = jnp.where(upper_mask_LM, 0.0, attn_BHJLM)

            eye_LM = jnp.eye(chunk_size, dtype=attn_BHJLM.dtype)
            lhs_BHJLM = eye_LM - attn_BHJLM
            rhs_BHJLM = jnp.broadcast_to(eye_LM, lhs_BHJLM.shape)
            # >>> THE ONLY DIFFERENCE: use solve_triangular <<<
            attn_BHJLM = jsp_linalg.solve_triangular(lhs_BHJLM, rhs_BHJLM, lower=True)

            v_corrected_BHJLU = jnp.einsum("BHJLM,BHJMU->BHJLU", attn_BHJLM, vb_BHJLU)
            k_cumdecay_BHJLA = jnp.einsum("BHJLM,BHJMA->BHJLA", attn_BHJLM, kb_BHJLA * jnp.exp(g_BHJL)[..., None])

            state_BHAU = jnp.zeros((B, H, A, U), dtype=jnp.float32)
            upper_mask_1_LM = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=1)

            def chunk_step(carry, chunk_idx):
                st_BHAU = carry
                q_j_BHLA = q_BHJLA[:, :, chunk_idx]
                k_j_BHMA = k_BHJLA[:, :, chunk_idx]
                v_j_BHLU = v_corrected_BHJLU[:, :, chunk_idx]
                g_j_BHL = g_BHJL[:, :, chunk_idx]
                kcd_j_BHLA = k_cumdecay_BHJLA[:, :, chunk_idx]
                dm_j_LM = decay_mask_LM[:, :, chunk_idx]

                intra_BHLM = (jnp.einsum("BHLA,BHMA->BHLM", q_j_BHLA, k_j_BHMA) * dm_j_LM)
                intra_BHLM = jnp.where(upper_mask_1_LM, 0.0, intra_BHLM)

                v_prime_BHLU = jnp.einsum("BHLA,BHAU->BHLU", kcd_j_BHLA, st_BHAU)
                v_new_BHLU = v_j_BHLU - v_prime_BHLU

                inter_BHLU = jnp.einsum("BHL,BHLU->BHLU", jnp.exp(g_j_BHL), jnp.einsum("BHLA,BHAU->BHLU", q_j_BHLA, st_BHAU))
                chunk_out_BHLU = inter_BHLU + jnp.einsum("BHLM,BHMU->BHLU", intra_BHLM, v_new_BHLU)

                g_last = g_j_BHL[:, :, -1, None, None]
                g_decay_BHL = jnp.exp(g_j_BHL[:, :, -1:] - g_j_BHL)
                k_decayed_BHMA = k_j_BHMA * g_decay_BHL[..., None]
                new_st_BHAU = st_BHAU * jnp.exp(g_last) + jnp.einsum("BHMA,BHMU->BHAU", k_decayed_BHMA, v_new_BHLU)

                return new_st_BHAU, chunk_out_BHLU

            state_BHAU, core_out_chunks = jax.lax.scan(
                chunk_step, state_BHAU, jnp.arange(J)
            )
            core_out_BHJLU = core_out_chunks.transpose(1, 2, 0, 3, 4)

            core_out_BHTU = core_out_BHJLU.reshape(B, H, -1, U)[:, :, :T, :]
            return core_out_BHTU.transpose(0, 2, 1, 3)

        dn.chunk_gated_delta_rule = chunk_gated_delta_rule_patched
        print(f"[patch] replaced chunk_gated_delta_rule with solve_triangular variant")

    else:
        print(f"[patch] using current code (jnp.linalg.solve)")


def mem_gb(device) -> float:
    s = device.memory_stats()
    return s.get("bytes_in_use", 0) / 1e9 if s else -1.0


def peak_mem_gb(device) -> float:
    s = device.memory_stats()
    return s.get("peak_bytes_in_use", 0) / 1e9 if s else -1.0


def main(_):
    jax.config.update("jax_compilation_cache_dir", FLAGS.jax_cache)
    variant = FLAGS.variant

    print(f"=== Qwen3.5 E2E benchmark: variant={variant} ===")
    print(f"devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    print(f"mesh: tp={FLAGS.tp_size} fsdp={FLAGS.fsdp_size} dp={FLAGS.dp_size}")
    print(f"batch_size={FLAGS.batch_size} seq_len={FLAGS.seq_len}")
    print()

    # Patch solver BEFORE model init
    _monkey_patch_solver(variant)

    mesh = ensure_mesh(tp_size=FLAGS.tp_size, fsdp_size=FLAGS.fsdp_size, dp_size=FLAGS.dp_size)

    # ── resolve config ────────────────────────────────────────────────────────
    print("[1/5] resolving config ...", flush=True)
    cfg = text_api.resolve_config(FLAGS.model_id)
    cfg = text_api.align_config_to_mesh(cfg, mesh)
    print(f"  config: layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads}")
    n_linear = sum(1 for lt in cfg.layer_types if lt == "linear_attention")
    n_full = sum(1 for lt in cfg.layer_types if lt == "full_attention")
    print(f"  layer_types: {n_linear} linear_attention + {n_full} full_attention")

    # ── model init (random weights) ───────────────────────────────────────────
    print("[2/5] init model (random weights) ...", flush=True)
    t0 = time.time()
    rng = jax.device_put(jax.random.key(0), NamedSharding(mesh, P()))
    axis_rules = axis_rules_for_mesh(mesh)

    from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM
    model = init_model_sharded(Qwen3_5ForCausalLM, cfg, rng, mesh, axis_rules)
    jax.effects_barrier()
    n_params = sum(np.prod(v.shape) for v in jax.tree.leaves(nnx.state(model)) if hasattr(v, "shape"))
    print(f"  model init: {time.time()-t0:.1f}s  params={n_params/1e9:.2f}B  "
          f"mem={max(mem_gb(d) for d in jax.devices()):.1f} GB/device", flush=True)

    # ── optimizer ─────────────────────────────────────────────────────────────
    print("[3/5] init optimizer ...", flush=True)
    t0 = time.time()
    with mesh_rules(mesh):
        tx = optax.adamw(learning_rate=1e-4, weight_decay=0.01, mu_dtype=jnp.float32)
        optimizer = MixedPrecisionOptimizer(model, tx)
    jax.effects_barrier()
    print(f"  optimizer init: {time.time()-t0:.1f}s  "
          f"mem={max(mem_gb(d) for d in jax.devices()):.1f} GB/device", flush=True)

    # ── JIT train step ────────────────────────────────────────────────────────
    from omegalax.trainers.text import make_sft_train_step
    train_step = make_sft_train_step(cfg, pad_id=0)

    batch_raw = {
        "token_ids_BT": np.ones((FLAGS.batch_size, FLAGS.seq_len), dtype=np.int32),
        "attention_mask_BT": np.ones((FLAGS.batch_size, FLAGS.seq_len), dtype=np.int32),
        "loss_mask_BT": np.ones((FLAGS.batch_size, FLAGS.seq_len), dtype=np.int32),
    }
    batch = runtime_shard_batch_dict(batch_raw, cfg.shd_cfg, mesh)

    per_device_flops = per_device_flops_per_step(cfg, FLAGS.seq_len, FLAGS.batch_size)

    # ── warmup ────────────────────────────────────────────────────────────────
    print(f"[4/5] warmup ({FLAGS.warmup_steps} steps, first = compile) ...", flush=True)
    t0 = time.time()
    for _ in range(FLAGS.warmup_steps):
        loss, metrics = train_step(optimizer, batch)
    jax.tree.map(lambda x: x.block_until_ready(), metrics)
    print(f"  warmup: {time.time()-t0:.1f}s  "
          f"peak_mem={max(peak_mem_gb(d) for d in jax.devices()):.1f} GB/device", flush=True)

    # ── timed steps ───────────────────────────────────────────────────────────
    N = FLAGS.measure_steps
    print(f"[5/5] measuring {N} steps ...", flush=True)
    step_times = []
    for i in range(N):
        t_start = time.perf_counter()
        loss, metrics = train_step(optimizer, batch)
        jax.tree.map(lambda x: x.block_until_ready(), metrics)
        elapsed = time.perf_counter() - t_start
        step_times.append(elapsed)

    step_times = np.array(step_times)
    mean_s = np.mean(step_times)
    std_s = np.std(step_times)
    global_tokens = FLAGS.seq_len * FLAGS.batch_size

    peak_tflops = PEAK_TFLOPS["h100_sxm"]
    m = step_metrics(per_device_flops, datetime.timedelta(seconds=mean_s), global_tokens, peak_tflops)

    print()
    print(f"{'=' * 60}")
    print(f"  VARIANT:            {variant}")
    print(f"  model:              {FLAGS.model_id} ({n_params/1e9:.2f}B params)")
    print(f"  mesh:               tp={FLAGS.tp_size} fsdp={FLAGS.fsdp_size} dp={FLAGS.dp_size}")
    print(f"  batch x seq:        {FLAGS.batch_size} x {FLAGS.seq_len}")
    print(f"  step_time:          {mean_s*1000:.1f} ± {std_s*1000:.1f} ms")
    print(f"  tokens/s (global):  {m['global_tokens_per_sec']:.0f}")
    print(f"  tokens/s/device:    {m['tokens_per_sec_per_device']:.0f}")
    print(f"  TFLOPS/device:      {m['tflops_per_device']:.1f}")
    print(f"  MFU:                {m['mfu']*100:.2f}%")
    print(f"  per-step times:     {[f'{t*1000:.1f}' for t in step_times]} ms")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    app.run(main)
