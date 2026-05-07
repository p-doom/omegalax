#!/usr/bin/env python3
"""Correctness test: solve_triangular vs linalg.solve in deltanet.

Compares forward outputs AND backward gradients on the same model/input
to verify the two solvers produce identical results in practice.

Runs on all available GPUs with FSDP.

Usage:
    python scripts/bench_solve_correctness.py
"""

from __future__ import annotations

import copy
import time

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
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

P = PartitionSpec

MODEL_ID = "Qwen/Qwen3.5-4B"
BATCH_SIZE = 2
SEQ_LEN = 512


def main():
    ndev = jax.device_count()
    print(f"JAX devices: {ndev} x {jax.devices()[0].device_kind}")

    mesh = ensure_mesh(tp_size=1, fsdp_size=ndev, dp_size=1)
    cfg = text_api.resolve_config(MODEL_ID)
    cfg = text_api.align_config_to_mesh(cfg, mesh)

    n_linear = sum(1 for lt in cfg.layer_types if lt == "linear_attention")
    n_full = sum(1 for lt in cfg.layer_types if lt == "full_attention")
    print(f"Config: {cfg.num_hidden_layers} layers ({n_linear} linear + {n_full} full attn)")

    # ── init model ────────────────────────────────────────────────────────────
    print("Initializing model (random weights) ...", flush=True)
    rng = jax.device_put(jax.random.key(42), NamedSharding(mesh, P()))
    axis_rules = axis_rules_for_mesh(mesh)

    from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM
    model = init_model_sharded(Qwen3_5ForCausalLM, cfg, rng, mesh, axis_rules)
    jax.effects_barrier()
    print("  model ready", flush=True)

    # ── create batch ──────────────────────────────────────────────────────────
    np.random.seed(123)
    token_ids = np.random.randint(1, 1000, (BATCH_SIZE, SEQ_LEN), dtype=np.int32)
    batch_raw = {
        "token_ids_BT": token_ids,
        "attention_mask_BT": np.ones_like(token_ids),
        "loss_mask_BT": np.ones_like(token_ids),
    }
    batch = runtime_shard_batch_dict(batch_raw, cfg.shd_cfg, mesh)
    token_ids_BT = batch["token_ids_BT"]
    loss_mask_BT = batch["loss_mask_BT"]

    # ── Save original chunk_gated_delta_rule ──────────────────────────────────
    import omegalax.models.qwen3_5.deltanet as dn
    original_fn = dn.chunk_gated_delta_rule

    # ── build the SAME patched fn used in the benchmark ───────────────────────
    def make_tri_fn():
        """Build a solve_triangular variant of chunk_gated_delta_rule."""
        def chunk_gated_delta_rule_tri(
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

        return chunk_gated_delta_rule_tri

    # ── JIT-compiled loss+grad (shard_map requires JIT) ─────────────────────
    def make_jitted_loss_and_grad():
        @nnx.jit
        def compute(model):
            def loss_fn(m):
                hidden_BTD, aux_loss = text_api.forward(m, token_ids_BT, pad_id=0, cfg=cfg)
                lm_weight = m.lm_head.kernel[...]
                loss = chunked_cross_entropy_loss(
                    hidden_BTD, lm_weight, token_ids_BT, loss_mask_BT,
                    num_tiles=4,
                    logits_out_sharding=cfg.shd_cfg.logits_btv,
                ) + aux_loss
                return loss
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            grad_norm = jnp.sqrt(sum(
                jnp.sum(g**2) for g in jax.tree.leaves(grads) if isinstance(g, jax.Array)
            ))
            return loss, grad_norm, grads
        return compute

    # ── Run with linalg.solve (current code) ──────────────────────────────────
    print("\n=== Running forward+backward with linalg.solve ===", flush=True)
    dn.chunk_gated_delta_rule = original_fn
    compute_solve = make_jitted_loss_and_grad()
    loss_solve, grad_norm_solve, grads_solve = compute_solve(model)
    loss_solve_val = float(loss_solve)
    grad_norm_solve = float(grad_norm_solve)
    print(f"  loss = {loss_solve_val:.6f}")
    print(f"  grad_norm = {grad_norm_solve:.6f}")

    # ── Run with solve_triangular ─────────────────────────────────────────────
    print("\n=== Running forward+backward with solve_triangular ===", flush=True)
    dn.chunk_gated_delta_rule = make_tri_fn()
    compute_tri = make_jitted_loss_and_grad()
    loss_tri, grad_norm_tri, grads_tri = compute_tri(model)
    loss_tri_val = float(loss_tri)
    grad_norm_tri = float(grad_norm_tri)
    print(f"  loss = {loss_tri_val:.6f}")
    print(f"  grad_norm = {grad_norm_tri:.6f}")

    # ── Compare ───────────────────────────────────────────────────────────────
    print("\n=== Comparison ===")
    loss_diff = abs(loss_solve_val - loss_tri_val)
    grad_norm_diff = abs(grad_norm_solve - grad_norm_tri)
    print(f"  |loss_diff|     = {loss_diff:.2e}")
    print(f"  |grad_norm_diff| = {grad_norm_diff:.2e}")
    print(f"  loss_solve      = {loss_solve_val:.8f}")
    print(f"  loss_tri        = {loss_tri_val:.8f}")
    print(f"  grad_norm_solve = {grad_norm_solve:.8f}")
    print(f"  grad_norm_tri   = {grad_norm_tri:.8f}")

    # Per-parameter gradient comparison
    leaves_solve = jax.tree.leaves(grads_solve)
    leaves_tri = jax.tree.leaves(grads_tri)
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    for s, t in zip(leaves_solve, leaves_tri):
        if not isinstance(s, jax.Array):
            continue
        abs_diff = float(jnp.max(jnp.abs(s - t)))
        denom = float(jnp.maximum(jnp.max(jnp.abs(s)), 1e-12))
        rel_diff = abs_diff / denom
        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)

    print(f"\n  max |grad_abs_diff| across all params = {max_abs_diff:.2e}")
    print(f"  max |grad_rel_diff| across all params = {max_rel_diff:.2e}")

    if loss_diff < 1e-4 and max_rel_diff < 1e-3:
        print("\n  PASS: results match within tolerance")
    else:
        print("\n  WARNING: results differ beyond tolerance — investigate!")

    # Restore original
    dn.chunk_gated_delta_rule = original_fn


if __name__ == "__main__":
    main()
