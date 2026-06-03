"""Pure-JAX reference for chunk_gated_delta_rule.

Mirror of the original implementation that was inlined in ``deltanet.py``,
moved here so the Pallas kernel can use it as a correctness oracle without a
circular import. The math is unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _l2norm(x: jax.Array, axis: int = -1, eps: float = 1e-6) -> jax.Array:
    inv_norm = jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)
    return x * inv_norm


def chunk_gated_delta_rule_xla(
    q_BTHA: jax.Array,
    k_BTHA: jax.Array,
    v_BTHU: jax.Array,
    g_BTH: jax.Array,
    beta_BTH: jax.Array,
    chunk_size: int = 64,
) -> jax.Array:
    """Chunked gated delta rule (XLA reference).

    Inputs:
        q_BTHA, k_BTHA: (B, T, H, A)
        v_BTHU:         (B, T, H, U)
        g_BTH, beta_BTH: (B, T, H)
    Returns:
        out_BTHU: (B, T, H, U)

    Internally l2-normalizes q and k. Pads T up to a multiple of ``chunk_size``
    and trims at the end.
    """
    q_BTHA = _l2norm(q_BTHA, axis=-1)
    k_BTHA = _l2norm(k_BTHA, axis=-1)

    q_BHTA, k_BHTA, v_BHTU = [
        x.transpose(0, 2, 1, 3).astype(jnp.float32) for x in (q_BTHA, k_BTHA, v_BTHU)
    ]
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

    scale = A**-0.5
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
    attn_BHJLM = jax.scipy.linalg.solve_triangular(lhs_BHJLM, rhs_BHJLM, lower=True)

    v_corrected_BHJLU = jnp.einsum("BHJLM,BHJMU->BHJLU", attn_BHJLM, vb_BHJLU)
    k_cumdecay_BHJLA = jnp.einsum(
        "BHJLM,BHJMA->BHJLA", attn_BHJLM, kb_BHJLA * jnp.exp(g_BHJL)[..., None]
    )

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

        intra_BHLM = jnp.einsum("BHLA,BHMA->BHLM", q_j_BHLA, k_j_BHMA) * dm_j_LM
        intra_BHLM = jnp.where(upper_mask_1_LM, 0.0, intra_BHLM)

        v_prime_BHLU = jnp.einsum("BHLA,BHAU->BHLU", kcd_j_BHLA, st_BHAU)
        v_new_BHLU = v_j_BHLU - v_prime_BHLU

        inter_BHLU = jnp.einsum(
            "BHL,BHLU->BHLU",
            jnp.exp(g_j_BHL),
            jnp.einsum("BHLA,BHAU->BHLU", q_j_BHLA, st_BHAU),
        )
        chunk_out_BHLU = inter_BHLU + jnp.einsum("BHLM,BHMU->BHLU", intra_BHLM, v_new_BHLU)

        g_last = g_j_BHL[:, :, -1, None, None]
        g_decay_BHL = jnp.exp(g_j_BHL[:, :, -1:] - g_j_BHL)
        k_decayed_BHMA = k_j_BHMA * g_decay_BHL[..., None]
        new_st_BHAU = st_BHAU * jnp.exp(g_last) + jnp.einsum(
            "BHMA,BHMU->BHAU",
            k_decayed_BHMA,
            v_new_BHLU,
        )

        return new_st_BHAU, chunk_out_BHLU

    state_BHAU, core_out_chunks = jax.lax.scan(chunk_step, state_BHAU, jnp.arange(J))
    core_out_BHJLU = core_out_chunks.transpose(1, 2, 0, 3, 4)
    core_out_BHTU = core_out_BHJLU.reshape(B, H, -1, U)[:, :, :T, :]
    return core_out_BHTU.transpose(0, 2, 1, 3)
