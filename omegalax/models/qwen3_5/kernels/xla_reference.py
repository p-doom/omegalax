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
    init_NHAU: jax.Array,
    init_idx_BJ: jax.Array,
    final_idx_BW: jax.Array,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Chunked gated delta rule (XLA reference).

    Inputs:
        q_BTHA, k_BTHA: (B, T, H, A)
        v_BTHU:         (B, T, H, U)
        g_BTH, beta_BTH: (B, T, H)
        init_NHAU:      (N, H, A, U) initial states
        init_idx_BJ:    (B, J) chunk ``j`` of row ``b`` starts from ``init_NHAU[i]`` when
                        ``i = init_idx_BJ[b, j] >= 0``, else from the previous chunk's state
        final_idx_BW:   (B, W) chunks whose end states to return
    Returns:
        out_BTHU: (B, T, H, U)
        state_BWHAU: (B, W, H, A, U) state after chunk ``final_idx_BW[b, w]``

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
    init_NHAU = init_NHAU.astype(jnp.float32)

    def chunk_step(carry, chunk_idx):
        idx_B = init_idx_BJ[:, chunk_idx]
        st_BHAU = jnp.where(
            (idx_B >= 0)[:, None, None, None], init_NHAU[jnp.maximum(idx_B, 0)], carry
        )
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

        return new_st_BHAU, (chunk_out_BHLU, new_st_BHAU)

    _, (core_out_chunks, state_JBHAU) = jax.lax.scan(chunk_step, state_BHAU, jnp.arange(J))
    core_out_BHJLU = core_out_chunks.transpose(1, 2, 0, 3, 4)
    core_out_BHTU = core_out_BHJLU.reshape(B, H, -1, U)[:, :, :T, :]
    state_BJHAU = jnp.swapaxes(state_JBHAU, 0, 1)
    state_BWHAU = jnp.take_along_axis(state_BJHAU, final_idx_BW[:, :, None, None, None], axis=1)
    return core_out_BHTU.transpose(0, 2, 1, 3), state_BWHAU


def recurrent_gated_delta_step_xla(
    q_BHA: jax.Array,
    k_BHA: jax.Array,
    v_BHU: jax.Array,
    g_BH: jax.Array,
    beta_BH: jax.Array,
    state_SHAU: jax.Array,
    slot_B: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """One gated delta rule step per row on state ``state_SHAU[slot_B[b]]`` (XLA reference).

    Returns ``(out_BHU, updated state_SHAU)``; same math as one token of the chunked rule.
    """
    q_BHA = _l2norm(q_BHA).astype(jnp.float32) * q_BHA.shape[-1] ** -0.5
    k_BHA = _l2norm(k_BHA).astype(jnp.float32)
    st_BHAU = state_SHAU[slot_B] * jnp.exp(g_BH.astype(jnp.float32))[..., None, None]
    kv_BHU = jnp.einsum("BHAU,BHA->BHU", st_BHAU, k_BHA)
    delta_BHU = (v_BHU.astype(jnp.float32) - kv_BHU) * beta_BH.astype(jnp.float32)[..., None]
    st_BHAU = st_BHAU + k_BHA[..., None] * delta_BHU[..., None, :]
    out_BHU = jnp.einsum("BHAU,BHA->BHU", st_BHAU, q_BHA)
    return out_BHU, state_SHAU.at[slot_B].set(st_BHAU)
