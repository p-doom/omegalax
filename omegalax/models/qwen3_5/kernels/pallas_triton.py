"""Pallas (Triton lowering) chunked gated delta rule on H100.

Two-kernel split:

  1. **State-pass kernel** (sequential per ``(B, H)``): walks J chunks
     producing the per-chunk pre-update state ``state_in[j]`` and the
     corrected ``v_new[j] = u_pre[j] - kcd[j] @ state_in[j]``. The state
     stays in registers across the J iterations; this is the loop the
     pure-JAX scan was paying ~6 small dispatches per chunk for.

  2. **Output computation** (parallel across chunks): once both
     ``state_in`` and ``v_new`` are materialized, the per-chunk output
     reduces to two batched einsums plus a decay multiply, all of which
     XLA already maps to cuBLAS GEMMs efficiently. No Pallas needed.

The chunked WY/UT solve (computing ``A_inv``, the corrected ``u`` and
``k_cumdecay``) is parallel across chunks and remains in JAX as a single
batched ``solve_triangular``.

Mosaic-GPU lowering rejects per-(B,H) GMEM block scheduling; Triton
lowering accepts our shape and reaches WGMMA codegen on Hopper. We pin
the lowering explicitly via ``triton.CompilerParams``.

Public entry point: :func:`chunk_gated_delta_rule_pallas`.
"""

from __future__ import annotations

import functools

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp


def _l2norm(x: jax.Array, axis: int = -1, eps: float = 1e-6) -> jax.Array:
    inv_norm = jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)
    return x * inv_norm


# --------------------------------------------------------------------------- #
# Pallas state-pass kernel
# --------------------------------------------------------------------------- #


def _state_pass_kernel(
    kcd_ref,  # (J, C, A)  fp32  — A_inv @ (kb * exp(g_cum))
    k_dec_ref,  # (J, C, A)  fp32  — k * exp(g_last - g_cum)
    u_pre_ref,  # (J, C, U)  fp32  — A_inv @ vb
    df_ref,  # (J,)       fp32  — exp(g_last)
    initial_state_ref,  # (A, U)    fp32
    state_out_ref,  # (J, A, U)  fp32  — state at the *start* of each chunk
    v_new_out_ref,  # (J, C, U)  fp32  — corrected v
    final_state_ref,  # (A, U)    fp32
    *,
    A: int,
    U: int,
    C: int,
    J: int,
):
    """One program per (B, H). state stays in registers across the J chunks."""
    state_init = initial_state_ref[:, :].astype(jnp.float32)

    def body(j, state):
        kcd = kcd_ref[j]  # (C, A)
        k_dec = k_dec_ref[j]  # (C, A)
        u_pre = u_pre_ref[j]  # (C, U)
        df = df_ref[j]  # scalar

        # Save state at the start of this chunk for the parallel output pass.
        state_out_ref[j] = state.astype(state_out_ref.dtype)

        v_prime = pl.dot(kcd, state)
        v_new = u_pre - v_prime
        v_new_out_ref[j] = v_new.astype(v_new_out_ref.dtype)

        new_state = df * state + pl.dot(k_dec.T, v_new)
        return new_state

    final_state = jax.lax.fori_loop(0, J, body, state_init)
    final_state_ref[:, :] = final_state.astype(final_state_ref.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=())
def _state_pass_pallas(
    kcd_BHJCA: jax.Array,
    k_dec_BHJCA: jax.Array,
    u_pre_BHJCU: jax.Array,
    df_BHJ: jax.Array,
    initial_state_BHAU: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sequential gated-delta state pass.

    Forward equations (per (B, H), J chunks):
      state[0]    = initial_state
      state_in[j] = state[j]
      v_new[j]    = u_pre[j] - kcd[j] @ state[j]
      state[j+1]  = df[j] * state[j] + k_dec[j].T @ v_new[j]

    Returns ``(state_in, v_new, final_state)``.
    """
    return _state_pass_pallas_fwd_only(
        kcd_BHJCA, k_dec_BHJCA, u_pre_BHJCU, df_BHJ, initial_state_BHAU
    )


def _state_pass_pallas_fwd_only(kcd, k_dec, u_pre, df, initial_state):
    B, H, J, C, A = kcd.shape
    U = u_pre.shape[-1]
    state_out_shape = jax.ShapeDtypeStruct((B, H, J, A, U), jnp.float32)
    v_new_out_shape = jax.ShapeDtypeStruct((B, H, J, C, U), jnp.float32)
    final_state_shape = jax.ShapeDtypeStruct((B, H, A, U), jnp.float32)
    kernel = functools.partial(_state_pass_kernel, A=A, U=U, C=C, J=J)

    spec_a = pl.BlockSpec((None, None, J, C, A), lambda b, h: (b, h, 0, 0, 0))
    spec_u = pl.BlockSpec((None, None, J, C, U), lambda b, h: (b, h, 0, 0, 0))
    spec_s = pl.BlockSpec((None, None, J, A, U), lambda b, h: (b, h, 0, 0, 0))
    spec_d = pl.BlockSpec((None, None, J), lambda b, h: (b, h, 0))
    spec_init = pl.BlockSpec((None, None, A, U), lambda b, h: (b, h, 0, 0))

    return pl.pallas_call(
        kernel,
        out_shape=(state_out_shape, v_new_out_shape, final_state_shape),
        grid=(B, H),
        in_specs=(spec_a, spec_a, spec_u, spec_d, spec_init),
        out_specs=(spec_s, spec_u, spec_init),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=2),
    )(kcd, k_dec, u_pre, df, initial_state)


def _state_pass_fwd_for_vjp(kcd, k_dec, u_pre, df, initial_state):
    state_in, v_new, final_state = _state_pass_pallas_fwd_only(kcd, k_dec, u_pre, df, initial_state)
    residuals = (kcd, k_dec, u_pre, df, initial_state)
    return (state_in, v_new, final_state), residuals


def _state_pass_bwd(residuals, cotangents):
    """Reverse-scan VJP for the sequential state pass.

    Backward derivation per chunk j (notation: dX = ∂L/∂X):
      Forward:
        v_new[j] = u_pre[j] - kcd[j] @ state[j]
        new_state = df[j] * state[j] + k_dec[j].T @ v_new[j]

      Receives in reverse: ``dstate`` (carry, ∂L/∂new_state) plus the
      cotangents ``dstate_in[j]`` and ``dv_new[j]`` from saved outputs.

      Through new_state:
        ddf[j]            = ⟨state[j], dstate⟩  (element-wise sum)
        dv_new += k_dec[j] @ dstate            (shape (C, U))
        dk_dec[j]          = v_new[j] @ dstate.T (shape (C, A))
        dstate_via_df      = df[j] * dstate

      Through v_new:
        du_pre[j]          = dv_new            (shape (C, U))
        dkcd[j]            = -dv_new @ state[j].T (shape (C, A))
        dstate_via_kcd     = -kcd[j].T @ dv_new (shape (A, U))

      Total dstate carry = dstate_via_df + dstate_via_kcd + dstate_in[j]
    """
    kcd, k_dec, u_pre, df, initial_state = residuals
    # Avoid retaining these per-chunk buffers across every BPTT segment.
    state_in, v_new, _ = _state_pass_pallas_fwd_only(kcd, k_dec, u_pre, df, initial_state)
    dstate_in, dv_new, dfinal_state = cotangents
    B, H, J, C, A = kcd.shape

    def body(dstate, j):
        kcd_j = kcd[:, :, j]  # (B, H, C, A)
        k_dec_j = k_dec[:, :, j]
        v_new_j = v_new[:, :, j]
        df_j = df[:, :, j]  # (B, H)
        state_j = state_in[:, :, j]  # (B, H, A, U)
        dstate_in_j = dstate_in[:, :, j]
        dv_new_saved_j = dv_new[:, :, j]

        # gradients through new_state
        ddf_j = jnp.einsum("BHAU,BHAU->BH", state_j, dstate)
        dk_dec_j = jnp.einsum("BHCU,BHAU->BHCA", v_new_j, dstate)
        dv_new_via_new_state = jnp.einsum("BHCA,BHAU->BHCU", k_dec_j, dstate)
        dstate_via_df = df_j[..., None, None] * dstate

        # total cotangent flowing into v_new[j]
        dv_new_total = dv_new_saved_j + dv_new_via_new_state

        # gradients through v_new
        du_pre_j = dv_new_total
        dkcd_j = -jnp.einsum("BHCU,BHAU->BHCA", dv_new_total, state_j)
        dstate_via_kcd = -jnp.einsum("BHCA,BHCU->BHAU", kcd_j, dv_new_total)

        new_dstate = dstate_via_df + dstate_via_kcd + dstate_in_j
        return new_dstate, (dkcd_j, dk_dec_j, du_pre_j, ddf_j)

    dstate_init = dfinal_state.astype(jnp.float32)
    dinitial_state, (dkcd_JBHCA, dk_dec_JBHCA, du_pre_JBHCU, ddf_JBH) = jax.lax.scan(
        body,
        dstate_init,
        jnp.arange(J),
        reverse=True,
    )
    # scan stacks outputs along axis 0 → reorder to (B, H, J, ...)
    dkcd_BHJCA = jnp.transpose(dkcd_JBHCA, (1, 2, 0, 3, 4))
    dk_dec_BHJCA = jnp.transpose(dk_dec_JBHCA, (1, 2, 0, 3, 4))
    du_pre_BHJCU = jnp.transpose(du_pre_JBHCU, (1, 2, 0, 3, 4))
    ddf_BHJ = jnp.transpose(ddf_JBH, (1, 2, 0))
    return dkcd_BHJCA, dk_dec_BHJCA, du_pre_BHJCU, ddf_BHJ, dinitial_state


_state_pass_pallas.defvjp(_state_pass_fwd_for_vjp, _state_pass_bwd)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def chunk_gated_delta_rule_pallas(
    q_BTHA: jax.Array,
    k_BTHA: jax.Array,
    v_BTHU: jax.Array,
    g_BTH: jax.Array,
    beta_BTH: jax.Array,
    chunk_size: int = 64,
    initial_state_BHAU: jax.Array | None = None,
    *,
    return_final_state: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Chunked gated delta rule with a Pallas state-pass kernel on Hopper.

    Same numeric contract as the XLA reference. The WY/UT solve runs in JAX
    (vectorized across chunks via batched ``solve_triangular``); the
    sequential cross-chunk state propagation runs in one Pallas program per
    ``(B, H)``; the per-chunk output computation runs as parallel JAX
    einsums.
    """
    out_dtype = jnp.float32  # match xla_reference's output dtype
    q_BTHA = _l2norm(q_BTHA, axis=-1)
    k_BTHA = _l2norm(k_BTHA, axis=-1)

    q_BHTA = q_BTHA.transpose(0, 2, 1, 3).astype(jnp.float32)
    k_BHTA = k_BTHA.transpose(0, 2, 1, 3).astype(jnp.float32)
    v_BHTU = v_BTHU.transpose(0, 2, 1, 3).astype(jnp.float32)
    beta_BHT = beta_BTH.transpose(0, 2, 1).astype(jnp.float32)
    g_BHT = g_BTH.transpose(0, 2, 1).astype(jnp.float32)

    B, H, T, A = k_BHTA.shape
    U = v_BHTU.shape[-1]
    C = chunk_size
    if initial_state_BHAU is None:
        initial_state_BHAU = jnp.zeros((B, H, A, U), dtype=jnp.float32)
    else:
        initial_state_BHAU = initial_state_BHAU.astype(jnp.float32)

    pad = (-T) % C
    if pad:
        pad_t = ((0, 0), (0, 0), (0, pad), (0, 0))
        pad_g = ((0, 0), (0, 0), (0, pad))
        q_BHTA = jnp.pad(q_BHTA, pad_t)
        k_BHTA = jnp.pad(k_BHTA, pad_t)
        v_BHTU = jnp.pad(v_BHTU, pad_t)
        beta_BHT = jnp.pad(beta_BHT, pad_g)
        g_BHT = jnp.pad(g_BHT, pad_g)
    Tp = T + pad
    J = Tp // C

    scale = A**-0.5
    q_BHTA = q_BHTA * scale

    kb_BHTA = k_BHTA * beta_BHT[..., None]
    vb_BHTU = v_BHTU * beta_BHT[..., None]

    q_BHJCA = q_BHTA.reshape(B, H, J, C, A)
    k_BHJCA = k_BHTA.reshape(B, H, J, C, A)
    kb_BHJCA = kb_BHTA.reshape(B, H, J, C, A)
    vb_BHJCU = vb_BHTU.reshape(B, H, J, C, U)
    g_BHJC = g_BHT.reshape(B, H, J, C)

    g_cum_BHJC = jnp.cumsum(g_BHJC, axis=-1)

    g_row = g_cum_BHJC[..., :, None]
    g_col = g_cum_BHJC[..., None, :]
    tril = jnp.tril(jnp.ones((C, C)))
    decay_mask = jnp.exp((g_row - g_col) * tril) * tril

    upper = jnp.triu(jnp.ones((C, C), dtype=jnp.bool_))
    A_pre = -(jnp.einsum("BHJLA,BHJMA->BHJLM", kb_BHJCA, k_BHJCA) * decay_mask)
    A_pre = jnp.where(upper, 0.0, A_pre)

    eye = jnp.eye(C, dtype=A_pre.dtype)
    lhs = eye - A_pre
    rhs = jnp.broadcast_to(eye, lhs.shape)
    A_inv = jax.scipy.linalg.solve_triangular(lhs, rhs, lower=True)

    u_pre_BHJCU = jnp.einsum("BHJLM,BHJMU->BHJLU", A_inv, vb_BHJCU)
    kb_decayed = kb_BHJCA * jnp.exp(g_cum_BHJC)[..., None]
    kcd_BHJCA = jnp.einsum("BHJLM,BHJMA->BHJLA", A_inv, kb_decayed)

    g_last_BHJ = g_cum_BHJC[..., -1]
    df_BHJ = jnp.exp(g_last_BHJ)
    decay_to_end_BHJC = jnp.exp(g_last_BHJ[..., None] - g_cum_BHJC)
    inter_decay_BHJC = jnp.exp(g_cum_BHJC)
    k_dec_BHJCA = k_BHJCA * decay_to_end_BHJC[..., None]

    # Sequential kernel: produces start-of-chunk state and corrected v_new.
    state_in_BHJAU, v_new_BHJCU, final_state_BHAU = _state_pass_pallas(
        kcd_BHJCA.astype(jnp.float32),
        k_dec_BHJCA.astype(jnp.float32),
        u_pre_BHJCU.astype(jnp.float32),
        df_BHJ.astype(jnp.float32),
        initial_state_BHAU,
    )

    # Output (parallel across chunks via batched JAX einsums).
    upper1 = jnp.triu(jnp.ones((C, C), dtype=jnp.bool_), k=1)
    qkt = jnp.einsum("BHJLA,BHJMA->BHJLM", q_BHJCA, k_BHJCA) * decay_mask
    intra = jnp.where(upper1, 0.0, qkt)
    inter = inter_decay_BHJC[..., None] * jnp.einsum(
        "BHJLA,BHJAU->BHJLU",
        q_BHJCA,
        state_in_BHJAU,
    )
    out_BHJCU = inter + jnp.einsum("BHJLM,BHJMU->BHJLU", intra, v_new_BHJCU)

    out_BHTU = out_BHJCU.reshape(B, H, Tp, U)[:, :, :T, :]
    out_BTHU = out_BHTU.transpose(0, 2, 1, 3).astype(out_dtype)
    if return_final_state:
        return out_BTHU, final_state_BHAU
    return out_BTHU
