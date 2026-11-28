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
import numpy as np

# Value columns per state-pass program: more programs when B * H is small (packed extend).
_STATE_U_BLOCK = 32


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
    idx_ref,  # (J,)       int32 — initial state index, or -1 to continue
    init_ref,  # (N, A, U)  fp32  — initial states
    state_out_ref,  # (J, A, U)  fp32  — state at the *start* of each chunk
    v_new_out_ref,  # (J, C, U)  fp32  — corrected v
    *,
    A: int,
    U: int,
    C: int,
    J: int,
):
    """One program per (B, H, U block); value columns of the state evolve independently.

    The state stays in registers across the J chunks.
    """
    state_init = jnp.zeros((A, U), dtype=jnp.float32)

    def body(j, state):
        idx = idx_ref[j]
        state = jax.lax.cond(idx >= 0, lambda: init_ref[jnp.maximum(idx, 0)], lambda: state)
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

    jax.lax.fori_loop(0, J, body, state_init)


def _state_pass_bwd_kernel(
    kcd_ref,  # (J, C, A)  fp32
    k_dec_ref,  # (J, C, A)  fp32
    dv_new_ref,  # (J, C, U)  fp32  — cotangent of v_new
    df_ref,  # (J,)       fp32
    idx_ref,  # (J,)       int32
    dstate_in_ref,  # (J, A, U)  fp32  — cotangent of state_in
    dstate_in_out_ref,  # (J, A, U)  fp32  — total gradient w.r.t. state_in[j]
    dv_total_out_ref,  # (J, C, U)  fp32  — total gradient w.r.t. v_new[j]
    *,
    A: int,
    U: int,
    J: int,
):
    """Reverse of :func:`_state_pass_kernel`; the carry is the gradient w.r.t. the state after
    chunk ``j``, which a reset at ``j`` disconnects from the chunks before it."""

    def body(i, dstate):
        j = J - 1 - i
        dv_total = dv_new_ref[j] + pl.dot(k_dec_ref[j], dstate)
        dv_total_out_ref[j] = dv_total
        dstate_in = df_ref[j] * dstate - pl.dot(kcd_ref[j].T, dv_total) + dstate_in_ref[j]
        dstate_in_out_ref[j] = dstate_in
        return jnp.where(idx_ref[j] >= 0, 0.0, dstate_in)

    jax.lax.fori_loop(0, J, body, jnp.zeros((A, U), jnp.float32))


@functools.partial(jax.custom_vjp, nondiff_argnums=())
def _state_pass_pallas(
    kcd_BHJCA: jax.Array,
    k_dec_BHJCA: jax.Array,
    u_pre_BHJCU: jax.Array,
    df_BHJ: jax.Array,
    init_idx_BJ: jax.Array,
    init_NHAU: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sequential gated-delta state pass.

    Forward equations (per (B, H), J chunks):
      state[0]    = 0
      state_in[j] = init[idx[j]] if idx[j] >= 0 else state[j]
      v_new[j]    = u_pre[j] - kcd[j] @ state_in[j]
      state[j+1]  = df[j] * state_in[j] + k_dec[j].T @ v_new[j]

    Returns ``(state_in, v_new)`` — both shape ``(B, H, J, ...)``.
    """
    return _state_pass_pallas_fwd_only(
        kcd_BHJCA, k_dec_BHJCA, u_pre_BHJCU, df_BHJ, init_idx_BJ, init_NHAU
    )


def _state_pass_pallas_fwd_only(kcd, k_dec, u_pre, df, init_idx, init):
    B, H, J, C, A = kcd.shape
    U = u_pre.shape[-1]
    N = init.shape[0]
    state_out_shape = jax.ShapeDtypeStruct((B, H, J, A, U), jnp.float32)
    v_new_out_shape = jax.ShapeDtypeStruct((B, H, J, C, U), jnp.float32)
    UB = min(U, _STATE_U_BLOCK)
    kernel = functools.partial(_state_pass_kernel, A=A, U=UB, C=C, J=J)

    spec_a = pl.BlockSpec((None, None, J, C, A), lambda b, h, u: (b, h, 0, 0, 0))
    spec_u = pl.BlockSpec((None, None, J, C, UB), lambda b, h, u: (b, h, 0, 0, u))
    spec_s = pl.BlockSpec((None, None, J, A, UB), lambda b, h, u: (b, h, 0, 0, u))
    spec_d = pl.BlockSpec((None, None, J), lambda b, h, u: (b, h, 0))
    spec_i = pl.BlockSpec((None, J), lambda b, h, u: (b, 0))
    spec_init = pl.BlockSpec((N, None, A, UB), lambda b, h, u: (0, h, 0, u))

    return pl.pallas_call(
        kernel,
        out_shape=(state_out_shape, v_new_out_shape),
        grid=(B, H, U // UB),
        in_specs=(spec_a, spec_a, spec_u, spec_d, spec_i, spec_init),
        out_specs=(spec_s, spec_u),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=2),
    )(kcd, k_dec, u_pre, df, init_idx, init.astype(jnp.float32))


def _state_pass_fwd_for_vjp(kcd, k_dec, u_pre, df, init_idx, init):
    state_in, v_new = _state_pass_pallas_fwd_only(kcd, k_dec, u_pre, df, init_idx, init)
    residuals = (kcd, k_dec, u_pre, df, state_in, v_new, init_idx, init)
    return (state_in, v_new), residuals


def _state_pass_bwd(residuals, cotangents):
    """VJP of the state pass: the sequential reverse recurrence runs in
    :func:`_state_pass_bwd_kernel`; the per-chunk parameter gradients are batched einsums.

    With ``D[j]`` the gradient w.r.t. the state after chunk ``j`` (``dstate_in[j + 1]`` unless
    chunk ``j + 1`` resets) and ``dv[j]`` the total gradient w.r.t. ``v_new[j]``:
      ddf[j] = <state_in[j], D[j]>,  dk_dec[j] = v_new[j] @ D[j].T,
      dkcd[j] = -dv[j] @ state_in[j].T,  du_pre[j] = dv[j],
    and each reset chunk adds its ``dstate_in`` to the gradient of the initial state it read.
    """
    kcd, k_dec, u_pre, df, state_in, v_new, init_idx, init = residuals
    dstate_in, dv_new = cotangents
    B, H, J, C, A = kcd.shape
    U = u_pre.shape[-1]
    UB = min(U, _STATE_U_BLOCK)
    spec_a = pl.BlockSpec((None, None, J, C, A), lambda b, h, u: (b, h, 0, 0, 0))
    spec_u = pl.BlockSpec((None, None, J, C, UB), lambda b, h, u: (b, h, 0, 0, u))
    spec_s = pl.BlockSpec((None, None, J, A, UB), lambda b, h, u: (b, h, 0, 0, u))
    spec_d = pl.BlockSpec((None, None, J), lambda b, h, u: (b, h, 0))
    spec_i = pl.BlockSpec((None, J), lambda b, h, u: (b, 0))
    dstate_in_total, dv_total = pl.pallas_call(
        functools.partial(_state_pass_bwd_kernel, A=A, U=UB, J=J),
        out_shape=(
            jax.ShapeDtypeStruct((B, H, J, A, U), jnp.float32),
            jax.ShapeDtypeStruct((B, H, J, C, U), jnp.float32),
        ),
        grid=(B, H, U // UB),
        in_specs=(spec_a, spec_a, spec_u, spec_d, spec_i, spec_s),
        out_specs=(spec_s, spec_u),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=2),
    )(kcd, k_dec, dv_new.astype(jnp.float32), df, init_idx, dstate_in.astype(jnp.float32))

    reset_BJ = init_idx >= 0
    after_BHJAU = jnp.where(reset_BJ[:, None, 1:, None, None], 0.0, dstate_in_total[:, :, 1:])
    after_BHJAU = jnp.concatenate([after_BHJAU, jnp.zeros_like(after_BHJAU[:, :, :1])], axis=2)
    ddf_BHJ = jnp.einsum("BHJAU,BHJAU->BHJ", state_in, after_BHJAU)
    dk_dec_BHJCA = jnp.einsum("BHJCU,BHJAU->BHJCA", v_new, after_BHJAU)
    dkcd_BHJCA = -jnp.einsum("BHJCU,BHJAU->BHJCA", dv_total, state_in)
    dinit = (
        jnp.zeros(init.shape, jnp.float32)
        .at[jnp.where(reset_BJ, init_idx, init.shape[0])]
        .add(dstate_in_total.transpose(0, 2, 1, 3, 4), mode="drop")
    )
    dinit_idx = np.zeros(init_idx.shape, jax.dtypes.float0)
    return dkcd_BHJCA, dk_dec_BHJCA, dv_total, ddf_BHJ, dinit_idx, dinit.astype(init.dtype)


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
    init_NHAU: jax.Array,
    init_idx_BJ: jax.Array,
    final_idx_BW: jax.Array,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
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
    state_in_BHJAU, v_new_BHJCU = _state_pass_pallas(
        kcd_BHJCA.astype(jnp.float32),
        k_dec_BHJCA.astype(jnp.float32),
        u_pre_BHJCU.astype(jnp.float32),
        df_BHJ.astype(jnp.float32),
        init_idx_BJ,
        init_NHAU,
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

    def at(x):
        idx = final_idx_BW.reshape(B, 1, -1, *[1] * (x.ndim - 3))
        return jnp.take_along_axis(x, idx, axis=2)

    state_BHWAU = at(df_BHJ[..., None, None]) * at(state_in_BHJAU) + jnp.einsum(
        "BHWCA,BHWCU->BHWAU", at(k_dec_BHJCA), at(v_new_BHJCU)
    )
    return out_BHTU.transpose(0, 2, 1, 3).astype(out_dtype), state_BHWAU.transpose(0, 2, 1, 3, 4)


def _conv_step_kernel(slot_ref, x_ref, w_ref, pool_ref, out_ref, pool_out_ref, *, num_q, num_qk):
    """One program per (row, head-sized channel block): the causal conv over the slot's K - 1
    previous inputs and this one, SiLU, and for q/k blocks the L2 norm (q also scaled). The
    block's window in the conv pool shifts in place; each channel belongs to one program."""
    b, j = pl.program_id(0), pl.program_id(1)
    slot = slot_ref[b]
    D = x_ref.shape[-1]
    cols = pl.ds(j * D, D)
    K1 = pool_ref.shape[1]
    taps = [pool_ref[slot, k, cols] for k in range(K1)]
    # Triton loads no single column of a block; mask the (D, K) tile instead.
    w_DK = w_ref[...].astype(jnp.float32)
    col_DK = jax.lax.broadcasted_iota(jnp.int32, w_DK.shape, 1)
    tap_w = lambda k: jnp.sum(jnp.where(col_DK == k, w_DK, 0.0), axis=1)
    # The conv and SiLU round every op to the input dtype, as the trainer's XLA ops do.
    rnd = lambda v: v.astype(x_ref.dtype).astype(jnp.float32)
    acc = jnp.zeros_like(tap_w(0))
    for k, tap in enumerate([*taps, x_ref[...]]):
        acc = rnd(acc + rnd(tap_w(k) * tap.astype(jnp.float32)))
    act = rnd(acc * rnd(1.0 / (1.0 + jnp.exp(-acc))))
    for k in range(K1 - 1):
        pool_out_ref[slot, k, cols] = taps[k + 1]
    pool_out_ref[slot, K1 - 1, cols] = x_ref[...]
    # q and k blocks are L2-normed in the input dtype like the trainer's chunked rule (q also
    # scaled), v blocks pass; selected arithmetically, as Triton lowers a select on the scalar
    # program id to a mask.
    is_q, is_qk = (j < num_q).astype(jnp.float32), (j < num_qk).astype(jnp.float32)
    unit = rnd(act * rnd(jax.lax.rsqrt(rnd(rnd(jnp.sum(rnd(act * act))) + 1e-6))))
    unit = unit * (1.0 + is_q * (D**-0.5 - 1.0))
    out_ref[...] = act + is_qk * (unit - act)


def _decode_step_kernel(
    slot_ref, q_ref, k_ref, v_ref, b_ref, a_ref, alog_ref, dt_ref, z_ref, w_ref, state_ref,
    out_ref, state_out_ref, *, eps,
):
    """One program per (row, value head): the gates, one delta rule step on the head's (A, U)
    state (rewritten in place) and the gated RMSNorm of its output."""
    b, h = pl.program_id(0), pl.program_id(1)
    slot = slot_ref[b]
    b_x = b_ref[b, h]
    beta = (1.0 / (1.0 + jnp.exp(-b_x.astype(jnp.float32)))).astype(b_x.dtype).astype(jnp.float32)
    x = a_ref[b, h].astype(jnp.float32) + dt_ref[h]
    softplus = jnp.maximum(x, 0.0) + jnp.log(1.0 + jnp.exp(-jnp.abs(x)))
    st_AU = state_ref[slot, h] * jnp.exp(-jnp.exp(alog_ref[h]) * softplus)
    k_A = k_ref[...]
    kv_U = jnp.sum(st_AU * k_A[:, None], axis=0)
    st_AU = st_AU + k_A[:, None] * ((v_ref[...] - kv_U) * beta)[None, :]
    state_out_ref[slot, h] = st_AU
    o_U = jnp.sum(st_AU * q_ref[...][:, None], axis=0)
    normed_U = o_U / jnp.sqrt(jnp.sum(o_U * o_U) / o_U.shape[0] + eps)
    z_U = z_ref[...].astype(jnp.float32)
    out_ref[...] = (w_ref[...].astype(jnp.float32) * normed_U * z_U / (1.0 + jnp.exp(-z_U))).astype(
        out_ref.dtype
    )


def gated_delta_decode_pallas(
    x_NC: jax.Array,
    z_NHU: jax.Array,
    b_NH: jax.Array,
    a_NH: jax.Array,
    conv_SKC: jax.Array,
    weight_CK: jax.Array,
    state_SHAU: jax.Array,
    A_log_H: jax.Array,
    dt_H: jax.Array,
    norm_U: jax.Array,
    slot_N: jax.Array,
    *,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pallas version of :func:`gated_delta_decode_xla`; the conv and state pools are aliased."""
    N, C = x_NC.shape
    H, U = z_NHU.shape[1:]
    num_k = (C - H * U) // (2 * U)
    if C != (2 * num_k + H) * U or state_SHAU.shape[2:] != (U, U):
        raise ValueError(f"Fused decode needs head_k_dim == head_v_dim; got C={C}, H={H}, U={U}.")
    gqa = H // num_k
    whole = pl.BlockSpec()
    act_NC, conv_SKC = pl.pallas_call(
        functools.partial(_conv_step_kernel, num_q=num_k, num_qk=2 * num_k),
        grid=(N, C // U),
        in_specs=[
            whole,
            pl.BlockSpec((None, U), lambda n, j: (n, j)),
            pl.BlockSpec((U, weight_CK.shape[1]), lambda n, j: (j, 0)),
            whole,
        ],
        out_specs=[pl.BlockSpec((None, U), lambda n, j: (n, j)), whole],
        out_shape=[
            jax.ShapeDtypeStruct((N, C), jnp.float32),
            jax.ShapeDtypeStruct(conv_SKC.shape, conv_SKC.dtype),
        ],
        input_output_aliases={3: 1},
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
        name="gated_delta_decode_conv",
    )(slot_N, x_NC, weight_CK, conv_SKC)
    head = lambda block: pl.BlockSpec((None, U), lambda n, h: (n, block(h)))
    out_NHU, state_SHAU = pl.pallas_call(
        functools.partial(_decode_step_kernel, eps=eps),
        grid=(N, H),
        in_specs=[
            whole,
            head(lambda h: h // gqa),
            head(lambda h: num_k + h // gqa),
            head(lambda h: 2 * num_k + h),
            whole,
            whole,
            whole,
            whole,
            pl.BlockSpec((None, None, U), lambda n, h: (n, h, 0)),
            whole,
            whole,
        ],
        out_specs=[pl.BlockSpec((None, None, U), lambda n, h: (n, h, 0)), whole],
        out_shape=[
            jax.ShapeDtypeStruct((N, H, U), x_NC.dtype),
            jax.ShapeDtypeStruct(state_SHAU.shape, state_SHAU.dtype),
        ],
        input_output_aliases={10: 1},
        compiler_params=plt.CompilerParams(num_warps=8, num_stages=1),
        name="gated_delta_decode_step",
    )(
        slot_N, act_NC, act_NC, act_NC, b_NH, a_NH, A_log_H.astype(jnp.float32),
        dt_H.astype(jnp.float32), z_NHU, norm_U, state_SHAU,
    )
    return out_NHU.reshape(N, H * U), conv_SKC, state_SHAU
