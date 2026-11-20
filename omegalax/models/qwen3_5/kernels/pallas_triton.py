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
import os as _os

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp


# Set MFU_TRIINV=1 to replace the batched triangular solve in the WY/UT transform
# with matmuls.
#
# That solve computes the FULL inverse of ``I - A_pre`` -- its right-hand side is
# the identity -- and ``A_pre`` is strictly lower triangular, hence nilpotent with
# ``A_pre**C == 0``. For such a matrix the inverse has an exact finite form,
#
#     (I - N)^-1 = sum_{i<C} N^i = (I+N)(I+N^2)(I+N^4)...(I+N^(C/2))
#
# which is log2(C) squarings plus log2(C) products, so 10 batched 64x64 matmuls at
# C=64. cuBLAS ``batch_trsm_left_kernel`` costs 92.5 ms/step here because forward
# substitution is inherently sequential; matmuls are what this hardware is good at.
#
# Mathematically exact, but a different rounding path than the solve, so results
# are equivalent rather than bit-identical. Same dtype throughout, no precision
# change.
# !!! KAPUTT — NICHT VERWENDEN. Erzeugt loss=nan auf dem echten Modell ab Step 1.
#
# Das Neumann-Produkt bildet N^2, N^4, ... N^32. Fuer die Wertebereiche der echten
# Delta-Rule ueberlaeuft das in fp32; die Vorwaertssubstitution von
# solve_triangular bildet solche Potenzen nie und ist deshalb stabil.
#
# Warum es die Pruefungen passiert hat: der CPU-Unit-Test benutzte synthetische
# Normalverteilungen, und der GPU-Aequivalenztest ein Mock-Modell mit J=2 Chunks
# statt der 192 in Produktion. Beide trafen den echten Wertebereich nicht. Und die
# Benchmarks meldeten tok/s, ohne dass der Loss geprueft wurde -- NaN rechnet
# genauso schnell. Siehe E49.
#
# Eine stabile Variante muesste blockweise Vorwaertssubstitution verwenden statt
# hoher Potenzen. Nicht implementiert.
_TRI_INV = _os.environ.get("MFU_TRIINV", "0") == "1"
_TRI_BLOCKED = _os.environ.get("MFU_TRIBLOCK", "0") == "1"


# Block size for the blocked inverse. 16 measured fastest of 8/16/32 on the
# production shape (15360 matrices of 64x64): 1.35 ms against 2.40 ms for cuBLAS
# trsm, so 1.78x, at 1.79e-07 relative accuracy -- fp32 epsilon is 1.2e-07.
_TRI_BLOCK = int(_os.environ.get("MFU_TRI_BLOCK", "16"))


def _blocked_unit_lower_inverse(n_mat, block: int):
    """Inverse of ``eye - n_mat`` by block forward substitution.

    Splits the matrix into block rows and solves them against the already
    inverted blocks above. Only the small diagonal blocks go through
    ``solve_triangular``; the coupling is matmuls, which is where the speedup
    comes from.

    Unlike the doubling identity (see _unit_lower_inverse) this never forms
    matrix powers, so the catastrophic cancellation that made that version
    unusable cannot arise -- measured 1.79e-07 against trsm versus three orders of
    magnitude for the doubling form (E50). Precision is pinned to HIGHEST because
    JAX defaults float32 matmuls on GPU to TF32, which alone costs 3.8e-04.
    """
    hp = jax.lax.Precision.HIGHEST
    c = n_mat.shape[-1]
    nb = c // block
    eye_b = jnp.eye(block, dtype=n_mat.dtype)
    blk = [[n_mat[..., i * block:(i + 1) * block, j * block:(j + 1) * block]
            for j in range(nb)] for i in range(nb)]
    inv = [[None] * nb for _ in range(nb)]
    for i in range(nb):
        inv[i][i] = jax.scipy.linalg.solve_triangular(
            eye_b - blk[i][i], jnp.broadcast_to(eye_b, blk[i][i].shape), lower=True
        )
    for i in range(nb):
        for j in range(i - 1, -1, -1):
            acc = jnp.zeros_like(blk[i][j])
            for k in range(j, i):
                acc = acc + jnp.einsum("...ij,...jk->...ik", blk[i][k], inv[k][j],
                                       precision=hp)
            inv[i][j] = jnp.einsum("...ij,...jk->...ik", inv[i][i], acc, precision=hp)
    zero = jnp.zeros_like(inv[0][0])
    rows = [jnp.concatenate([inv[i][j] if j <= i else zero for j in range(nb)], axis=-1)
            for i in range(nb)]
    return jnp.concatenate(rows, axis=-2)


def _unit_lower_inverse(n_mat: jax.Array, eye: jax.Array) -> jax.Array:
    """Inverse of ``eye - n_mat`` for strictly lower triangular ``n_mat``."""
    acc = eye + n_mat
    p = n_mat
    k = 2
    while k < n_mat.shape[-1]:
        p = p @ p
        # Kept as acc @ (eye + p) although `eye + p` is a materialised
        # (B,H,J,C,C) fp32 intermediate costing 30.1 ms/step. The algebraically
        # identical `acc + acc @ p`, which never builds it, measured 2.3 % SLOWER
        # on one GPU and 2.0 % on two: XLA folds the identity add into the matmul
        # more cheaply than a separate add of the full result. Measured, not
        # assumed -- see E44.
        acc = acc @ (eye + p)
        k *= 2
    return acc


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
        compiler_params=plt.CompilerParams(num_warps=_WARPS, num_stages=_STAGES),
    )(kcd, k_dec, u_pre, df, initial_state)


def _state_pass_fwd_for_vjp(kcd, k_dec, u_pre, df, initial_state):
    state_in, v_new, final_state = _state_pass_pallas_fwd_only(kcd, k_dec, u_pre, df, initial_state)
    residuals = (kcd, k_dec, u_pre, df, initial_state)
    return (state_in, v_new, final_state), residuals


# Set MFU_BWD_PALLAS=1 to run the state-pass VJP as a Pallas kernel instead of a
# pure-JAX scan.
#
# The forward already uses a Pallas kernel precisely because, as the module
# docstring notes, "a pure-JAX scan was paying ~6 small dispatches per chunk". The
# backward below still pays exactly that: measured on this pipeline it costs
# 127.9 ms/step (5.8 %) spread over 23552 kernel launches, each 2.5-9.6 us, so it
# is bound by launch latency rather than arithmetic. This kernel mirrors the
# forward -- one program per (B, H), the dstate carry resident in registers across
# the J chunks, walked in reverse.
#
# Same equations in the same order as _state_pass_bwd, same fp32 throughout, so
# this is a scheduling change and not a precision change.
_BWD_PALLAS = _os.environ.get("MFU_BWD_PALLAS", "0") == "1"

# Launch geometry of both state-pass kernels. The values below are the ones that
# were hard-coded here; nothing in the repo records that they were measured, and
# the tiles are 128x128 with a 64-row chunk, so 4 warps (128 threads) is not
# obviously the right choice. Made tunable to find out.
_WARPS = int(_os.environ.get("MFU_PALLAS_WARPS", "4"))
_STAGES = int(_os.environ.get("MFU_PALLAS_STAGES", "2"))


def _state_pass_bwd_kernel(
    kcd_ref,          # (J, C, A)
    k_dec_ref,        # (J, C, A)
    df_ref,           # (J,)
    dstate_in_ref,    # (J, A, U)
    dv_new_ref,       # (J, C, U)
    dfinal_state_ref, # (A, U)
    dstate_traj_ref,  # (J, A, U)  dstate am ANFANG jedes Chunks
    du_pre_ref,       # (J, C, U)
    dinit_ref,        # (A, U)
    *,
    J: int,
):
    """Only the sequential part of the VJP: the dstate recurrence and du_pre.

    Bisecting the kernel showed that adding the dkcd and dk_dec writes to this
    loop makes Triton lowering segfault, while this variant lowers cleanly. Those
    three outputs (dkcd, dk_dec, ddf) do not need the recurrence at all -- they
    are pointwise in j once the dstate trajectory is known. Emitting the
    trajectory here and computing them outside turns 3 x J tiny per-chunk ops into
    three batched einsums over all J at once, which is both faster and avoids the
    construct that crashes.
    """
    def body(i, dstate):
        j = J - 1 - i
        dstate_traj_ref[j] = dstate.astype(dstate_traj_ref.dtype)
        # allow_tf32=False: pl.dot permits TF32 by default, which has ~10 mantissa
        # bits. Against the fp32 einsums of the scan it produced a 6.5e-04
        # deviation -- right at TF32's resolution and far above the ~1e-06 that
        # fp32 rounding gives. Matching the reference exactly matters more here
        # than the speed of the dot, since changing precision is out of scope.
        dv = dv_new_ref[j] + pl.dot(k_dec_ref[j], dstate, allow_tf32=False)
        du_pre_ref[j] = dv.astype(du_pre_ref.dtype)
        return (df_ref[j] * dstate
                - pl.dot(kcd_ref[j], dv, trans_a=True, allow_tf32=False)
                + dstate_in_ref[j])

    dinit_ref[:, :] = jax.lax.fori_loop(
        0, J, body, dfinal_state_ref[:, :].astype(jnp.float32)
    ).astype(dinit_ref.dtype)


def _state_pass_bwd_pallas(kcd, k_dec, state_in, v_new, df, dstate_in, dv_new, dfinal_state):
    B, H, J, C, A = kcd.shape
    U = v_new.shape[-1]
    spec_a = pl.BlockSpec((None, None, J, C, A), lambda b, h: (b, h, 0, 0, 0))
    spec_u = pl.BlockSpec((None, None, J, C, U), lambda b, h: (b, h, 0, 0, 0))
    spec_s = pl.BlockSpec((None, None, J, A, U), lambda b, h: (b, h, 0, 0, 0))
    spec_d = pl.BlockSpec((None, None, J), lambda b, h: (b, h, 0))
    spec_init = pl.BlockSpec((None, None, A, U), lambda b, h: (b, h, 0, 0))
    dstate_traj, du_pre, dinit = pl.pallas_call(
        functools.partial(_state_pass_bwd_kernel, J=J),
        out_shape=(
            jax.ShapeDtypeStruct((B, H, J, A, U), jnp.float32),
            jax.ShapeDtypeStruct((B, H, J, C, U), jnp.float32),
            jax.ShapeDtypeStruct((B, H, A, U), jnp.float32),
        ),
        grid=(B, H),
        in_specs=(spec_a, spec_a, spec_d, spec_s, spec_u, spec_init),
        out_specs=(spec_s, spec_u, spec_init),
        compiler_params=plt.CompilerParams(num_warps=_WARPS, num_stages=_STAGES),
    )(kcd, k_dec, df, dstate_in, dv_new, dfinal_state)

    # Pointwise in j, so batched over all chunks at once instead of inside the loop.
    hp = jax.lax.Precision.HIGHEST
    dk_dec = jnp.einsum("BHJCU,BHJAU->BHJCA", v_new, dstate_traj, precision=hp)
    dkcd = -jnp.einsum("BHJCU,BHJAU->BHJCA", du_pre, state_in, precision=hp)
    ddf = jnp.einsum("BHJAU,BHJAU->BHJ", state_in, dstate_traj, precision=hp)
    return dkcd, dk_dec, du_pre, ddf, dinit


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

    if _BWD_PALLAS:
        return _state_pass_bwd_pallas(
            kcd, k_dec, state_in, v_new, df, dstate_in, dv_new, dfinal_state
        )

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
    if _os.environ.get("MFU_TRIINV_DEBUG") == "1":
        # Beide Wege rechnen und vergleichen, statt die Ursache zu vermuten.
        _u = _blocked_unit_lower_inverse(A_pre, _TRI_BLOCK)
        _t = jax.scipy.linalg.solve_triangular(
            eye - A_pre, jnp.broadcast_to(eye, A_pre.shape), lower=True
        )
        _pw = A_pre
        for _ in range(5):
            _pw = _pw @ _pw
        jax.debug.print(
            "[triinv] max_A_pre={a} max_N32={p} max_inv_trsm={t} max_inv_neumann={u} "
            "nan_trsm={ft} nan_neumann={fu}",
            a=jnp.max(jnp.abs(A_pre)),
            p=jnp.max(jnp.abs(_pw)),
            t=jnp.max(jnp.where(jnp.isfinite(_t), jnp.abs(_t), 0.0)),
            u=jnp.max(jnp.where(jnp.isfinite(_u), jnp.abs(_u), 0.0)),
            ft=jnp.sum(1.0 - jnp.isfinite(_t).astype(jnp.float32)),
            fu=jnp.sum(1.0 - jnp.isfinite(_u).astype(jnp.float32)),
        )
    if _TRI_BLOCKED:
        A_inv = _blocked_unit_lower_inverse(A_pre, _TRI_BLOCK)
    elif _TRI_INV:
        A_inv = _unit_lower_inverse(A_pre, eye)
    else:
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
