"""Context-Parallel (CP Stage 2) gated-delta-rule over contiguous seq segments.

The gated-delta recurrence is LINEAR in the recurrent state, which is what makes
sequence-sharded (context-parallel) DeltaNet exact rather than approximate.
Splitting a sequence into contiguous per-cp-rank segments, each rank's local
chunked scan is an affine map on the incoming boundary state:

    state_final[r] = A_r @ S_in[r] + B_r

where ``(A_r, B_r)`` is the segment's AGGREGATE transition (``A_r`` a per-head
(A,A) matrix, ``B_r`` = the (A,U) final state started from zero). The incoming
boundary states chain across the cp ring:

    S_in[0] = 0,   S_in[r] = A_{r-1} @ S_in[r-1] + B_{r-1}

i.e. ``S_in[r]`` is the affine composition of segments 0..r-1 applied to 0. Since
affine composition ``(A2,B2)∘(A1,B1) = (A2@A1, A2@B1+B2)`` is associative, the
cross-rank resolution is an ``all_gather`` of ``(A_r, B_r)`` over ``cp`` followed
by an ``associative_scan`` (exclusive prefix) — cheap because ``cp`` is small
(<= ~8). Each rank then runs its LOCAL output kernel seeded with ``S_in[r]``.

This is bit-identical to the full-sequence result (verified) because nothing is
approximated: the affine algebra is exact and the per-segment kernel is the same
one used non-CP, just started from ``S_in`` instead of zeros.

Stage-2 scope note: contiguous sequence sharding (Stage 1a layout). Zig-zag
causal load-balancing (Stage 1b) and full document masking remain follow-ons.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .xla_reference import _l2norm


def _segment_state_transition(
    q_BTHA: jax.Array,
    k_BTHA: jax.Array,
    v_BTHU: jax.Array,
    g_BTH: jax.Array,
    beta_BTH: jax.Array,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Aggregate affine state transition ``(A_seg, B_seg)`` of one seq segment.

    ``state_final = A_seg @ state_init + B_seg`` for ANY incoming ``state_init``
    (verified bit-identical against seeding the kernel with ``state_init``).
    ``B_seg`` equals the final state started from zero.

    Returns:
        A_seg: (B, H, A, A) per-head aggregate transition on the key axis.
        B_seg: (B, H, A, U) per-head aggregate bias (= final state from zero).

    Uses the same WY/UT prep as the kernel, then folds the per-chunk affine maps
    ``M[j] = df[j]*I - k_dec[j].T @ kcd[j]`` / ``b[j] = k_dec[j].T @ u_pre[j]``
    into the segment aggregate via a J-scan. Pure XLA (differentiable); cp is
    small so this runs once per segment and is cheap vs the output kernel.
    """
    q_BTHA = _l2norm(q_BTHA, axis=-1)
    k_BTHA = _l2norm(k_BTHA, axis=-1)
    k_BHTA = k_BTHA.transpose(0, 2, 1, 3).astype(jnp.float32)
    v_BHTU = v_BTHU.transpose(0, 2, 1, 3).astype(jnp.float32)
    beta_BHT = beta_BTH.transpose(0, 2, 1).astype(jnp.float32)
    g_BHT = g_BTH.transpose(0, 2, 1).astype(jnp.float32)

    B, H, T, A = k_BHTA.shape
    U = v_BHTU.shape[-1]
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad:
        k_BHTA = jnp.pad(k_BHTA, ((0, 0), (0, 0), (0, pad), (0, 0)))
        v_BHTU = jnp.pad(v_BHTU, ((0, 0), (0, 0), (0, pad), (0, 0)))
        beta_BHT = jnp.pad(beta_BHT, ((0, 0), (0, 0), (0, pad)))
        g_BHT = jnp.pad(g_BHT, ((0, 0), (0, 0), (0, pad)))
    total_T = T + pad
    vb = v_BHTU * beta_BHT[..., None]
    kb = k_BHTA * beta_BHT[..., None]
    J = total_T // chunk_size
    k_ = k_BHTA.reshape(B, H, J, chunk_size, A)
    kb_ = kb.reshape(B, H, J, chunk_size, A)
    vb_ = vb.reshape(B, H, J, chunk_size, U)
    g_ = g_BHT.reshape(B, H, J, chunk_size)
    g_ = jnp.cumsum(g_, axis=-1)

    diff = g_[..., :, None] - g_[..., None, :]
    tril = jnp.tril(jnp.ones((chunk_size, chunk_size)))
    decay_mask = jnp.exp(diff * tril) * tril
    upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    attn = -(jnp.einsum("BHJLA,BHJMA->BHJLM", kb_, k_) * decay_mask)
    attn = jnp.where(upper, 0.0, attn)
    eye = jnp.eye(chunk_size, dtype=attn.dtype)
    lhs = eye - attn
    rhs = jnp.broadcast_to(eye, lhs.shape)
    attn = jax.scipy.linalg.solve_triangular(lhs, rhs, lower=True)

    u_pre = jnp.einsum("BHJLM,BHJMU->BHJLU", attn, vb_)
    kcd = jnp.einsum("BHJLM,BHJMA->BHJLA", attn, kb_ * jnp.exp(g_)[..., None])
    g_last = g_[..., -1]
    df = jnp.exp(g_last)
    decay_to_end = jnp.exp(g_last[..., None] - g_)
    k_dec = k_ * decay_to_end[..., None]

    # Per-chunk affine map on the state:  state[j+1] = M[j] @ state[j] + b[j].
    I_A = jnp.eye(A)
    M_j = df[..., None, None] * I_A - jnp.einsum("BHJMX,BHJMY->BHJXY", k_dec, kcd)
    b_j = jnp.einsum("BHJMX,BHJMU->BHJXU", k_dec, u_pre)

    def acc(carry, j):
        A_acc, B_acc = carry
        A_next = jnp.einsum("BHXY,BHYZ->BHXZ", M_j[:, :, j], A_acc)
        B_next = jnp.einsum("BHXY,BHYU->BHXU", M_j[:, :, j], B_acc) + b_j[:, :, j]
        return (A_next, B_next), None

    A0 = jnp.broadcast_to(I_A, (B, H, A, A))
    B0 = jnp.zeros((B, H, A, U))
    (A_seg, B_seg), _ = jax.lax.scan(acc, (A0, B0), jnp.arange(J))
    return A_seg, B_seg


def _exclusive_prefix_S_in(
    A_r_BHAA: jax.Array,
    B_r_BHAU: jax.Array,
    cp_axis: str,
) -> jax.Array:
    """Resolve this rank's incoming boundary state ``S_in`` across the cp ring.

    ``all_gather`` the per-rank ``(A_r, B_r)`` over ``cp`` (small axis), then take
    the EXCLUSIVE affine prefix so ``S_in[r]`` is the composition of segments
    0..r-1 applied to zero (``S_in[0] = 0``). Because the prefix is applied to the
    zero state, ``S_in[r]`` is exactly the accumulated bias of segments 0..r-1
    (the ``A`` part multiplies zero), so we only need the bias channel of the
    inclusive scan, shifted by one rank.

    Must be called INSIDE a shard_map over ``cp`` (uses ``jax.lax.all_gather`` /
    ``axis_index``). Returns this rank's ``S_in`` of shape ``(B, H, A, U)``.
    """
    # Gather all ranks' (A_r, B_r) along a new leading cp axis: (cp, B, H, A, A/U).
    A_all = jax.lax.all_gather(A_r_BHAA, cp_axis, axis=0, tiled=False)
    B_all = jax.lax.all_gather(B_r_BHAU, cp_axis, axis=0, tiled=False)

    def compose(x, y):
        # affine (A2,B2) ∘ (A1,B1) = (A2@A1, A2@B1 + B2); scanned over cp (axis 0).
        A1, B1 = x
        A2, B2 = y
        A_c = jnp.einsum("cBHXY,cBHYZ->cBHXZ", A2, A1)
        B_c = jnp.einsum("cBHXY,cBHYU->cBHXU", A2, B1) + B2
        return A_c, B_c

    # Inclusive prefix over cp: incl[r] = compose(segs 0..r).
    _, B_incl = jax.lax.associative_scan(compose, (A_all, B_all), axis=0)

    cp_size = A_all.shape[0]
    r = jax.lax.axis_index(cp_axis)
    B, H, A, U = A_r_BHAA.shape[0], A_r_BHAA.shape[1], A_r_BHAA.shape[2], B_r_BHAU.shape[-1]
    zero = jnp.zeros((B, H, A, U), dtype=B_all.dtype)
    # Exclusive prefix bias: S_in[r] = B_incl[r-1] for r>0, else 0.
    # (Applied-to-zero, so only the accumulated bias matters.)
    B_excl = jnp.concatenate([zero[None], B_incl[: cp_size - 1]], axis=0)  # (cp,B,H,A,U)
    S_in = B_excl[r]
    return S_in
