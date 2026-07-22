"""Attention mask utilities for tokamax integration.

Sequence packing (multiple independent training examples concatenated into one
``max_length`` row) requires *block-diagonal causal* attention: a query token must
attend only to earlier keys **within its own segment**, never across a segment
boundary. This is expressed to tokamax as ``Mask(k_start=<segment start>,
is_causal=True)`` — ``k_start`` supplies the per-query lower key bound (the first
index of that query's segment) and ``is_causal`` supplies the upper bound (the
diagonal). Both the ``mosaic_gpu`` (sm90/sm100) kernel — which additionally skips
whole KV tiles below ``k_start`` for throughput — and the ``xla`` reference honour
``k_start`` as a hard per-element mask, so cross-segment attention is impossible,
not merely discouraged.
"""

import jax
import jax.numpy as jnp

from tokamax._src.ops.attention.api import IMPLEMENTATIONS as _TOKAMAX_IMPLEMENTATIONS
from tokamax._src.ops.attention.base import Mask as _TokamaxMask


def segmented_causal_attention(
    q_BTHK: jax.Array,
    k_BTGK: jax.Array,
    v_BTGK: jax.Array,
    *,
    scale: float,
    backend: str,
    q_sharding,
    k_start_B1T: jax.Array,
) -> jax.Array:
    """Block-diagonal causal attention for packed sequences.

    Each query token attends only to keys ``[k_start[t], t]`` — i.e. keys at or
    after the start of its own segment (``k_start``) and at or before itself
    (causal). Tokens of one packed sub-sequence therefore can NEVER attend to
    another sub-sequence. This is the single, non-optional mechanism that
    enforces cross-segment isolation; there is no fallback to full attention.

    Args:
        q_BTHK, k_BTGK, v_BTGK: attention inputs, already RoPE'd, ``(B, T, N, H)``.
        scale: logits scale (``head_dim ** -0.5``).
        backend: a key of tokamax's attention ``IMPLEMENTATIONS`` (e.g.
            ``"mosaic_gpu"`` on H100, ``"xla"`` on CPU). No silent fallback: an
            unsupported backend raises.
        q_sharding: ``NamedSharding`` for the query, or ``None``.
        k_start_B1T: int32 ``(B, 1, T)`` — the first key index of the segment
            containing each query token. Broadcasts over the head axis.

    Returns:
        Attention output ``(B, T, N, H)`` in the input dtype.
    """
    if backend not in _TOKAMAX_IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown attention backend {backend!r} for packed attention; "
            f"expected one of {sorted(_TOKAMAX_IMPLEMENTATIONS)}."
        )
    op = _TOKAMAX_IMPLEMENTATIONS[backend]
    mask = _TokamaxMask(k_start=k_start_B1T.astype(jnp.int32), is_causal=True)
    in_dtype = q_BTHK.dtype
    # tokamax attention kernels require fp16/bf16 (matches the causal path).
    out = op(
        q_BTHK.astype(jnp.bfloat16),
        k_BTGK.astype(jnp.bfloat16),
        v_BTGK.astype(jnp.bfloat16),
        mask=mask,
        logits_scale=scale,
        q_sharding=q_sharding,
    )
    return out.astype(in_dtype)


def segment_ids_to_kstart(segment_ids_BT: jax.Array) -> jax.Array:
    """Convert segment IDs to k_start for tokamax Mask.

    For each position, computes the start index of its segment using
    boundary detection + cumulative max. Handles left-padded single
    sequences and multi-document packing in O(T).

    Args:
        segment_ids_BT: (B, T) where 0=padding, 1+=document ID.

    Returns:
        k_start_BT: (B, T) where k_start[b, t] is the start position
        of the segment containing token t in batch element b.
    """
    B, T = segment_ids_BT.shape
    pos = jnp.arange(T)[None, :]
    changes = jnp.concatenate(
        [
            jnp.ones((B, 1), dtype=jnp.bool_),
            segment_ids_BT[:, 1:] != segment_ids_BT[:, :-1],
        ],
        axis=1,
    )
    boundary_positions = jnp.where(changes, pos, 0)
    return jax.lax.cummax(boundary_positions, axis=1)


def cu_seqlens_to_kstart(cu_seqlens: jax.Array, N: int) -> jax.Array:
    """Convert cumulative sequence lengths to k_start.

    Args:
        cu_seqlens: (num_segments + 1,) cumulative token counts, e.g. [0, 100, 250].
        N: total number of tokens.

    Returns:
        k_start_N: (N,) where k_start[t] is the start of the segment containing token t.
    """
    seg_ids = jnp.searchsorted(cu_seqlens[1:], jnp.arange(N), side="right")
    return cu_seqlens[seg_ids]
