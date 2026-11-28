"""Attention mask utilities for tokamax integration."""

import jax
import jax.numpy as jnp


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
    T = segment_ids_BT.shape[1]
    pos = jnp.arange(T)[None, :]
    # Pad, not concatenate with a fresh column: it keeps segment_ids_BT's (explicit) sharding.
    changes = jnp.pad(segment_ids_BT[:, 1:] != segment_ids_BT[:, :-1], ((0, 0), (1, 0)), constant_values=True)
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
