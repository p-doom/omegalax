"""Attention mask utilities and Context-Parallel (CP) attention for tokamax.

Context parallelism (Stage 1: all-gather-KV)
--------------------------------------------
tokamax CANNOT do seq-sharded KV on GPU (its top-level ``__call__`` raises
"Sharding along seq_k_axis unsupported"; the ring path exists only for TPU). So
CP here is **all-gather-KV**, not ring:

  * A ``shard_map`` over the ``cp`` axis (composed with the existing ``tp`` head
    sharding). Inside the body every tensor is UNSHARDED, so tokamax's own head /
    seq sharding logic is bypassed entirely -- we own the mesh.
  * Q stays sharded on the ``cp`` sequence axis (each rank holds its contiguous
    slice of tokens); K and V are ``jax.lax.all_gather``-ed across ``cp`` along
    the sequence axis so every rank sees the FULL K/V.
  * Causality is enforced with an explicit boolean mask built from GLOBAL token
    positions: ``q_pos`` = this rank's absolute positions, ``k_pos`` = all
    positions. We pass this mask with ``is_causal=False`` -- we do NOT rely on
    tokamax's shard-local causal auto-fill (it would apply a wrong, shard-local
    ``q_indices = arange(T_local)`` mask; see base.DotProductAttention.__call__).

Stage 1a is CONTIGUOUS sequence sharding (correctness first). Zig-zag causal
load-balancing (so late-shard ranks are not idle under a causal mask) is Stage
1b and is NOT implemented here. Full document / block-diagonal masking (using
``segment_ids``) is also a follow-on: Stage 1 masks causal-only but threads
``segment_ids`` as far as the global-index construction. DeltaNet-layer CP is
Stage 2.
"""

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from tokamax import dot_product_attention


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


# --- Context-Parallel (CP) all-gather-KV attention ---------------------------


def context_parallel_attention(
    q_BTHK: jax.Array,
    k_BTGK: jax.Array,
    v_BTGK: jax.Array,
    *,
    cp_axis: str,
    scale: float,
    heads_spec: P,
    implementation: str = "xla",
) -> jax.Array:
    """All-gather-KV context-parallel attention (Stage 1a, contiguous sharding).

    Runs a ``shard_map`` over ``(cp_axis, tp)`` (the head-parallel ``tp`` axis is
    part of ``heads_spec`` and is passed through unchanged, so this composes with
    tensor parallelism). Inside the body:

      1. all-gather K, V across ``cp_axis`` along the sequence axis (axis 1), so
         each rank holds the full ``(B, T_full, G_local, K)`` K/V while Q stays
         the local ``(B, T_local, H_local, K)`` slice;
      2. build a causal mask from GLOBAL token positions -- q positions are this
         rank's absolute offsets ``cp_index * T_local + arange(T_local)`` and k
         positions are ``arange(T_full)`` -- so ``q_pos >= k_pos`` is exact
         across the gathered KV (bypassing tokamax's shard-local causal fill);
      3. call ``tokamax.dot_product_attention`` on local-Q + gathered-KV with
         that mask and ``is_causal=False``.

    Args:
        q_BTHK: query, ``(B, T_local, H_local, K)``, sequence-sharded on ``cp``.
        k_BTGK, v_BTGK: key / value, ``(B, T_local, G_local, K)``, seq-sharded on ``cp``.
        cp_axis: name of the mesh axis carrying the sequence (context) shard.
        scale: attention logits scale (``1/sqrt(head_dim)``).
        heads_spec: the ``act_btnh`` PartitionSpec ``(batch, cp, tp, None)`` used
            for q/k/v; its cp entry must equal ``cp_axis``.
        implementation: tokamax backend for the local attention (``"xla"`` on
            CPU/A100, ``"mosaic_gpu"`` on H100). The mask path works with any.

    Returns:
        attn_BTHK: ``(B, T_local, H_local, K)``, sequence-sharded on ``cp`` (same
        layout as ``q_BTHK``).
    """
    mesh = jax.sharding.get_abstract_mesh()
    cp_size = mesh.shape[cp_axis]

    def _body(q_l, k_l, v_l):
        # Gather full K/V across cp along the sequence axis (axis=1).
        # tiled=True concatenates the cp shards in axis order -> contiguous global
        # sequence, matching the contiguous Stage-1a sharding of Q.
        k_full = jax.lax.all_gather(k_l, cp_axis, axis=1, tiled=True)
        v_full = jax.lax.all_gather(v_l, cp_axis, axis=1, tiled=True)

        t_local = q_l.shape[1]
        t_full = k_full.shape[1]
        cp_index = jax.lax.axis_index(cp_axis)

        # GLOBAL positions. Contiguous sharding: this rank owns tokens
        # [cp_index*t_local, (cp_index+1)*t_local).
        q_pos = cp_index * t_local + jnp.arange(t_local, dtype=jnp.int32)
        k_pos = jnp.arange(t_full, dtype=jnp.int32)

        # Causal mask over gathered KV, broadcast over (B, H): (1, 1, T_local, T_full).
        causal_TS = q_pos[:, None] >= k_pos[None, :]
        mask_11TS = causal_TS[None, None, :, :]

        return dot_product_attention(
            q_l,
            k_full,
            v_full,
            mask=mask_11TS,
            is_causal=False,
            scale=scale,
            implementation=implementation,
        )

    if cp_size == 1:
        # Defensive: a size-1 cp axis should never reach here (callers gate on
        # cp_size > 1). With no sharding to gather over, this is plain causal
        # attention -- and axis_index/all_gather are only valid inside a
        # shard_map, so we take the direct path.
        return dot_product_attention(
            q_BTHK,
            k_BTGK,
            v_BTGK,
            is_causal=True,
            scale=scale,
            implementation=implementation,
        )

    return shard_map(
        _body,
        mesh,
        in_specs=(heads_spec, heads_spec, heads_spec),
        out_specs=heads_spec,
        check_rep=False,
    )(q_BTHK, k_BTGK, v_BTGK)
