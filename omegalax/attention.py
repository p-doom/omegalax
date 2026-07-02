"""Attention mask utilities and Context-Parallel (CP) attention for tokamax.

Context parallelism (all-gather-KV)
-----------------------------------
tokamax CANNOT do seq-sharded KV on GPU (its top-level ``__call__`` raises
"Sharding along seq_k_axis unsupported"; the ring path exists only for TPU). So
CP here is **all-gather-KV**, not ring:

  * A ``shard_map`` over the ``cp`` axis (composed with the existing ``tp`` head
    sharding). Inside the body every tensor is UNSHARDED, so tokamax's own head /
    seq sharding logic is bypassed entirely -- we own the mesh.
  * Q stays sharded on the ``cp`` sequence axis (each rank holds its slice of
    tokens); K and V are ``jax.lax.all_gather``-ed across ``cp`` along the
    sequence axis so every rank sees the FULL K/V. The per-token POSITION ids
    (and, for document masking, SEGMENT ids) are all-gathered alongside K so the
    mask is built from GLOBAL, layout-independent values.
  * The mask is an explicit boolean array with ``is_causal=False`` -- we do NOT
    rely on tokamax's shard-local causal auto-fill (it would apply a wrong,
    shard-local ``q_indices = arange(T_local)`` mask). It combines:
      - causal ``q_pos >= k_pos`` over GLOBAL positions, and
      - (optional) block-diagonal document mask ``q_seg == k_seg`` (packed
        multi-document sequences must not attend across a document boundary).

Because the mask uses GLOBAL positions rather than ``arange`` of the local slice,
the sharding LAYOUT is arbitrary: contiguous (Stage 1a) OR zig-zag load-balanced
(Stage 1b) sharding both work with no change here — only the per-rank
``position_ids`` differ, and the callers pass those in. Zig-zag balances the
causal-triangle work across ranks (rank 0 would otherwise attend to almost
nothing, the last rank to everything); see
:func:`omegalax.distributed.zigzag`.

DeltaNet-layer CP is Stage 2 (omegalax/models/qwen3_5/kernels/cp.py).
"""

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P, reshard
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
    q_positions_BT: jax.Array,
    *,
    cp_axis: str,
    scale: float,
    heads_spec: P,
    seq_spec: P,
    q_segment_ids_BT: jax.Array | None = None,
    implementation: str = "xla",
) -> jax.Array:
    """All-gather-KV context-parallel attention (layout-agnostic).

    Runs a ``shard_map`` over ``(cp_axis, tp)`` (the head-parallel ``tp`` axis is
    part of ``heads_spec`` and is passed through unchanged, so this composes with
    tensor parallelism). Inside the body:

      1. all-gather K, V, the k-side POSITION ids, and (if given) the k-side
         SEGMENT ids across ``cp_axis`` along the sequence axis, so each rank
         holds the full ``(B, T_full, G_local, K)`` K/V + global (B, T_full)
         positions/segments while Q stays the local ``(B, T_local, H_local, K)``
         slice with its local ``q_positions``;
      2. build the mask from GLOBAL, layout-independent values:
           - causal  ``q_pos >= k_pos``  (positions, NOT arange, so contiguous
             AND zig-zag shardings both work), and
           - (optional) block-diagonal document mask ``q_seg == k_seg`` so packed
             multi-document sequences never attend across a document boundary;
      3. call ``tokamax.dot_product_attention`` on local-Q + gathered-KV with
         that mask and ``is_causal=False``.

    Args:
        q_BTHK: query, ``(B, T_local, H_local, K)``, sequence-sharded on ``cp``.
        k_BTGK, v_BTGK: key / value, ``(B, T_local, G_local, K)``, seq-sharded on ``cp``.
        q_positions_BT: ``(B, T_local)`` GLOBAL/original position id of each local
            q token (== ``arange`` under contiguous sharding, or the permuted
            original positions under zig-zag). Also used for the k-side (gathered).
        cp_axis: name of the mesh axis carrying the sequence (context) shard.
        scale: attention logits scale (``1/sqrt(head_dim)``).
        heads_spec: the ``act_btnh`` PartitionSpec ``(batch, cp, tp, None)`` for q/k/v.
        seq_spec: the ``(B, T)`` PartitionSpec ``(batch, cp)`` for positions/segments.
        q_segment_ids_BT: optional ``(B, T_local)`` document/segment id per token
            (0 = padding). When given, adds the block-diagonal document mask.
        implementation: tokamax backend (``"xla"`` on CPU/A100, ``"mosaic_gpu"`` on H100).

    Returns:
        attn_BTHK: ``(B, T_local, H_local, K)``, seq-sharded on ``cp`` (q's layout).
    """
    mesh = jax.sharding.get_abstract_mesh()
    cp_size = mesh.shape[cp_axis]
    use_seg = q_segment_ids_BT is not None

    def _body(q_l, k_l, v_l, q_pos_l, q_seg_l):
        # Gather full K/V + global k-side positions/segments across cp (axis 1 for
        # kv, axis 1 for the (B,T) position/segment arrays). tiled=True keeps the
        # cp shards in axis order (matching how Q's layout was sharded).
        k_full = jax.lax.all_gather(k_l, cp_axis, axis=1, tiled=True)
        v_full = jax.lax.all_gather(v_l, cp_axis, axis=1, tiled=True)
        k_pos = jax.lax.all_gather(q_pos_l, cp_axis, axis=1, tiled=True)  # (B, T_full)

        q_pos = q_pos_l  # (B, T_local)
        # Causal over GLOBAL positions: (B, 1, T_local, T_full).
        mask = q_pos[:, None, :, None] >= k_pos[:, None, None, :]
        if use_seg:
            k_seg = jax.lax.all_gather(q_seg_l, cp_axis, axis=1, tiled=True)  # (B, T_full)
            # Block-diagonal document mask: same non-padding segment id. Padding
            # (seg == 0) attends to nothing (and is not attended to).
            same_seg = (q_seg_l[:, None, :, None] == k_seg[:, None, None, :]) & (
                q_seg_l[:, None, :, None] != 0
            )
            mask = mask & same_seg

        return dot_product_attention(
            q_l,
            k_full,
            v_full,
            mask=mask,
            is_causal=False,
            scale=scale,
            implementation=implementation,
        )

    if cp_size == 1:
        # Defensive: a size-1 cp axis should never reach here (callers gate on
        # cp_size > 1); axis_index/all_gather are only valid inside a shard_map.
        if use_seg:
            same = (q_segment_ids_BT[:, None, :, None] == q_segment_ids_BT[:, None, None, :]) & (
                q_segment_ids_BT[:, None, :, None] != 0
            )
            causal = q_positions_BT[:, None, :, None] >= q_positions_BT[:, None, None, :]
            return dot_product_attention(
                q_BTHK, k_BTGK, v_BTGK, mask=causal & same, is_causal=False,
                scale=scale, implementation=implementation,
            )
        return dot_product_attention(
            q_BTHK, k_BTGK, v_BTGK, is_causal=True, scale=scale,
            implementation=implementation,
        )

    seg_in = q_segment_ids_BT if use_seg else jnp.zeros_like(q_positions_BT)
    # position/segment ids may arrive replicated (model-internal arange) or already
    # cp-sharded (from the batch); reshard both to the cp seq layout so the
    # shard_map in_specs match and each rank gets its slice. Under contiguous CP a
    # resharded arange gives contiguous positions; under zig-zag the caller already
    # supplies the permuted true positions.
    q_positions_BT = reshard(q_positions_BT, seq_spec)
    seg_in = reshard(seg_in, seq_spec)
    return shard_map(
        _body,
        mesh,
        in_specs=(heads_spec, heads_spec, heads_spec, seq_spec, seq_spec),
        out_specs=heads_spec,
        check_rep=False,
    )(q_BTHK, k_BTGK, v_BTGK, q_positions_BT, seg_in)
