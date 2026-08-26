"""Attention mask utilities and Context-Parallel (CP) attention for tokamax.

CP here is all-gather-KV (not ring): tokamax cannot do seq-sharded KV on GPU. A
``shard_map`` over ``cp`` (composed with the ``tp`` head sharding) keeps Q on the
local sequence slice and ``all_gather``s the full K/V plus the k-side position and
segment ids. The mask is an explicit boolean array (``is_causal=False``, since
tokamax's shard-local causal auto-fill would use a wrong ``arange(T_local)``):
causal ``q_pos >= k_pos`` over GLOBAL positions, optionally AND a block-diagonal
``q_seg == k_seg`` document mask.

Because the mask is built from GLOBAL positions (not ``arange`` of the local
slice), the sharding LAYOUT is arbitrary -- contiguous or zig-zag load-balanced
both work unchanged, only the per-rank ``position_ids`` (passed in) differ. See
:func:`omegalax.distributed.zigzag`. DeltaNet-layer CP is Stage 2
(omegalax/models/qwen3_5/kernels/cp.py).
"""

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P, reshard
from tokamax import dot_product_attention


def segment_ids_to_kstart(segment_ids_BT: jax.Array) -> jax.Array:
    """Start position of each token's segment (via boundary detect + cummax), O(T).

    ``segment_ids_BT``: (B, T), 0=padding, 1+=document id. Returns (B, T) k_start.
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
    """Start position of each of ``N`` tokens' segment from cumulative seqlens
    (``cu_seqlens``: ``(num_segments + 1,)``, e.g. [0, 100, 250]). Returns (N,)."""
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

    ``shard_map`` over ``(cp_axis, tp)`` (``tp`` rides ``heads_spec`` unchanged, so
    this composes with tensor parallelism): all-gather K, V and the k-side
    position/segment ids across ``cp``, keep local Q, then run tokamax attention on
    local-Q + gathered-KV with an explicit mask (causal over GLOBAL positions AND,
    if ``q_segment_ids_BT`` given, a block-diagonal document mask).

    ``q_positions_BT`` (B, T_local) are the GLOBAL/original positions (== arange
    under contiguous sharding, permuted under zig-zag). ``heads_spec`` is the
    ``act_btnh`` spec ``(batch, cp, tp, None)``; ``seq_spec`` the ``(batch, cp)``
    spec for positions/segments. ``implementation`` is the tokamax backend ("xla"
    on CPU/A100, "mosaic_gpu" on H100). Returns q's ``(B, T_local, H_local, K)``.
    """
    mesh = jax.sharding.get_abstract_mesh()
    cp_size = mesh.shape[cp_axis]
    use_seg = q_segment_ids_BT is not None

    def _body(q_l, k_l, v_l, q_pos_l, q_seg_l):
        # Gather full K/V + global k-side positions/segments across cp; tiled=True
        # keeps the cp shards in axis order (matching Q's layout).
        k_full = jax.lax.all_gather(k_l, cp_axis, axis=1, tiled=True)
        v_full = jax.lax.all_gather(v_l, cp_axis, axis=1, tiled=True)
        k_pos = jax.lax.all_gather(q_pos_l, cp_axis, axis=1, tiled=True)  # (B, T_full)

        q_pos = q_pos_l  # (B, T_local)
        # Causal over GLOBAL positions: (B, 1, T_local, T_full).
        mask = q_pos[:, None, :, None] >= k_pos[:, None, None, :]
        if use_seg:
            k_seg = jax.lax.all_gather(q_seg_l, cp_axis, axis=1, tiled=True)  # (B, T_full)
            # Block-diagonal document mask; padding (seg == 0) attends to nothing.
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
        # Defensive: callers gate on cp_size > 1, so this path is not normally hit.
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
    # position/segment ids may arrive replicated or already cp-sharded; reshard both
    # to the cp seq layout so the shard_map in_specs match and each rank gets its slice.
    q_positions_BT = reshard(q_positions_BT, seq_spec)
    seg_in = reshard(seg_in, seq_spec)
    return shard_map(
        _body,
        mesh,
        in_specs=(heads_spec, heads_spec, heads_spec, seq_spec, seq_spec),
        out_specs=heads_spec,
        check_rep=False,
    )(q_BTHK, k_BTGK, v_BTGK, q_positions_BT, seg_in)
