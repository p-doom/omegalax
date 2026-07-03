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
from tokamax._src.ops.attention import base as tokamax_base
from tokamax._src.ops.attention.api import IMPLEMENTATIONS as _TOKAMAX_IMPLEMENTATIONS


def segment_ids_to_kstart(segment_ids_BT: jax.Array) -> jax.Array:
    """Start position of each token's segment (via boundary detect + cummax), O(T).

    ``segment_ids_BT``: (B, T), 0=padding, 1+=document id. Returns (B, T) k_start.
    """
    T = segment_ids_BT.shape[1]
    pos = jnp.arange(T, dtype=jnp.int32)[None, :]
    # Boundary where the segment id differs from the previous token (the first
    # token is always a boundary). The shifted-previous array is built from SLICES
    # of ``segment_ids_BT`` (``first - 1`` forces the first token to differ) so it
    # inherits the SAME layout -- a replicated ``ones`` column would clash with a
    # batch-sharded ``segment_ids_BT`` under an explicit-sharding mesh.
    first_col = segment_ids_BT[:, :1]
    prev_BT = jnp.concatenate([first_col - 1, segment_ids_BT[:, :-1]], axis=1)
    changes = segment_ids_BT != prev_BT
    boundary_positions = jnp.where(changes, pos, 0)
    return jax.lax.cummax(boundary_positions, axis=1)


def cu_seqlens_to_kstart(cu_seqlens: jax.Array, N: int) -> jax.Array:
    """Start position of each of ``N`` tokens' segment from cumulative seqlens
    (``cu_seqlens``: ``(num_segments + 1,)``, e.g. [0, 100, 250]). Returns (N,)."""
    seg_ids = jnp.searchsorted(cu_seqlens[1:], jnp.arange(N), side="right")
    return cu_seqlens[seg_ids]


# --- Non-CP block-diagonal (document) causal attention -----------------------


def _resolve_attention_impl(implementation: str):
    """Resolve a tokamax backend name to its ``Op`` instance.

    Mirrors the private dispatch inside ``tokamax.dot_product_attention``; the
    PUBLIC wrapper only accepts a boolean ``mask`` and hides ``k_start``/``k_end``,
    so to feed a per-row ``k_start`` (the nested-mask patch's whole point) we build
    the ``Mask`` ourselves and call the resolved implementation directly -- exactly
    what the public wrapper's tail does, just with a ``Mask`` instead of a bool.
    """
    if implementation == "mosaic":
        from jax.extend import backend

        kind = backend.get_default_device().device_kind
        implementation = "mosaic_gpu" if "NVIDIA" in kind else "mosaic_tpu"
    try:
        return _TOKAMAX_IMPLEMENTATIONS[implementation]
    except KeyError as exc:
        raise ValueError(f"Unknown tokamax attention implementation: {implementation!r}") from exc


def document_causal_attention(
    q_BTHK: jax.Array,
    k_BTGK: jax.Array,
    v_BTGK: jax.Array,
    segment_ids_BT: jax.Array,
    *,
    scale: float,
    implementation: str = "xla",
    q_sharding=None,
) -> jax.Array:
    """Block-diagonal (per-document) causal attention for packed sequences.

    Builds tokamax's per-row ``k_start`` (each query's segment start, via
    :func:`segment_ids_to_kstart`) and runs attention with
    ``Mask(k_start=k_start, is_causal=True)``: query row ``t`` may attend keys in
    ``[segment_start(t), t]`` -- causal WITHIN its document, nothing across a
    document boundary. When there is a single segment ``k_start`` is all-zero, so
    this reduces EXACTLY (bit-identical) to ``is_causal=True``; hence it is applied
    unconditionally -- the masking is DATA-DRIVEN off ``segment_ids_BT`` with no
    flag. Padding tokens (segment id 0) form a trailing segment that real queries
    never reach (they sit past the causal frontier) and whose rows the loss mask
    discards.
    """
    # tokamax's Mask.k_start is ``*#B #h #T``; a bare (B, T) is read as (#h, #T)
    # (B mistaken for a head axis, which also breaks the sharded shard_map). Insert
    # an explicit broadcast head axis -> (B, 1, T): per-batch, per-query k_start.
    k_start_B1T = segment_ids_to_kstart(segment_ids_BT)[:, None, :]
    impl = _resolve_attention_impl(implementation)
    mask = tokamax_base.Mask(k_start=k_start_B1T, is_causal=True)
    return impl(q_BTHK, k_BTGK, v_BTGK, mask=mask, logits_scale=scale, q_sharding=q_sharding)


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
