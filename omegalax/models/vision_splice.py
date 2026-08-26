"""Destination indices for splicing vision embeddings into text token positions.

The vision tower emits one merged embedding per ``merge_size**2`` block of
``pixel_values`` rows, in row order. The text side wants those embeddings at the
``<|image_pad|>`` positions of ``token_ids_BT``, in row-major ``(batch, seq)``
order. Pairing the k-th embedding with the k-th image token positionally is only
correct when every embedding before it is a real one.

That assumption breaks under multi-process data loading. Each process's collator
pads its own local block to the static vision budget and appends the dummy patch
rows at the end of *its* block, so
``jax.make_array_from_process_local_data`` builds the global layout
``[b0_real | b0_pad | b1_real | b1_pad | ...]`` while the global image-token mask
is a gapless ``[b0 | b1 | ...]``. From the first process boundary on, every image
token would receive a padding-derived embedding.

``vision_patch_valid`` (emitted by the collator, one flag per ``pixel_values``
row, sharded on the same axis) restores the correspondence: an embedding's
destination is decided by its *rank among the real embeddings*, which is
invariant to where each process parked its padding.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard


def merged_embed_valid(
    vision_patch_valid: jax.Array | None,
    num_embeds: int,
) -> jax.Array | None:
    """Reduce a per-patch-row validity mask to one flag per merged embedding.

    Padding is appended at whole-image granularity and every image contributes a
    multiple of ``merge_size**2`` patch rows, so validity is constant inside a
    merge block and the first row of each block decides it.
    """
    if vision_patch_valid is None:
        return None
    if num_embeds == 0:
        return jnp.zeros((0,), dtype=bool)
    valid_P = reshard(vision_patch_valid, P()).astype(bool)
    return valid_P.reshape(num_embeds, -1)[:, 0]


def image_embed_destinations(
    image_mask_BT: jax.Array,
    num_embeds: int,
    valid_N: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(batch_idx_N, seq_idx_N)`` scatter destinations for each embedding.

    Slot ``k`` targets the ``rank(k)``-th image token in row-major order, where
    ``rank(k)`` counts the real embeddings at or before ``k``. Padding slots are
    aimed at the out-of-range column ``seq_len`` so a ``mode="drop"`` scatter
    discards them.

    ``valid_N=None`` reproduces the positional pairing exactly, which is what an
    all-real (single-process, or unpadded) batch resolves to anyway.
    """
    seq_len = image_mask_BT.shape[1]
    mask_replicated = reshard(image_mask_BT, P())
    batch_idx_N, seq_idx_N = jnp.where(
        mask_replicated,
        size=num_embeds,
        fill_value=(0, seq_len),
    )
    if valid_N is None:
        return batch_idx_N, seq_idx_N

    dest_rank_N = jnp.maximum(jnp.cumsum(valid_N.astype(jnp.int32)) - 1, 0)
    return (
        batch_idx_N[dest_rank_N],
        jnp.where(valid_N, seq_idx_N[dest_rank_N], seq_len),
    )
