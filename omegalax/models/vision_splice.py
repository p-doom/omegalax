"""Scatter index that splices vision embeddings onto their image-pad tokens."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def vision_scatter_index(image_mask_BT: jax.Array, n_embeds: int) -> tuple[jax.Array, jax.Array]:
    """Return ``(batch_idx, seq_idx)`` placing embedding row ``k`` on its own token.

    ``pixel_values`` is padded *per sample*, so the embeddings split into equal
    blocks: rows ``[b*E, (b+1)*E)`` belong to batch row ``b``, and the k-th of
    them fills that row's k-th ``<|image_pad|>`` token. Rows past a sample's real
    image tokens are padding and get column ``T``, which an out-of-bounds-dropping
    scatter discards.

    Pairing the rows positionally instead — one flat ``jnp.where`` over the whole
    mask, consuming embeddings in token order — assumes the embeddings hold no
    interior padding. They do: the collator pads each process's block locally and
    ``make_array_from_process_local_data`` concatenates the blocks, so the pads
    sit *between* samples. Every sample after the first then reads off the end of
    an earlier sample's block, shifted by the accumulated padding. Single-process
    runs have no interior pads and so never show it.

    The block mapping holds under any mesh ordering: ``pixel_values`` and the
    batch are sharded along the same axis with a constant ratio ``E``, so shard
    ``k`` of the patches always covers exactly the samples in shard ``k`` of the
    batch.
    """
    batch_size, seq_len = image_mask_BT.shape
    embeds_per_sample, remainder = divmod(n_embeds, batch_size)
    if remainder:
        raise ValueError(
            f"vision embeddings ({n_embeds}) are not a whole number per sample for "
            f"batch_size={batch_size}. pixel_values must be padded per sample — "
            "set max_vision_patches_per_sample / max_vision_images_per_sample."
        )

    seq_idx_BE = jax.vmap(
        lambda mask_T: jnp.where(mask_T, size=embeds_per_sample, fill_value=seq_len)[0]
    )(image_mask_BT)
    batch_idx_N = jnp.repeat(jnp.arange(batch_size), embeds_per_sample)
    return batch_idx_N, seq_idx_BE.reshape(-1)
