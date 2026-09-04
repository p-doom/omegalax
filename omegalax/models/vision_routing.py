from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, reshard

P = PartitionSpec


def _vision_token_destinations(
    image_mask_BT: jax.Array,
    vision_patch_valid: jax.Array,
    spatial_merge_size: int,
) -> tuple[jax.Array, jax.Array]:
    image_mask_BT = reshard(image_mask_BT, P())
    vision_patch_valid = reshard(vision_patch_valid, P())
    merged_valid_N = jnp.all(
        vision_patch_valid.reshape(-1, spatial_merge_size**2),
        axis=1,
    )
    embedding_rank_N = jnp.cumsum(merged_valid_N.astype(jnp.int32)) - 1
    num_embeddings = merged_valid_N.shape[0]
    seq_len = image_mask_BT.shape[1]
    image_batch_N, image_seq_N = jnp.where(
        image_mask_BT,
        size=num_embeddings,
        fill_value=(0, seq_len),
    )
    destination_N = jnp.maximum(embedding_rank_N, 0)
    return (
        jnp.where(merged_valid_N, image_batch_N[destination_N], 0),
        jnp.where(merged_valid_N, image_seq_N[destination_N], seq_len),
    )
