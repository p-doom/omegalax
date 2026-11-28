import jax
import jax.numpy as jnp


def compute_positions_from_segment_ids(seg_ids_BT: jax.Array) -> jax.Array:
    token_positions = jnp.arange(seg_ids_BT.shape[1], dtype=jnp.int32)[None, :]
    row_offsets = jnp.argmax(seg_ids_BT, axis=1, keepdims=True)
    relative_positions = token_positions - row_offsets
    default_positions = jnp.full_like(relative_positions, jnp.int32(2**30))
    return jax.lax.select(seg_ids_BT != 0, relative_positions, default_positions)
