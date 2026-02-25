import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, get_abstract_mesh, reshard
from jaxtyping import Array

from .config import ShardingSpec

P = PartitionSpec


def shard(x, s: ShardingSpec):
    mesh = get_abstract_mesh()
    if not mesh.empty and len(mesh.axis_names) > 0 and isinstance(x, (jax.Array, jnp.ndarray)):
        return reshard(x, s)
    return x


def count_left_pads(token_ids_BT: Array) -> Array:
    return jnp.sum(jnp.cumsum(token_ids_BT != 0, axis=-1) == 0, -1)


def count_right_pads(token_ids_BT: jax.Array, pad_id: int) -> jax.Array:
    result = jnp.where(
        jnp.all(token_ids_BT == pad_id, axis=1),
        token_ids_BT.shape[1],
        jnp.argmin(jnp.flip(token_ids_BT == pad_id, axis=1).astype(jnp.int32), axis=1),
    )
    return jnp.max(result)


def compute_positions_from_segment_ids(seg_ids_BT: jax.Array) -> jax.Array:
    token_positions = jnp.arange(seg_ids_BT.shape[1], dtype=jnp.int32)[None, :]
    row_offsets = jnp.argmax(seg_ids_BT, axis=1, keepdims=True)
    relative_positions = token_positions - row_offsets
    default_positions = jnp.full_like(relative_positions, jnp.int32(2**30))
    return jax.lax.select(seg_ids_BT != 0, relative_positions, default_positions)
