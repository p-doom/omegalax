import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, reshard
from jaxtyping import Array


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
    # ``arange`` over the FULL (static) T is the GLOBAL absolute position, so RoPE
    # is positionwise-correct even when seg_ids_BT is sequence-sharded on "cp"
    # (context parallelism): each cp shard sees the full arange and its own slice
    # of seg_ids, so position[t] == t globally.
    token_positions = jnp.arange(seg_ids_BT.shape[1], dtype=jnp.int32)[None, :]
    row_offsets = jnp.argmax(seg_ids_BT, axis=1, keepdims=True)
    relative_positions = token_positions - row_offsets
    default_positions = jnp.full_like(relative_positions, jnp.int32(2**30))
    # Under context parallelism seg_ids_BT is sequence-sharded (P(batch, "cp")),
    # but the arange-derived cases above are replicated, which makes ``select``
    # raise on the sharding mismatch. Re-tie the T-axis sharding of the select
    # cases to seg_ids' sharding so the (positionwise) result stays cp-sharded.
    # No-op when seg_ids_BT is unsharded (seq entry of the spec is None).
    seg_sharding = getattr(jax.typeof(seg_ids_BT), "sharding", None)
    seg_spec = getattr(seg_sharding, "spec", None)
    if seg_spec is not None and len(seg_spec) >= 2 and seg_spec[1] is not None:
        case_spec = PartitionSpec(seg_spec[0], seg_spec[1])
        relative_positions = reshard(relative_positions, case_spec)
        default_positions = reshard(default_positions, case_spec)
    return jax.lax.select(seg_ids_BT != 0, relative_positions, default_positions)
