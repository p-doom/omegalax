from __future__ import annotations

from typing import Any

from flax import nnx
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from omegalax.models.shard_config import ShardConfig

P = PartitionSpec


def init_model_sharded(
    model_cls: type[nnx.Module],
    cfg: Any,
    rng: jax.Array,
    mesh: Mesh,
    axis_rules: tuple[tuple[str, str | None], ...],
) -> nnx.Module:
    """Create a model with params born sharded. jax.jit is mandatory to avoid
    materializing a full unsharded copy (OOM for large models)."""
    with jax.set_mesh(mesh), nnx.logical_axis_rules(axis_rules):
        return jax.jit(lambda rng: model_cls(cfg, rngs=nnx.Rngs(rng)))(rng)


def batch_partition_spec(shd_cfg: ShardConfig) -> PartitionSpec:
    return P(shd_cfg.act_btd[0], None)


def shard_batch(token_ids_BT: jax.Array, shd_cfg: ShardConfig, mesh: Mesh) -> jax.Array:
    sharding = NamedSharding(mesh, batch_partition_spec(shd_cfg))
    return jax.make_array_from_process_local_data(sharding, token_ids_BT)
