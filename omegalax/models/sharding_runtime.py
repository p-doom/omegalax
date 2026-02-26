from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flax import nnx
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from omegalax.models.shard_config import ShardConfig

P = PartitionSpec

_ModelStateShardingFn = Callable[[nnx.State, ShardConfig, Mesh], nnx.State]
_ShardConfigGetter = Callable[[Any], ShardConfig]


def _merge_axes(left: object, right: object) -> object:
    if left is None:
        return right
    if right is None:
        return left
    return (left, right)


def ndh_to_linear_kernel(spec_ndh: PartitionSpec) -> PartitionSpec:
    """Conceptual layout N D H -> linear kernel layout D (N*H)."""
    return P(spec_ndh[1], _merge_axes(spec_ndh[0], spec_ndh[2]))


def nhd_to_linear_kernel(spec_nhd: PartitionSpec) -> PartitionSpec:
    """Conceptual layout N H D -> linear kernel layout (N*H) D."""
    return P(_merge_axes(spec_nhd[0], spec_nhd[1]), spec_nhd[2])


def vision_axis_spec(shd_cfg: ShardConfig) -> PartitionSpec:
    return P(shd_cfg.act_btd[2])


def vision_none_axis_spec(shd_cfg: ShardConfig) -> PartitionSpec:
    return P(None, shd_cfg.act_btd[2])


def ffw_df_param_spec(shd_cfg: ShardConfig) -> PartitionSpec:
    return P(None, *shd_cfg.ffw_weight_df)


def ffw_fd_param_spec(shd_cfg: ShardConfig) -> PartitionSpec:
    return P(None, *shd_cfg.ffw_weight_fd)


def init_sharded_model(
    model_cls: type[nnx.Module],
    cfg: Any,
    rng: jax.Array,
    mesh: Mesh,
    model_state_sharding_fn: _ModelStateShardingFn,
    shd_cfg_getter: _ShardConfigGetter,
) -> nnx.Module:
    model_shape = nnx.eval_shape(lambda: model_cls(cfg, rngs=nnx.Rngs(rng)))
    state_shape = nnx.state(model_shape)
    state_sharding = model_state_sharding_fn(state_shape, shd_cfg_getter(cfg), mesh)
    replicated = NamedSharding(mesh, P())

    def init_state(init_rng: jax.Array):
        model = model_cls(cfg, rngs=nnx.Rngs(init_rng))
        return nnx.state(model)

    state = jax.jit(
        init_state,
        in_shardings=replicated,
        out_shardings=state_sharding,
    )(rng)
    return nnx.merge(nnx.graphdef(model_shape), state)


def apply_sharding_to_model_state(
    model: nnx.Module,
    shd_cfg: ShardConfig,
    mesh: Mesh,
    model_state_sharding_fn: _ModelStateShardingFn,
) -> nnx.Module:
    graphdef, state = nnx.split(model)
    state_sharding = model_state_sharding_fn(state, shd_cfg, mesh)
    sharded_state = jax.tree.map(jax.device_put, state, state_sharding)
    return nnx.merge(graphdef, sharded_state)


def batch_partition_spec(shd_cfg: ShardConfig) -> PartitionSpec:
    return P(shd_cfg.act_btd[0], None)


def shard_batch(token_ids_BT: jax.Array, shd_cfg: ShardConfig, mesh: Mesh) -> jax.Array:
    sharding = NamedSharding(mesh, batch_partition_spec(shd_cfg))
    return jax.make_array_from_process_local_data(sharding, token_ids_BT)
