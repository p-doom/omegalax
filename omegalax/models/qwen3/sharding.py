"""Qwen3 sharding helpers based on ShardConfig."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from flax import nnx
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from omegalax.models.shard_config import ShardConfig, ShardingSpec
from omegalax.models.sharding_runtime import (
    batch_partition_spec,
    ffw_df_param_spec,
    ffw_fd_param_spec,
    ndh_to_linear_kernel,
    nhd_to_linear_kernel,
    shard_batch,
)

P = PartitionSpec

_SpecResolver = Callable[[ShardConfig], ShardingSpec]


_DIRECT_RULES: dict[str, _SpecResolver] = {
    "embedder.embedding": lambda shd_cfg: shd_cfg.emb_vd,
    "lm_head.kernel": lambda shd_cfg: shd_cfg.emb_dv,
    "final_norm.scale": lambda shd_cfg: shd_cfg.rms_norm,
}


_LAYER_RULES: dict[tuple[str, ...], _SpecResolver] = {
    ("input_layernorm", "scale"): lambda shd_cfg: shd_cfg.rms_norm,
    ("post_attention_layernorm", "scale"): lambda shd_cfg: shd_cfg.rms_norm,
    ("attn", "q_norm", "scale"): lambda _shd_cfg: P(None),
    ("attn", "k_norm", "scale"): lambda _shd_cfg: P(None),
    ("attn", "q_proj", "kernel"): lambda shd_cfg: ndh_to_linear_kernel(shd_cfg.q_weight_ndh),
    ("attn", "k_proj", "kernel"): lambda shd_cfg: ndh_to_linear_kernel(shd_cfg.kv_weight_ndh),
    ("attn", "v_proj", "kernel"): lambda shd_cfg: ndh_to_linear_kernel(shd_cfg.kv_weight_ndh),
    ("attn", "o_proj", "kernel"): lambda shd_cfg: nhd_to_linear_kernel(shd_cfg.o_weight_nhd),
    ("mlp", "gate_proj", "kernel"): lambda shd_cfg: shd_cfg.ffw_weight_df,
    ("mlp", "up_proj", "kernel"): lambda shd_cfg: shd_cfg.ffw_weight_df,
    ("mlp", "down_proj", "kernel"): lambda shd_cfg: shd_cfg.ffw_weight_fd,
    ("mlp", "gate_proj"): ffw_df_param_spec,
    ("mlp", "up_proj"): ffw_df_param_spec,
    ("mlp", "down_proj"): ffw_fd_param_spec,
    ("mlp", "router", "kernel"): lambda shd_cfg: shd_cfg.ffw_weight_df,
}


def model_state_sharding(
    model_state_shape: nnx.State,
    shd_cfg: ShardConfig,
    mesh: Mesh,
) -> nnx.State:
    def to_sharding(path: Sequence[Any], leaf: Any):
        parts = [str(part.key) if hasattr(part, "key") else str(part.name) for part in path]
        path_key = ".".join(parts[:-1])

        resolver = _DIRECT_RULES.get(path_key)
        if resolver is None and len(parts) >= 5 and parts[0] == "layers" and parts[1].isdigit():
            resolver = _LAYER_RULES.get(tuple(parts[2:-1]))
        if resolver is None:
            raise ValueError(f"Unmapped Qwen3 state path: {'.'.join(parts)}")
        return NamedSharding(mesh, resolver(shd_cfg))

    return jax.tree_util.tree_map_with_path(to_sharding, model_state_shape)
