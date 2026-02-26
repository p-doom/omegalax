"""Sharding helpers for Qwen3-VL models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from flax import nnx
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from omegalax.models.shard_config import ShardConfig
from omegalax.models.sharding_runtime import (
    batch_partition_spec,
    ffw_df_param_spec,
    ffw_fd_param_spec,
    ndh_to_linear_kernel,
    nhd_to_linear_kernel,
    shard_batch,
    vision_axis_spec,
    vision_none_axis_spec,
)

P = PartitionSpec


_SpecResolver = Callable[[ShardConfig], PartitionSpec]


_DIRECT_RULES: dict[str, _SpecResolver] = {
    "text.embedder.embedding": lambda shd_cfg: shd_cfg.emb_vd,
    "lm_head.kernel": lambda shd_cfg: shd_cfg.emb_dv,
    "text.final_norm.scale": lambda shd_cfg: shd_cfg.rms_norm,
    "vision.patch_embed.proj.kernel": vision_none_axis_spec,
    "vision.patch_embed.proj.bias": vision_axis_spec,
    "vision.pos_embed.embedding": vision_none_axis_spec,
}


_TEXT_LAYER_RULES: dict[tuple[str, ...], _SpecResolver] = {
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


_VISION_BLOCK_RULES: dict[tuple[str, ...], _SpecResolver] = {
    ("norm1", "scale"): vision_axis_spec,
    ("norm1", "bias"): vision_axis_spec,
    ("norm2", "scale"): vision_axis_spec,
    ("norm2", "bias"): vision_axis_spec,
    ("attn", "qkv", "kernel"): vision_none_axis_spec,
    ("attn", "qkv", "bias"): vision_axis_spec,
    ("attn", "proj", "kernel"): vision_none_axis_spec,
    ("attn", "proj", "bias"): vision_axis_spec,
    ("mlp", "fc1", "kernel"): vision_none_axis_spec,
    ("mlp", "fc1", "bias"): vision_axis_spec,
    ("mlp", "fc2", "kernel"): vision_none_axis_spec,
    ("mlp", "fc2", "bias"): vision_axis_spec,
}


_VISION_MERGER_RULES: dict[tuple[str, ...], _SpecResolver] = {
    ("norm", "scale"): vision_axis_spec,
    ("norm", "bias"): vision_axis_spec,
    ("fc1", "kernel"): lambda _shd_cfg: P(None, None),
    ("fc1", "bias"): lambda _shd_cfg: P(None),
    ("fc2", "kernel"): vision_none_axis_spec,
    ("fc2", "bias"): vision_axis_spec,
}


def model_state_sharding(model_state_shape: nnx.State, shd_cfg: ShardConfig, mesh: Mesh) -> nnx.State:
    def to_sharding(path: Sequence[Any], leaf: Any):
        parts = [str(part.key) if hasattr(part, "key") else str(part.name) for part in path]
        path_key = ".".join(parts[:-1])

        resolver = _DIRECT_RULES.get(path_key)
        if resolver is None:
            if len(parts) >= 6 and parts[0] == "text" and parts[1] == "layers" and parts[2].isdigit():
                resolver = _TEXT_LAYER_RULES.get(tuple(parts[3:-1]))
            elif len(parts) >= 6 and parts[0] == "vision" and parts[1] == "blocks" and parts[2].isdigit():
                resolver = _VISION_BLOCK_RULES.get(tuple(parts[3:-1]))
            elif len(parts) >= 6 and parts[0] == "vision" and parts[1] == "deepstack_mergers" and parts[2].isdigit():
                resolver = _VISION_MERGER_RULES.get(tuple(parts[3:-1]))
            elif len(parts) >= 5 and parts[0] == "vision" and parts[1] == "merger":
                resolver = _VISION_MERGER_RULES.get(tuple(parts[2:-1]))
        if resolver is None:
            raise ValueError(f"Unmapped Qwen3-VL state path: {'.'.join(parts)}")
        return NamedSharding(mesh, resolver(shd_cfg))

    return jax.tree_util.tree_map_with_path(to_sharding, model_state_shape)
