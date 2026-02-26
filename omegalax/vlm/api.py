"""Architecture-specific API for vision-language models."""

from __future__ import annotations

import dataclasses
from typing import Union

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, PartitionSpec

from omegalax.distributed.mesh import ensure_mesh, required_batch_multiple as mesh_required_batch_multiple
from omegalax.models.shard_config import shard_config_for_mesh
from omegalax.models.sharding_runtime import init_sharded_model as init_sharded_model_runtime
from omegalax.models.qwen3_vl import Qwen3VL, make_vl_config
from omegalax.models.qwen3_vl.config import (
    Qwen3VLConfig,
    is_supported_qwen3_vl_model_id,
    list_supported_qwen3_vl_model_ids,
)
from omegalax.models.qwen3_vl.sharding import (
    batch_partition_spec as qwen3_vl_batch_partition_spec,
    model_state_sharding as qwen3_vl_model_state_sharding,
    shard_batch as qwen3_vl_shard_batch,
)
from omegalax.models.qwen3_5 import Qwen3_5Config
from omegalax.models.qwen3_5 import make_config as make_qwen3_5_config
from omegalax.models.qwen3_5.config import is_supported_qwen3_5_model_id, list_supported_qwen3_5_model_ids
from omegalax.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from omegalax.models.qwen3_5.sharding import (
    batch_partition_spec as qwen3_5_batch_partition_spec,
    model_state_sharding as qwen3_5_model_state_sharding,
    shard_batch as qwen3_5_shard_batch,
)

VLMConfig = Union[Qwen3_5Config, Qwen3VLConfig]
P = PartitionSpec


def _is_qwen3_vl(model: nnx.Module) -> bool:
    return isinstance(model, Qwen3VL)


def _is_qwen3_5_vlm(model: nnx.Module) -> bool:
    return isinstance(model, Qwen3_5ForConditionalGeneration)


def resolve_config(model_or_id: str | VLMConfig) -> VLMConfig:
    """Resolve VLM config from model id (Qwen3.5 or Qwen3-VL) or pass through."""
    if not isinstance(model_or_id, str):
        return model_or_id

    if is_supported_qwen3_5_model_id(model_or_id):
        return make_qwen3_5_config(model_or_id)
    if is_supported_qwen3_vl_model_id(model_or_id):
        return make_vl_config(model_or_id)

    supported_qwen3_5 = list_supported_qwen3_5_model_ids()
    supported_qwen3_vl = list_supported_qwen3_vl_model_ids()
    raise ValueError(
        f"Unsupported VLM model id '{model_or_id}'. "
        f"Supported Qwen3.5 ids: {supported_qwen3_5}; supported Qwen3-VL ids: {supported_qwen3_vl}."
    )


def align_config_to_mesh(cfg: VLMConfig, mesh: Mesh) -> VLMConfig:
    """Drop singleton mesh axes from sharding specs to avoid degenerate constraints."""
    if isinstance(cfg, Qwen3_5Config):
        return dataclasses.replace(
            cfg,
            text_config=dataclasses.replace(
                cfg.text_config,
                shd_cfg=shard_config_for_mesh(cfg.text_config.shd_cfg, mesh),
            ),
        )
    if isinstance(cfg, Qwen3VLConfig):
        return dataclasses.replace(cfg, shd_cfg=shard_config_for_mesh(cfg.shd_cfg, mesh))
    raise TypeError(f"Unsupported VLM config type: {type(cfg)}")


def batch_partition_spec(cfg: VLMConfig) -> PartitionSpec:
    if isinstance(cfg, Qwen3_5Config):
        return qwen3_5_batch_partition_spec(cfg.text_config.shd_cfg)
    if isinstance(cfg, Qwen3VLConfig):
        return qwen3_vl_batch_partition_spec(cfg.shd_cfg)
    raise TypeError(f"Unsupported VLM config type: {type(cfg)}")


def required_batch_multiple(cfg: VLMConfig, mesh: Mesh) -> int:
    return mesh_required_batch_multiple(batch_partition_spec(cfg), mesh)


def shard_batch(token_ids_BT: jax.Array, cfg: VLMConfig, mesh: Mesh) -> jax.Array:
    if isinstance(cfg, Qwen3_5Config):
        return qwen3_5_shard_batch(token_ids_BT, cfg.text_config.shd_cfg, mesh)
    if isinstance(cfg, Qwen3VLConfig):
        return qwen3_vl_shard_batch(token_ids_BT, cfg.shd_cfg, mesh)
    raise TypeError(f"Unsupported VLM config type: {type(cfg)}")


def vocab_size(cfg: VLMConfig) -> int:
    if isinstance(cfg, Qwen3_5Config):
        return int(cfg.text_config.vocab_size)
    if isinstance(cfg, Qwen3VLConfig):
        return int(cfg.vocab_size)
    raise TypeError(f"Unsupported VLM config type: {type(cfg)}")


def init_model(
    model_or_id: str | VLMConfig,
    rng: jax.Array,
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
) -> tuple[nnx.Module, VLMConfig]:
    """Initialize a vision-language model."""
    cfg = resolve_config(model_or_id)
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size)
    cfg = align_config_to_mesh(cfg, mesh)

    if isinstance(cfg, Qwen3_5Config):
        model = init_sharded_model_runtime(
            Qwen3_5ForConditionalGeneration,
            cfg,
            rng,
            mesh,
            qwen3_5_model_state_sharding,
            lambda x: x.text_config.shd_cfg,
        )
        return model, cfg
    if isinstance(cfg, Qwen3VLConfig):
        model = init_sharded_model_runtime(
            Qwen3VL,
            cfg,
            rng,
            mesh,
            qwen3_vl_model_state_sharding,
            lambda x: x.shd_cfg,
        )
        return model, cfg
    raise ValueError(f"Unsupported VLM config type: {type(cfg)}")


def forward(
    model: nnx.Module,
    token_ids_BT: jax.Array,
    pad_id: int,
    cfg,
    *,
    attention_mask_BT: jax.Array | None = None,
    pixel_values: jax.Array | None = None,
    image_grid_thw: jax.Array | None = None,
    position_ids_ZBT: jax.Array | None = None,
):
    """Forward pass for VLMs; supports text-only or multimodal batches."""
    if attention_mask_BT is None:
        attention_mask_BT = (token_ids_BT != pad_id).astype(jnp.int32)

    if _is_qwen3_5_vlm(model):
        segment_ids_BT = attention_mask_BT.astype(jnp.int32)
        logits_BTV, aux_loss = model(
            token_ids_BT,
            segment_ids_BT,
            None,
            jnp.array(0, dtype=jnp.int32),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids_ZBT=position_ids_ZBT,
        )
        return logits_BTV, aux_loss

    if _is_qwen3_vl(model):
        outputs = model(
            token_ids_BT,
            attention_mask_BT,
            position_ids_ZBT=position_ids_ZBT,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        if isinstance(outputs, tuple):
            logits_BTV, aux_loss = outputs
            return logits_BTV, aux_loss
        else:
            return outputs, jnp.array(0.0, dtype=jnp.float32)

    raise ValueError(f"Unsupported VLM model type: {type(model)}")


def make_cache(*_args, **_kwargs):
    """Placeholder for cache creation to keep the interface symmetric."""
    return None


def decode(*_args, **_kwargs):
    raise NotImplementedError("decode is not implemented for vision-language models.")
