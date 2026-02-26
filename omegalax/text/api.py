"""Architecture-specific API for text-only causal language models."""

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
from omegalax.models.qwen3 import registry as qwen3_registry
from omegalax.models.qwen3.cache import Cache, init_cache
from omegalax.models.qwen3.dense.model import Qwen3Dense
from omegalax.models.qwen3.moe.model import Qwen3Moe
from omegalax.models.qwen3.sharding import (
    batch_partition_spec as qwen3_batch_partition_spec,
    model_state_sharding as qwen3_model_state_sharding,
    shard_batch as qwen3_shard_batch,
)
from omegalax.models.qwen3.utils import count_right_pads
from omegalax.models.qwen3_5 import Qwen3_5TextConfig
from omegalax.models.qwen3_5.config import (
    is_supported_qwen3_5_model_id,
    list_supported_qwen3_5_model_ids,
    make_config as make_qwen3_5_config,
)
from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM
from omegalax.models.qwen3_5.sharding import (
    batch_partition_spec as qwen3_5_batch_partition_spec,
    model_state_sharding as qwen3_5_model_state_sharding,
    shard_batch as qwen3_5_shard_batch,
)

P = PartitionSpec

ModelConfig = qwen3_registry.Qwen3Config
TextConfig = Union[ModelConfig, Qwen3_5TextConfig]
registry = qwen3_registry

list_qwen3_dense_model_ids = qwen3_registry.list_qwen3_dense_model_ids
list_qwen3_moe_model_ids = qwen3_registry.list_qwen3_moe_model_ids


def resolve_config(model_or_id: str | TextConfig) -> TextConfig:
    """Resolve text config from model id (Qwen3 or Qwen3.5) or pass through."""
    if not isinstance(model_or_id, str):
        return model_or_id

    if qwen3_registry.is_supported_model_id(model_or_id):
        return qwen3_registry.build_config(model_or_id)
    if is_supported_qwen3_5_model_id(model_or_id):
        return make_qwen3_5_config(model_or_id).text_config

    supported_qwen3 = sorted(set(qwen3_registry.list_qwen3_dense_model_ids() + qwen3_registry.list_qwen3_moe_model_ids()))
    supported_qwen3_5 = list_supported_qwen3_5_model_ids()
    raise ValueError(
        f"Unsupported text model_id '{model_or_id}'. "
        f"Supported Qwen3 ids: {supported_qwen3}; supported Qwen3.5 ids: {supported_qwen3_5}."
    )


def align_config_to_mesh(cfg: TextConfig, mesh: Mesh) -> TextConfig:
    """Drop singleton mesh axes from sharding specs to avoid degenerate constraints."""
    if isinstance(cfg, qwen3_registry.Qwen3Config) or isinstance(cfg, Qwen3_5TextConfig):
        return dataclasses.replace(cfg, shd_cfg=shard_config_for_mesh(cfg.shd_cfg, mesh))
    raise TypeError(f"Unsupported text config type: {type(cfg)}")


def batch_partition_spec(cfg: TextConfig) -> PartitionSpec:
    """Return the token batch partition spec for a text config."""
    if isinstance(cfg, qwen3_registry.Qwen3Config):
        return qwen3_batch_partition_spec(cfg.shd_cfg)
    if isinstance(cfg, Qwen3_5TextConfig):
        return qwen3_5_batch_partition_spec(cfg.shd_cfg)
    raise TypeError(f"Unsupported text config type: {type(cfg)}")


def required_batch_multiple(cfg: TextConfig, mesh: Mesh) -> int:
    """Global batch size must be divisible by this multiple for input sharding."""
    return mesh_required_batch_multiple(batch_partition_spec(cfg), mesh)


def shard_batch(token_ids_BT: jax.Array, cfg: TextConfig, mesh: Mesh) -> jax.Array:
    """Shard a token batch for model families that implement input sharding."""
    if isinstance(cfg, qwen3_registry.Qwen3Config):
        return qwen3_shard_batch(token_ids_BT, cfg.shd_cfg, mesh)
    if isinstance(cfg, Qwen3_5TextConfig):
        return qwen3_5_shard_batch(token_ids_BT, cfg.shd_cfg, mesh)
    raise TypeError(f"Unsupported text config type: {type(cfg)}")


def init_model(
    model_or_id: str | TextConfig,
    rng: jax.Array,
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
) -> tuple[nnx.Module, TextConfig]:
    """Initialize a text-only model (Qwen3 or Qwen3.5 text) and return (model, cfg)."""
    cfg = resolve_config(model_or_id)
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size)
    cfg = align_config_to_mesh(cfg, mesh)

    if isinstance(cfg, qwen3_registry.Qwen3Config):
        model_cls = qwen3_registry.get_model_cls(cfg.variant)
        model = init_sharded_model_runtime(
            model_cls,
            cfg,
            rng,
            mesh,
            qwen3_model_state_sharding,
            lambda x: x.shd_cfg,
        )
        return model, cfg
    if isinstance(cfg, Qwen3_5TextConfig):
        model = init_sharded_model_runtime(
            Qwen3_5ForCausalLM,
            cfg,
            rng,
            mesh,
            qwen3_5_model_state_sharding,
            lambda x: x.shd_cfg,
        )
        return model, cfg

    raise ValueError(f"Unsupported text config type: {type(cfg)}")


def forward(model: nnx.Module, token_ids_BT: jax.Array, pad_id: int, cfg: TextConfig):
    """Forward pass for text-only models; returns logits and aux loss."""
    segment_ids_BT = 1 * (token_ids_BT != pad_id)

    if isinstance(model, (Qwen3Dense, Qwen3Moe)):
        if cfg.variant == "moe":
            logits_BTV, aux_loss = model(token_ids_BT, segment_ids_BT, None, jnp.array(0, dtype=jnp.int32))
            return logits_BTV, aux_loss
        logits_BTV = model(token_ids_BT, segment_ids_BT, None, jnp.array(0, dtype=jnp.int32))
        return logits_BTV, jnp.array(0.0, dtype=jnp.float32)

    if isinstance(model, Qwen3_5ForCausalLM):
        logits_BTV, aux_loss = model(token_ids_BT, segment_ids_BT, None, jnp.array(0, dtype=jnp.int32))
        return logits_BTV, aux_loss

    raise ValueError(f"Unsupported text model type: {type(model)}")


def decode(model: nnx.Module, cache: Cache, token_ids_BT: jax.Array, pad_id: int, cfg: TextConfig):
    """Decode step for autoregressive generation (Qwen3 only)."""
    if not isinstance(model, (Qwen3Dense, Qwen3Moe)):
        raise NotImplementedError("decode is only implemented for Qwen3 text models.")

    segment_ids_BT = 1 * (token_ids_BT != pad_id)
    num_right_pads = count_right_pads(token_ids_BT, pad_id)
    outputs = model(token_ids_BT, segment_ids_BT, cache, jnp.array(num_right_pads, dtype=jnp.int32))

    if cfg.variant == "moe":
        logits_BTV, aux_loss = outputs
    else:
        logits_BTV, aux_loss = outputs, jnp.array(0.0, dtype=jnp.float32)

    target_ind = token_ids_BT.shape[-1] - num_right_pads - 1
    return logits_BTV[:, target_ind], cache, aux_loss


def make_cache(cfg: TextConfig, batch_size: int, token_len: int, generate_steps: int, dtype: jnp.dtype = jnp.bfloat16):
    """Create KV cache for generation; only supported for Qwen3."""
    if not isinstance(cfg, qwen3_registry.Qwen3Config):
        raise NotImplementedError("Cache is only available for Qwen3 text models.")
    return init_cache(cfg, batch_size, token_len, generate_steps, dtype)
