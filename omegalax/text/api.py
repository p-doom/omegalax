"""Architecture-specific API for text-only causal language models."""

from __future__ import annotations

from typing import Union

import jax
import jax.numpy as jnp
from flax import nnx

from omegalax.models.qwen3 import registry as qwen3_registry
from omegalax.models.qwen3.cache import Cache, init_cache
from omegalax.models.qwen3.dense.model import Qwen3Dense
from omegalax.models.qwen3.moe.model import Qwen3Moe
from omegalax.models.qwen3.utils import count_right_pads
from omegalax.models.qwen3_5 import Qwen3_5TextConfig
from omegalax.models.qwen3_5.config import make_config as make_qwen3_5_config
from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM

ModelConfig = qwen3_registry.Qwen3Config
TextConfig = Union[ModelConfig, Qwen3_5TextConfig]
registry = qwen3_registry

list_qwen3_dense_model_ids = qwen3_registry.list_qwen3_dense_model_ids
list_qwen3_moe_model_ids = qwen3_registry.list_qwen3_moe_model_ids


def init_model(model_or_id: str | TextConfig, rng: jax.Array, *, use_sharding: bool = False) -> tuple[nnx.Module, TextConfig]:
    """Initialize a text-only model (Qwen3 or Qwen3.5 text) and return (model, cfg)."""
    if isinstance(model_or_id, str):
        key = model_or_id.lower()
        if "qwen3.5" in key or "qwen3_5" in key:
            cfg = make_qwen3_5_config(model_or_id).text_config
            return Qwen3_5ForCausalLM(cfg, rngs=nnx.Rngs(rng)), cfg
        cfg = qwen3_registry.build_config(model_or_id, use_sharding=use_sharding)
    else:
        cfg = model_or_id

    if isinstance(cfg, qwen3_registry.Qwen3Config):
        model_cls = qwen3_registry.get_model_cls(cfg.variant)
        return model_cls(cfg, rngs=nnx.Rngs(rng)), cfg
    if isinstance(cfg, Qwen3_5TextConfig):
        return Qwen3_5ForCausalLM(cfg, rngs=nnx.Rngs(rng)), cfg

    raise ValueError(f"Unsupported text config type: {type(cfg)}")


def forward(model: nnx.Module, tokens: jax.Array, pad_id: int, cfg: TextConfig):
    """Forward pass for text-only models; returns logits and aux loss."""
    segment_ids = 1 * (tokens != pad_id)

    if isinstance(model, (Qwen3Dense, Qwen3Moe)):
        if cfg.variant == "moe":
            logits, aux_loss = model(tokens, segment_ids, None, jnp.array(0, dtype=jnp.int32))
            return logits, aux_loss
        logits = model(tokens, segment_ids, None, jnp.array(0, dtype=jnp.int32))
        return logits, jnp.array(0.0, dtype=jnp.float32)

    if isinstance(model, Qwen3_5ForCausalLM):
        logits, aux_loss = model(tokens, segment_ids, None, jnp.array(0, dtype=jnp.int32))
        return logits, aux_loss

    raise ValueError(f"Unsupported text model type: {type(model)}")


def decode(model: nnx.Module, cache: Cache, tokens: jax.Array, pad_id: int, cfg: TextConfig):
    """Decode step for autoregressive generation (Qwen3 only)."""
    if not isinstance(model, (Qwen3Dense, Qwen3Moe)):
        raise NotImplementedError("decode is only implemented for Qwen3 text models.")

    segment_ids = 1 * (tokens != pad_id)
    num_right_pads = count_right_pads(tokens, pad_id)
    outputs = model(tokens, segment_ids, cache, jnp.array(num_right_pads, dtype=jnp.int32))

    if cfg.variant == "moe":
        logits, aux_loss = outputs
    else:
        logits, aux_loss = outputs, jnp.array(0.0, dtype=jnp.float32)

    target_ind = tokens.shape[-1] - num_right_pads - 1
    return logits[:, target_ind], cache, aux_loss


def make_cache(cfg: TextConfig, batch_size: int, token_len: int, generate_steps: int, dtype: jnp.dtype = jnp.bfloat16):
    """Create KV cache for generation; only supported for Qwen3."""
    if not isinstance(cfg, qwen3_registry.Qwen3Config):
        raise NotImplementedError("Cache is only available for Qwen3 text models.")
    return init_cache(cfg, batch_size, token_len, generate_steps, dtype)
