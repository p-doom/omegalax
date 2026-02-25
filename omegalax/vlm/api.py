"""Architecture-specific API for vision-language models."""

from __future__ import annotations

from typing import Union

import jax
import jax.numpy as jnp
from flax import nnx

from omegalax.models.qwen3_vl import Qwen3VL, make_vl_config
from omegalax.models.qwen3_vl.config import Qwen3VLConfig
from omegalax.models.qwen3_5 import Qwen3_5Config
from omegalax.models.qwen3_5 import make_config as make_qwen3_5_config
from omegalax.models.qwen3_5.model import Qwen3_5ForConditionalGeneration

VLMConfig = Union[Qwen3_5Config, Qwen3VLConfig]


def _is_qwen3_vl(model: nnx.Module) -> bool:
    return isinstance(model, Qwen3VL)


def _is_qwen3_5_vlm(model: nnx.Module) -> bool:
    return isinstance(model, Qwen3_5ForConditionalGeneration)


def init_model(model_or_id: str | VLMConfig, rng: jax.Array) -> tuple[nnx.Module, VLMConfig]:
    """Initialize a vision-language model."""
    if isinstance(model_or_id, str):
        key = model_or_id.lower()
        # FIXME (f.srambical): this should not be heuristics-based
        if "qwen3.5" in key or "qwen3_5" in key:
            cfg = make_qwen3_5_config(model_or_id)
            return Qwen3_5ForConditionalGeneration(cfg, rngs=nnx.Rngs(rng)), cfg
        if "qwen3-vl" in key or "vl" in key:
            cfg = make_vl_config(model_or_id)
            return Qwen3VL(cfg, rngs=nnx.Rngs(rng)), cfg
        raise ValueError(f"Unsupported VLM model id '{model_or_id}'")

    cfg = model_or_id
    if isinstance(cfg, Qwen3_5Config):
        return Qwen3_5ForConditionalGeneration(cfg, rngs=nnx.Rngs(rng)), cfg
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
        if cfg.variant == "moe":
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
