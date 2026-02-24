"""Dispatch helpers for loading HuggingFace Qwen3 checkpoints."""

from __future__ import annotations

from omegalax.models.qwen3 import registry
from omegalax.models.qwen3.dense.params_dense import create_qwen3_dense_from_safetensors
from omegalax.models.qwen3.moe.params_moe import create_qwen3_moe_from_safetensors


def create_qwen3_from_safetensors(file_dir: str, model_id: str, use_sharding: bool = False):
    variant = registry.infer_variant(model_id)
    if variant == "dense":
        return create_qwen3_dense_from_safetensors(file_dir, model_id, use_sharding=use_sharding)
    if variant == "moe":
        return create_qwen3_moe_from_safetensors(file_dir, model_id, use_sharding=use_sharding)
    raise ValueError(f"Unknown Qwen3 variant for model_id={model_id}")
