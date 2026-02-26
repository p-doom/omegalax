"""Dispatch helpers for loading HuggingFace Qwen3 checkpoints."""

from __future__ import annotations

from omegalax.models.qwen3 import registry
from omegalax.models.qwen3.dense.params_dense import create_qwen3_dense_from_safetensors
from omegalax.models.qwen3.moe.params_moe import create_qwen3_moe_from_safetensors


def create_qwen3_from_safetensors(
    file_dir: str,
    model_id: str,
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
):
    variant = registry.infer_variant(model_id)
    if variant == "dense":
        return create_qwen3_dense_from_safetensors(
            file_dir,
            model_id,
            tp_size=tp_size,
            fsdp_size=fsdp_size,
        )
    if variant == "moe":
        return create_qwen3_moe_from_safetensors(
            file_dir,
            model_id,
            tp_size=tp_size,
            fsdp_size=fsdp_size,
        )
    raise ValueError(f"Unknown Qwen3 variant for model_id={model_id}")
