"""Unified HuggingFace export dispatcher."""

from __future__ import annotations

from pathlib import Path

from omegalax.models.qwen3.config import Qwen3Config
from omegalax.models.qwen3.dense.model import Qwen3Dense
from omegalax.models.qwen3.dense.params_dense import export_qwen3_dense_to_safetensors
from omegalax.models.qwen3.moe.config import Qwen3MoeConfig
from omegalax.models.qwen3.moe.model import Qwen3Moe
from omegalax.models.qwen3.moe.params_moe import export_qwen3_moe_to_safetensors
from omegalax.models.qwen3_vl.config import Qwen3VLConfig
from omegalax.models.qwen3_vl.model import Qwen3VL
from omegalax.models.qwen3_vl.params import export_qwen3_vl_to_safetensors
from omegalax.models.qwen3_5.config import Qwen3_5Config
from omegalax.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from omegalax.models.qwen3_5.params import export_qwen3_5_to_safetensors


def export_model_to_hf(model, cfg, out_dir: str | Path) -> Path:
    """Route to the correct exporter based on model/config type."""
    if isinstance(cfg, Qwen3Config):
        if cfg.variant == "dense" and isinstance(model, Qwen3Dense):
            return export_qwen3_dense_to_safetensors(model, cfg, out_dir)
        if cfg.variant == "moe" and isinstance(cfg, Qwen3MoeConfig) and isinstance(model, Qwen3Moe):
            return export_qwen3_moe_to_safetensors(model, cfg, out_dir)
        raise ValueError(f"Unsupported Qwen3 variant '{cfg.variant}' for export.")

    if isinstance(cfg, Qwen3VLConfig) and isinstance(model, Qwen3VL):
        return export_qwen3_vl_to_safetensors(model, cfg, out_dir)

    if isinstance(cfg, Qwen3_5Config) and isinstance(model, Qwen3_5ForConditionalGeneration):
        return export_qwen3_5_to_safetensors(model, cfg, out_dir)

    raise ValueError(f"Unsupported model/config combination for export: {type(model)} / {type(cfg)}")
