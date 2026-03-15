"""Unified HuggingFace export dispatcher."""

from __future__ import annotations

from pathlib import Path

from omegalax.models.qwen3.config import Qwen3Config
from omegalax.models.qwen3.model import Qwen3
from omegalax.models.qwen3.params import export_qwen3_to_safetensors, qwen3_to_hf_config_dict
from omegalax.models.qwen3_vl.config import Qwen3VLConfig
from omegalax.models.qwen3_vl.model import Qwen3VL
from omegalax.models.qwen3_vl.params import export_qwen3_vl_to_safetensors, qwen3_vl_to_hf_config_dict
from omegalax.models.qwen3_5.config import Qwen3_5Config
from omegalax.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from omegalax.models.qwen3_5.params import export_qwen3_5_to_safetensors, qwen3_5_to_hf_config_dict


def export_model_to_hf(model, cfg, out_dir: str | Path) -> Path:
    """Route to the correct exporter based on model/config type."""
    if isinstance(cfg, Qwen3Config) and isinstance(model, Qwen3):
        return export_qwen3_to_safetensors(model, cfg, out_dir)

    if isinstance(cfg, Qwen3VLConfig) and isinstance(model, Qwen3VL):
        return export_qwen3_vl_to_safetensors(model, cfg, out_dir)

    if isinstance(cfg, Qwen3_5Config) and isinstance(model, Qwen3_5ForConditionalGeneration):
        return export_qwen3_5_to_safetensors(model, cfg, out_dir)

    raise ValueError(f"Unsupported model/config combination for export: {type(model)} / {type(cfg)}")


def model_config_to_hf_dict(cfg) -> dict:
    """Serialize a runtime config to HF config.json format."""
    if isinstance(cfg, Qwen3Config):
        return qwen3_to_hf_config_dict(cfg)
    if isinstance(cfg, Qwen3VLConfig):
        return qwen3_vl_to_hf_config_dict(cfg)
    if isinstance(cfg, Qwen3_5Config):
        return qwen3_5_to_hf_config_dict(cfg)
    raise ValueError(f"Unsupported config type for HF serialization: {type(cfg)}")
