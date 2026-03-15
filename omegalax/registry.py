"""Lightweight registry for routing model ids to architectures."""

from __future__ import annotations

from enum import Enum

from omegalax.models.params_utils import load_hf_config_from_source
from omegalax.models.qwen3.config import is_supported_model_id as is_supported_qwen3_model_id
from omegalax.models.qwen3.config import resolve_qwen3_repo_id
from omegalax.models.qwen3_5.config import is_supported_qwen3_5_model_id, resolve_qwen3_5_repo_id
from omegalax.models.qwen3_vl.config import is_supported_qwen3_vl_model_id, resolve_qwen3_vl_repo_id


class Arch(str, Enum):
    TEXT = "text"
    VLM = "vlm"


_TEXT_MODEL_TYPES = {"qwen3", "qwen3_moe"}
_VLM_MODEL_TYPES = {"qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe"}


def _load_resolved_hf_config(model_id: str) -> dict:
    if is_supported_qwen3_model_id(model_id):
        source = resolve_qwen3_repo_id(model_id)
    elif is_supported_qwen3_5_model_id(model_id):
        source = resolve_qwen3_5_repo_id(model_id)
    elif is_supported_qwen3_vl_model_id(model_id):
        source = resolve_qwen3_vl_repo_id(model_id)
    else:
        raise ValueError(f"Unsupported model id '{model_id}'")
    return load_hf_config_from_source(source)


def infer_arch(model_id: str) -> Arch:
    """Determine the architecture from a smoke alias or HF-format config source."""
    if model_id.startswith("qwen3-smoke"):
        return Arch.TEXT
    if model_id.startswith("qwen3.5-smoke") or model_id.startswith("qwen3-vl-smoke"):
        return Arch.VLM

    hf_cfg = _load_resolved_hf_config(model_id)
    model_type = hf_cfg.get("model_type")
    if model_type in _TEXT_MODEL_TYPES:
        return Arch.TEXT
    if model_type in _VLM_MODEL_TYPES:
        return Arch.VLM
    raise ValueError(f"Cannot infer architecture for model/config source '{model_id}'")


def resolve_hf_repo_id(model_id: str) -> str:
    """Map a short alias to the canonical HuggingFace repo id when applicable."""
    if "/" in model_id:
        return model_id
    if is_supported_qwen3_model_id(model_id):
        return resolve_qwen3_repo_id(model_id)
    if is_supported_qwen3_5_model_id(model_id):
        return resolve_qwen3_5_repo_id(model_id)
    if is_supported_qwen3_vl_model_id(model_id):
        return resolve_qwen3_vl_repo_id(model_id)
    return model_id


def resolve(model_id: str) -> Arch:
    """Return the architecture classification for a given model id."""
    return infer_arch(model_id)
