"""Registry for Qwen3 variants (dense + MoE)."""

from .config import (
    Qwen3Config,
    get_spec,
    is_supported_model_id,
    list_qwen3_dense_model_ids,
    list_qwen3_moe_model_ids,
    make_config,
)
from .model import Qwen3


def build_config(model_id: str, *, variant: str | None = None) -> Qwen3Config:
    return make_config(model_id)


def get_model_cls(variant: str | None = None) -> type:
    return Qwen3


def infer_variant(model_id: str) -> str:
    cfg = make_config(model_id)
    return cfg.variant


__all__ = [
    "Qwen3Config",
    "Qwen3",
    "build_config",
    "get_model_cls",
    "infer_variant",
    "is_supported_model_id",
    "list_qwen3_dense_model_ids",
    "list_qwen3_moe_model_ids",
]
