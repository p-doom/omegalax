"""Public loader entrypoints for Qwen3 (dense + MoE)."""

from __future__ import annotations

from .loader import create_qwen3_from_safetensors

__all__ = ["create_qwen3_from_safetensors"]
