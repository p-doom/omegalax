"""Public loader entrypoint for Qwen3.5."""

from __future__ import annotations

from .loader import create_qwen3_5_from_safetensors

__all__ = ["create_qwen3_5_from_safetensors"]
