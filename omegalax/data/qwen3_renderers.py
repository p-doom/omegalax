"""Omegalax message renderers for supported Qwen model families."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

MessageRenderer = Callable[[dict[str, Any], str], tuple[str, str]]


def _block(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


def _render_qwen3(message: dict[str, Any], content: str) -> tuple[str, str]:
    role = message["role"]
    if role == "assistant":
        if content.startswith("\n"):
            raise ValueError("Qwen3 assistant content must not start with a newline")
        if "</think>" in content:
            raise ValueError("pre-rendered reasoning content is not supported")
    historical = _block(role, content)
    if role != "assistant":
        return historical, historical
    reasoning = (message.get("reasoning_content") or "").strip("\n")
    terminal = _block(role, f"<think>\n{reasoning}\n</think>\n\n{content}")
    return historical, terminal


def _render_qwen3_vl(message: dict[str, Any], content: str) -> tuple[str, str]:
    text = _block(message["role"], content)
    return text, text


def _render_qwen35(message: dict[str, Any], content: str) -> tuple[str, str]:
    role = message["role"]
    content = content.strip()
    if role == "assistant" and "</think>" in content:
        raise ValueError("pre-rendered reasoning content is not supported")
    historical = _block(role, content)
    if role != "assistant":
        return historical, historical
    reasoning = (message.get("reasoning_content") or "").strip()
    terminal = _block(role, f"<think>\n{reasoning}\n</think>\n\n{content}")
    return historical, terminal


_RENDERERS: dict[str, tuple[MessageRenderer, bool]] = {
    "qwen3": (_render_qwen3, False),
    "qwen3_moe": (_render_qwen3, False),
    "qwen3_vl": (_render_qwen3_vl, True),
    "qwen3_vl_moe": (_render_qwen3_vl, True),
    "qwen3_5": (_render_qwen35, True),
    "qwen3_5_moe": (_render_qwen35, True),
}


def renderer_for_model_type(model_type: str) -> tuple[MessageRenderer, bool]:
    try:
        return _RENDERERS[model_type]
    except KeyError:
        raise ValueError(f"unsupported Qwen model_type: {model_type!r}") from None
