"""Omegalax conversation renderers for supported Qwen model families."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ConversationRenderer = Callable[[list[dict[str, Any]], list[str]], list[str | None]]


def _block(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


def _render_qwen3(messages: list[dict[str, Any]], contents: list[str]) -> list[str | None]:
    last_user = next(
        (index for index in range(len(messages) - 1, -1, -1) if messages[index]["role"] == "user"),
        len(messages) - 1,
    )
    rendered: list[str | None] = []
    for index, (message, content) in enumerate(zip(messages, contents, strict=True)):
        if message["role"] == "assistant":
            reasoning = ""
            if isinstance(message.get("reasoning_content"), str):
                reasoning = message["reasoning_content"]
            elif "</think>" in content:
                before, content = content.split("</think>", 1)
                reasoning = before.split("<think>")[-1].lstrip("\n")
                reasoning = reasoning.rstrip("\n")
                content = content.lstrip("\n")
            if index > last_user and (index == len(messages) - 1 or reasoning):
                reasoning = reasoning.strip("\n")
                content = content.lstrip("\n")
                content = f"<think>\n{reasoning}\n</think>\n\n{content}"
        rendered.append(_block(message["role"], content))
    return rendered


def _render_qwen3_vl(
    messages: list[dict[str, Any]], contents: list[str]
) -> list[str | None]:
    return [
        None if message["role"] == "system" and index != 0 else _block(message["role"], content)
        for index, (message, content) in enumerate(zip(messages, contents, strict=True))
    ]


def _render_qwen35(messages: list[dict[str, Any]], contents: list[str]) -> list[str | None]:
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("Qwen3.5 requires a user query")
    last_user = max(index for index, message in enumerate(messages) if message["role"] == "user")
    rendered: list[str | None] = []
    for index, (message, raw_content) in enumerate(zip(messages, contents, strict=True)):
        role = message["role"]
        if role == "system" and index != 0:
            raise ValueError("Qwen3.5 system messages must be first")
        content = raw_content.strip()
        if role == "assistant":
            reasoning = ""
            if isinstance(message.get("reasoning_content"), str):
                reasoning = message["reasoning_content"]
            elif "</think>" in content:
                before, content = content.split("</think>", 1)
                reasoning = before.split("<think>")[-1].lstrip("\n")
                reasoning = reasoning.rstrip("\n")
                content = content.lstrip("\n")
            if index > last_user:
                content = f"<think>\n{reasoning.strip()}\n</think>\n\n{content}"
        rendered.append(_block(role, content))
    return rendered


_RENDERERS: dict[str, tuple[ConversationRenderer, bool]] = {
    "qwen3": (_render_qwen3, False),
    "qwen3_moe": (_render_qwen3, False),
    "qwen3_vl": (_render_qwen3_vl, True),
    "qwen3_vl_moe": (_render_qwen3_vl, True),
    "qwen3_5": (_render_qwen35, True),
    "qwen3_5_moe": (_render_qwen35, True),
}


def renderer_for_model_type(model_type: str) -> tuple[ConversationRenderer, bool]:
    try:
        return _RENDERERS[model_type]
    except KeyError:
        raise ValueError(f"unsupported Qwen model_type: {model_type!r}") from None
