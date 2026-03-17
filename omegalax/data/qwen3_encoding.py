"""Shared Qwen3/Qwen3.5 message serialization and encoding helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from transformers import BaseImageProcessor, PreTrainedTokenizer


def build_chatml_text(
    messages: list[dict[str, Any]],
    image_grids: list[tuple[int, int, int]],
    merge_size: int,
) -> str:
    """Build a ChatML string from messages, inserting image pad tokens."""

    parts: list[str] = []
    img_idx = 0

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        parts.append(f"<|im_start|>{role}\n")

        if isinstance(content, str):
            parts.append(content)
        else:
            for block in content:
                if block["type"] == "text":
                    parts.append(block["text"])
                elif block["type"] == "image":
                    grid_t, grid_h, grid_w = image_grids[img_idx]
                    img_idx += 1
                    n_tokens = grid_t * (grid_h // merge_size) * (grid_w // merge_size)
                    parts.append(
                        "<|vision_start|>"
                        + "<|image_pad|>" * n_tokens
                        + "<|vision_end|>"
                    )

        parts.append("<|im_end|>\n")

    return "".join(parts)


def extract_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
    """Pull PIL images out of Qwen structured-content blocks."""

    images: list[Image.Image] = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            continue
        for block in content:
            if block["type"] != "image":
                continue
            if "image" in block:
                img = block["image"]
                images.append(img if isinstance(img, Image.Image) else Image.open(img))
            elif "url" in block:
                images.append(Image.open(block["url"]))
    return images


def encode_qwen_messages(
    messages: list[dict[str, Any]],
    *,
    tokenizer: PreTrainedTokenizer,
    image_processor: BaseImageProcessor | None = None,
    include_pixels: bool = False,
) -> dict[str, np.ndarray]:
    """Encode a Qwen chat example exactly as the collators expect."""

    image_grids: list[tuple[int, int, int]] = []
    result: dict[str, np.ndarray] = {}
    if image_processor is not None:
        imgs = extract_images(messages)
        if imgs:
            processed = image_processor.preprocess(imgs, return_tensors="np")
            result["image_grid_thw"] = processed["image_grid_thw"]
            if include_pixels:
                result["pixel_values"] = processed["pixel_values"]
            image_grids = [tuple(row) for row in result["image_grid_thw"].tolist()]

    merge_size = int(getattr(image_processor, "merge_size", 1))
    text = build_chatml_text(messages, image_grids, merge_size)
    result["input_ids"] = np.asarray(
        tokenizer.encode(text, add_special_tokens=False),
        dtype=np.int32,
    )
    return result
