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


def _message_has_images(message: dict[str, Any]) -> bool:
    content = message.get("content", "")
    if isinstance(content, str):
        return False
    return any(block.get("type") == "image" for block in content)


def make_message_length_fn(
    tokenizer: PreTrainedTokenizer,
    image_processor: BaseImageProcessor | None = None,
):
    """Return a ``message -> token_count`` callable for use with ``build_chunk_index``.

    Suitable for ChatML-formatted models (Qwen3 / Qwen3.5).  Token lengths are
    exactly additive at message boundaries: ``<|im_start|>``/``<|im_end|>`` act
    as hard BPE split points and ``add_special_tokens=False`` suppresses any
    per-sequence overhead, so ``sum(lengths)`` equals the full-sequence length
    exactly.  For a different chat template, implement an analogous factory and
    swap it in.
    """
    merge_size = int(getattr(image_processor, "merge_size", 1)) if image_processor else 1

    def _measure(message: dict[str, Any]) -> int | dict[str, Any]:
        if image_processor is None and _message_has_images(message):
            raise ValueError(
                "Encountered image content in message but no image_processor was provided. "
                "Pass image_processor= to make_message_length_fn."
            )
        encoded = encode_qwen_messages(
            [message],
            tokenizer=tokenizer,
            image_processor=image_processor,
            include_pixels=False,
        )
        length = int(len(encoded["input_ids"]))

        grid_thw = encoded.get("image_grid_thw", np.empty((0, 3), dtype=np.int64))
        num_images = int(grid_thw.shape[0])
        vision_tokens = 0
        vision_patches = 0
        for row in grid_thw:
            t, h, w = int(row[0]), int(row[1]), int(row[2])
            vision_tokens += t * (h // merge_size) * (w // merge_size)
            vision_patches += t * h * w

        return {
            "length": length,
            "vision_tokens": vision_tokens,
            "vision_patches": vision_patches,
            "num_images": num_images,
            "image_grid_thw": grid_thw.tolist(),
        }

    return _measure


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
