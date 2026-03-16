"""Qwen3 / Qwen3.5 SFT collators (ChatML format, vision tokens).

Tokenize chat conversations and produce loss masks. Both collators output
numpy dicts ready for ``shard_batch_dict``. Model-specific to Qwen3-VL and
Qwen3.5 (ChatML delimiters, assistant-based loss, Qwen image processor).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image
from transformers import BaseImageProcessor, PreTrainedTokenizer


def _build_assistant_loss_mask(
    input_ids: np.ndarray,
    im_start_id: int,
    im_end_id: int,
    assistant_token_id: int,
) -> np.ndarray:
    """Mask with 1 on assistant-content tokens, 0 elsewhere.

    ChatML ``<|im_start|>``/``<|im_end|>`` pair 1:1 in sequence order.
    We find which pairs are assistant turns, then fill the content spans
    via cumsum — no Python loops.
    """
    n = len(input_ids)
    starts = np.where(input_ids == im_start_id)[0]
    ends = np.where(input_ids == im_end_id)[0]
    k = min(len(starts), len(ends))
    if k == 0:
        return np.zeros(n, dtype=np.int32)
    starts, ends = starts[:k], ends[:k]

    # Which <|im_start|> are followed by `assistant`?
    is_asst = (starts + 1 < n) & (input_ids[starts + 1] == assistant_token_id)
    # Content starts after <|im_start|> assistant \n  (3 tokens).
    content_starts = starts[is_asst] + 3
    content_ends = ends[is_asst]

    # +1 at content start, -1 at <|im_end|> → cumsum gives the mask.
    signal = np.zeros(n, dtype=np.int32)
    valid = content_starts < n
    np.add.at(signal, content_starts[valid], 1)
    np.add.at(signal, content_ends[valid], -1)
    return np.cumsum(signal).astype(np.int32)


def _build_chatml_text(
    messages: list[dict[str, Any]],
    image_grids: list[tuple[int, int, int]],
    merge_size: int,
) -> str:
    """Build a ChatML string from messages, inserting image pad tokens.

    ``image_grids`` contains one ``(grid_t, grid_h, grid_w)`` tuple per
    image, in the order images appear across all messages.  For each image
    the number of ``<|image_pad|>`` tokens is
    ``grid_t * (grid_h // merge_size) * (grid_w // merge_size)``.
    """
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


def _extract_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
    """Pull PIL images out of Qwen-style structured content blocks.

    Only messages whose ``content`` is a list (structured-content format)
    are inspected; plain-string content means text-only and is skipped.
    """
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


class TextSFTCollator:
    """Collate Qwen ChatML chat examples into padded numpy arrays with loss masks.

    Outputs ``{"token_ids_BT", "attention_mask_BT", "loss_mask_BT"}``, all
    ``(B, max_length)`` int32.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        assert tokenizer.pad_token_id is not None, "tokenizer must have pad_token_id set (e.g. Qwen3-VL, Qwen3.5)"

        self._im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._assistant_token_id = tokenizer.encode("assistant", add_special_tokens=False)[0]

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
        batch_ids: list[np.ndarray] = []
        batch_attn: list[np.ndarray] = []
        batch_mask: list[np.ndarray] = []

        for ex in examples:
            messages = ex["messages"]

            result = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
            )
            full_ids = result["input_ids"]
            if len(full_ids) > self.max_length:
                full_ids = full_ids[-self.max_length:]

            seq_len = len(full_ids)
            pad_len = self.max_length - seq_len
            token_ids = np.array(full_ids, dtype=np.int32)
            attn_mask = np.ones(seq_len, dtype=np.int32)

            loss_mask = _build_assistant_loss_mask(
                token_ids, self._im_start_id, self._im_end_id,
                self._assistant_token_id,
            )

            if pad_len > 0:
                token_ids = np.pad(token_ids, (0, pad_len), constant_values=self.tokenizer.pad_token_id)
                attn_mask = np.pad(attn_mask, (0, pad_len), constant_values=0)
                loss_mask = np.pad(loss_mask, (0, pad_len), constant_values=0)

            batch_ids.append(token_ids)
            batch_attn.append(attn_mask)
            batch_mask.append(loss_mask)

        return {
            "token_ids_BT": np.stack(batch_ids).astype(np.int32),
            "attention_mask_BT": np.stack(batch_attn).astype(np.int32),
            "loss_mask_BT": np.stack(batch_mask).astype(np.int32),
        }


def _compute_vision_cu_seqlens(image_grid_thw: np.ndarray) -> np.ndarray:
    """Return cumulative per-frame token counts for the vision tower.

    For each ``(t, h, w)`` row, append ``h*w`` exactly ``t`` times, then prefix-sum
    with a leading zero. This is derived execution metadata, analogous to
    ``position_ids_ZBT``.
    """
    frame_token_counts: list[int] = []
    for t, h, w in image_grid_thw.tolist():
        frame_token_counts.extend([int(h) * int(w)] * int(t))
    return np.concatenate(
        [np.zeros(1, dtype=np.int32), np.cumsum(np.asarray(frame_token_counts, dtype=np.int32), dtype=np.int32)]
    )


class VLMSFTCollator:
    """Collate Qwen multimodal chat examples into padded numpy arrays with loss masks.

    Expects messages in the Qwen structured-content format where images are
    inline ``{"type": "image", "url": "..."}`` blocks inside ``content``
    lists.  Builds ChatML text with the correct number of image pad tokens
    and encodes via ``tokenizer.encode``.  Images are preprocessed by the
    HF image processor (slow path, NumPy backend).

    Outputs ``{"token_ids_BT", "attention_mask_BT", "loss_mask_BT"}`` plus
    ``"pixel_values"``, ``"image_grid_thw"``, and ``"position_ids_ZBT"``
    when images are present.

    ``position_ids_ZBT`` is precomputed here (on CPU, via numpy) so the
    model's ``get_rope_index`` never needs to run inside ``jax.jit``.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        image_processor: BaseImageProcessor,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_processor = image_processor
        assert tokenizer.pad_token_id is not None, "tokenizer must have pad_token_id set (e.g. Qwen3-VL, Qwen3.5)"

        self._im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._assistant_token_id = tokenizer.encode("assistant", add_special_tokens=False)[0]

        self._image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self._video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self._vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
        from omegalax.models.qwen3_vl.model import get_rope_index

        batch_ids: list[np.ndarray] = []
        batch_attn: list[np.ndarray] = []
        batch_mask: list[np.ndarray] = []
        all_pixel_values: list[np.ndarray] = []
        all_grid_thw: list[np.ndarray] = []
        has_images = False

        for ex in examples:
            messages = ex["messages"]

            imgs = _extract_images(messages)
            image_grids: list[tuple[int, int, int]] = []
            if imgs:
                has_images = True
                processed = self.image_processor.preprocess(imgs, return_tensors="np")
                pv = processed["pixel_values"]
                grid_thw = processed["image_grid_thw"]
                all_pixel_values.append(pv)
                all_grid_thw.append(grid_thw)
                image_grids = [tuple(row) for row in grid_thw.tolist()]

            text = _build_chatml_text(messages, image_grids, self.image_processor.merge_size)
            full_ids = self.tokenizer.encode(text, add_special_tokens=False)

            if len(full_ids) > self.max_length:
                full_ids = full_ids[-self.max_length:]

            seq_len = len(full_ids)
            pad_len = self.max_length - seq_len
            token_ids = np.array(full_ids, dtype=np.int32)
            attn_mask = np.ones(seq_len, dtype=np.int32)

            loss_mask = _build_assistant_loss_mask(
                token_ids, self._im_start_id, self._im_end_id,
                self._assistant_token_id,
            )

            if pad_len > 0:
                token_ids = np.pad(token_ids, (0, pad_len), constant_values=self.tokenizer.pad_token_id)
                attn_mask = np.pad(attn_mask, (0, pad_len), constant_values=0)
                loss_mask = np.pad(loss_mask, (0, pad_len), constant_values=0)

            batch_ids.append(token_ids)
            batch_attn.append(attn_mask)
            batch_mask.append(loss_mask)

        result: dict[str, np.ndarray] = {
            "token_ids_BT": np.stack(batch_ids).astype(np.int32),
            "attention_mask_BT": np.stack(batch_attn).astype(np.int32),
            "loss_mask_BT": np.stack(batch_mask).astype(np.int32),
        }

        if has_images and all_pixel_values:
            result["pixel_values"] = np.concatenate(all_pixel_values, axis=0)
        if has_images and all_grid_thw:
            image_grid_thw = np.concatenate(all_grid_thw, axis=0)
            result["image_grid_thw"] = image_grid_thw
            result["vision_cu_seqlens"] = _compute_vision_cu_seqlens(image_grid_thw)

            position_ids, _ = get_rope_index(
                result["token_ids_BT"],
                image_grid_thw=image_grid_thw,
                attention_mask=result["attention_mask_BT"],
                spatial_merge_size=self.image_processor.merge_size,
                image_token_id=self._image_token_id,
                video_token_id=self._video_token_id,
                vision_start_token_id=self._vision_start_token_id,
            )
            result["position_ids_ZBT"] = position_ids.astype(np.int32)

        return result
