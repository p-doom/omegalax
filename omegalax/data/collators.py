"""SFT collators that tokenize chat conversations and produce loss masks.

Both collators output numpy dicts ready for ``shard_batch_dict``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizer, ProcessorMixin


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

    NOTE: hard-coded for the ChatML delimiters used by Qwen3-VL / Qwen3.5.
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


class TextSFTCollator:
    """Collate chat-format examples into padded numpy arrays with loss masks.

    Outputs ``{"token_ids_BT", "attention_mask_BT", "loss_mask_BT"}``, all
    ``(B, max_length)`` int32.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

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


class VLMSFTCollator:
    """Collate multimodal chat examples into padded numpy arrays with loss masks.

    Expects messages in the Qwen structured-content format where images are
    inline ``{"type": "image", "url": "..."}`` blocks inside ``content``
    lists.  Uses ``processor.apply_chat_template`` in a single call to
    tokenize text and process images together.

    Outputs ``{"token_ids_BT", "attention_mask_BT", "loss_mask_BT"}`` plus
    optional ``"pixel_values"`` and ``"image_grid_thw"`` when images are present.
    """

    def __init__(self, processor: ProcessorMixin, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length
        self.tokenizer: PreTrainedTokenizer = processor.tokenizer  # type: ignore[attr-defined]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._assistant_token_id = self.tokenizer.encode("assistant", add_special_tokens=False)[0]

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
        batch_ids: list[np.ndarray] = []
        batch_attn: list[np.ndarray] = []
        batch_mask: list[np.ndarray] = []
        all_pixel_values: list[np.ndarray] = []
        all_grid_thw: list[np.ndarray] = []
        has_images = False

        for ex in examples:
            messages = ex["messages"]

            processed = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="np",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )

            input_ids = processed["input_ids"][0].astype(np.int32)
            attn_mask = processed["attention_mask"][0].astype(np.int32)

            loss_mask = _build_assistant_loss_mask(
                input_ids, self._im_start_id, self._im_end_id,
                self._assistant_token_id,
            )
            loss_mask = loss_mask * attn_mask

            batch_ids.append(input_ids)
            batch_attn.append(attn_mask)
            batch_mask.append(loss_mask)

            if "pixel_values" in processed:
                has_images = True
                all_pixel_values.append(processed["pixel_values"][0])
            if "image_grid_thw" in processed:
                all_grid_thw.append(processed["image_grid_thw"][0])

        result: dict[str, np.ndarray] = {
            "token_ids_BT": np.stack(batch_ids).astype(np.int32),
            "attention_mask_BT": np.stack(batch_attn).astype(np.int32),
            "loss_mask_BT": np.stack(batch_mask).astype(np.int32),
        }

        if has_images and all_pixel_values:
            result["pixel_values"] = np.concatenate(all_pixel_values, axis=0)
        if has_images and all_grid_thw:
            result["image_grid_thw"] = np.concatenate(all_grid_thw, axis=0)

        return result
