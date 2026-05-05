"""Qwen3 / Qwen3.5 SFT collators (ChatML format, vision tokens).

Tokenize chat conversations and produce loss masks. Both collators output
numpy dicts ready for ``shard_batch_dict``. Model-specific to Qwen3-VL and
Qwen3.5 (ChatML delimiters, assistant-based loss, Qwen image processor).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from transformers import BaseImageProcessor, PreTrainedTokenizer

from omegalax.data.qwen3_encoding import (
    build_chatml_text as _build_chatml_text,
    encode_qwen_messages as _encode_qwen_messages,
    extract_images as _extract_images,
)


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

    # +1 at content start, -1 after <|im_end|> → cumsum gives the mask.
    # content_ends points at <|im_end|> itself, which must be a supervised
    # target so the model learns to terminate; the \n that follows is not.
    signal = np.zeros(n, dtype=np.int32)
    valid = content_starts < n
    ends_plus_one = content_ends[valid] + 1
    np.add.at(signal, content_starts[valid], 1)
    np.add.at(signal, ends_plus_one[ends_plus_one < n], -1)
    return np.cumsum(signal).astype(np.int32)


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
            encoded = _encode_qwen_messages(
                messages,
                tokenizer=self.tokenizer,
            )
            full_ids = encoded["input_ids"]
            if len(full_ids) > self.max_length:
                raise ValueError(
                    f"Encoded example length {len(full_ids)} exceeds max_length={self.max_length}; "
                    "rebuild the chunk index for this profile."
                )

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


def _pad_vision_arrays(
    pixel_values: np.ndarray,
    image_grid_thw: np.ndarray,
    merge_size: int,
    max_patches: int,
    max_images: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad vision arrays to exact ``(max_patches, max_images)`` target.

    Uses "absorber" dummy images whose grid entries are chosen to make the
    total patch count equal ``max_patches`` exactly, preserving the invariant
    ``pixel_values.shape[0] == sum(t*h*w for image_grid_thw)``.
    """
    real_images = image_grid_thw.shape[0]
    real_patches = pixel_values.shape[0]
    feat_dim = pixel_values.shape[1]
    ms2 = merge_size * merge_size

    num_dummies = max_images - real_images
    extra_patches = max_patches - real_patches

    if num_dummies < 0 or extra_patches < 0:
        raise ValueError(
            f"Batch exceeds padding budget: real_images={real_images} > "
            f"max_images={max_images} or real_patches={real_patches} > "
            f"max_patches={max_patches}. Increase the per-sample limits."
        )

    if num_dummies == 0 and extra_patches == 0:
        return pixel_values, image_grid_thw, _compute_vision_cu_seqlens(image_grid_thw)

    # Build dummy grid entries: one absorber + simple (1, ms, ms) dummies.
    dummy_grids: list[list[int]] = []
    if num_dummies == 1:
        dummy_grids.append([1, merge_size, extra_patches // merge_size])
    else:
        num_simple = num_dummies - 1
        absorber_patches = extra_patches - num_simple * ms2
        dummy_grids.append([1, merge_size, absorber_patches // merge_size])
        dummy_grids.extend([[1, merge_size, merge_size]] * num_simple)

    padded_grid = np.concatenate(
        [image_grid_thw, np.array(dummy_grids, dtype=np.int32)], axis=0,
    )
    padded_pv = np.concatenate(
        [pixel_values, np.zeros((extra_patches, feat_dim), dtype=pixel_values.dtype)],
        axis=0,
    )
    padded_cu = _compute_vision_cu_seqlens(padded_grid)
    return padded_pv, padded_grid, padded_cu


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
        *,
        max_vision_patches_per_sample: int | None = None,
        max_vision_images_per_sample: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_processor = image_processor
        self._max_vision_patches_per_sample = max_vision_patches_per_sample
        self._max_vision_images_per_sample = max_vision_images_per_sample
        assert tokenizer.pad_token_id is not None, "tokenizer must have pad_token_id set (e.g. Qwen3-VL, Qwen3.5)"

        self._im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._assistant_token_id = tokenizer.encode("assistant", add_special_tokens=False)[0]

        self._image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self._video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self._vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")

        # Per-patch feat dim from the HF image processor's (T, C, P, P) flatten,
        # used to shape the all-text-only placeholder so ``pixel_values`` stays
        # in the batch dict at fixed shape (else ``train_step`` recompiles).
        self._patch_feat_dim = (
            image_processor.temporal_patch_size
            * len(image_processor.image_mean)
            * image_processor.patch_size
            * image_processor.patch_size
        )

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
            encoded = _encode_qwen_messages(
                messages,
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                include_pixels=True,
            )
            full_ids = encoded["input_ids"]
            if len(full_ids) > self.max_length:
                raise ValueError(
                    f"Encoded example length {len(full_ids)} exceeds max_length={self.max_length}; "
                    "rebuild the chunk index for this profile."
                )

            if "pixel_values" in encoded:
                has_images = True
                all_pixel_values.append(encoded["pixel_values"])
                all_grid_thw.append(encoded["image_grid_thw"])

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

        if all_pixel_values:
            pixel_values = np.concatenate(all_pixel_values, axis=0)
            image_grid_thw = np.concatenate(all_grid_thw, axis=0)
        else:
            pixel_values = np.zeros((0, self._patch_feat_dim), dtype=np.float32)
            image_grid_thw = np.zeros((0, 3), dtype=np.int32)

        # Compute position_ids from REAL (unpadded) grid — these only
        # depend on real <|image_pad|> positions in token_ids_BT.
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

        # Pad vision arrays to static shapes so JAX JIT never recompiles.
        # Per-sample limits are multiplied by batch size so the user
        # doesn't need to recompute when changing batch_size.
        if self._max_vision_patches_per_sample is not None and self._max_vision_images_per_sample is not None:
            bs = len(examples)
            pixel_values, image_grid_thw, vision_cu_seqlens = _pad_vision_arrays(
                pixel_values, image_grid_thw,
                merge_size=self.image_processor.merge_size,
                max_patches=self._max_vision_patches_per_sample * bs,
                max_images=self._max_vision_images_per_sample * bs,
            )
        else:
            vision_cu_seqlens = _compute_vision_cu_seqlens(image_grid_thw)

        result["pixel_values"] = pixel_values
        result["image_grid_thw"] = image_grid_thw
        result["vision_cu_seqlens"] = vision_cu_seqlens

        return result
