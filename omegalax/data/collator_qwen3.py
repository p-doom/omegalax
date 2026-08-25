"""Qwen3 and Qwen3.5 SFT collators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import ml_dtypes
import numpy as np
from transformers import BaseImageProcessor, PreTrainedTokenizer

from omegalax.data.qwen3_encoding import Qwen3MessageEncoder


class TextSFTCollator:
    """Collate Qwen ChatML chat examples into padded numpy arrays with loss masks.

    Outputs ``{"token_ids_BT", "attention_mask_BT", "loss_mask_BT"}``, all
    ``(B, max_length)`` int32.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id")
        self._encoder = Qwen3MessageEncoder(tokenizer, None)

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
        batch_ids: list[np.ndarray] = []
        batch_attn: list[np.ndarray] = []
        batch_mask: list[np.ndarray] = []

        for ex in examples:
            messages = ex["messages"]
            encoded = self._encoder.encode(messages)
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
            loss_mask = encoded["loss_mask"]

            if pad_len > 0:
                token_ids = np.pad(
                    token_ids, (0, pad_len), constant_values=self.tokenizer.pad_token_id
                )
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

    Fills the ``num_dummies`` padding image slots with ``(1, ms, ms*k_i)`` rows
    whose patch counts ``ms2*k_i`` are spread as evenly as possible so the total
    lands on ``max_patches`` exactly. Preserves
    ``pixel_values.shape[0] == sum(t*h*w for image_grid_thw)``.

    Each grid row contributes at least ``ms2`` patches, so padding requires
    ``extra_patches == num_dummies == 0`` (nothing to do) or
    ``num_dummies >= 1 and extra_patches >= num_dummies * ms2``. Other budget
    combinations are infeasible — increase the per-sample limits.
    """
    real_images = image_grid_thw.shape[0]
    real_patches = pixel_values.shape[0]
    feat_dim = pixel_values.shape[1]
    ms2 = merge_size * merge_size

    num_dummies = max_images - real_images
    extra_patches = max_patches - real_patches

    exceeded = []
    if num_dummies < 0:
        exceeded.append(f"real_images={real_images} > max_images={max_images}")
    if extra_patches < 0:
        exceeded.append(f"real_patches={real_patches} > max_patches={max_patches}")
    if exceeded:
        raise ValueError(
            f"Batch exceeds padding budget: {', '.join(exceeded)}. Increase the per-sample limits."
        )

    if num_dummies == 0 and extra_patches == 0:
        return pixel_values, image_grid_thw, _compute_vision_cu_seqlens(image_grid_thw)

    if num_dummies == 0 or extra_patches < num_dummies * ms2:
        raise ValueError(
            f"Vision budgets are infeasible for this batch: real_images="
            f"{real_images}, real_patches={real_patches}, max_images="
            f"{max_images}, max_patches={max_patches}, ms2={ms2}. Padding "
            f"needs num_dummies>=1 and extra_patches>=num_dummies*ms2 (each "
            f"dummy row costs at least ms2 patches). Increase "
            f"max_vision_images_per_sample or max_vision_patches_per_sample "
            f"so this invariant holds for every batch."
        )

    if extra_patches % ms2 != 0:
        raise ValueError(
            f"extra_patches={extra_patches} not divisible by ms2={ms2}; "
            f"max_patches and every real image (t*h*w) must be multiples of "
            f"ms2 so padding rows are clean (1, ms, ms*k) images."
        )
    total_cells = extra_patches // ms2
    base, remainder = divmod(total_cells, num_dummies)
    cells_per_dummy = [base + 1] * remainder + [base] * (num_dummies - remainder)
    dummy_grids: list[list[int]] = [[1, merge_size, merge_size * k] for k in cells_per_dummy]

    padded_grid = np.concatenate(
        [image_grid_thw, np.array(dummy_grids, dtype=np.int32)],
        axis=0,
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
        [
            np.zeros(1, dtype=np.int32),
            np.cumsum(np.asarray(frame_token_counts, dtype=np.int32), dtype=np.int32),
        ]
    )


class VLMSFTCollator:
    """Collate Qwen multimodal chat examples into padded numpy arrays with loss masks.

    Expects messages in the Qwen structured-content format where images are
    inline ``{"type": "image", "image": ...}`` blocks inside ``content`` lists.

    Every key is always present at a fixed shape, images or not, so ``train_step``
    never recompiles: ``token_ids_BT``, ``attention_mask_BT``, ``loss_mask_BT``,
    ``pixel_values``, ``image_grid_thw``, ``vision_cu_seqlens``, ``position_ids_ZBT``.

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
        pixel_values_dtype: Any = ml_dtypes.bfloat16,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_processor = image_processor
        self._max_vision_patches_per_sample = max_vision_patches_per_sample
        self._max_vision_images_per_sample = max_vision_images_per_sample
        self._pixel_values_dtype = pixel_values_dtype
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id")
        self._encoder = Qwen3MessageEncoder(tokenizer, image_processor)

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
        batch_mm_type: list[np.ndarray] = []
        all_pixel_values: list[np.ndarray] = []
        all_grid_thw: list[np.ndarray] = []

        for ex in examples:
            messages = ex["messages"]
            encoded = self._encoder.encode(messages)
            full_ids = encoded["input_ids"]
            if len(full_ids) > self.max_length:
                raise ValueError(
                    f"Encoded example length {len(full_ids)} exceeds max_length={self.max_length}; "
                    "rebuild the chunk index for this profile."
                )

            if "pixel_values" in encoded:
                all_pixel_values.append(encoded["pixel_values"])
                all_grid_thw.append(encoded["image_grid_thw"])

            seq_len = len(full_ids)
            pad_len = self.max_length - seq_len
            token_ids = np.array(full_ids, dtype=np.int32)
            attn_mask = np.ones(seq_len, dtype=np.int32)
            loss_mask = encoded["loss_mask"]
            mm_type = encoded["mm_token_type_ids"]

            if pad_len > 0:
                token_ids = np.pad(
                    token_ids, (0, pad_len), constant_values=self.tokenizer.pad_token_id
                )
                attn_mask = np.pad(attn_mask, (0, pad_len), constant_values=0)
                loss_mask = np.pad(loss_mask, (0, pad_len), constant_values=0)
                mm_type = np.pad(mm_type, (0, pad_len), constant_values=0)

            batch_ids.append(token_ids)
            batch_attn.append(attn_mask)
            batch_mask.append(loss_mask)
            batch_mm_type.append(mm_type)

        result: dict[str, np.ndarray] = {
            "token_ids_BT": np.stack(batch_ids).astype(np.int32),
            "attention_mask_BT": np.stack(batch_attn).astype(np.int32),
            "loss_mask_BT": np.stack(batch_mask).astype(np.int32),
            "mm_token_type_ids_BT": np.stack(batch_mm_type).astype(np.int32),
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
        if (
            self._max_vision_patches_per_sample is not None
            and self._max_vision_images_per_sample is not None
        ):
            bs = len(examples)
            pixel_values, image_grid_thw, vision_cu_seqlens = _pad_vision_arrays(
                pixel_values,
                image_grid_thw,
                merge_size=self.image_processor.merge_size,
                max_patches=self._max_vision_patches_per_sample * bs,
                max_images=self._max_vision_images_per_sample * bs,
            )
        else:
            vision_cu_seqlens = _compute_vision_cu_seqlens(image_grid_thw)

        result["pixel_values"] = pixel_values.astype(self._pixel_values_dtype, copy=False)
        result["image_grid_thw"] = image_grid_thw
        result["vision_cu_seqlens"] = vision_cu_seqlens

        return result
