"""Qwen3 / Qwen3.5 SFT collators (ChatML format, vision tokens).

Serialization and loss masking are delegated to ``renderers``; this module owns
only the JAX-side batching (padding to static shapes, vision-array packing,
``position_ids_ZBT`` precompute). No ``role_to_mask`` is passed, so the
supervised span is the renderers default, pinned by
``test_renderers_loss_mask_gate``.

The TEXT collator must NOT use ``Qwen3VLRendererConfig``: the VL renderer omits
the ``<think>\\n\\n</think>\\n\\n`` block that the Qwen3 / Qwen3.5 template emits
on the final assistant turn, which shrinks the supervised span (measured: 4 vs 8
mask tokens on a two-assistant-turn conversation) and is train/serve skew
against vLLM / prime-rl, which render the model's own template.

Video is not supported: the Qwen3-VL renderer raises ``NotImplementedError`` on
``{"type": "video"}`` parts. We feed frames as individual image parts, so this
never fires on our data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import ml_dtypes
import numpy as np
from renderers import (
    Qwen35RendererConfig,
    Qwen3RendererConfig,
    Qwen3VLRenderer,
    Qwen3VLRendererConfig,
    build_training_sample,
    create_renderer,
)
from transformers import BaseImageProcessor, PreTrainedTokenizer

from omegalax.data.arrayrecord_images import resolve_message_images

#: Probe conversation for the text-renderer template-parity self-check. Two
#: assistant turns on purpose: Qwen3's template appends the empty
#: ``<think>\n\n</think>\n\n`` block to the FINAL assistant turn only, so a
#: single-turn probe cannot tell a correct renderer from one that drops it.
_TEXT_TEMPLATE_PROBE = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "a"},
    {"role": "assistant", "content": "b"},
    {"role": "user", "content": "c"},
    {"role": "assistant", "content": "d"},
]


def resolve_text_renderer_config(
    model_id: str | None = None,
) -> Qwen3RendererConfig | Qwen35RendererConfig:
    """Pick the renderer config for a TEXT Qwen model.

    Not ``AutoRendererConfig``: resolution there is an exact match on
    ``tokenizer.name_or_path``, which misses every fine-tuned checkpoint export we
    train from and silently falls back to ``DefaultRenderer`` for text models.

    ``Qwen3RendererConfig`` and ``Qwen35RendererConfig`` were measured to render
    byte-identically for both the Qwen3 and Qwen3.5 tokenizers across plain,
    multi-turn and inline-``<think>`` conversations, so the family split changes
    nothing today. It is kept only so a future template divergence lands on the
    right config; ``assert_text_template_parity`` is what actually catches a
    wrong pick. The registry predicate is the single spelling of "is this
    Qwen3.5" -- a ``"qwen3.5" in model_id`` substring test would guess, and guess
    wrong for any fine-tune not named after its base.
    """
    if model_id is None:
        return Qwen3RendererConfig()

    from omegalax.models.qwen3_5.config import is_supported_qwen3_5_model_id  # noqa: PLC0415

    if is_supported_qwen3_5_model_id(model_id):
        return Qwen35RendererConfig()
    return Qwen3RendererConfig()


def assert_text_template_parity(
    tokenizer: PreTrainedTokenizer,
    renderer_config: Any,
) -> None:
    """Fail at construction if the renderer diverges from the tokenizer's own template.

    This is the property the module docstring promises and that nothing asserted:
    what we train on must be what the serving stack renders. Checked only for the
    auto-resolved default — passing ``renderer_config`` explicitly is the escape
    hatch for a checkpoint whose chat template legitimately differs.
    """
    reference = tokenizer.apply_chat_template(
        _TEXT_TEMPLATE_PROBE, tokenize=False, add_generation_prompt=False
    )
    sample = build_training_sample(
        create_renderer(tokenizer, renderer_config), _TEXT_TEMPLATE_PROBE
    )
    rendered = tokenizer.decode(list(sample.token_ids))
    if rendered != reference:
        raise ValueError(
            f"{type(renderer_config).__name__} does not reproduce the chat template of "
            f"{getattr(tokenizer, 'name_or_path', '<unknown>')!r}, so SFT would train on a "
            f"different stream than the serving stack renders.\n"
            f"  renderer: {rendered!r}\n"
            f"  template: {reference!r}\n"
            "Pass renderer_config= explicitly if this checkpoint's template really does "
            "differ and you have confirmed the rollout side matches."
        )


class _ImageProcessorAsProcessor:
    """Adapt a bare ``BaseImageProcessor`` to the ``.image_processor`` attribute
    the Qwen3-VL renderer reads off a full ``Qwen3VLProcessor``.

    omegalax constructs the image processor directly (``AutoImageProcessor``,
    ``use_fast=False``, optionally with an overridden ``preprocessor_config``),
    and that exact instance must be the one the renderer calls — otherwise a
    lazily ``AutoProcessor.from_pretrained``-loaded default would silently
    change patch geometry.
    """

    __slots__ = ("image_processor",)

    def __init__(self, image_processor: BaseImageProcessor) -> None:
        self.image_processor = image_processor


class Qwen3RendererEncoder:
    """Picklable ``messages -> (token_ids, loss_mask, vision arrays)`` encoder.

    Defined at module scope with a lazily-built renderer so it survives
    ``spawn`` multiprocessing (``grain_pipeline._compute_message_lengths_from_chat``
    re-pickles the measure fn into each worker; the HF tokenizer pickles, the
    renderer's caches need not travel).

    The config is pinned here, never auto-resolved: ``AutoRendererConfig`` is an
    exact match on ``tokenizer.name_or_path`` against ``MODEL_RENDERER_MAP`` and
    *raises* for a VLM that misses the map, which is every fine-tuned checkpoint
    export we train from.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        image_processor: BaseImageProcessor | None = None,
        renderer_config: Qwen3VLRendererConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.renderer_config = (
            Qwen3VLRendererConfig() if renderer_config is None else renderer_config
        )
        self._renderer = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_renderer"] = None
        return state

    @property
    def renderer(self):
        if self._renderer is None:
            if self.image_processor is not None:
                # Constructed directly so OUR image processor is the one used;
                # ``create_renderer`` has no processor seam and the renderer would
                # otherwise lazily ``AutoProcessor.from_pretrained`` a default,
                # silently changing patch geometry.
                self._renderer = Qwen3VLRenderer(
                    self.tokenizer,
                    self.renderer_config,
                    processor=_ImageProcessorAsProcessor(self.image_processor),
                )
            else:
                self._renderer = create_renderer(self.tokenizer, self.renderer_config)
        return self._renderer

    def render(self, messages: list[dict[str, Any]]):
        """``messages -> RenderedTrainingSample``.

        This class deliberately defines NO ``__call__``. ``_MessageLengthFn``
        subclasses it and is invoked as ``fn(message)`` with a different
        signature, so a base ``__call__`` that forwards to ``render`` plus an
        ``encode`` that self-calls via ``self(...)`` is unbounded mutual
        recursion -- that shipped once and killed every chunk-index job.
        """
        return build_training_sample(self.renderer, resolve_message_images(messages))

    def encode(self, messages: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        """``token_ids`` + ``loss_mask`` + concatenated vision arrays for one example."""
        sample = self.render(messages)
        out: dict[str, np.ndarray] = {
            "input_ids": np.asarray(sample.token_ids, dtype=np.int32),
            "loss_mask": np.asarray(sample.loss_mask, dtype=np.int32),
        }
        if sample.mm_token_type_ids is not None:
            out["mm_token_type_ids"] = np.asarray(sample.mm_token_type_ids, dtype=np.int32)
        items = sample.multi_modal_data.mm_items.get("image", []) if sample.multi_modal_data else []
        if items:
            out["pixel_values"] = np.concatenate([i["pixel_values"] for i in items], axis=0)
            out["image_grid_thw"] = np.concatenate(
                [np.asarray(i["image_grid_thw"]).reshape(-1, 3) for i in items], axis=0
            )
        return out


class _MessageLengthFn(Qwen3RendererEncoder):
    """Picklable ``message -> measurement`` callable (see ``make_message_length_fn``)."""

    def reject_unmeasurable(self, message: dict[str, Any]) -> None:
        """Raise if this fn cannot measure ``message``.

        The single definition of measurability. ``grain_pipeline`` calls it over
        the whole task list in the parent before spawning workers, so the failure
        lands on the operator's terminal instead of inside a pool child; keeping
        the predicate here means the two boundaries cannot drift apart.
        """
        if self.image_processor is None and _message_has_images(message):
            raise ValueError(
                "message has image content but the measure fn has no image_processor. "
                "Pass --processor <hf-repo> (the scripts' flag; --preprocessor_config "
                "to override its geometry), or image_processor= if calling "
                "make_message_length_fn directly."
            )

    def __call__(self, message: dict[str, Any]) -> int | dict[str, Any]:
        self.reject_unmeasurable(message)
        encoded = self.encode([message])
        merge_size = (
            int(getattr(self.image_processor, "merge_size", 1)) if self.image_processor else 1
        )
        grid_thw = encoded.get("image_grid_thw", np.empty((0, 3), dtype=np.int64))
        vision_tokens = 0
        vision_patches = 0
        for row in grid_thw:
            t, h, w = int(row[0]), int(row[1]), int(row[2])
            vision_tokens += t * (h // merge_size) * (w // merge_size)
            vision_patches += t * h * w
        return {
            "length": int(len(encoded["input_ids"])),
            "vision_tokens": vision_tokens,
            "vision_patches": vision_patches,
            "num_images": int(grid_thw.shape[0]),
            "image_grid_thw": grid_thw.tolist(),
        }


def _message_has_images(message: dict[str, Any]) -> bool:
    content = message.get("content", "")
    if isinstance(content, str):
        return False
    return any(block.get("type") in ("image", "image_url") for block in content)


def make_message_length_fn(
    tokenizer: PreTrainedTokenizer,
    image_processor: BaseImageProcessor | None = None,
    renderer_config: Qwen3VLRendererConfig | None = None,
):
    """Return a ``message -> measurement`` callable for the record builders.

    Token lengths are exactly additive at message boundaries:
    ``<|im_start|>``/``<|im_end|>`` are hard BPE split points, so
    ``sum(lengths)`` equals the full-sequence length exactly. Returns a
    picklable instance so it can be shipped to ``spawn`` workers.
    """
    return _MessageLengthFn(tokenizer, image_processor, renderer_config)


class TextSFTCollator:
    """Collate Qwen ChatML chat examples into padded numpy arrays with loss masks.

    Outputs ``{"token_ids_BT", "attention_mask_BT", "loss_mask_BT"}``, all
    ``(B, max_length)`` int32.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        renderer_config: Any | None = None,
        model_id: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        assert tokenizer.pad_token_id is not None, (
            "tokenizer must have pad_token_id set (e.g. Qwen3-VL, Qwen3.5)"
        )
        if renderer_config is None:
            renderer_config = resolve_text_renderer_config(model_id)
            assert_text_template_parity(tokenizer, renderer_config)
        self._encoder = Qwen3RendererEncoder(tokenizer, None, renderer_config)

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

    if num_dummies < 0 or extra_patches < 0:
        raise ValueError(
            f"Batch exceeds padding budget: real_images={real_images} > "
            f"max_images={max_images} or real_patches={real_patches} > "
            f"max_patches={max_patches}. Increase the per-sample limits."
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
    inline ``{"type": "image", "url": "..."}`` blocks inside ``content`` lists.

    Every key is always present at a fixed shape, images or not, so ``train_step``
    never recompiles: ``token_ids_BT``, ``attention_mask_BT``, ``loss_mask_BT``,
    ``mm_token_type_ids_BT``, ``pixel_values``, ``image_grid_thw``,
    ``vision_cu_seqlens``, ``position_ids_ZBT``.

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
        renderer_config: Qwen3VLRendererConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_processor = image_processor
        self._max_vision_patches_per_sample = max_vision_patches_per_sample
        self._max_vision_images_per_sample = max_vision_images_per_sample
        self._pixel_values_dtype = pixel_values_dtype
        assert tokenizer.pad_token_id is not None, (
            "tokenizer must have pad_token_id set (e.g. Qwen3-VL, Qwen3.5)"
        )
        self._encoder = Qwen3RendererEncoder(tokenizer, image_processor, renderer_config)

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

        batch_mm_type: list[np.ndarray] = []

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
            # 0=text / 1=image / 2=video — the vision encoder slices image
            # tokens out of the packed stream with this. Text-only samples
            # through the VLM renderer get no sidecar; synthesize zeros so the
            # batch key is always present at a fixed shape.
            mm_type = encoded.get("mm_token_type_ids")
            if mm_type is None:
                mm_type = np.zeros(seq_len, dtype=np.int32)

            if pad_len > 0:
                mm_type = np.pad(mm_type, (0, pad_len), constant_values=0)
                token_ids = np.pad(
                    token_ids, (0, pad_len), constant_values=self.tokenizer.pad_token_id
                )
                attn_mask = np.pad(attn_mask, (0, pad_len), constant_values=0)
                loss_mask = np.pad(loss_mask, (0, pad_len), constant_values=0)

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
