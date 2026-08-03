"""Shared Qwen3/Qwen3.5 message serialization and encoding helpers."""

from __future__ import annotations

import atexit
import io
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image
from transformers import BaseImageProcessor, PreTrainedTokenizer

_ARRAYRECORD_IMAGE_URI_SCHEME = "ar"
_ARRAYRECORD_IMAGE_CACHE_SIZE = int(os.environ.get("OMEGALAX_ARRAYRECORD_IMAGE_CACHE_SIZE", "128"))
_ARRAYRECORD_IMAGE_SOURCES: OrderedDict[str, Any] = OrderedDict()


def _close_arrayrecord_reader(reader: Any) -> None:
    close = getattr(reader, "close", None)
    if close is not None:
        close()


def _close_arrayrecord_image_sources() -> None:
    while _ARRAYRECORD_IMAGE_SOURCES:
        _, reader = _ARRAYRECORD_IMAGE_SOURCES.popitem(last=False)
        _close_arrayrecord_reader(reader)


def _get_arrayrecord_image_reader(path: str) -> Any:
    reader = _ARRAYRECORD_IMAGE_SOURCES.get(path)
    if reader is not None:
        _ARRAYRECORD_IMAGE_SOURCES.move_to_end(path)
        return reader

    from array_record.python.array_record_module import ArrayRecordReader  # noqa: PLC0415

    reader = ArrayRecordReader(path)
    if _ARRAYRECORD_IMAGE_CACHE_SIZE <= 0:
        return reader

    while len(_ARRAYRECORD_IMAGE_SOURCES) >= _ARRAYRECORD_IMAGE_CACHE_SIZE:
        _, old_reader = _ARRAYRECORD_IMAGE_SOURCES.popitem(last=False)
        _close_arrayrecord_reader(old_reader)
    _ARRAYRECORD_IMAGE_SOURCES[path] = reader
    return reader


# A ChatML turn is ``<|im_start|>{role}\n{content}<|im_end|>\n``. Because
# ``<|im_start|>``/``<|im_end|>`` are registered special tokens (hard BPE split
# points), the assistant header ``<|im_start|>assistant\n`` always tokenizes to
# exactly three tokens and the ``<|im_end|>\n`` footer to two, independent of the
# surrounding text. These offsets let us mask assistant content structurally.
_ASSISTANT_ROLE = "assistant"
_CHATML_HEADER_TOKENS = 3  # <|im_start|> , role , \n
_CHATML_TRAILING_TOKENS = 1  # the \n after <|im_end|> (the <|im_end|> itself is supervised)


def build_chatml_blocks(
    messages: list[dict[str, Any]],
    image_grids: list[tuple[int, int, int]],
    merge_size: int,
) -> list[tuple[str, str]]:
    """Return one ``(role, block_text)`` per message.

    Each block is a complete ChatML turn ``<|im_start|>{role}\n{content}<|im_end|>\n``.
    Concatenating the block texts reproduces :func:`build_chatml_text` exactly, and
    because ``<|im_start|>``/``<|im_end|>`` are registered special tokens (hard BPE
    split points) each block also tokenizes independently: the concatenation of the
    per-block token ids equals ``tokenizer.encode(build_chatml_text(...))``. Callers
    use this to build the assistant loss mask from message *structure* rather than by
    scanning the final token stream for ChatML specials -- the latter is corrupted
    when user/context text contains literal ``<|im_start|>`` / ``<|im_end|>`` markers.
    """

    blocks: list[tuple[str, str]] = []
    img_idx = 0

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        parts: list[str] = [f"<|im_start|>{role}\n"]

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
                    parts.append("<|vision_start|>" + "<|image_pad|>" * n_tokens + "<|vision_end|>")

        parts.append("<|im_end|>\n")
        blocks.append((role, "".join(parts)))

    return blocks


def build_chatml_text(
    messages: list[dict[str, Any]],
    image_grids: list[tuple[int, int, int]],
    merge_size: int,
) -> str:
    """Build a ChatML string from messages, inserting image pad tokens."""

    return "".join(
        block_text for _, block_text in build_chatml_blocks(messages, image_grids, merge_size)
    )


def _assistant_block_loss_mask(block_ids: np.ndarray, is_assistant: bool) -> np.ndarray:
    """Loss mask for a single ChatML block: 1 on supervised tokens, 0 elsewhere.

    A block is ``<|im_start|>{role}\n{content}<|im_end|>\n``. Non-assistant turns are
    never supervised. For assistant turns everything between the 3-token header
    (``<|im_start|>``, ``assistant``, ``\n``) and the trailing ``\n`` is supervised --
    i.e. the content plus the terminating ``<|im_end|>`` so the model learns to stop.

    Because the mask is scoped to one block built from message structure, literal
    ``<|im_start|>`` / ``<|im_end|>`` markers appearing inside ``content`` cannot flip
    neighbouring (user/system) turns -- or image pad tokens -- to supervised.
    """

    mask = np.zeros(len(block_ids), dtype=np.int32)
    if not is_assistant:
        return mask
    start = _CHATML_HEADER_TOKENS
    end = len(block_ids) - _CHATML_TRAILING_TOKENS  # exclude trailing \n, keep <|im_end|>
    if end > start:
        mask[start:end] = 1
    return mask


def _is_arrayrecord_image_uri(value: object) -> bool:
    return isinstance(value, str) and value.startswith(f"{_ARRAYRECORD_IMAGE_URI_SCHEME}://")


def _parse_arrayrecord_image_uri(uri: str) -> tuple[Path, int]:
    parsed = urlparse(uri)
    if parsed.scheme != _ARRAYRECORD_IMAGE_URI_SCHEME:
        raise ValueError(f"not an ArrayRecord image URI: {uri!r}")
    if parsed.netloc:
        raise ValueError(
            f"unsupported named ArrayRecord image URI {uri!r}; expected ar:///path#record"
        )
    if not parsed.path or not parsed.fragment:
        raise ValueError(f"malformed ArrayRecord image URI: {uri!r}")
    try:
        record_index = int(parsed.fragment)
    except ValueError as e:
        raise ValueError(f"ArrayRecord URI fragment must be an integer: {uri!r}") from e
    if record_index < 0:
        raise ValueError(f"ArrayRecord URI record index must be non-negative: {uri!r}")
    return Path(unquote(parsed.path)), record_index


def _open_arrayrecord_image(uri: str) -> Image.Image:
    shard_path, record_index = _parse_arrayrecord_image_uri(uri)
    key = str(shard_path)
    reader = _get_arrayrecord_image_reader(key)
    try:
        jpeg_bytes = reader.read([record_index])[0]
        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            return img.convert("RGB")
    finally:
        if _ARRAYRECORD_IMAGE_CACHE_SIZE <= 0:
            _close_arrayrecord_reader(reader)


atexit.register(_close_arrayrecord_image_sources)


def _open_image_ref(ref: Any) -> Image.Image:
    if isinstance(ref, Image.Image):
        return ref
    if _is_arrayrecord_image_uri(ref):
        return _open_arrayrecord_image(ref)
    return Image.open(ref)


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
                images.append(_open_image_ref(block["image"]))
            elif "url" in block:
                images.append(_open_image_ref(block["url"]))
    return images


def _message_has_images(message: dict[str, Any]) -> bool:
    content = message.get("content", "")
    if isinstance(content, str):
        return False
    return any(block.get("type") == "image" for block in content)


class _MessageLengthFn:
    """Picklable ``message -> measurement`` callable (see ``make_message_length_fn``).

    Defined at module scope rather than as a closure so it can be pickled and
    sent to ``spawn`` multiprocessing workers. The measure pass
    (``grain_pipeline._compute_message_lengths_from_chat``) uses ``spawn`` --
    workers must not inherit the parent's thread-tainted native ArrayRecord image
    readers, which segfault -- and ``spawn`` re-pickles the measure fn into each
    worker; a nested closure cannot cross that boundary, an instance of this
    class can.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        image_processor: BaseImageProcessor | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.merge_size = int(getattr(image_processor, "merge_size", 1)) if image_processor else 1

    def __call__(self, message: dict[str, Any]) -> int | dict[str, Any]:
        if self.image_processor is None and _message_has_images(message):
            raise ValueError(
                "Encountered image content in message but no image_processor was provided. "
                "Pass image_processor= to make_message_length_fn."
            )
        encoded = encode_qwen_messages(
            [message],
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            include_pixels=False,
        )
        length = int(len(encoded["input_ids"]))

        grid_thw = encoded.get("image_grid_thw", np.empty((0, 3), dtype=np.int64))
        num_images = int(grid_thw.shape[0])
        vision_tokens = 0
        vision_patches = 0
        for row in grid_thw:
            t, h, w = int(row[0]), int(row[1]), int(row[2])
            vision_tokens += t * (h // self.merge_size) * (w // self.merge_size)
            vision_patches += t * h * w

        return {
            "length": length,
            "vision_tokens": vision_tokens,
            "vision_patches": vision_patches,
            "num_images": num_images,
            "image_grid_thw": grid_thw.tolist(),
        }


def make_message_length_fn(
    tokenizer: PreTrainedTokenizer,
    image_processor: BaseImageProcessor | None = None,
):
    """Return a ``message -> token_count`` callable for use with the record builders.

    Suitable for ChatML-formatted models (Qwen3 / Qwen3.5).  Token lengths are
    exactly additive at message boundaries: ``<|im_start|>``/``<|im_end|>`` act
    as hard BPE split points and ``add_special_tokens=False`` suppresses any
    per-sequence overhead, so ``sum(lengths)`` equals the full-sequence length
    exactly.  Returns a picklable :class:`_MessageLengthFn` instance so it can be
    shipped to ``spawn`` workers. For a different chat template, implement an
    analogous factory and swap it in.
    """
    return _MessageLengthFn(tokenizer, image_processor)


def encode_qwen_messages(
    messages: list[dict[str, Any]],
    *,
    tokenizer: PreTrainedTokenizer,
    image_processor: BaseImageProcessor | None = None,
    include_pixels: bool = False,
) -> dict[str, np.ndarray]:
    """Encode a Qwen chat example exactly as the collators expect.

    Returns ``input_ids`` and a matching assistant ``loss_mask`` (both 1-D int32),
    plus ``image_grid_thw`` / ``pixel_values`` when an image processor is given. The
    loss mask is built from message structure per ChatML turn, so it is unaffected by
    literal ``<|im_start|>`` / ``<|im_end|>`` markers embedded in user/context text.
    """

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
    blocks = build_chatml_blocks(messages, image_grids, merge_size)

    # Encode each ChatML turn independently and build its loss mask from message
    # structure. The additive property of the ChatML specials (see
    # ``build_chatml_blocks``) guarantees the concatenated ids are identical to a
    # single ``tokenizer.encode`` of the whole conversation, while masking per block
    # is immune to literal ``<|im_start|>`` / ``<|im_end|>`` markers in user text.
    id_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    for role, block_text in blocks:
        block_ids = np.asarray(
            tokenizer.encode(block_text, add_special_tokens=False),
            dtype=np.int32,
        )
        id_parts.append(block_ids)
        mask_parts.append(_assistant_block_loss_mask(block_ids, role == _ASSISTANT_ROLE))

    result["input_ids"] = np.concatenate(id_parts) if id_parts else np.zeros(0, dtype=np.int32)
    result["loss_mask"] = np.concatenate(mask_parts) if mask_parts else np.zeros(0, dtype=np.int32)
    return result
