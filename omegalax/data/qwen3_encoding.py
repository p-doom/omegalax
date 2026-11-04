"""Direct Qwen message encoding for SFT."""

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

from omegalax.data.qwen_renderers import renderer_for_model_type

_ARRAYRECORD_IMAGE_CACHE_SIZE = int(os.environ.get("OMEGALAX_ARRAYRECORD_IMAGE_CACHE_SIZE", "128"))
_ARRAYRECORD_IMAGE_SOURCES: OrderedDict[str, Any] = OrderedDict()

_CHATML_HEADER_TOKENS = 3
_CHATML_TRAILING_TOKENS = 1


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

    from array_record.python.array_record_module import ArrayRecordReader

    reader = ArrayRecordReader(path)
    if _ARRAYRECORD_IMAGE_CACHE_SIZE <= 0:
        return reader

    while len(_ARRAYRECORD_IMAGE_SOURCES) >= _ARRAYRECORD_IMAGE_CACHE_SIZE:
        _, old_reader = _ARRAYRECORD_IMAGE_SOURCES.popitem(last=False)
        _close_arrayrecord_reader(old_reader)
    _ARRAYRECORD_IMAGE_SOURCES[path] = reader
    return reader


def _parse_arrayrecord_image_uri(uri: str) -> tuple[Path, int]:
    parsed = urlparse(uri)
    if parsed.scheme != "ar":
        raise ValueError(f"not an ArrayRecord image URI: {uri!r}")
    if parsed.netloc:
        raise ValueError(
            f"unsupported named ArrayRecord image URI {uri!r}; expected ar:///path#record"
        )
    if not parsed.path or not parsed.fragment:
        raise ValueError(f"malformed ArrayRecord image URI: {uri!r}")
    try:
        record_index = int(parsed.fragment)
    except ValueError as error:
        raise ValueError(f"ArrayRecord URI fragment must be an integer: {uri!r}") from error
    if record_index < 0:
        raise ValueError(f"ArrayRecord URI record index must be non-negative: {uri!r}")
    return Path(unquote(parsed.path)), record_index


def _open_arrayrecord_image(uri: str) -> Image.Image:
    shard_path, record_index = _parse_arrayrecord_image_uri(uri)
    reader = _get_arrayrecord_image_reader(str(shard_path))
    try:
        jpeg_bytes = reader.read([record_index])[0]
        with Image.open(io.BytesIO(jpeg_bytes)) as image:
            return image.convert("RGB")
    finally:
        if _ARRAYRECORD_IMAGE_CACHE_SIZE <= 0:
            _close_arrayrecord_reader(reader)


def _open_image_ref(ref: Any) -> Image.Image:
    if isinstance(ref, Image.Image):
        return ref
    if isinstance(ref, str) and ref.startswith("ar://"):
        return _open_arrayrecord_image(ref)
    if not isinstance(ref, (str, os.PathLike)):
        raise TypeError("image reference must be a PIL image, local path, or ar:// URI")
    if isinstance(ref, str) and "://" in ref:
        raise ValueError("remote image references are not supported")
    return Image.open(ref)


def _extract_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            continue
        for part in content:
            if part["type"] == "image":
                images.append(_open_image_ref(part["image"]))
    return images


atexit.register(_close_arrayrecord_image_sources)


def _validate_message(message: dict[str, Any], *, multimodal: bool) -> None:
    if (
        not isinstance(message, dict)
        or not {"role", "content"} <= set(message)
        or set(message) - {"role", "content", "reasoning_content", "loss"}
    ):
        raise ValueError(
            "messages must contain role and content; only reasoning_content and loss are optional"
        )
    role = message["role"]
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"unsupported message role: {role!r}")
    if "loss" in message:
        if role != "assistant":
            raise ValueError("loss is supported only on assistant turns")
        if not isinstance(message["loss"], bool):
            raise ValueError("loss must be a boolean")
    reasoning = message.get("reasoning_content")
    if reasoning is not None and (role != "assistant" or not isinstance(reasoning, str)):
        raise ValueError("reasoning_content must be a string on an assistant turn")

    content = message["content"]
    if isinstance(content, str):
        if "<|video_pad|>" in content:
            raise ValueError("video content is not supported")
        return
    if not isinstance(content, list) or not content:
        raise ValueError("message content must be a string or a non-empty multimodal part list")
    if not multimodal and any(
        isinstance(part, dict) and part.get("type") == "image" for part in content
    ):
        raise ValueError("image content requires an image processor")
    if not multimodal:
        raise ValueError("structured content requires an image processor")
    for part in content:
        if isinstance(part, dict) and (part.get("type") == "video" or "video" in part):
            raise ValueError("video content is not supported")
        if not isinstance(part, dict) or part.get("type") not in {"text", "image"}:
            raise ValueError("multimodal parts must be text or image")
        if part["type"] == "text":
            if set(part) != {"type", "text"} or not isinstance(part["text"], str):
                raise ValueError("text parts must contain exactly type and string text")
            if "<|video_pad|>" in part["text"]:
                raise ValueError("video content is not supported")
        elif set(part) != {"type", "image"}:
            raise ValueError("image parts must contain exactly type and image")
        elif role != "user":
            raise ValueError("images are supported only in user turns")


def _assistant_loss_mask(block_ids: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(block_ids), dtype=np.int32)
    mask[_CHATML_HEADER_TOKENS:-_CHATML_TRAILING_TOKENS] = 1
    return mask


def _message_is_supervised(message: dict[str, Any]) -> bool:
    return message["role"] == "assistant" and message.get("loss", True)


class Qwen3MessageEncoder:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        image_processor: BaseImageProcessor | None,
        model_type: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self._renderer, supports_images = renderer_for_model_type(model_type)
        if image_processor is not None and not supports_images:
            raise ValueError("Qwen3 text encoding does not accept an image processor")
        self.merge_size = int(getattr(image_processor, "merge_size", 1))

    def _process_images(self, images: list[Image.Image]) -> tuple[np.ndarray | None, np.ndarray]:
        if not images:
            return None, np.empty((0, 3), dtype=np.int32)
        processed = self.image_processor(images=images, return_tensors="np")
        return (
            np.asarray(processed["pixel_values"]),
            np.asarray(processed["image_grid_thw"], dtype=np.int32).reshape(-1, 3),
        )

    def _content(self, message: dict[str, Any], grids: np.ndarray) -> str:
        content = message["content"]
        if isinstance(content, str):
            text = content
        else:
            parts: list[str] = []
            image_index = 0
            for part in content:
                if part["type"] == "text":
                    parts.append(part["text"])
                else:
                    grid = grids[image_index]
                    image_index += 1
                    image_tokens = int(np.prod(grid, dtype=np.int64)) // self.merge_size**2
                    parts.append(
                        "<|vision_start|>" + "<|image_pad|>" * image_tokens + "<|vision_end|>"
                    )
            text = "".join(parts)
        return text

    def _encode_block(self, message: dict[str, Any], text: str) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(self.tokenizer.encode(text, add_special_tokens=False), dtype=np.int32)
        mask = (
            _assistant_loss_mask(ids)
            if _message_is_supervised(message)
            else np.zeros(len(ids), np.int32)
        )
        return ids, mask

    def _validate(self, messages: list[dict[str, Any]]) -> None:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        multimodal = self.image_processor is not None
        for message in messages:
            _validate_message(message, multimodal=multimodal)

    def _prepare_turns(
        self, messages: list[dict[str, Any]]
    ) -> tuple[np.ndarray | None, list[np.ndarray], list[str]]:
        images_by_turn = [_extract_images([message]) for message in messages]
        images = [image for turn_images in images_by_turn for image in turn_images]
        pixel_values, grids = self._process_images(images)
        grids_by_turn: list[np.ndarray] = []
        contents: list[str] = []
        grid_offset = 0
        for message, turn_images in zip(messages, images_by_turn, strict=True):
            turn_grids = grids[grid_offset : grid_offset + len(turn_images)]
            grid_offset += len(turn_images)
            grids_by_turn.append(turn_grids)
            contents.append(self._content(message, turn_grids))
        return pixel_values, grids_by_turn, contents

    def _encode_prepared(
        self,
        messages: list[dict[str, Any]],
        grids_by_turn: list[np.ndarray],
        contents: list[str],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        blocks = self._renderer(messages, contents)
        id_parts: list[np.ndarray] = []
        mask_parts: list[np.ndarray] = []
        for message, block in zip(messages, blocks, strict=True):
            if block is None:
                ids = np.empty(0, dtype=np.int32)
                mask = np.empty(0, dtype=np.int32)
            else:
                ids, mask = self._encode_block(message, block)
            id_parts.append(ids)
            mask_parts.append(mask)
        return id_parts, mask_parts

    def _measure_prepared(
        self,
        messages: list[dict[str, Any]],
        grids_by_turn: list[np.ndarray],
        contents: list[str],
    ) -> dict[str, Any]:
        id_parts, mask_parts = self._encode_prepared(messages, grids_by_turn, contents)
        message_measurements: list[dict[str, Any]] = []
        for ids, mask, grids in zip(id_parts, mask_parts, grids_by_turn, strict=True):
            vision_patches = sum(int(np.prod(grid, dtype=np.int64)) for grid in grids)
            message_measurements.append(
                {
                    "length": len(ids),
                    "supervised_tokens": int(mask.sum()),
                    "vision_tokens": vision_patches // self.merge_size**2,
                    "vision_patches": vision_patches,
                    "num_images": len(grids),
                    "image_grid_thw": grids.tolist(),
                }
            )
        return {
            "length": sum(item["length"] for item in message_measurements),
            "supervised_tokens": sum(
                item["supervised_tokens"] for item in message_measurements
            ),
            "vision_tokens": sum(item["vision_tokens"] for item in message_measurements),
            "vision_patches": sum(item["vision_patches"] for item in message_measurements),
            "num_images": sum(item["num_images"] for item in message_measurements),
            "image_grid_thw": [
                grid for item in message_measurements for grid in item["image_grid_thw"]
            ],
            "message_measurements": message_measurements,
        }

    def encode(self, messages: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        self._validate(messages)
        pixel_values, grids_by_turn, contents = self._prepare_turns(messages)
        id_parts, mask_parts = self._encode_prepared(messages, grids_by_turn, contents)

        result = {
            "input_ids": np.concatenate(id_parts),
            "loss_mask": np.concatenate(mask_parts),
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
            result["image_grid_thw"] = np.concatenate(grids_by_turn)
        return result

    def measure(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self._validate(messages)
        _, grids_by_turn, contents = self._prepare_turns(messages)
        return self._measure_prepared(messages, grids_by_turn, contents)

    def prepare(self, messages: list[dict[str, Any]]) -> _PreparedConversation:
        self._validate(messages)
        _, grids_by_turn, contents = self._prepare_turns(messages)
        return _PreparedConversation(self, messages, grids_by_turn, contents)


class _PreparedConversation:
    def __init__(
        self,
        encoder: Qwen3MessageEncoder,
        messages: list[dict[str, Any]],
        grids_by_turn: list[np.ndarray],
        contents: list[str],
    ) -> None:
        self._encoder = encoder
        self._messages = messages
        self._grids_by_turn = grids_by_turn
        self._contents = contents
        self._cache: dict[tuple[int, int], dict[str, Any]] = {}
        self.message_measurements = self(0, len(messages))["message_measurements"]

    def __call__(self, start: int, end: int) -> dict[str, Any]:
        if not 0 <= start < end <= len(self._messages):
            raise ValueError(f"invalid conversation span [{start}, {end})")
        key = (start, end)
        if key not in self._cache:
            self._cache[key] = self._encoder._measure_prepared(
                self._messages[start:end],
                self._grids_by_turn[start:end],
                self._contents[start:end],
            )
        return self._cache[key]


class _ConversationMeasureFn:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        image_processor: BaseImageProcessor | None,
        model_type: str,
    ) -> None:
        self.encoder = Qwen3MessageEncoder(tokenizer, image_processor, model_type)

    def __call__(self, messages: list[dict[str, Any]]) -> _PreparedConversation:
        return self.encoder.prepare(messages)


def make_conversation_measure_fn(
    tokenizer: PreTrainedTokenizer,
    image_processor: BaseImageProcessor | None,
    model_type: str,
) -> _ConversationMeasureFn:
    return _ConversationMeasureFn(tokenizer, image_processor, model_type)
