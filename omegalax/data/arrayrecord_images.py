"""ArrayRecord image-reference resolution."""

from __future__ import annotations

import atexit
import io
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from PIL import Image

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

    from array_record.python.array_record_module import ArrayRecordReader

    reader = ArrayRecordReader(path)
    if _ARRAYRECORD_IMAGE_CACHE_SIZE <= 0:
        return reader

    while len(_ARRAYRECORD_IMAGE_SOURCES) >= _ARRAYRECORD_IMAGE_CACHE_SIZE:
        _, old_reader = _ARRAYRECORD_IMAGE_SOURCES.popitem(last=False)
        _close_arrayrecord_reader(old_reader)
    _ARRAYRECORD_IMAGE_SOURCES[path] = reader
    return reader


def is_arrayrecord_image_uri(value: object) -> bool:
    return isinstance(value, str) and value.startswith(f"{_ARRAYRECORD_IMAGE_URI_SCHEME}://")


def parse_arrayrecord_image_uri(uri: str) -> tuple[Path, int]:
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


def open_arrayrecord_image(uri: str) -> Image.Image:
    shard_path, record_index = parse_arrayrecord_image_uri(uri)
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


def extract_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") not in ("image", "image_url"):
                continue
            if block.get("type") == "image":
                if set(block) != {"type", "image"}:
                    raise ValueError("image parts must contain exactly type and image")
                ref = block["image"]
            else:
                if set(block) != {"type", "image_url"}:
                    raise ValueError("image_url parts must contain exactly type and image_url")
                ref = block["image_url"]
                if isinstance(ref, dict) and set(ref) == {"url"}:
                    ref = ref["url"]
            images.append(ref if isinstance(ref, Image.Image) else _open_image_ref(ref))
    return images


def _open_image_ref(ref: Any) -> Image.Image:
    if isinstance(ref, Image.Image):
        return ref
    if is_arrayrecord_image_uri(ref):
        return open_arrayrecord_image(ref)
    if not isinstance(ref, (str, os.PathLike)):
        raise TypeError("image reference must be a PIL image, local path, or ar:// URI")
    if isinstance(ref, str) and "://" in ref:
        raise ValueError("remote image references are not supported")
    return Image.open(ref)
