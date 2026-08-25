"""Validated local artifact custody for vision-language model snapshots."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Self

from safetensors import SafetensorError, safe_open

_MANIFEST_NAME = "omegalax-vlm-snapshot.json"
_FORMAT = "omegalax.vlm_snapshot.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_BYTES = 8 << 20
_MAX_JSON_ASSET_BYTES = 256 << 20
_COPY_CHUNK_BYTES = 8 << 20
_REQUIRED_ASSETS = frozenset(
    {
        "chat_template.json",
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
_TOKEN = object()


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _canonical_absolute_directory(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("VLM snapshot path must be absolute")
    lexical = Path(os.path.normpath(os.fspath(candidate)))
    try:
        resolved = Path(os.path.realpath(candidate, strict=True))
    except OSError as error:
        raise ValueError(f"VLM snapshot does not exist: {candidate}") from error
    if lexical != resolved:
        raise ValueError(f"VLM snapshot path must not traverse symlinks: {candidate}")
    return resolved


def _open_directory(path: Path) -> int:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError(f"VLM snapshot is not an accessible directory: {path}") from error
    metadata = os.fstat(fd)
    if metadata.st_mode & 0o222:
        os.close(fd)
        raise ValueError(f"VLM snapshot directory must be read-only: {path}")
    return fd


def _open_regular_at(directory_fd: int, name: str, *, require_read_only: bool) -> int:
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        raise ValueError(f"Invalid VLM snapshot child name: {name!r}")
    try:
        fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(
            f"VLM snapshot child is not an accessible regular file: {name!r}"
        ) from error
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(fd)
        raise ValueError(f"VLM snapshot child is not a regular file: {name!r}")
    if require_read_only and metadata.st_mode & 0o222:
        os.close(fd)
        raise ValueError(f"VLM snapshot child must be read-only: {name!r}")
    return fd


def _read_bounded(fd: int, size: int, limit: int, label: str) -> bytes:
    if size > limit:
        raise ValueError(f"{label} exceeds {limit} bytes")
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(size - offset, _COPY_CHUNK_BYTES), offset)
        if not chunk:
            raise ValueError(f"{label} ended while reading")
        chunks.append(chunk)
        offset += len(chunk)
    if os.pread(fd, 1, size):
        raise ValueError(f"{label} grew while reading")
    return b"".join(chunks)


def _parse_json(data: bytes, label: str) -> dict:
    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON key {key!r} in {label}")
            value[key] = item
        return value

    def reject_constant(token):
        raise ValueError(f"Non-finite JSON value {token!r} in {label}")

    try:
        value = json.loads(
            data,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON in {label}: {error}") from error
    if type(value) is not dict:
        raise ValueError(f"Expected a JSON object in {label}")
    return value


def _sha256(fd: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(size - offset, _COPY_CHUNK_BYTES), offset)
        if not chunk:
            raise ValueError("VLM snapshot child ended while hashing")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(fd, 1, size):
        raise ValueError("VLM snapshot child grew while hashing")
    return digest.hexdigest()


def _validate_manifest(value: dict) -> dict[str, tuple[int, str]]:
    if set(value) != {"format", "files"} or value["format"] != _FORMAT:
        raise ValueError("VLM snapshot manifest has an invalid schema")
    files = value["files"]
    if type(files) is not dict or not files:
        raise ValueError("VLM snapshot manifest files must be a non-empty object")
    result: dict[str, tuple[int, str]] = {}
    for name, entry in files.items():
        if (
            type(name) is not str
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\x00" in name
        ):
            raise ValueError(f"Invalid VLM snapshot manifest child name: {name!r}")
        if type(entry) is not dict or set(entry) != {"sha256", "size_bytes"}:
            raise ValueError(f"Invalid VLM snapshot manifest entry for {name!r}")
        size = entry["size_bytes"]
        digest = entry["sha256"]
        if type(size) is not int or size < 0:
            raise ValueError(f"Invalid VLM snapshot size for {name!r}")
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"Invalid VLM snapshot digest for {name!r}")
        result[name] = (size, digest)
    return result


def _validate_inventory(files: dict[str, tuple[int, str]]) -> None:
    names = set(files)
    missing = _REQUIRED_ASSETS - names
    if missing:
        raise ValueError(f"VLM snapshot is missing required assets: {sorted(missing)}")
    weight_names = sorted(name for name in names if name.endswith(".safetensors"))
    if not weight_names:
        raise ValueError("VLM snapshot contains no safetensors weights")
    if any(name.endswith(".safetensors.index.json") for name in names):
        raise ValueError("VLM snapshot must not contain a safetensors index")


class LocalVLMSnapshot:
    """An owning handle to one validated, descriptor-pinned local snapshot."""

    __slots__ = (
        "_closed",
        "_consumer_directory",
        "_directory_fd",
        "_directory_identity",
        "_file_fds",
        "_file_identities",
        "_manifest",
        "_path",
    )

    def __init__(
        self,
        token: object,
        path: Path,
        directory_fd: int,
        file_fds: dict[str, int],
        manifest: dict[str, tuple[int, str]],
    ) -> None:
        if token is not _TOKEN:
            raise TypeError("LocalVLMSnapshot must be created by open_local_vlm_snapshot")
        self._path = path
        self._directory_fd = directory_fd
        self._directory_identity = _identity(os.fstat(directory_fd))
        self._file_fds = file_fds
        self._file_identities = {name: _identity(os.fstat(fd)) for name, fd in file_fds.items()}
        self._manifest = manifest
        self._consumer_directory = tempfile.TemporaryDirectory(prefix="omegalax-vlm-snapshot-")
        self._closed = False
        alias = Path(self._consumer_directory.name)
        for name, fd in file_fds.items():
            os.symlink(f"/proc/self/fd/{fd}", alias / name)
        os.chmod(alias, 0o500)

    @property
    def path(self) -> Path:
        return self._path

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("VLM snapshot handle is closed")

    def assert_unchanged(self) -> None:
        self._require_open()
        if _identity(os.fstat(self._directory_fd)) != self._directory_identity:
            raise RuntimeError("VLM snapshot directory changed after validation")
        if set(os.listdir(self._directory_fd)) != set(self._file_fds) | {_MANIFEST_NAME}:
            raise RuntimeError("VLM snapshot inventory changed after validation")
        for name, fd in self._file_fds.items():
            metadata = os.fstat(fd)
            if _identity(metadata) != self._file_identities[name]:
                raise RuntimeError(f"VLM snapshot child changed after validation: {name!r}")
            expected_size, expected_digest = self._manifest[name]
            if metadata.st_size != expected_size or _sha256(fd, expected_size) != expected_digest:
                raise RuntimeError(f"VLM snapshot child content changed after validation: {name!r}")

    @contextlib.contextmanager
    def consume(self) -> Iterator[str]:
        self.assert_unchanged()
        try:
            yield self._consumer_directory.name
        finally:
            self.assert_unchanged()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.chmod(self._consumer_directory.name, 0o700)
        self._consumer_directory.cleanup()
        for fd in self._file_fds.values():
            os.close(fd)
        os.close(self._directory_fd)

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def open_local_vlm_snapshot(path: str | os.PathLike[str]) -> LocalVLMSnapshot:
    resolved = _canonical_absolute_directory(path)
    directory_fd = _open_directory(resolved)
    file_fds: dict[str, int] = {}
    try:
        names = set(os.listdir(directory_fd))
        manifest_fd = _open_regular_at(directory_fd, _MANIFEST_NAME, require_read_only=True)
        try:
            manifest_metadata = os.fstat(manifest_fd)
            manifest_value = _parse_json(
                _read_bounded(
                    manifest_fd,
                    manifest_metadata.st_size,
                    _MAX_MANIFEST_BYTES,
                    "VLM snapshot manifest",
                ),
                "VLM snapshot manifest",
            )
        finally:
            os.close(manifest_fd)
        manifest = _validate_manifest(manifest_value)
        _validate_inventory(manifest)
        if names != set(manifest) | {_MANIFEST_NAME}:
            raise ValueError("VLM snapshot directory inventory does not match its manifest")

        for name, (expected_size, expected_digest) in manifest.items():
            fd = _open_regular_at(directory_fd, name, require_read_only=True)
            metadata = os.fstat(fd)
            if metadata.st_size != expected_size or _sha256(fd, expected_size) != expected_digest:
                os.close(fd)
                raise ValueError(f"VLM snapshot identity mismatch for {name!r}")
            file_fds[name] = fd

        config_fd = file_fds["config.json"]
        config_metadata = os.fstat(config_fd)
        config = _parse_json(
            _read_bounded(
                config_fd,
                config_metadata.st_size,
                _MAX_JSON_ASSET_BYTES,
                "VLM snapshot config.json",
            ),
            "VLM snapshot config.json",
        )
        if config.get("model_type") not in {"qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe"}:
            raise ValueError(f"Unsupported VLM snapshot model_type: {config.get('model_type')!r}")

        tensor_names: set[str] = set()
        for name in sorted(name for name in manifest if name.endswith(".safetensors")):
            try:
                with safe_open(f"/proc/self/fd/{file_fds[name]}", framework="numpy") as weights:
                    current_names = set(weights.keys())
            except SafetensorError as error:
                raise ValueError(f"Invalid safetensors file {name!r}: {error}") from error
            if not current_names:
                raise ValueError(f"Safetensors file contains no tensors: {name!r}")
            duplicates = tensor_names & current_names
            if duplicates:
                raise ValueError(
                    f"Safetensors tensor names occur in multiple files: {sorted(duplicates)[:3]}"
                )
            tensor_names.update(current_names)

        return LocalVLMSnapshot(_TOKEN, resolved, directory_fd, file_fds, manifest)
    except BaseException:
        for fd in file_fds.values():
            os.close(fd)
        os.close(directory_fd)
        raise


def write_local_vlm_snapshot_manifest(path: str | os.PathLike[str]) -> None:
    resolved = _canonical_absolute_directory(path)
    directory_fd = os.open(
        resolved,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        names = set(os.listdir(directory_fd))
        if _MANIFEST_NAME in names:
            raise ValueError(f"VLM snapshot manifest already exists under {resolved}")
        files: dict[str, dict[str, int | str]] = {}
        for name in sorted(names):
            fd = _open_regular_at(directory_fd, name, require_read_only=False)
            try:
                metadata = os.fstat(fd)
                files[name] = {
                    "sha256": _sha256(fd, metadata.st_size),
                    "size_bytes": metadata.st_size,
                }
            finally:
                os.close(fd)
        manifest = {"format": _FORMAT, "files": files}
        _validate_inventory(_validate_manifest(manifest))
        data = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        output_fd = os.open(
            _MANIFEST_NAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o440,
            dir_fd=directory_fd,
        )
        try:
            offset = 0
            while offset < len(data):
                offset += os.write(output_fd, data[offset:])
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
        for name in names:
            os.chmod(name, 0o440, dir_fd=directory_fd, follow_symlinks=False)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.chmod(resolved, 0o550)
