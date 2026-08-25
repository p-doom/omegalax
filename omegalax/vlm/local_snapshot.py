"""Manifest-bound local VLM snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Self

MANIFEST_NAME = "omegalax-vlm-snapshot.json"
SNAPSHOT_FORMAT = "omegalax.vlm_snapshot.v1"
IDENTITY_ASSETS = frozenset(
    {
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    }
)
REQUIRED_IDENTITY_ASSETS = frozenset(
    {
        "chat_template.json",
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
_MAX_MANIFEST_BYTES = 1 << 20
_CHUNK_BYTES = 8 << 20


def _sha256_fd(fd: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(_CHUNK_BYTES, size - offset), offset)
        if not chunk:
            raise ValueError("Snapshot file ended while hashing")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(fd, 1, size):
        raise ValueError("Snapshot file grew while hashing")
    return digest.hexdigest()


def _read_fd(fd: int, size: int, limit: int, label: str) -> bytes:
    if size > limit:
        raise ValueError(f"{label} exceeds {limit} bytes")
    chunks = []
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(_CHUNK_BYTES, size - offset), offset)
        if not chunk:
            raise ValueError(f"{label} ended while reading")
        chunks.append(chunk)
        offset += len(chunk)
    if os.pread(fd, 1, size):
        raise ValueError(f"{label} grew while reading")
    return b"".join(chunks)


def _open_regular(directory_fd: int, name: str) -> int:
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        raise ValueError(f"Invalid snapshot file name: {name!r}")
    try:
        fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(f"Snapshot child is not an accessible regular file: {name!r}") from error
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f"Snapshot child is not a regular file: {name!r}")
    return fd


def _parse_manifest(payload: bytes) -> dict[str, tuple[int, str]]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate snapshot manifest key: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid snapshot manifest JSON") from error
    if type(value) is not dict or set(value) != {"format", "files"}:
        raise ValueError("Snapshot manifest fields are invalid")
    if value["format"] != SNAPSHOT_FORMAT or type(value["files"]) is not dict:
        raise ValueError("Snapshot manifest format is invalid")

    files = {}
    for name, entry in value["files"].items():
        if (
            type(name) is not str
            or not name
            or name in {".", "..", MANIFEST_NAME}
            or "/" in name
            or "\x00" in name
        ):
            raise ValueError(f"Invalid snapshot manifest file: {name!r}")
        if type(entry) is not dict or set(entry) != {"size_bytes", "sha256"}:
            raise ValueError(f"Invalid snapshot manifest entry: {name!r}")
        size = entry["size_bytes"]
        digest = entry["sha256"]
        if type(size) is not int or size < 0:
            raise ValueError(f"Invalid snapshot size: {name!r}")
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Invalid snapshot digest: {name!r}")
        files[name] = (size, digest)
    if not files:
        raise ValueError("Snapshot manifest contains no files")
    missing = REQUIRED_IDENTITY_ASSETS - set(files)
    if missing:
        raise ValueError(f"Snapshot is missing required identity assets: {sorted(missing)}")
    weights = {name for name in files if name.endswith(".safetensors")}
    if not weights:
        raise ValueError("Snapshot contains no safetensors weights")
    unsupported = set(files) - IDENTITY_ASSETS - weights
    if unsupported:
        raise ValueError(f"Snapshot contains unsupported files: {sorted(unsupported)}")
    return files


class LocalVLMSnapshot:
    """Owns verified descriptors for one local snapshot."""

    def __init__(
        self,
        path: Path,
        directory_fd: int,
        manifest_fd: int,
        file_fds: dict[str, int],
        sha256: str,
    ) -> None:
        self.path = path
        self.sha256 = sha256
        self._directory_fd = directory_fd
        self._directory_identity = os.fstat(directory_fd)
        self._manifest_fd = manifest_fd
        self._file_fds = file_fds
        self._files = MappingProxyType(
            {name: f"/proc/self/fd/{fd}" for name, fd in sorted(file_fds.items())}
        )
        self._closed = False

    @property
    def names(self) -> tuple[str, ...]:
        self._require_open()
        return tuple(self._files)

    @property
    def identity_assets(self) -> tuple[str, ...]:
        return tuple(name for name in self.names if name in IDENTITY_ASSETS)

    def files(self) -> Mapping[str, str]:
        self._require_open()
        current = self.path.lstat()
        if not stat.S_ISDIR(current.st_mode):
            raise RuntimeError("Snapshot path changed after validation")
        if (current.st_dev, current.st_ino) != (
            self._directory_identity.st_dev,
            self._directory_identity.st_ino,
        ):
            raise RuntimeError("Snapshot path changed after validation")
        return self._files

    def copy_identity_assets(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        for name in self.identity_assets:
            destination = out_dir / name
            with destination.open("xb") as output:
                source_fd = self._file_fds[name]
                size = os.fstat(source_fd).st_size
                offset = 0
                while offset < size:
                    chunk = os.pread(source_fd, min(_CHUNK_BYTES, size - offset), offset)
                    if not chunk:
                        raise RuntimeError(f"Snapshot asset ended while copying: {name!r}")
                    output.write(chunk)
                    offset += len(chunk)
                output.flush()
                os.fsync(output.fileno())

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Snapshot is closed")

    def close(self) -> None:
        if self._closed:
            return
        errors = []
        for fd in (*self._file_fds.values(), self._manifest_fd, self._directory_fd):
            try:
                os.close(fd)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
        self._closed = True
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Snapshot cleanup failed", errors)

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, _error_type, error, _traceback) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if error is None:
                raise
            error.add_note(f"Snapshot cleanup also failed: {cleanup_error!r}")


def open_local_vlm_snapshot(path: str | Path) -> LocalVLMSnapshot:
    path = Path(path)
    if not path.is_absolute() or path == Path("/") or path != Path(os.path.normpath(path)):
        raise ValueError("Snapshot path must be a canonical absolute directory")
    try:
        directory_fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError(f"Snapshot is not an accessible no-follow directory: {path}") from error
    manifest_fd = -1
    file_fds = {}
    try:
        manifest_fd = _open_regular(directory_fd, MANIFEST_NAME)
        metadata = os.fstat(manifest_fd)
        manifest_payload = _read_fd(
            manifest_fd,
            metadata.st_size,
            _MAX_MANIFEST_BYTES,
            "Snapshot manifest",
        )
        manifest = _parse_manifest(manifest_payload)
        if set(os.listdir(directory_fd)) != set(manifest) | {MANIFEST_NAME}:
            raise ValueError("Snapshot directory does not match its manifest")
        for name, (expected_size, expected_digest) in manifest.items():
            fd = _open_regular(directory_fd, name)
            file_fds[name] = fd
            size = os.fstat(fd).st_size
            if size != expected_size or _sha256_fd(fd, size) != expected_digest:
                raise ValueError(f"Snapshot file does not match its manifest: {name!r}")
        return LocalVLMSnapshot(
            path,
            directory_fd,
            manifest_fd,
            file_fds,
            hashlib.sha256(manifest_payload).hexdigest(),
        )
    except BaseException:
        for fd in file_fds.values():
            os.close(fd)
        if manifest_fd >= 0:
            os.close(manifest_fd)
        os.close(directory_fd)
        raise
