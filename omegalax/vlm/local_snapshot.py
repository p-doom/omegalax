"""Validated local artifact custody for vision-language model snapshots."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Self

from safetensors import SafetensorError, safe_open

_MANIFEST_NAME = "omegalax-vlm-snapshot.json"
_FORMAT = "omegalax.vlm_snapshot.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_BYTES = 8 << 20
_MAX_JSON_ASSET_BYTES = 24 << 20
_MAX_TOTAL_JSON_ASSET_BYTES = 32 << 20
_MAX_FILES = 128
_MAX_PATH_COMPONENTS = 64
_COPY_CHUNK_BYTES = 8 << 20
SNAPSHOT_IDENTITY_ASSETS = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    }
)
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


def _open_canonical_directory(
    path: str | os.PathLike[str],
    *,
    label: str,
    require_read_only: bool,
) -> tuple[Path, int, int, str]:
    raw = os.fspath(path)
    if not os.path.isabs(raw):
        raise ValueError("VLM snapshot path must be absolute")
    normalized = os.path.normpath(raw)
    if raw != normalized or normalized == "/":
        raise ValueError(f"{label} must be a canonical absolute directory")
    components = normalized.removeprefix("/").split("/")
    if len(components) > _MAX_PATH_COMPONENTS:
        raise ValueError(f"{label} exceeds {_MAX_PATH_COMPONENTS} path components")
    current_fd = os.open(
        "/",
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        for component in components[:-1]:
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=current_fd,
                )
            except OSError as error:
                raise ValueError(f"{label} contains a missing or symlinked parent") from error
            os.close(current_fd)
            current_fd = next_fd
        name = components[-1]
        try:
            directory_fd = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
        except OSError as error:
            raise ValueError(f"{label} is not an accessible no-follow directory") from error
        metadata = os.fstat(directory_fd)
        if require_read_only and metadata.st_mode & 0o222:
            os.close(directory_fd)
            raise ValueError(f"{label} directory must be read-only")
        return Path(normalized), directory_fd, current_fd, name
    except BaseException:
        os.close(current_fd)
        raise


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
    if len(files) > _MAX_FILES:
        raise ValueError(f"VLM snapshot manifest exceeds {_MAX_FILES} files")
    result: dict[str, tuple[int, str]] = {}
    for name, entry in files.items():
        if (
            type(name) is not str
            or not name
            or name in {".", "..", _MANIFEST_NAME}
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


def _validate_inventory(files: Mapping[str, object]) -> None:
    names = set(files)
    missing = _REQUIRED_ASSETS - names
    if missing:
        raise ValueError(f"VLM snapshot is missing required assets: {sorted(missing)}")
    weight_names = sorted(name for name in names if name.endswith(".safetensors"))
    if not weight_names:
        raise ValueError("VLM snapshot contains no safetensors weights")
    if any(name.endswith(".safetensors.index.json") for name in names):
        raise ValueError("VLM snapshot must not contain a safetensors index")
    unsupported = names - SNAPSHOT_IDENTITY_ASSETS - set(weight_names)
    if unsupported:
        raise ValueError(f"VLM snapshot contains unsupported assets: {sorted(unsupported)}")


class LocalVLMSnapshot:
    """An owning handle to one validated, descriptor-pinned local snapshot."""

    __slots__ = (
        "_closed",
        "_directory_fd",
        "_directory_identity",
        "_directory_name",
        "_directory_parent_fd",
        "_file_fds",
        "_file_identities",
        "_files",
        "_manifest_fd",
        "_manifest_identity",
        "_open_fds",
        "_path",
        "_sha256",
    )

    def __init__(
        self,
        token: object,
        path: Path,
        directory_fd: int,
        directory_parent_fd: int,
        directory_name: str,
        manifest_fd: int,
        manifest_sha256: str,
        file_fds: dict[str, int],
    ) -> None:
        if token is not _TOKEN:
            raise TypeError("LocalVLMSnapshot must be created by open_local_vlm_snapshot")
        self._path = path
        self._directory_fd = directory_fd
        self._directory_identity = _identity(os.fstat(directory_fd))
        self._directory_parent_fd = directory_parent_fd
        self._directory_name = directory_name
        self._manifest_fd = manifest_fd
        self._manifest_identity = _identity(os.fstat(manifest_fd))
        self._sha256 = manifest_sha256
        self._file_fds = file_fds
        self._file_identities = {name: _identity(os.fstat(fd)) for name, fd in file_fds.items()}
        self._files = MappingProxyType(
            {name: f"/proc/self/fd/{fd}" for name, fd in sorted(file_fds.items())}
        )
        self._open_fds = {
            *file_fds.values(),
            manifest_fd,
            directory_fd,
            directory_parent_fd,
        }
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def sha256(self) -> str:
        return self._sha256

    @property
    def names(self) -> tuple[str, ...]:
        self._require_open()
        return tuple(self._files)

    @property
    def identity_assets(self) -> tuple[str, ...]:
        self._require_open()
        return tuple(name for name in self._files if name in SNAPSHOT_IDENTITY_ASSETS)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("VLM snapshot handle is closed")

    def assert_unchanged(self) -> None:
        self._require_open()
        path_metadata = os.stat(
            self._directory_name,
            dir_fd=self._directory_parent_fd,
            follow_symlinks=False,
        )
        if _identity(path_metadata) != self._directory_identity:
            raise RuntimeError("VLM snapshot path changed after validation")
        if _identity(os.fstat(self._directory_fd)) != self._directory_identity:
            raise RuntimeError("VLM snapshot directory changed after validation")
        if set(os.listdir(self._directory_fd)) != set(self._file_fds) | {_MANIFEST_NAME}:
            raise RuntimeError("VLM snapshot inventory changed after validation")
        manifest_metadata = os.fstat(self._manifest_fd)
        if (
            _identity(manifest_metadata) != self._manifest_identity
            or _sha256(self._manifest_fd, manifest_metadata.st_size) != self._sha256
        ):
            raise RuntimeError("VLM snapshot manifest changed after validation")
        for name, fd in self._file_fds.items():
            metadata = os.fstat(fd)
            path_metadata = os.stat(name, dir_fd=self._directory_fd, follow_symlinks=False)
            if (
                _identity(metadata) != self._file_identities[name]
                or _identity(path_metadata) != self._file_identities[name]
            ):
                raise RuntimeError(f"VLM snapshot child changed after validation: {name!r}")

    @contextlib.contextmanager
    def files(self) -> Iterator[Mapping[str, str]]:
        self.assert_unchanged()
        try:
            yield self._files
        finally:
            self.assert_unchanged()

    def copy_identity_asset_to(self, name: str, destination_fd: int) -> None:
        if name not in SNAPSHOT_IDENTITY_ASSETS or name not in self._file_fds:
            raise ValueError(f"VLM snapshot has no allowed identity asset {name!r}")
        destination = os.fstat(destination_fd)
        if not stat.S_ISREG(destination.st_mode) or destination.st_size != 0:
            raise ValueError("VLM snapshot identity destination must be a new empty regular file")
        self.assert_unchanged()
        source_fd = self._file_fds[name]
        source = os.fstat(source_fd)
        offset = 0
        while offset < source.st_size:
            chunk = os.pread(source_fd, min(source.st_size - offset, _COPY_CHUNK_BYTES), offset)
            if not chunk:
                raise RuntimeError(f"VLM snapshot identity asset ended while copying: {name!r}")
            written = 0
            while written < len(chunk):
                count = os.pwrite(destination_fd, chunk[written:], offset + written)
                if count == 0:
                    raise OSError(f"VLM snapshot identity copy made no progress: {name!r}")
                written += count
            offset += len(chunk)
        if os.pread(source_fd, 1, source.st_size):
            raise RuntimeError(f"VLM snapshot identity asset grew while copying: {name!r}")
        if _identity(os.fstat(source_fd)) != self._file_identities[name]:
            raise RuntimeError(f"VLM snapshot identity asset changed while copying: {name!r}")
        self.assert_unchanged()

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        for fd in tuple(self._open_fds):
            try:
                os.close(fd)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
            else:
                self._open_fds.remove(fd)
        self._closed = not self._open_fds
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("VLM snapshot cleanup failed", errors)

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, _error_type, error, _traceback) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if error is None:
                raise
            error.add_note(f"VLM snapshot cleanup also failed: {cleanup_error!r}")


def _open_local_vlm_snapshot_at(
    path: Path,
    directory_fd: int,
    directory_parent_fd: int,
    directory_name: str,
) -> LocalVLMSnapshot:
    manifest_fd = -1
    file_fds: dict[str, int] = {}
    try:
        names = set(os.listdir(directory_fd))
        manifest_fd = _open_regular_at(directory_fd, _MANIFEST_NAME, require_read_only=True)
        manifest_metadata = os.fstat(manifest_fd)
        manifest_bytes = _read_bounded(
            manifest_fd,
            manifest_metadata.st_size,
            _MAX_MANIFEST_BYTES,
            "VLM snapshot manifest",
        )
        manifest_value = _parse_json(manifest_bytes, "VLM snapshot manifest")
        manifest = _validate_manifest(manifest_value)
        _validate_inventory(manifest)
        if names != set(manifest) | {_MANIFEST_NAME}:
            raise ValueError("VLM snapshot directory inventory does not match its manifest")

        json_sizes = [size for name, (size, _) in manifest.items() if name.endswith(".json")]
        if any(size > _MAX_JSON_ASSET_BYTES for size in json_sizes):
            raise ValueError(f"VLM snapshot JSON asset exceeds {_MAX_JSON_ASSET_BYTES} bytes")
        if sum(json_sizes) > _MAX_TOTAL_JSON_ASSET_BYTES:
            raise ValueError(
                f"VLM snapshot JSON assets exceed {_MAX_TOTAL_JSON_ASSET_BYTES} aggregate bytes"
            )

        for name, (expected_size, expected_digest) in manifest.items():
            fd = _open_regular_at(directory_fd, name, require_read_only=True)
            metadata = os.fstat(fd)
            if metadata.st_size != expected_size or _sha256(fd, expected_size) != expected_digest:
                os.close(fd)
                raise ValueError(f"VLM snapshot identity mismatch for {name!r}")
            file_fds[name] = fd

        for name in sorted(name for name in manifest if name.endswith(".json")):
            fd = file_fds[name]
            metadata = os.fstat(fd)
            _parse_json(
                _read_bounded(
                    fd,
                    metadata.st_size,
                    _MAX_JSON_ASSET_BYTES,
                    f"VLM snapshot {name}",
                ),
                f"VLM snapshot {name}",
            )

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

        return LocalVLMSnapshot(
            _TOKEN,
            path,
            directory_fd,
            directory_parent_fd,
            directory_name,
            manifest_fd,
            hashlib.sha256(manifest_bytes).hexdigest(),
            file_fds,
        )
    except BaseException:
        for fd in file_fds.values():
            os.close(fd)
        if manifest_fd >= 0:
            os.close(manifest_fd)
        os.close(directory_fd)
        os.close(directory_parent_fd)
        raise


def open_local_vlm_snapshot(path: str | os.PathLike[str]) -> LocalVLMSnapshot:
    resolved, directory_fd, directory_parent_fd, directory_name = _open_canonical_directory(
        path,
        label="VLM snapshot",
        require_read_only=True,
    )
    return _open_local_vlm_snapshot_at(
        resolved,
        directory_fd,
        directory_parent_fd,
        directory_name,
    )


def _write_local_vlm_snapshot_manifest_at(directory_fd: int, label: str) -> None:
    names = set(os.listdir(directory_fd))
    if _MANIFEST_NAME in names:
        raise ValueError(f"VLM snapshot manifest already exists under {label}")
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
            count = os.write(output_fd, data[offset:])
            if count == 0:
                raise OSError("VLM snapshot manifest write made no progress")
            offset += count
        os.fsync(output_fd)
    finally:
        os.close(output_fd)
    for name in names:
        os.chmod(name, 0o440, dir_fd=directory_fd, follow_symlinks=False)
    os.fchmod(directory_fd, 0o550)
    os.fsync(directory_fd)


def write_local_vlm_snapshot_manifest(path: str | os.PathLike[str]) -> None:
    resolved, directory_fd, directory_parent_fd, _ = _open_canonical_directory(
        path,
        label="VLM snapshot",
        require_read_only=False,
    )
    try:
        _write_local_vlm_snapshot_manifest_at(directory_fd, str(resolved))
    finally:
        os.close(directory_fd)
        os.close(directory_parent_fd)
