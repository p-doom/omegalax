"""Read the model-snapshot identity committed with a VLM checkpoint."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from omegalax.vlm.local_snapshot import (
    _identity,
    _open_canonical_directory,
    _parse_json,
    _read_bounded,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_SCHEMA_BYTES = 8 << 20


def _open_directory_at(parent_fd: int, name: str, label: str) -> int:
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        raise ValueError(f"Invalid {label} name")
    try:
        return os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except OSError as error:
        raise ValueError(f"Missing or symlinked {label}: {name!r}") from error


def _read_schema(save_dir: str | os.PathLike[str], step: int) -> dict:
    if type(step) is not int or step <= 0:
        raise ValueError("VLM checkpoint step must be a positive integer")
    _, root_fd, parent_fd, _ = _open_canonical_directory(
        save_dir,
        label="VLM checkpoint root",
        require_read_only=False,
    )
    step_fd = -1
    schema_fd = -1
    metadata_fd = -1
    try:
        step_fd = _open_directory_at(root_fd, f"{step:06d}", "VLM checkpoint step")
        schema_fd = _open_directory_at(step_fd, "schema", "VLM checkpoint schema directory")
        try:
            metadata_fd = os.open(
                "metadata",
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=schema_fd,
            )
        except OSError as error:
            raise ValueError("Missing or symlinked VLM checkpoint schema metadata") from error
        before = os.fstat(metadata_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("VLM checkpoint schema metadata is not a regular file")
        data = _read_bounded(
            metadata_fd,
            before.st_size,
            _MAX_SCHEMA_BYTES,
            "VLM checkpoint schema metadata",
        )
        value = _parse_json(data, "VLM checkpoint schema metadata")
        if _identity(os.fstat(metadata_fd)) != _identity(before):
            raise ValueError("VLM checkpoint schema metadata changed while reading")
        return value
    finally:
        for fd in (metadata_fd, schema_fd, step_fd, root_fd, parent_fd):
            if fd >= 0:
                os.close(fd)


def require_checkpoint_snapshot(
    save_dir: str | os.PathLike[str],
    step: int,
    expected_sha256: str,
) -> None:
    """Require an exact sealed-model identity before JAX startup or restore."""
    if type(expected_sha256) is not str or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("Expected VLM snapshot SHA-256 must be lowercase hexadecimal")
    schema = _read_schema(save_dir, step)
    actual = schema.get("model_snapshot_sha256")
    if schema.get("version") != 3 or type(actual) is not str or _SHA256.fullmatch(actual) is None:
        raise ValueError("VLM checkpoint has no valid sealed-model identity")
    if actual != expected_sha256:
        raise ValueError(f"VLM checkpoint model snapshot {actual} does not match {expected_sha256}")


def require_checkpoint_path_snapshot(
    checkpoint_path: str | os.PathLike[str],
    expected_sha256: str,
) -> None:
    raw = os.fspath(checkpoint_path)
    if not os.path.isabs(raw) or raw != os.path.normpath(raw):
        raise ValueError("VLM checkpoint path must be canonical and absolute")
    path = Path(raw)
    if not path.name.isdigit():
        raise ValueError("VLM checkpoint path must end in a numeric step")
    step = int(path.name)
    if path.name != f"{step:06d}":
        raise ValueError("VLM checkpoint path step is not canonical")
    require_checkpoint_snapshot(path.parent, step, expected_sha256)
