"""Create a manifest-bound local VLM snapshot from regular local files."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from absl import app, flags

from omegalax.vlm.local_snapshot import (
    SNAPSHOT_IDENTITY_ASSETS,
    _open_canonical_directory,
    _open_local_vlm_snapshot_at,
    _write_local_vlm_snapshot_manifest_at,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("source_dir", None, "Absolute local unsealed Hugging Face directory.")
flags.DEFINE_string("out_dir", None, "Absolute destination for the sealed snapshot.")

_CHUNK_BYTES = 8 << 20
_MANIFEST_NAME = "omegalax-vlm-snapshot.json"


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _copy_regular(source_fd: int, destination_fd: int, name: str) -> None:
    try:
        input_fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=source_fd)
    except OSError as error:
        raise ValueError(
            f"Source snapshot child is not an accessible regular file: {name!r}"
        ) from error
    try:
        before = os.fstat(input_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Source snapshot child is not a regular file: {name!r}")
        output_fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=destination_fd,
        )
        try:
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(input_fd, min(before.st_size - offset, _CHUNK_BYTES), offset)
                if not chunk:
                    raise ValueError(f"Source snapshot child ended while copying: {name!r}")
                written = 0
                while written < len(chunk):
                    count = os.write(output_fd, chunk[written:])
                    if count == 0:
                        raise OSError(f"Snapshot copy made no progress: {name!r}")
                    written += count
                offset += len(chunk)
            if os.pread(input_fd, 1, before.st_size):
                raise ValueError(f"Source snapshot child grew while copying: {name!r}")
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
        if _file_identity(os.fstat(input_fd)) != _file_identity(before):
            raise ValueError(f"Source snapshot child changed while copying: {name!r}")
    finally:
        os.close(input_fd)


def _remove_owned_output(
    parent_fd: int,
    destination_fd: int,
    name: str,
    identity: tuple[int, int],
) -> None:
    metadata = os.fstat(destination_fd)
    if (metadata.st_dev, metadata.st_ino) != identity:
        raise RuntimeError("Destination snapshot changed during failed sealing")
    os.fchmod(destination_fd, 0o700)
    for child in os.listdir(destination_fd):
        metadata = os.stat(child, dir_fd=destination_fd, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("Destination snapshot gained a non-regular child")
        os.unlink(child, dir_fd=destination_fd)
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (current.st_dev, current.st_ino) != identity:
        raise RuntimeError("Destination snapshot path changed during failed sealing")
    os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def seal_snapshot(source_dir: str, out_dir: str) -> Path:
    _, source_fd, source_parent_fd, _ = _open_canonical_directory(
        source_dir,
        label="--source_dir",
        require_read_only=False,
    )
    destination = Path(out_dir)
    if not destination.is_absolute():
        os.close(source_fd)
        os.close(source_parent_fd)
        raise ValueError("--out_dir must be absolute")
    raw_destination = os.fspath(destination)
    normalized_destination = Path(os.path.normpath(raw_destination))
    if raw_destination != os.fspath(normalized_destination):
        os.close(source_fd)
        os.close(source_parent_fd)
        raise ValueError("--out_dir must be canonical")
    destination = normalized_destination
    try:
        parent, parent_fd, parent_parent_fd, _ = _open_canonical_directory(
            destination.parent,
            label="--out_dir parent",
            require_read_only=False,
        )
    except BaseException:
        os.close(source_fd)
        os.close(source_parent_fd)
        raise
    destination = parent / destination.name

    complete = False
    destination_fd = -1
    destination_identity = None
    try:
        try:
            os.mkdir(destination.name, 0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise ValueError(f"--out_dir already exists: {destination}") from error
        destination_fd = os.open(
            destination.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        destination_metadata = os.fstat(destination_fd)
        destination_identity = (destination_metadata.st_dev, destination_metadata.st_ino)
        source_before = os.fstat(source_fd)
        names = sorted(os.listdir(source_fd))
        if _MANIFEST_NAME in names:
            raise ValueError("Source snapshot is already sealed")
        selected_names = [
            name
            for name in names
            if name in SNAPSHOT_IDENTITY_ASSETS or name.endswith(".safetensors")
        ]
        for name in selected_names:
            _copy_regular(source_fd, destination_fd, name)
        source_after = os.fstat(source_fd)
        if (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_mtime_ns,
            source_before.st_ctime_ns,
            names,
        ) != (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_mtime_ns,
            source_after.st_ctime_ns,
            sorted(os.listdir(source_fd)),
        ):
            raise ValueError("Source snapshot directory changed while copying")
        _write_local_vlm_snapshot_manifest_at(destination_fd, str(destination))
        with _open_local_vlm_snapshot_at(
            destination,
            os.dup(destination_fd),
            os.dup(parent_fd),
            destination.name,
        ) as snapshot:
            snapshot.assert_unchanged()
        current = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != destination_identity:
            raise RuntimeError("Destination snapshot path changed during sealing")
        if set(os.listdir(destination_fd)) != set(selected_names) | {_MANIFEST_NAME}:
            raise RuntimeError("Destination snapshot inventory changed during sealing")
        complete = True
        return destination
    finally:
        try:
            os.close(source_fd)
            os.close(source_parent_fd)
            if not complete and destination_fd >= 0 and destination_identity is not None:
                _remove_owned_output(
                    parent_fd,
                    destination_fd,
                    destination.name,
                    destination_identity,
                )
        finally:
            if destination_fd >= 0:
                os.close(destination_fd)
            os.close(parent_fd)
            os.close(parent_parent_fd)


def main(_) -> None:
    if FLAGS.source_dir is None or FLAGS.out_dir is None:
        raise ValueError("--source_dir and --out_dir are required")
    print(seal_snapshot(FLAGS.source_dir, FLAGS.out_dir))


if __name__ == "__main__":
    app.run(main)
