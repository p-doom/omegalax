"""Copy a Hugging Face directory into a manifest-bound local VLM snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from absl import app, flags

from omegalax.vlm.local_snapshot import (
    IDENTITY_ASSETS,
    MANIFEST_NAME,
    REQUIRED_IDENTITY_ASSETS,
    SNAPSHOT_FORMAT,
    open_local_vlm_snapshot,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("source_dir", None, "Absolute local Hugging Face source directory.")
flags.DEFINE_string("out_dir", None, "Absolute destination snapshot directory.")


def _file_entry(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def seal_vlm_snapshot(source_dir: str | Path, out_dir: str | Path) -> Path:
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    if not source_dir.is_absolute() or not out_dir.is_absolute():
        raise ValueError("Snapshot source and destination must be absolute")
    if not source_dir.is_dir():
        raise ValueError(f"Snapshot source is not a directory: {source_dir}")
    if out_dir.exists():
        raise ValueError(f"Snapshot destination already exists: {out_dir}")

    source_names = {path.name for path in source_dir.iterdir()}
    missing = REQUIRED_IDENTITY_ASSETS - source_names
    if missing:
        raise ValueError(f"Snapshot source is missing identity assets: {sorted(missing)}")
    selected = sorted(
        name for name in source_names if name in IDENTITY_ASSETS or name.endswith(".safetensors")
    )
    if not any(name.endswith(".safetensors") for name in selected):
        raise ValueError("Snapshot source contains no safetensors weights")

    out_dir.mkdir(mode=0o700)
    for name in selected:
        source = source_dir / name
        if not source.is_file():
            raise ValueError(f"Snapshot source child is not a file: {source}")
        destination = out_dir / name
        with source.open("rb") as input_file, destination.open("xb") as output_file:
            shutil.copyfileobj(input_file, output_file, length=8 << 20)
            output_file.flush()
            os.fsync(output_file.fileno())
        destination.chmod(0o444)

    manifest = {
        "format": SNAPSHOT_FORMAT,
        "files": {name: _file_entry(out_dir / name) for name in selected},
    }
    manifest_path = out_dir / MANIFEST_NAME
    with manifest_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())
    manifest_path.chmod(0o444)
    directory_fd = os.open(out_dir, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    out_dir.chmod(0o555)
    with open_local_vlm_snapshot(out_dir):
        pass
    return out_dir


def main(_) -> None:
    if FLAGS.source_dir is None or FLAGS.out_dir is None:
        raise ValueError("--source_dir and --out_dir are required")
    print(seal_vlm_snapshot(FLAGS.source_dir, FLAGS.out_dir))


if __name__ == "__main__":
    app.run(main)
