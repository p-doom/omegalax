"""Verify D7's recorded generator revision before the existing pollution audit."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


_D7_COMMIT = "9703b61ef7f32c5f8da9e117fb224d2cf4a05f8a"
_D7_CONTENT_SHA256 = "7c252369feaa946203abe5e50f3bf7344da1c16c67298b519123d2af17d4a821"
_TRAINING_FILES = (
    "scripts/train_memory_tasks.py",
    "omegalax/data/memory_tasks.py",
    "omegalax/trainers/pretrain.py",
    "omegalax/trainers/tail_pretrain.py",
)


def allow_verified_d7_training_content() -> None:
    """Accept only the archived D7 content after verifying the base generator is intact."""
    repo = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    historical_generator = None
    for relative_path in _TRAINING_FILES:
        historical = subprocess.run(
            ["git", "show", f"{_D7_COMMIT}:{relative_path}"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        digest.update(relative_path.encode() + b"\0")
        digest.update(historical)
        if relative_path == "omegalax/data/memory_tasks.py":
            historical_generator = historical
    if digest.hexdigest() != _D7_CONTENT_SHA256:
        raise ValueError("Archived D7 training source does not match its recorded content hash")
    current = (repo / "omegalax/data/memory_tasks.py").read_bytes()
    insertion = b"\ndef build_memory_interference_batch("
    next_definition = b"\n@dataclasses.dataclass(frozen=True, slots=True)\nclass _PairIndex:"
    if current.count(insertion) != 1 or current.count(next_definition) != 1:
        raise ValueError(
            "Cannot isolate the new interference generator from the original D7 generator"
        )
    before, added_and_after = current.split(insertion, 1)
    _, after = added_and_after.split(next_definition, 1)
    if before + next_definition + after != historical_generator:
        raise ValueError("The original D7 generator differs from the recorded training revision")

    from omegalax.evals import memory_pollution

    memory_pollution._COMPATIBLE_V2_CONTENT_SHA256 = (
        memory_pollution._COMPATIBLE_V2_CONTENT_SHA256 | frozenset({_D7_CONTENT_SHA256})
    )
