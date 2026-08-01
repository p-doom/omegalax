"""Deterministic IID-reference entropy token selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TOP_ENTROPY_FRACTION = 0.20
TOP_ENTROPY_SELECTION = "top_entropy_20pct"


@dataclass(frozen=True)
class EntropySelectorIdentity:
    reference_checkpoint_root: str
    reference_checkpoint_step: int
    reference_config_hash: str
    manifest_hash: str
    dataset_hash: str
    code_hash: str
    pad_id: int
    fraction: float
    state_mode: str
    position_mode: str
    tp_size: int
    fsdp_size: int
    dp_size: int
    batch_size: int
    deltanet_kernel: str
    attention_backend: str


@dataclass(frozen=True)
class EntropySelectionRow:
    chunk_position: int
    eligible_count: int
    selected_positions: tuple[int, ...]


_SELECTOR_SCHEMA = pa.schema(
    (
        pa.field("chunk_position", pa.int64(), nullable=False),
        pa.field("eligible_count", pa.int64(), nullable=False),
        pa.field("selected_positions", pa.list_(pa.int32()), nullable=False),
    )
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _identity_hash(identity: EntropySelectorIdentity) -> str:
    return hashlib.sha256(_canonical_json(asdict(identity))).hexdigest()


def _atomic_write(path: Path, write) -> None:
    descriptor, raw_temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temp_path = Path(raw_temp_path)
    try:
        write(temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def top_entropy_positions(
    entropies_T1: np.ndarray,
    loss_mask_T: np.ndarray,
    *,
    fraction: float = TOP_ENTROPY_FRACTION,
) -> tuple[int, ...]:
    """Select target positions, breaking equal-entropy ties by lower position."""
    entropies_T1 = np.asarray(entropies_T1, dtype=np.float64)
    loss_mask_T = np.asarray(loss_mask_T)
    if entropies_T1.shape != (loss_mask_T.shape[0] - 1,):
        raise ValueError(
            f"Entropy/mask shape mismatch: entropies={entropies_T1.shape}, "
            f"loss_mask={loss_mask_T.shape}"
        )
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    target_positions = np.flatnonzero(loss_mask_T[1:] != 0).astype(np.int64) + 1
    if target_positions.size == 0:
        return ()
    values = entropies_T1[target_positions - 1]
    if not np.all(np.isfinite(values)):
        raise ValueError("Eligible token entropies must be finite")
    selected_count = max(1, math.ceil(fraction * target_positions.size))
    order = np.lexsort((target_positions, -values))
    return tuple(int(position) for position in target_positions[order[:selected_count]])


@dataclass
class EntropySelectorStore:
    run_dir: Path
    identity: EntropySelectorIdentity

    @classmethod
    def open(
        cls,
        reference_checkpoint_root: str | Path,
        identity: EntropySelectorIdentity,
    ) -> EntropySelectorStore:
        reference_checkpoint_root = Path(reference_checkpoint_root).expanduser().resolve()
        if identity.reference_checkpoint_root != str(reference_checkpoint_root):
            raise ValueError("Selector identity does not match the reference checkpoint root")
        run_dir = (
            reference_checkpoint_root
            / "evals"
            / "entropy_selector_v1"
            / f"step_{identity.reference_checkpoint_step}"
            / _identity_hash(identity)
        )
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "selector_config.json"
        payload = {"identity": asdict(identity)}
        if config_path.is_file():
            if _canonical_json(json.loads(config_path.read_text())) != _canonical_json(payload):
                raise ValueError(f"Entropy selector identity conflict: {run_dir}")
        else:
            _atomic_write(
                config_path,
                lambda path: path.write_bytes(
                    json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
                ),
            )
        return cls(run_dir=run_dir, identity=identity)

    def _shard_path(self, shard_id: int) -> Path:
        return self.run_dir / "raw" / f"shard_{int(shard_id)}.parquet"

    def shard_is_complete(self, shard_id: int) -> bool:
        return self._shard_path(shard_id).is_file()

    def read_shard(self, shard_id: int) -> tuple[EntropySelectionRow, ...]:
        path = self._shard_path(shard_id)
        if not path.is_file():
            return ()
        return tuple(
            EntropySelectionRow(
                chunk_position=int(row["chunk_position"]),
                eligible_count=int(row["eligible_count"]),
                selected_positions=tuple(int(value) for value in row["selected_positions"]),
            )
            for row in pq.read_table(path).to_pylist()
        )

    def write_shard(
        self,
        shard_id: int,
        rows: Iterable[EntropySelectionRow],
    ) -> Path:
        path = self._shard_path(shard_id)
        rows = tuple(rows)
        if path.is_file():
            if self.read_shard(shard_id) == rows:
                return path
            raise ValueError(f"Selector shard already exists with conflicting rows: {path}")
        table = pa.Table.from_pylist([asdict(row) for row in rows], schema=_SELECTOR_SCHEMA)
        _atomic_write(path, lambda temp_path: pq.write_table(table, temp_path))
        return path

    def selection_mask(
        self,
        shard_id: int,
        *,
        num_chunks: int,
        chunk_length: int,
    ) -> np.ndarray:
        rows = self.read_shard(shard_id)
        expected_positions = tuple(range(1, num_chunks + 1))
        if tuple(row.chunk_position for row in rows) != expected_positions:
            raise ValueError(
                f"Incomplete selector shard {shard_id}: expected chunks={expected_positions}, "
                f"actual={tuple(row.chunk_position for row in rows)}"
            )
        mask_CT = np.zeros((num_chunks, chunk_length), dtype=np.int32)
        for row in rows:
            positions = np.asarray(row.selected_positions, dtype=np.int64)
            if positions.size and (positions.min() < 1 or positions.max() >= chunk_length):
                raise ValueError(f"Selector shard {shard_id} contains an invalid target position")
            mask_CT[row.chunk_position - 1, positions] = 1
        return mask_CT
