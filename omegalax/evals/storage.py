"""Resumable Parquet storage for state-usage evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from omegalax.evals.aggregation import SummaryPoint


_CONDITIONS = {
    "gdn": ("true_gdn", "zero_gdn", "shuffled_gdn"),
    "conv": ("true_conv", "zero_conv", "shuffled_conv"),
}

_METRIC_SCHEMA = pa.schema(
    (
        pa.field("experiment", pa.string(), nullable=False),
        pa.field("condition", pa.string(), nullable=False),
        pa.field("bucket_idx", pa.int64(), nullable=False),
        pa.field("record_idx", pa.int64(), nullable=False),
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("doc_num_chunks", pa.int64(), nullable=False),
        pa.field("chunk_position", pa.int64(), nullable=False),
        pa.field("nll_sum", pa.float64(), nullable=False),
        pa.field("token_count", pa.int64(), nullable=False),
    )
)

_SUMMARY_SCHEMA = pa.schema(
    (
        pa.field("experiment", pa.string(), nullable=False),
        pa.field("metric", pa.string(), nullable=False),
        pa.field("view", pa.string(), nullable=False),
        pa.field("doc_num_chunks", pa.int64(), nullable=True),
        pa.field("chunk_position", pa.int64(), nullable=False),
        pa.field("condition", pa.string(), nullable=True),
        pa.field("value", pa.float64(), nullable=False),
        pa.field("token_count", pa.float64(), nullable=False),
    )
)


@dataclass(frozen=True)
class EvalRunIdentity:
    dataset_hash: str
    manifest_hash: str
    checkpoint_root: str
    checkpoint_step: int
    code_hash: str
    eval_config: dict[str, Any]


@dataclass(frozen=True)
class MetricRow:
    experiment: str
    condition: str
    bucket_idx: int
    record_idx: int
    doc_id: str
    doc_num_chunks: int
    chunk_position: int
    nll_sum: float
    token_count: int


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
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


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _atomic_write(path, lambda temp_path: temp_path.write_bytes(payload))


@dataclass
class EvalRunStore:
    run_dir: Path
    config_path: Path
    status_path: Path
    identity: EvalRunIdentity

    @classmethod
    def open(
        cls,
        checkpoint_root: str | Path,
        checkpoint_step: int,
        identity: EvalRunIdentity,
    ) -> EvalRunStore:
        checkpoint_root = Path(checkpoint_root).expanduser().resolve()
        if identity.checkpoint_root != str(checkpoint_root):
            raise ValueError("Evaluation identity checkpoint_root does not match the run path")
        if identity.checkpoint_step != checkpoint_step:
            raise ValueError("Evaluation identity checkpoint_step does not match the run path")

        run_dir = checkpoint_root / "evals" / "state_usage_v1" / f"step_{checkpoint_step}"
        config_path = run_dir / "eval_config.json"
        status_path = run_dir / "status.json"
        identity_payload = asdict(identity)

        if config_path.is_file():
            stored_config = json.loads(config_path.read_text())
            stored_identity = stored_config.get("identity", stored_config)
            if _canonical_json(stored_identity) != _canonical_json(identity_payload):
                raise ValueError("Evaluation run identity conflicts with the stored identity")

        for relative_dir in (
            "raw/gdn",
            "raw/conv",
            "summary",
            "plots/gdn",
            "plots/conv",
            "comparisons",
        ):
            (run_dir / relative_dir).mkdir(parents=True, exist_ok=True)

        if not config_path.exists():
            _write_json(config_path, {"identity": identity_payload})
        if not status_path.exists():
            _write_json(status_path, {"complete": False})

        return cls(
            run_dir=run_dir,
            config_path=config_path,
            status_path=status_path,
            identity=identity,
        )

    def extend_identity(self, new_identity: EvalRunIdentity) -> None:
        if new_identity.checkpoint_root != self.identity.checkpoint_root:
            raise ValueError("Extended identity must keep checkpoint_root unchanged")
        if new_identity.checkpoint_step != self.identity.checkpoint_step:
            raise ValueError("Extended identity must keep checkpoint_step unchanged")

        _write_json(self.status_path, {"complete": False})
        _write_json(self.config_path, {"identity": asdict(new_identity)})
        self.identity = new_identity

    def _shard_path(self, experiment: str, condition: str, shard_id: int) -> Path:
        if experiment not in _CONDITIONS or condition not in _CONDITIONS[experiment]:
            raise ValueError(f"Unsupported evaluation condition: {experiment}/{condition}")
        return self.run_dir / "raw" / experiment / f"{condition}_shard_{int(shard_id)}.parquet"

    def shard_is_complete(self, experiment: str, condition: str, shard_id: int) -> bool:
        return self._shard_path(experiment, condition, shard_id).is_file()

    def read_shard(
        self,
        experiment: str,
        condition: str,
        shard_id: int,
    ) -> tuple[MetricRow, ...]:
        path = self._shard_path(experiment, condition, shard_id)
        if not path.is_file():
            return ()
        return tuple(MetricRow(**row) for row in pq.read_table(path).to_pylist())

    def write_shard(
        self,
        experiment: str,
        condition: str,
        shard_id: int,
        rows: Iterable[MetricRow],
    ) -> Path:
        path = self._shard_path(experiment, condition, shard_id)
        rows = tuple(rows)
        if any(row.experiment != experiment or row.condition != condition for row in rows):
            raise ValueError("Every metric row must match the shard experiment and condition")

        if path.is_file():
            if self.read_shard(experiment, condition, shard_id) == rows:
                return path
            raise ValueError(f"Shard already exists with conflicting rows: {path}")

        table = pa.Table.from_pylist([asdict(row) for row in rows], schema=_METRIC_SCHEMA)
        _atomic_write(path, lambda temp_path: pq.write_table(table, temp_path))
        return path

    def read_rows(
        self,
        *,
        experiment: str | None = None,
        condition: str | None = None,
    ) -> tuple[MetricRow, ...]:
        rows = []
        for candidate_experiment, candidate_conditions in _CONDITIONS.items():
            if experiment is not None and candidate_experiment != experiment:
                continue
            for candidate_condition in candidate_conditions:
                if condition is not None and candidate_condition != condition:
                    continue
                prefix = f"{candidate_condition}_shard_"
                paths = sorted(
                    (self.run_dir / "raw" / candidate_experiment).glob(f"{prefix}*.parquet")
                )
                for path in paths:
                    shard_id = int(path.stem.removeprefix(prefix))
                    rows.extend(
                        self.read_shard(candidate_experiment, candidate_condition, shard_id)
                    )
        return tuple(rows)

    def write_summary(self, points: Iterable[SummaryPoint]) -> Path:
        path = self.run_dir / "summary" / "metrics.parquet"
        points = tuple(points)
        if path.is_file() and self.read_summary() == points:
            return path

        table = pa.Table.from_pylist([asdict(point) for point in points], schema=_SUMMARY_SCHEMA)
        _atomic_write(path, lambda temp_path: pq.write_table(table, temp_path))
        return path

    def read_summary(self) -> tuple[SummaryPoint, ...]:
        from omegalax.evals.aggregation import SummaryPoint

        path = self.run_dir / "summary" / "metrics.parquet"
        if not path.is_file():
            return ()
        return tuple(SummaryPoint(**row) for row in pq.read_table(path).to_pylist())

    def mark_complete(self) -> None:
        expected_shards = self.identity.eval_config.get("expected_shards", {})
        missing = []
        for experiment, conditions in expected_shards.items():
            for condition, shard_ids in conditions.items():
                for shard_id in shard_ids:
                    if not self.shard_is_complete(experiment, condition, int(shard_id)):
                        missing.append((experiment, condition, int(shard_id)))
        if missing:
            raise ValueError(f"Evaluation is incomplete; missing shards: {missing}")

        status = json.loads(self.status_path.read_text())
        if not status.get("complete", False):
            _write_json(self.status_path, {"complete": True})
