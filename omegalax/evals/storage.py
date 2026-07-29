"""Resumable Parquet storage for state-usage evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
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

    @property
    def result_version(self) -> str:
        return str(self.identity.eval_config.get("result_version", "state_usage_v1"))

    @property
    def token_selections(self) -> tuple[str, ...]:
        return tuple(self.identity.eval_config.get("token_selections", ("all",)))

    def _ensure_directories(self) -> None:
        if self.result_version == "state_usage_v1":
            relative_dirs = (
                "raw/gdn",
                "raw/conv",
                "summary",
                "plots/gdn",
                "plots/conv",
                "comparisons",
            )
        elif self.result_version == "state_usage_v2":
            relative_dirs = tuple(
                relative
                for selection in self.token_selections
                for relative in (
                    f"raw/{selection}/gdn",
                    f"raw/{selection}/conv",
                    f"summary/{selection}",
                    f"plots/{selection}/gdn",
                    f"plots/{selection}/conv",
                    f"comparisons/{selection}",
                )
            )
        else:
            raise ValueError(f"Unsupported evaluation result version: {self.result_version!r}")
        for relative_dir in relative_dirs:
            (self.run_dir / relative_dir).mkdir(parents=True, exist_ok=True)

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

        result_version = str(identity.eval_config.get("result_version", "state_usage_v1"))
        if result_version not in {"state_usage_v1", "state_usage_v2"}:
            raise ValueError(f"Unsupported evaluation result version: {result_version!r}")
        run_dir = checkpoint_root / "evals" / result_version / f"step_{checkpoint_step}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "eval_config.json"
        status_path = run_dir / "status.json"
        identity_payload = asdict(identity)
        store = cls(
            run_dir=run_dir,
            config_path=config_path,
            status_path=status_path,
            identity=identity,
        )
        store._ensure_directories()

        if config_path.is_file():
            stored_config = json.loads(config_path.read_text())
            stored_identity = stored_config.get("identity", stored_config)
            if _canonical_json(stored_identity) != _canonical_json(identity_payload):
                raise ValueError("Evaluation run identity conflicts with the stored identity")

        if not config_path.exists():
            _write_json(config_path, {"identity": identity_payload})
        if not status_path.exists():
            _write_json(status_path, {"complete": False})
        return store

    def extend_identity(self, new_identity: EvalRunIdentity) -> None:
        if new_identity.checkpoint_root != self.identity.checkpoint_root:
            raise ValueError("Extended identity must keep checkpoint_root unchanged")
        if new_identity.checkpoint_step != self.identity.checkpoint_step:
            raise ValueError("Extended identity must keep checkpoint_step unchanged")

        self.invalidate_comparison_outputs()
        self.identity = new_identity
        self._ensure_directories()
        self.mark_incomplete()
        _write_json(self.config_path, {"identity": asdict(new_identity)})

    def register_comparison_outputs(self, directories: Iterable[str | Path]) -> None:
        path = self.run_dir / "comparison_outputs.json"
        registered = set()
        if path.is_file():
            registered.update(json.loads(path.read_text()).get("directories", ()))
        registered.update(str(Path(directory).expanduser().resolve()) for directory in directories)
        _write_json(path, {"directories": sorted(registered)})

    def invalidate_comparison_outputs(self) -> None:
        registry_path = self.run_dir / "comparison_outputs.json"
        directories = ()
        if registry_path.is_file():
            directories = json.loads(registry_path.read_text()).get("directories", ())
        for directory in directories:
            directory = Path(directory)
            if directory.exists():
                shutil.rmtree(directory)
        comparisons_dir = self.run_dir / "comparisons"
        if comparisons_dir.exists():
            shutil.rmtree(comparisons_dir)
        registry_path.unlink(missing_ok=True)

    def mark_incomplete(self) -> None:
        _write_json(self.status_path, {"complete": False})

    def _shard_path(
        self,
        experiment: str,
        condition: str,
        shard_id: int,
        *,
        selection: str = "all",
    ) -> Path:
        if experiment not in _CONDITIONS or condition not in _CONDITIONS[experiment]:
            raise ValueError(f"Unsupported evaluation condition: {experiment}/{condition}")
        if selection not in self.token_selections:
            raise ValueError(f"Unsupported token selection: {selection!r}")
        raw_dir = self.run_dir / "raw"
        if self.result_version == "state_usage_v2":
            raw_dir /= selection
        return raw_dir / experiment / f"{condition}_shard_{int(shard_id)}.parquet"

    def shard_is_complete(
        self,
        experiment: str,
        condition: str,
        shard_id: int,
        *,
        selection: str = "all",
    ) -> bool:
        return self._shard_path(experiment, condition, shard_id, selection=selection).is_file()

    def read_shard(
        self,
        experiment: str,
        condition: str,
        shard_id: int,
        *,
        selection: str = "all",
    ) -> tuple[MetricRow, ...]:
        path = self._shard_path(experiment, condition, shard_id, selection=selection)
        if not path.is_file():
            return ()
        return tuple(MetricRow(**row) for row in pq.read_table(path).to_pylist())

    def write_shard(
        self,
        experiment: str,
        condition: str,
        shard_id: int,
        rows: Iterable[MetricRow],
        *,
        selection: str = "all",
    ) -> Path:
        path = self._shard_path(experiment, condition, shard_id, selection=selection)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = tuple(rows)
        if any(row.experiment != experiment or row.condition != condition for row in rows):
            raise ValueError("Every metric row must match the shard experiment and condition")

        if path.is_file():
            if self.read_shard(experiment, condition, shard_id, selection=selection) == rows:
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
        selection: str = "all",
    ) -> tuple[MetricRow, ...]:
        if selection not in self.token_selections:
            raise ValueError(f"Unsupported token selection: {selection!r}")
        raw_dir = self.run_dir / "raw"
        if self.result_version == "state_usage_v2":
            raw_dir /= selection
        rows = []
        for candidate_experiment, candidate_conditions in _CONDITIONS.items():
            if experiment is not None and candidate_experiment != experiment:
                continue
            for candidate_condition in candidate_conditions:
                if condition is not None and candidate_condition != condition:
                    continue
                prefix = f"{candidate_condition}_shard_"
                paths = sorted((raw_dir / candidate_experiment).glob(f"{prefix}*.parquet"))
                for path in paths:
                    shard_id = int(path.stem.removeprefix(prefix))
                    rows.extend(
                        self.read_shard(
                            candidate_experiment,
                            candidate_condition,
                            shard_id,
                            selection=selection,
                        )
                    )
        return tuple(rows)

    def write_summary(
        self,
        points: Iterable[SummaryPoint],
        *,
        selection: str = "all",
    ) -> Path:
        summary_dir = self.run_dir / "summary"
        if self.result_version == "state_usage_v2":
            summary_dir /= selection
        summary_dir.mkdir(parents=True, exist_ok=True)
        path = summary_dir / "metrics.parquet"
        points = tuple(points)
        if path.is_file() and self.read_summary(selection=selection) == points:
            return path

        table = pa.Table.from_pylist([asdict(point) for point in points], schema=_SUMMARY_SCHEMA)
        _atomic_write(path, lambda temp_path: pq.write_table(table, temp_path))
        return path

    def read_summary(self, *, selection: str = "all") -> tuple[SummaryPoint, ...]:
        from omegalax.evals.aggregation import SummaryPoint

        summary_dir = self.run_dir / "summary"
        if self.result_version == "state_usage_v2":
            summary_dir /= selection
        path = summary_dir / "metrics.parquet"
        if not path.is_file():
            return ()
        return tuple(SummaryPoint(**row) for row in pq.read_table(path).to_pylist())

    def mark_complete(self) -> None:
        expected_shards = self.identity.eval_config.get("expected_shards", {})
        missing = []
        expected_by_selection = (
            expected_shards if self.result_version == "state_usage_v2" else {"all": expected_shards}
        )
        for selection, experiments in expected_by_selection.items():
            for experiment, conditions in experiments.items():
                for condition, shard_ids in conditions.items():
                    for shard_id in shard_ids:
                        if not self.shard_is_complete(
                            experiment,
                            condition,
                            int(shard_id),
                            selection=selection,
                        ):
                            missing.append((selection, experiment, condition, int(shard_id)))
        if missing:
            raise ValueError(f"Evaluation is incomplete; missing shards: {missing}")

        status = json.loads(self.status_path.read_text())
        if not status.get("complete", False):
            _write_json(self.status_path, {"complete": True})
