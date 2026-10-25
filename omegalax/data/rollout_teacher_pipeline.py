"""Pointer index and Grain iterator for rollout-teacher distillation batches.

The index stores only pointers into an existing ``rollout_teacher_v1`` root: sequence and
target ArrayRecord positions of every selected whole context chain, grouped into
segment-count buckets. Labels are joined once from the frozen source snapshot at build
time. Batches carry the stored token IDs of whole chains plus compact teacher rows.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import grain
import numpy as np

from omegalax.data import rollout_teacher as rt
from omegalax.data.pretrain_data_set import (
    BATCH_PRETRAIN_METADATA_KEY,
    COMPILED_METADATA_FILENAME,
    DEFAULT_PAD_ID,
    MAX_PRETRAIN_POSITIONS,
    load_arrayrecord_metadata,
    resolve_arrayrecord_paths,
    resolve_pretrain_dp,
    write_json_arrayrecord_dataset,
)
from omegalax.data.pretrain_statepassing import STATEPASSING_INDEX_SHUFFLE_ROUNDS

ROLLOUT_TEACHER_INDEX_FORMAT = "omegalax_rollout_teacher_index_v1"
ROLLOUT_TEACHER_INDEX_VERSION = 1
ROLLOUT_TEACHER_SELECTIONS = ("correct", "incorrect", "graded")
TARGET_CAPACITY_MULTIPLE = 1024
TEACHER_BATCH_KEYS = (
    "teacher_positions_BR",
    "teacher_mask_BR",
    "teacher_top512_ids_BRK",
    "teacher_top512_logits_BRK",
    "teacher_special_logits_BRS",
    "teacher_logz_all_BR",
    "teacher_logz_tail_BR",
)
_SELECTION_LABELS = {
    "correct": ["correct"],
    "incorrect": ["incorrect"],
    "graded": ["correct", "incorrect"],
}
_LABEL_RULE = (
    "snapshot attempts[alias attempt_id].result: status == 'completed' and correct is True "
    "-> correct; status == 'completed' and correct is False -> incorrect; otherwise unlabeled"
)
_RECORDS_PER_SHARD = 100_000


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_index_config(
    selection: str, segment_length: int, segment_buckets: Sequence[int]
) -> list[int]:
    _require(
        selection in ROLLOUT_TEACHER_SELECTIONS,
        f"selection must be one of {ROLLOUT_TEACHER_SELECTIONS}",
    )
    _require(
        type(segment_length) is int and segment_length > 0, "segment_length must be an int > 0"
    )
    buckets = [int(value) for value in segment_buckets]
    _require(
        buckets
        and buckets[0] > 0
        and all(left < right for left, right in zip(buckets, buckets[1:])),
        f"segment_buckets must be strictly increasing positive segment counts, got {list(segment_buckets)}",
    )
    return buckets


def _source_hashes(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Parsed manifest and sha256 of manifest, catalog, prepared index and label snapshot."""

    manifest_path = root / rt.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    paths = {
        "manifest_sha256": rt.MANIFEST_FILENAME,
        "catalog_sha256": rt.CATALOG_FILENAME,
        "prepared_sha256": manifest["prepared"]["path"],
        "snapshot_sha256": manifest["snapshot"]["path"],
    }
    for relative in paths.values():
        path = PurePosixPath(relative)
        _require(
            not path.is_absolute() and ".." not in path.parts,
            f"Source path must stay inside the root: {relative}",
        )
    return manifest, {key: _sha256_file(root / relative) for key, relative in paths.items()}


def _attempt_label(result: Mapping[str, Any]) -> str | None:
    if result["status"] == "completed" and result["correct"] is True:
        return "correct"
    if result["status"] == "completed" and result["correct"] is False:
        return "incorrect"
    return None


def build_rollout_teacher_index(
    dataset_root: str | Path,
    out_dir: str | Path,
    *,
    selection: str,
    segment_length: int,
    segment_buckets: Sequence[int],
) -> Path:
    """Write a pointer index of the selected whole chains of ``dataset_root``.

    Only manifest, catalog, prepared index and label snapshot are read; no sequence or
    target records. A selected chain longer than ``max(segment_buckets) * segment_length``
    raises instead of being cropped or dropped.
    """

    buckets = _validate_index_config(selection, segment_length, segment_buckets)
    out_path = Path(out_dir).expanduser().resolve()
    _require(
        not out_path.exists() or not any(out_path.iterdir()),
        f"Refusing to write into non-empty index directory {out_path}; remove a crashed partial build explicitly",
    )

    dataset = rt.TeacherDataset(dataset_root)
    root = dataset.root
    manifest, hashes = _source_hashes(root)
    _require(manifest == dataset.manifest, "Dataset manifest changed while building the index")
    _require(
        hashes["prepared_sha256"] == manifest["prepared"]["sha256"]
        and hashes["snapshot_sha256"] == manifest["snapshot"]["sha256"],
        "prepared index or label snapshot does not match the manifest sha256",
    )
    prepared = json.loads((root / manifest["prepared"]["path"]).read_text())
    snapshot = json.loads((root / manifest["snapshot"]["path"]).read_text())
    _require(
        prepared["snapshot_sha256"] == manifest["snapshot"]["sha256"]
        and prepared["binding_sha256"] == manifest["binding_sha256"],
        "prepared index is not bound to the manifest snapshot and builder",
    )
    prepared_by_id = {entry["sequence_id"]: entry for entry in prepared["sequences"]}
    attempts = {attempt["attempt_id"]: attempt for attempt in snapshot["attempts"]}
    _require(
        len(prepared_by_id) == len(prepared["sequences"]) == len(dataset.sequences)
        and len(attempts) == len(snapshot["attempts"]),
        "prepared index, catalog and snapshot must have unique, matching sequence and attempt ids",
    )

    labels = _SELECTION_LABELS[selection]
    split_records: dict[str, list[dict[str, Any]]] = {}
    too_long = []
    for entry in dataset.sequences:
        sequence_id = entry["sequence_id"]
        prepared_entry = prepared_by_id.get(sequence_id)
        _require(
            prepared_entry is not None
            and all(
                prepared_entry[key] == entry[key]
                for key in (
                    "split",
                    "family",
                    "task_key",
                    "token_count",
                    "target_count",
                    "aliases",
                    "path",
                    "record_index",
                )
            ),
            f"Catalog entry {sequence_id} does not match prepared.json",
        )
        alias_labels = set()
        for alias in entry["aliases"]:
            attempt = attempts.get(alias["attempt_id"])
            _require(
                attempt is not None
                and attempt["state"] == "included"
                and attempt["task_key"] == alias["task_key"],
                f"Alias attempt of {sequence_id} is not an included snapshot attempt of the same task",
            )
            alias_labels.add(_attempt_label(attempt["result"]))
        _require(
            len(alias_labels) == 1,
            f"Aliases of {sequence_id} disagree on the label: {sorted(map(str, alias_labels))}",
        )
        split_records.setdefault(entry["split"], [])
        (label,) = alias_labels
        if label not in labels:
            continue
        num_segments = -(-entry["token_count"] // segment_length)
        bucket_idx = bisect.bisect_left(buckets, num_segments)
        if bucket_idx == len(buckets):
            too_long.append((num_segments, sequence_id))
            continue
        split_records[entry["split"]].append(
            {
                "sequence_id": sequence_id,
                "split": entry["split"],
                "family": entry["family"],
                "task_key": entry["task_key"],
                "attempt_ids": [alias["attempt_id"] for alias in entry["aliases"]],
                "label": label,
                "token_count": entry["token_count"],
                "target_count": entry["target_count"],
                "num_segments": num_segments,
                "bucket_idx": bucket_idx,
                "path": entry["path"],
                "record_index": entry["record_index"],
                "targets_path": entry["targets_path"],
                "target_records": entry["target_records"],
            }
        )
    if too_long:
        too_long.sort(reverse=True)
        raise ValueError(
            f"{len(too_long)} selected samples need more than {buckets[-1]} segments of {segment_length} "
            f"tokens (largest needs {too_long[0][0]}: {[sid for _, sid in too_long[:3]]}). Whole chains are "
            f"never cropped or dropped; add a bucket >= {too_long[0][0]} or increase segment_length."
        )
    empty = sorted(split for split, records in split_records.items() if not records)
    _require(not empty, f"selection={selection!r} selects no samples in split(s) {empty}")

    max_targets = [0] * len(buckets)
    for records in split_records.values():
        records.sort(key=lambda record: (record["bucket_idx"], record["sequence_id"]))
        for sample_idx, record in enumerate(records):
            record["sample_idx"] = sample_idx
            max_targets[record["bucket_idx"]] = max(
                max_targets[record["bucket_idx"]], record["target_count"]
            )

    splits_metadata = {}
    for split, records in sorted(split_records.items()):
        split_path = write_json_arrayrecord_dataset(
            records,
            out_path / split,
            records_per_shard=_RECORDS_PER_SHARD,
            overwrite=False,
            metadata={"format": ROLLOUT_TEACHER_INDEX_FORMAT, "split": split},
        )
        shard_paths = resolve_arrayrecord_paths(split_path)
        bucket_metadata = []
        for bucket_idx, num_segments in enumerate(buckets):
            members = [record for record in records if record["bucket_idx"] == bucket_idx]
            bucket_metadata.append(
                {
                    "num_segments": num_segments,
                    "record_start": members[0]["sample_idx"] if members else len(records),
                    "num_samples": len(members),
                    "num_targets": sum(record["target_count"] for record in members),
                    "max_target_count": max(
                        (record["target_count"] for record in members), default=0
                    ),
                }
            )
        splits_metadata[split] = {
            "path": split,
            "shard_paths": [str(path.relative_to(out_path)) for path in shard_paths],
            "shard_sha256": [_sha256_file(path) for path in shard_paths],
            "num_samples": len(records),
            "num_context_tokens": sum(record["token_count"] for record in records),
            "num_targets": sum(record["target_count"] for record in records),
            "label_counts": {
                label: sum(record["label"] == label for record in records) for label in labels
            },
            "buckets": bucket_metadata,
        }

    metadata = {
        "format": ROLLOUT_TEACHER_INDEX_FORMAT,
        "version": ROLLOUT_TEACHER_INDEX_VERSION,
        "selection": selection,
        "labels": labels,
        "label_rule": _LABEL_RULE,
        "segment_length": segment_length,
        "segment_buckets": buckets,
        "target_capacities": [
            -(-value // TARGET_CAPACITY_MULTIPLE) * TARGET_CAPACITY_MULTIPLE
            for value in max_targets
        ],
        "target_capacity_multiple": TARGET_CAPACITY_MULTIPLE,
        "vocab_size": rt.VOCAB_SIZE,
        "top_k": rt.TOP_K,
        "special_ids": list(rt.SPECIAL_IDS),
        "source": {
            "dataset_root": str(root),
            "format": rt.FORMAT,
            **hashes,
            "dataset_binding": rt.dataset_source_binding(manifest),
            "scoring_binding": dataset.binding,
            "teacher_binding_sha256": manifest["teacher_binding_sha256"],
            "teacher_revision": manifest["teacher"]["revision"],
        },
        "splits": splits_metadata,
    }
    metadata["index_binding"] = _json_sha256(metadata)
    # Written last: a crashed build leaves no metadata.json and is never loadable.
    (out_path / COMPILED_METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n")
    return out_path


def load_rollout_teacher_index(
    index_dir: str | Path,
    *,
    selection: str,
    segment_length: int,
    segment_buckets: Sequence[int],
) -> dict[str, Any]:
    """Return the index metadata after metadata-only validation against the run and source.

    Hashes the index shards and the source manifest, catalog, prepared index and label
    snapshot; reads no sequence or target records.
    """

    buckets = _validate_index_config(selection, segment_length, segment_buckets)
    index_path = Path(index_dir).expanduser().resolve()
    metadata = load_arrayrecord_metadata(index_path)
    _require(
        metadata.get("format") == ROLLOUT_TEACHER_INDEX_FORMAT
        and metadata.get("version") == ROLLOUT_TEACHER_INDEX_VERSION,
        f"Expected {ROLLOUT_TEACHER_INDEX_FORMAT} version {ROLLOUT_TEACHER_INDEX_VERSION} at {index_path}",
    )
    _require(
        metadata.get("index_binding")
        == _json_sha256({k: v for k, v in metadata.items() if k != "index_binding"}),
        f"Index metadata does not match its index_binding: {index_path}",
    )
    for key, requested in (
        ("selection", selection),
        ("segment_length", segment_length),
        ("segment_buckets", buckets),
    ):
        _require(
            metadata[key] == requested,
            f"Rollout-teacher index {key} is {metadata[key]!r} but the run requested {requested!r}; "
            "build a matching index",
        )
    _require(
        metadata["vocab_size"] == rt.VOCAB_SIZE
        and metadata["top_k"] == rt.TOP_K
        and metadata["special_ids"] == list(rt.SPECIAL_IDS),
        "Index vocab, Top-K or special IDs differ from the reader format",
    )
    source = metadata["source"]
    _, hashes = _source_hashes(Path(source["dataset_root"]))
    changed = sorted(key for key, value in hashes.items() if source[key] != value)
    _require(
        not changed,
        f"Source dataset {source['dataset_root']} changed since the index was built: {changed}",
    )
    for split, split_metadata in metadata["splits"].items():
        _require(
            [_sha256_file(index_path / path) for path in split_metadata["shard_paths"]]
            == split_metadata["shard_sha256"],
            f"Index shards of split {split!r} do not match metadata",
        )
    return metadata


def _split_plan(
    metadata: Mapping[str, Any],
    *,
    split: str,
    batch_size: int,
    data_shards: int,
) -> list[tuple[int, int, int, int]]:
    """Per bucket: (record_start, num_samples, local samples per batch, global batches per epoch)."""

    _require(batch_size > 0, "batch_size must be > 0")
    _require(
        split in metadata["splits"],
        f"Index has no split {split!r}; available: {sorted(metadata['splits'])}",
    )
    plan = []
    for num_segments, bucket in zip(
        metadata["segment_buckets"], metadata["splits"][split]["buckets"]
    ):
        _require(
            batch_size % num_segments == 0,
            f"batch_size={batch_size} segments per data shard must be divisible by every segment bucket "
            f"{metadata['segment_buckets']}",
        )
        local_samples = batch_size // num_segments
        global_samples = local_samples * data_shards
        plan.append(
            (
                bucket["record_start"],
                bucket["num_samples"],
                local_samples,
                -(-bucket["num_samples"] // global_samples),
            )
        )
    return plan


def rollout_teacher_epoch_batches(
    index_metadata: Mapping[str, Any],
    *,
    split: str,
    batch_size: int,
    dp_size: int,
    fsdp_size: int,
) -> int:
    """Batches per epoch of one data shard; equal on every shard and to global batches per epoch."""

    data_shards, _ = resolve_pretrain_dp(dp_size=dp_size, fsdp_size=fsdp_size, process_index=0)
    plan = _split_plan(index_metadata, split=split, batch_size=batch_size, data_shards=data_shards)
    return sum(num_batches for *_, num_batches in plan)


def _shuffle_seed(seed: int, epoch: int, stream: str) -> int:
    digest = hashlib.sha256(f"rollout_teacher/{seed}/{epoch}/{stream}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


class _RolloutTeacherBatchBuilder:
    """Maps a global batch position to this data shard's whole-chain batch."""

    def __init__(
        self,
        *,
        index_path: Path,
        metadata: Mapping[str, Any],
        split: str,
        plan: Sequence[tuple[int, int, int, int]],
        data_shards: int,
        shard_index: int,
        shuffle: bool,
        seed: int,
        pad_id: int,
    ) -> None:
        self.dataset_root = Path(metadata["source"]["dataset_root"])
        self.scoring_binding = str(metadata["source"]["scoring_binding"])
        self.shard_paths = [
            str(index_path / path) for path in metadata["splits"][split]["shard_paths"]
        ]
        self.split = split
        self.segment_length = int(metadata["segment_length"])
        self.buckets = [int(value) for value in metadata["segment_buckets"]]
        self.capacities = [int(value) for value in metadata["target_capacities"]]
        self.plan = list(plan)
        self.batch_offsets = np.cumsum(
            [0] + [num_batches for *_, num_batches in plan[:-1]]
        ).tolist()
        self.epoch_batches = sum(num_batches for *_, num_batches in plan)
        self.data_shards = int(data_shards)
        self.shard_index = int(shard_index)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.pad_id = int(pad_id)
        self._source: grain.sources.ArrayRecordDataSource | None = None

    def _index_source(self) -> grain.sources.ArrayRecordDataSource:
        if self._source is None:
            self._source = grain.sources.ArrayRecordDataSource(self.shard_paths)
        return self._source

    def __call__(self, global_batch_idx: int) -> dict[str, Any]:
        epoch, position = divmod(int(global_batch_idx), self.epoch_batches)
        batch_id = position
        if self.shuffle:
            batch_id = grain.experimental.index_shuffle(
                position,
                self.epoch_batches - 1,
                _shuffle_seed(self.seed, epoch, "batches"),
                STATEPASSING_INDEX_SHUFFLE_ROUNDS,
            )
        bucket_idx = bisect.bisect_right(self.batch_offsets, batch_id) - 1
        record_start, num_samples, local_samples, _ = self.plan[bucket_idx]
        first_slot = (batch_id - self.batch_offsets[bucket_idx]) * local_samples * self.data_shards
        first_slot += self.shard_index * local_samples
        records = []
        for slot in range(first_slot, first_slot + local_samples):
            if slot >= num_samples:
                records.append(None)
                continue
            if self.shuffle:
                slot = grain.experimental.index_shuffle(
                    slot,
                    num_samples - 1,
                    _shuffle_seed(self.seed, epoch, f"bucket{bucket_idx}"),
                    STATEPASSING_INDEX_SHUFFLE_ROUNDS,
                )
            records.append(json.loads(self._index_source()[record_start + slot]))
        batch = self._make_batch(records, bucket_idx)
        batch[BATCH_PRETRAIN_METADATA_KEY].update(
            split=self.split,
            epoch=epoch,
            global_batch_idx=int(global_batch_idx),
            bucket_idx=bucket_idx,
        )
        return batch

    def _make_batch(
        self, records: Sequence[dict[str, Any] | None], bucket_idx: int
    ) -> dict[str, Any]:
        B, C, T, R = (
            len(records),
            self.buckets[bucket_idx],
            self.segment_length,
            self.capacities[bucket_idx],
        )
        K, S = rt.TOP_K, len(rt.SPECIAL_IDS)
        token_ids = np.full((B, C * T), self.pad_id, dtype=np.int32)
        attention_mask = np.zeros((B, C * T), dtype=np.int32)
        loss_mask = np.zeros((B, C * T), dtype=np.int32)
        teacher = {
            "teacher_positions_BR": np.zeros((B, R), dtype=np.int32),
            "teacher_mask_BR": np.zeros((B, R), dtype=np.bool_),
            "teacher_top512_ids_BRK": np.zeros((B, R, K), dtype=np.int32),
            "teacher_top512_logits_BRK": np.zeros((B, R, K), dtype=np.float32),
            "teacher_special_logits_BRS": np.zeros((B, R, S), dtype=np.float32),
            "teacher_logz_all_BR": np.zeros((B, R), dtype=np.float32),
            "teacher_logz_tail_BR": np.zeros((B, R), dtype=np.float32),
        }
        metadata = {
            "num_segments": C,
            "target_capacity": R,
            "sequence_ids": [],
            "families": [],
            "labels": [],
            "sample_idx_B": np.full((B,), -1, dtype=np.int32),
            "valid_sample_B": np.zeros((B,), dtype=np.bool_),
            "token_count_B": np.zeros((B,), dtype=np.int32),
            "target_count_B": np.zeros((B,), dtype=np.int32),
            "valid_segments_B": np.zeros((B,), dtype=np.int32),
            "teacher_target_positions_BR": np.full((B, R), -1, dtype=np.int32),
        }
        for row, record in enumerate(records):
            metadata["sequence_ids"].append(None if record is None else record["sequence_id"])
            metadata["families"].append(None if record is None else record["family"])
            metadata["labels"].append(None if record is None else record["label"])
            if record is None:
                continue
            sequence_id = record["sequence_id"]
            _, sequence = rt.read_sequence(self.dataset_root, record)
            targets = rt.read_targets(self.dataset_root, record, self.scoring_binding)
            n = sequence["token_ids"].shape[0]
            positions = targets["target_positions"]
            m = positions.shape[0]
            _require(
                record["split"] == self.split
                and record["bucket_idx"] == bucket_idx
                and -(-n // T) == record["num_segments"] <= C
                and m <= R
                and np.array_equal(positions, sequence["target_positions"]),
                f"Index record of {sequence_id} does not match its split, bucket, capacity or stored targets",
            )
            token_ids[row, :n] = sequence["token_ids"]
            attention_mask[row, :n] = 1
            loss_mask[row, positions] = 1
            teacher["teacher_positions_BR"][row, :m] = positions - 1
            teacher["teacher_mask_BR"][row, :m] = True
            teacher["teacher_top512_ids_BRK"][row, :m] = targets["top512_ids"]
            teacher["teacher_top512_logits_BRK"][row, :m] = targets["top512_logits"]
            teacher["teacher_special_logits_BRS"][row, :m] = targets["special_logits"]
            teacher["teacher_logz_all_BR"][row, :m] = targets["logz_all"]
            teacher["teacher_logz_tail_BR"][row, :m] = targets["logz_tail"]
            metadata["sample_idx_B"][row] = record["sample_idx"]
            metadata["valid_sample_B"][row] = True
            metadata["token_count_B"][row] = n
            metadata["target_count_B"][row] = m
            metadata["valid_segments_B"][row] = record["num_segments"]
            metadata["teacher_target_positions_BR"][row, :m] = positions

        reset_state = np.zeros((B, C), dtype=np.bool_)
        reset_state[:, 0] = True
        return {
            "token_ids_BCT": token_ids.reshape(B, C, T),
            "attention_mask_BCT": attention_mask.reshape(B, C, T),
            "loss_mask_BCT": loss_mask.reshape(B, C, T),
            "reset_state_BC": reset_state,
            "chunk_idx_BC": np.tile(np.arange(C, dtype=np.int32), (B, 1)),
            **teacher,
            BATCH_PRETRAIN_METADATA_KEY: metadata,
        }


def make_rollout_teacher_iterator(
    index_dir: str | Path,
    *,
    split: str,
    selection: str,
    segment_length: int,
    segment_buckets: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
    num_epochs: int | None = None,
    dp_size: int,
    fsdp_size: int,
    dp_index: int | None = None,
    process_index: int | None = None,
    pad_id: int = DEFAULT_PAD_ID,
    grain_workers: int = 8,
    grain_worker_buffer_size: int = 1,
    grain_read_threads: int = 2,
    grain_read_prefetch_buffer_size: int = 4,
) -> grain.DatasetIterator:
    """Deterministic bucket-homogeneous whole-chain batches of one split for one data shard.

    ``batch_size`` is the segment budget per data shard; a bucket with C segments yields
    ``batch_size // C`` samples per batch. Every valid sample appears exactly once per epoch
    over all data shards; residual batches are filled with fully masked padding samples.
    """

    if grain_workers < 0:
        raise ValueError("grain_workers must be >= 0")
    if grain_worker_buffer_size <= 0:
        raise ValueError("grain_worker_buffer_size must be > 0")
    if grain_read_threads <= 0:
        raise ValueError("grain_read_threads must be > 0")
    if grain_read_prefetch_buffer_size <= 0:
        raise ValueError("grain_read_prefetch_buffer_size must be > 0")

    index_path = Path(index_dir).expanduser().resolve()
    metadata = load_rollout_teacher_index(
        index_path,
        selection=selection,
        segment_length=segment_length,
        segment_buckets=segment_buckets,
    )
    data_shards, shard_index = resolve_pretrain_dp(
        dp_size=dp_size, fsdp_size=fsdp_size, process_index=process_index
    )
    if dp_index is not None:
        shard_index = int(dp_index)
    if shard_index < 0 or shard_index >= data_shards:
        raise ValueError(f"dp_index must be in [0, {data_shards}), got {shard_index}")
    plan = _split_plan(metadata, split=split, batch_size=batch_size, data_shards=data_shards)
    epoch_batches = sum(num_batches for *_, num_batches in plan)
    if num_epochs is not None and num_epochs <= 0:
        raise ValueError("num_epochs must be > 0")
    total_batches = (
        MAX_PRETRAIN_POSITIONS if num_epochs is None else epoch_batches * int(num_epochs)
    )

    dataset = grain.MapDataset.range(total_batches).map(
        _RolloutTeacherBatchBuilder(
            index_path=index_path,
            metadata=metadata,
            split=split,
            plan=plan,
            data_shards=data_shards,
            shard_index=shard_index,
            shuffle=shuffle,
            seed=seed,
            pad_id=pad_id,
        )
    )
    iter_dataset = dataset.to_iter_dataset(
        grain.ReadOptions(
            num_threads=grain_read_threads,
            prefetch_buffer_size=grain_read_prefetch_buffer_size,
        )
    )
    if grain_workers > 0:
        iter_dataset = iter_dataset.mp_prefetch(
            grain.MultiprocessingOptions(
                num_workers=grain_workers,
                per_worker_buffer_size=grain_worker_buffer_size,
            )
        )
    return iter(iter_dataset)


ROLLOUT_UPDATE_PLAN_FORMAT = "omegalax_rollout_update_plan_v1"
PLANNED_MICROBATCH_CHAINS = 4


def build_rollout_update_plan(dataset_root: str | Path, out_path: str | Path) -> dict[str, Any]:
    """Freeze whole-chain membership once, without reading the large target arrays."""
    root = Path(dataset_root).resolve()
    manifest, hashes = _source_hashes(root)
    dataset = rt.TeacherDataset(root)
    prepared = json.loads((root / manifest["prepared"]["path"]).read_text())
    snapshot = json.loads((root / manifest["snapshot"]["path"]).read_text())
    _require(
        hashes["prepared_sha256"] == manifest["prepared"]["sha256"]
        and hashes["snapshot_sha256"] == manifest["snapshot"]["sha256"],
        "Dataset metadata does not match frozen manifest",
    )
    by_id = {e["sequence_id"]: e for e in prepared["sequences"]}
    attempts = {a["attempt_id"]: a for a in snapshot["attempts"]}
    records = []
    for entry in dataset.sequences:
        original = by_id[entry["sequence_id"]]
        _require(all(entry[k] == v for k, v in original.items()), "Catalog/prepared mismatch")
        if not 20000 <= entry["token_count"] <= 55000:
            continue
        if set(entry["target_counts"]) != {"complete"}:
            continue
        labels = set()
        for alias in entry["aliases"]:
            attempt = attempts[alias["attempt_id"]]
            _require(
                attempt["state"] == "included" and attempt["task_key"] == entry["task_key"],
                "Alias is not an included same-task attempt",
            )
            result = attempt["result"]
            label = _attempt_label(result)
            if (
                result.get("error") is not None
                or result.get("finish_reason") != "stop"
                or result.get("token_count_complete") is not True
            ):
                label = None
            labels.add(label)
        if len(labels) != 1 or None in labels:
            continue
        records.append({**entry, "label": labels.pop()})
    records.sort(key=lambda e: e["sequence_id"])
    train = [i for i, r in enumerate(records) if r["split"] == "train"]
    val = [i for i, r in enumerate(records) if r["split"] == "val"]
    _require(train and val, "Filtered train/val splits must both be nonempty")
    _require(
        not ({records[i]["task_key"] for i in train} & {records[i]["task_key"] for i in val}),
        "Task split leakage",
    )
    mean = sum(records[i]["token_count"] for i in train) / len(train)
    chains_per_update = PLANNED_MICROBATCH_CHAINS * max(
        1, round(500000 / (mean * PLANNED_MICROBATCH_CHAINS))
    )
    updates = []
    cumulative = 0
    for epoch in range(2):
        order = sorted(
            train, key=lambda i: _json_sha256(["comparison", 0, epoch, records[i]["sequence_id"]])
        )
        for start in range(0, len(order), chains_per_update):
            members = order[start : start + chains_per_update]
            tokens = sum(records[i]["token_count"] for i in members)
            cumulative += tokens
            identity = [
                {
                    k: records[i][k]
                    for k in ("sequence_id", "content_key", "token_count", "target_count")
                }
                for i in members
            ]
            updates.append(
                {
                    "epoch": epoch,
                    "members": members,
                    "context_tokens": tokens,
                    "cumulative_tokens": cumulative,
                    "identity_sha256": _json_sha256(identity),
                }
            )
    # The schedule uses each completed update's cumulative REAL tokens. Final applied LR is zero.
    # warmup_end counts completed warmup updates; first post-warmup update has index warmup_end+1.
    warmup_end = next(
        i + 1 for i, u in enumerate(updates) if u["cumulative_tokens"] >= 0.02 * cumulative
    )
    # Save immediately BEFORE the first update whose endpoint enters the final 10%.
    preanneal = next(i for i, u in enumerate(updates) if u["cumulative_tokens"] > 0.9 * cumulative)
    checkpoints = sorted(
        {*range(warmup_end + 500, preanneal + 1, 500), preanneal, len(updates)} - {0}
    )
    plan = {
        "format": ROLLOUT_UPDATE_PLAN_FORMAT,
        "source": {
            "dataset_root": str(root),
            **hashes,
            "scoring_binding": dataset.binding,
            "dataset_binding": rt.dataset_source_binding(manifest),
        },
        "selection": "completed_boolean_graded_complete_turns_20000_55000",
        "seed": 0,
        "epochs": 2,
        "chains_per_update": chains_per_update,
        "microbatch_chains": PLANNED_MICROBATCH_CHAINS,
        "microbatches_per_update": chains_per_update // PLANNED_MICROBATCH_CHAINS,
        "max_context_tokens": 55000,
        "target_capacity": -(-max(r["target_count"] for r in records) // 1024) * 1024,
        "records": records,
        "validation_members": val,
        "updates": updates,
        "total_context_tokens": cumulative,
        "warmup_end_update": warmup_end,
        "preanneal_update": preanneal,
        "checkpoint_updates": checkpoints,
        "lr": {
            "peak": 2e-5,
            "warmup_token_fraction": 0.02,
            "decay_token_fraction": 0.1,
            "position": "completed_update_cumulative_real_tokens",
            "final": 0.0,
        },
    }
    plan["plan_sha256"] = _json_sha256(plan)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(plan, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    return plan


def load_rollout_update_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text())
    _require(plan["format"] == ROLLOUT_UPDATE_PLAN_FORMAT, "Unknown common update plan")
    _require(
        plan["plan_sha256"] == _json_sha256({k: v for k, v in plan.items() if k != "plan_sha256"}),
        "Common update plan digest mismatch",
    )
    _, hashes = _source_hashes(Path(plan["source"]["dataset_root"]))
    _require(all(plan["source"][k] == v for k, v in hashes.items()), "Common plan source changed")
    return plan


class _PlannedTeacherBatchBuilder:
    def __init__(self, plan: Mapping[str, Any], segment_length: int, split: str, pad_id: int):
        self.plan = plan
        self.split = split
        self.segment_length = segment_length
        C = -(-plan["max_context_tokens"] // segment_length)
        self.builder = _RolloutTeacherBatchBuilder(
            index_path=Path("."),
            metadata={
                "source": plan["source"],
                "segment_length": segment_length,
                "segment_buckets": [C],
                "target_capacities": [plan["target_capacity"]],
                "splits": {split: {"shard_paths": []}},
            },
            split=split,
            plan=[(0, len(plan["records"]), 1, len(plan["records"]))],
            data_shards=1,
            shard_index=0,
            shuffle=False,
            seed=0,
            pad_id=pad_id,
        )

    def __call__(self, cursor: int) -> dict[str, Any]:
        microbatches = -(-self.plan["chains_per_update"] // PLANNED_MICROBATCH_CHAINS)
        if self.split == "train":
            update_idx, micro = divmod(cursor, microbatches)
            members = self.plan["updates"][update_idx]["members"]
        else:
            update_idx, micro = -1, cursor
            members = self.plan["validation_members"]
        selected = members[
            micro * PLANNED_MICROBATCH_CHAINS : (micro + 1) * PLANNED_MICROBATCH_CHAINS
        ]
        records = [
            {
                **self.plan["records"][member],
                "sample_idx": member,
                "bucket_idx": 0,
                "num_segments": -(
                    -self.plan["records"][member]["token_count"] // self.segment_length
                ),
            }
            for member in selected
        ]
        records += [None] * (PLANNED_MICROBATCH_CHAINS - len(records))
        batch = self.builder._make_batch(records, 0)
        batch[BATCH_PRETRAIN_METADATA_KEY].update(
            common_plan_sha256=self.plan["plan_sha256"],
            common_update_idx=update_idx,
            common_micro_idx=micro,
            common_cursor=cursor,
        )
        return batch


def make_planned_rollout_teacher_iterator(
    plan: Mapping[str, Any],
    *,
    segment_length: int,
    split: str = "train",
    pad_id: int = DEFAULT_PAD_ID,
):
    """Single loading controller, four complete chains per microbatch; checkpointable Grain cursor."""
    _require(segment_length in (8192, 2048, 512), "Controlled segment length must be 8192/2048/512")
    _require(split in ("train", "val"), "Expected train/val split")
    count = (
        len(plan["updates"]) * -(-plan["chains_per_update"] // PLANNED_MICROBATCH_CHAINS)
        if split == "train"
        else -(-len(plan["validation_members"]) // PLANNED_MICROBATCH_CHAINS)
    )
    return iter(
        grain.MapDataset.range(count)
        .map(_PlannedTeacherBatchBuilder(plan, segment_length, split, pad_id))
        .to_iter_dataset(grain.ReadOptions(num_threads=1, prefetch_buffer_size=1))
    )
