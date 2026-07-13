"""Fixed-window pretraining iterators for state passing."""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import grain
import numpy as np

from omegalax.data.pretrain_data_set import (
    DEFAULT_DATA_SET_SPLIT,
    DEFAULT_EOS_ID,
    DEFAULT_PAD_ID,
    DEFAULT_CHUNK_LENGTH,
    BATCH_PRETRAIN_METADATA_KEY,
    COMPILED_METADATA_FILENAME,
    DataSetReader,
    DOC_CHAIN_DATASET_VERSION,
    WindowMetadata,
    build_window_arrays,
    iter_document_window_metadata,
    load_arrayrecord_metadata,
    make_dataset_index,
    resolve_arrayrecord_paths,
    resolve_pretrain_dp,
    rewrite_data_set_root_path,
    window_metadata_to_record,
    write_json_arrayrecord_dataset,
)

STATEPASSING_WINDOW_INDEX_FORMAT = "omegalax_pretrain_statepassing_window_index_v1"
STATEPASSING_CURRICULUM_INDEX_FORMAT = "omegalax_pretrain_statepassing_curriculum_index_v1"
STATEPASSING_FIXED_C_INDEX_FORMAT = "omegalax_pretrain_statepassing_fixed_c_index_v1"
STATEPASSING_INDEX_SHUFFLE_ROUNDS = 4


def _window_metadata_from_index_record(record: dict[str, Any]) -> WindowMetadata:
    eos_token_idx = record.get("eos_token_idx")
    return WindowMetadata(
        bucket_idx=int(record["bucket_idx"]),
        record_idx=int(record["record_idx"]),
        doc_id=str(record["doc_id"]),
        window_idx=int(record["window_idx"]),
        start_chunk=int(record["start_chunk"]),
        num_segments=int(record["num_segments"]),
        doc_token_count=int(record["doc_token_count"]),
        doc_num_chunks=int(record["doc_num_chunks"]),
        eos_token_idx=None if eos_token_idx is None else int(eos_token_idx),
    )


def build_statepassing_window_index(
    root: str | Path,
    out_dir: str | Path,
    *,
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
    num_segments: int,
    eos_id: int | None = DEFAULT_EOS_ID,
    split: str = DEFAULT_DATA_SET_SPLIT,
    records_per_shard: int = 100_000,
    overwrite: bool = False,
) -> Path:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be > 0")
    if num_segments <= 0:
        raise ValueError("num_segments must be > 0")

    reader = DataSetReader(root, split=split)
    dynamic_metadata: dict[str, Any] = {
        "format": STATEPASSING_WINDOW_INDEX_FORMAT,
        "data_set_root": str(reader.root),
        "split": reader.split,
        "bucket_names": list(reader.bucket_names),
        "chunk_length": int(chunk_length),
        "num_segments": int(num_segments),
        "eos_id": eos_id,
        "num_windows": 0,
        "num_residual_chunks": 0,
        "num_bucket_records": 0,
        "bucket_record_counts": [],
    }

    def _iter_index_records() -> Iterator[dict[str, Any]]:
        bucket_record_counts = [0 for _ in reader.bucket_paths]
        num_windows = 0
        num_residual_chunks = 0
        for bucket_idx, record_idx, doc in reader.iter_records():
            bucket_record_counts[bucket_idx] += 1
            doc_num_chunks = (doc.doc_token_count + chunk_length - 1) // chunk_length
            num_residual_chunks += doc_num_chunks % int(num_segments)
            for window_metadata in iter_document_window_metadata(
                doc,
                chunk_length=chunk_length,
                num_segments=num_segments,
                bucket_idx=bucket_idx,
                record_idx=record_idx,
                eos_id=eos_id,
            ):
                num_windows += 1
                yield window_metadata_to_record(window_metadata)
        dynamic_metadata["num_windows"] = num_windows
        dynamic_metadata["num_residual_chunks"] = num_residual_chunks
        dynamic_metadata["num_bucket_records"] = sum(bucket_record_counts)
        dynamic_metadata["bucket_record_counts"] = bucket_record_counts

    return write_json_arrayrecord_dataset(
        _iter_index_records(),
        out_dir,
        records_per_shard=records_per_shard,
        overwrite=overwrite,
        metadata=dynamic_metadata,
    )


def _validate_curriculum_orders(
    allocation_order: Sequence[int],
    train_order: Sequence[int],
) -> tuple[list[int], list[int]]:
    allocation = [int(value) for value in allocation_order]
    train = [int(value) for value in train_order]
    if not allocation:
        raise ValueError("allocation_order must not be empty")
    if any(value <= 0 for value in allocation):
        raise ValueError("allocation_order values must be > 0")
    if any(left <= right for left, right in zip(allocation, allocation[1:], strict=False)):
        raise ValueError("allocation_order must be strictly descending")
    if sorted(train) != sorted(allocation) or len(set(train)) != len(train):
        raise ValueError("train_order must contain exactly the same segment values")
    return allocation, train


def _curriculum_windows_per_step(
    *,
    num_segments: int,
    batch_size: int,
    grad_accum_steps: int,
) -> int:
    if batch_size <= 0:
        raise ValueError("curriculum_trim_batch_size must be > 0")
    if grad_accum_steps <= 0:
        raise ValueError("curriculum_trim_grad_accum_steps must be > 0")
    if int(batch_size) % int(num_segments):
        raise ValueError(
            f"curriculum_trim_batch_size must be divisible by num_segments={num_segments}"
        )
    segments_per_step = int(batch_size) * int(grad_accum_steps)
    if segments_per_step % int(num_segments):
        raise ValueError(
            "curriculum_trim_batch_size * curriculum_trim_grad_accum_steps "
            f"must be divisible by num_segments={num_segments}"
        )
    return segments_per_step // int(num_segments)


def _fixed_c_windows_per_step(
    *,
    num_segments: int,
    batch_size: int,
    grad_accum_steps: int,
) -> int:
    if num_segments <= 0:
        raise ValueError("num_segments must be > 0")
    if batch_size <= 0:
        raise ValueError("fixed_trim_batch_size must be > 0")
    if grad_accum_steps <= 0:
        raise ValueError("fixed_trim_grad_accum_steps must be > 0")
    if int(batch_size) % int(num_segments):
        raise ValueError(f"fixed_trim_batch_size must be divisible by num_segments={num_segments}")
    segments_per_step = int(batch_size) * int(grad_accum_steps)
    if segments_per_step % int(num_segments):
        raise ValueError(
            "fixed_trim_batch_size * fixed_trim_grad_accum_steps "
            f"must be divisible by num_segments={num_segments}"
        )
    return segments_per_step // int(num_segments)


def _prepare_curriculum_out_dir(out_dir: str | Path, *, overwrite: bool) -> Path:
    out_path = Path(out_dir).expanduser().resolve()
    if out_path.exists():
        has_contents = any(out_path.iterdir())
        if has_contents and not overwrite:
            raise ValueError(f"Refusing to overwrite non-empty output directory: {out_path}")
        if has_contents and overwrite:
            shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def _write_json_arrayrecord_dataset_or_empty(
    records: Iterator[dict[str, Any]],
    out_dir: str | Path,
    *,
    records_per_shard: int,
    metadata: dict[str, Any],
    num_records: int,
) -> Path:
    if num_records:
        return write_json_arrayrecord_dataset(
            records,
            out_dir,
            records_per_shard=records_per_shard,
            overwrite=False,
            metadata=metadata,
        )

    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    final_metadata = dict(metadata)
    final_metadata.update(
        {
            "version": DOC_CHAIN_DATASET_VERSION,
            "num_records": 0,
            "num_shards": 0,
            "shard_paths": [],
        }
    )
    (out_path / COMPILED_METADATA_FILENAME).write_text(json.dumps(final_metadata, indent=2) + "\n")
    return out_path


def _iter_jsonl_records(path: Path, *, limit: int) -> Iterator[dict[str, Any]]:
    with path.open() as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            yield json.loads(line)


def _record_chunk_end(record: dict[str, Any], *, chunk_length: int) -> int:
    end_chunk = int(record["start_chunk"]) + int(record["num_segments"])
    return min(end_chunk * int(chunk_length), int(record["doc_token_count"]))


def _eos_repair_by_doc(
    raw_paths: dict[int, Path],
    *,
    trim_counts: dict[int, int],
    allocation_order: Sequence[int],
    chunk_length: int,
) -> dict[tuple[int, int], int]:
    retained_ends: dict[tuple[int, int], tuple[int, int]] = {}
    for num_segments in allocation_order:
        for record in _iter_jsonl_records(
            raw_paths[int(num_segments)], limit=trim_counts[num_segments]
        ):
            if not record.get("_doc_ends_with_eos"):
                continue
            key = (int(record["bucket_idx"]), int(record["record_idx"]))
            retained_end = _record_chunk_end(record, chunk_length=chunk_length)
            previous = retained_ends.get(key)
            if previous is None or retained_end > previous[0]:
                retained_ends[key] = (retained_end, int(record["doc_token_count"]))

    return {
        key: retained_end - 1
        for key, (retained_end, doc_token_count) in retained_ends.items()
        if retained_end > 0 and retained_end < doc_token_count
    }


def _final_curriculum_record(
    raw_record: dict[str, Any],
    *,
    chunk_length: int,
    eos_repair: dict[tuple[int, int], int],
) -> dict[str, Any]:
    record = {key: value for key, value in raw_record.items() if not str(key).startswith("_")}
    key = (int(record["bucket_idx"]), int(record["record_idx"]))
    eos_token_idx = eos_repair.get(key)
    start = int(record["start_chunk"]) * int(chunk_length)
    end = _record_chunk_end(record, chunk_length=chunk_length)
    record["eos_token_idx"] = (
        eos_token_idx if eos_token_idx is not None and start <= eos_token_idx < end else None
    )
    return record


def _iter_final_curriculum_records(
    path: Path,
    *,
    limit: int,
    chunk_length: int,
    eos_repair: dict[tuple[int, int], int],
) -> Iterator[dict[str, Any]]:
    for raw_record in _iter_jsonl_records(path, limit=limit):
        yield _final_curriculum_record(
            raw_record,
            chunk_length=chunk_length,
            eos_repair=eos_repair,
        )


def _iter_iid_records_from_phase_records(
    raw_paths: dict[int, Path],
    *,
    trim_counts: dict[int, int],
    allocation_order: Sequence[int],
    chunk_length: int,
    eos_repair: dict[tuple[int, int], int],
) -> Iterator[dict[str, Any]]:
    iid_window_idx = 0
    for num_segments in allocation_order:
        for raw_record in _iter_jsonl_records(
            raw_paths[int(num_segments)], limit=trim_counts[num_segments]
        ):
            record = _final_curriculum_record(
                raw_record,
                chunk_length=chunk_length,
                eos_repair=eos_repair,
            )
            for chunk_offset in range(int(record["num_segments"])):
                chunk_idx = int(record["start_chunk"]) + chunk_offset
                start = chunk_idx * int(chunk_length)
                end = min(start + int(chunk_length), int(record["doc_token_count"]))
                eos_token_idx = record.get("eos_token_idx")
                yield {
                    "bucket_idx": int(record["bucket_idx"]),
                    "record_idx": int(record["record_idx"]),
                    "doc_id": str(record["doc_id"]),
                    "window_idx": iid_window_idx,
                    "start_chunk": chunk_idx,
                    "num_segments": 1,
                    "doc_token_count": int(record["doc_token_count"]),
                    "doc_num_chunks": int(record["doc_num_chunks"]),
                    "eos_token_idx": (
                        int(eos_token_idx)
                        if eos_token_idx is not None and start <= int(eos_token_idx) < end
                        else None
                    ),
                }
                iid_window_idx += 1


def build_statepassing_curriculum_indexes(
    root: str | Path,
    out_dir: str | Path,
    *,
    allocation_order: Sequence[int],
    train_order: Sequence[int],
    max_tokens_by_num_segments: dict[int, int] | None = None,
    trim_batch_size: int,
    trim_grad_accum_steps: int,
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
    eos_id: int | None = DEFAULT_EOS_ID,
    splits: Sequence[str] = (DEFAULT_DATA_SET_SPLIT,),
    records_per_shard: int = 100_000,
    overwrite: bool = False,
) -> Path:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be > 0")

    allocation, train = _validate_curriculum_orders(allocation_order, train_order)
    max_tokens_by_num_segments = {
        int(key): int(value) for key, value in (max_tokens_by_num_segments or {}).items()
    }
    unknown_caps = sorted(set(max_tokens_by_num_segments) - set(allocation))
    if unknown_caps:
        raise ValueError(f"curriculum_max_tokens contains unknown segment values: {unknown_caps}")
    if any(value < 0 for value in max_tokens_by_num_segments.values()):
        raise ValueError("curriculum_max_tokens values must be >= 0")

    windows_per_step = {
        num_segments: _curriculum_windows_per_step(
            num_segments=num_segments,
            batch_size=trim_batch_size,
            grad_accum_steps=trim_grad_accum_steps,
        )
        for num_segments in allocation
    }
    iid_chunks_per_step = int(trim_batch_size) * int(trim_grad_accum_steps)
    out_path = _prepare_curriculum_out_dir(out_dir, overwrite=overwrite)
    root_metadata: dict[str, Any] = {
        "format": STATEPASSING_CURRICULUM_INDEX_FORMAT,
        "allocation_order": allocation,
        "train_order": train,
        "chunk_length": int(chunk_length),
        "eos_id": eos_id,
        "trim_batch_size": int(trim_batch_size),
        "trim_grad_accum_steps": int(trim_grad_accum_steps),
        "windows_per_step": {str(key): value for key, value in windows_per_step.items()},
        "max_tokens_by_num_segments": {
            str(key): value for key, value in max_tokens_by_num_segments.items()
        },
        "splits": {},
    }

    for split in splits:
        reader = DataSetReader(root, split=split)
        if "data_set_root" not in root_metadata:
            root_metadata["data_set_root"] = str(reader.root)
            root_metadata["bucket_names"] = list(reader.bucket_names)
        elif root_metadata["data_set_root"] != str(reader.root):
            raise ValueError("All curriculum splits must use the same data_set_root")

        cap_windows_remaining = {
            num_segments: max_tokens // (num_segments * int(chunk_length))
            for num_segments, max_tokens in max_tokens_by_num_segments.items()
        }
        raw_counts = {num_segments: 0 for num_segments in allocation}
        bucket_record_counts = [0 for _ in reader.bucket_paths]

        with tempfile.TemporaryDirectory(dir=out_path) as tmp_name:
            tmp_path = Path(tmp_name)
            raw_paths = {
                num_segments: tmp_path / f"c{num_segments}.jsonl" for num_segments in allocation
            }
            with contextlib.ExitStack() as stack:
                writers = {
                    num_segments: stack.enter_context(raw_paths[num_segments].open("w"))
                    for num_segments in allocation
                }

                for bucket_idx, record_idx, doc in reader.iter_records():
                    bucket_record_counts[bucket_idx] += 1
                    if doc.doc_token_count <= 0:
                        continue
                    doc_num_chunks = (doc.doc_token_count + chunk_length - 1) // chunk_length
                    used_until = 0
                    window_idx_by_num_segments = {num_segments: 0 for num_segments in allocation}
                    doc_ends_with_eos = (
                        eos_id is not None
                        and doc.token_ids.size > 0
                        and int(doc.token_ids[-1]) == int(eos_id)
                    )

                    for num_segments in allocation:
                        remaining = doc_num_chunks - used_until
                        if remaining < num_segments:
                            continue
                        num_windows = remaining // num_segments
                        cap_remaining = cap_windows_remaining.get(num_segments)
                        if cap_remaining is not None:
                            num_windows = min(num_windows, cap_remaining)
                        if num_windows <= 0:
                            continue

                        for offset in range(num_windows):
                            window_metadata = WindowMetadata(
                                bucket_idx=bucket_idx,
                                record_idx=record_idx,
                                doc_id=doc.doc_id,
                                window_idx=window_idx_by_num_segments[num_segments] + offset,
                                start_chunk=used_until + offset * num_segments,
                                num_segments=num_segments,
                                doc_token_count=doc.doc_token_count,
                                doc_num_chunks=doc_num_chunks,
                                eos_token_idx=None,
                            )
                            record = window_metadata_to_record(window_metadata)
                            record["_doc_ends_with_eos"] = doc_ends_with_eos
                            writers[num_segments].write(json.dumps(record, sort_keys=True) + "\n")

                        raw_counts[num_segments] += num_windows
                        window_idx_by_num_segments[num_segments] += num_windows
                        if num_segments in cap_windows_remaining:
                            cap_windows_remaining[num_segments] -= num_windows
                        used_until += num_windows * num_segments

            trim_counts = {
                num_segments: (raw_counts[num_segments] // windows_per_step[num_segments])
                * windows_per_step[num_segments]
                for num_segments in allocation
            }
            phase_steps = {
                num_segments: trim_counts[num_segments] // windows_per_step[num_segments]
                for num_segments in allocation
            }
            eos_repair = _eos_repair_by_doc(
                raw_paths,
                trim_counts=trim_counts,
                allocation_order=allocation,
                chunk_length=chunk_length,
            )
            split_metadata: dict[str, Any] = {
                "bucket_record_counts": bucket_record_counts,
                "num_bucket_records": sum(bucket_record_counts),
                "phases": {},
            }

            for num_segments in allocation:
                phase_metadata = {
                    "format": STATEPASSING_WINDOW_INDEX_FORMAT,
                    "data_set_root": str(reader.root),
                    "split": reader.split,
                    "bucket_names": list(reader.bucket_names),
                    "chunk_length": int(chunk_length),
                    "num_segments": int(num_segments),
                    "eos_id": eos_id,
                    "num_windows": int(trim_counts[num_segments]),
                    "num_residual_chunks": 0,
                    "num_bucket_records": sum(bucket_record_counts),
                    "bucket_record_counts": bucket_record_counts,
                    "curriculum_format": STATEPASSING_CURRICULUM_INDEX_FORMAT,
                    "phase_steps": int(phase_steps[num_segments]),
                    "windows_per_step": int(windows_per_step[num_segments]),
                }
                phase_path = out_path / f"c{num_segments}" / reader.split
                _write_json_arrayrecord_dataset_or_empty(
                    _iter_final_curriculum_records(
                        raw_paths[num_segments],
                        limit=trim_counts[num_segments],
                        chunk_length=chunk_length,
                        eos_repair=eos_repair,
                    ),
                    phase_path,
                    records_per_shard=records_per_shard,
                    metadata=phase_metadata,
                    num_records=trim_counts[num_segments],
                )
                split_metadata["phases"][str(num_segments)] = {
                    "path": str(phase_path.relative_to(out_path)),
                    "num_windows": int(trim_counts[num_segments]),
                    "raw_num_windows": int(raw_counts[num_segments]),
                    "phase_steps": int(phase_steps[num_segments]),
                    "windows_per_step": int(windows_per_step[num_segments]),
                }

            iid_num_chunks = sum(
                trim_counts[num_segments] * num_segments for num_segments in allocation
            )
            iid_steps = iid_num_chunks // iid_chunks_per_step
            iid_metadata = {
                "format": STATEPASSING_WINDOW_INDEX_FORMAT,
                "data_set_root": str(reader.root),
                "split": reader.split,
                "bucket_names": list(reader.bucket_names),
                "chunk_length": int(chunk_length),
                "num_segments": 1,
                "eos_id": eos_id,
                "num_windows": int(iid_num_chunks),
                "num_residual_chunks": 0,
                "num_bucket_records": sum(bucket_record_counts),
                "bucket_record_counts": bucket_record_counts,
                "curriculum_format": STATEPASSING_CURRICULUM_INDEX_FORMAT,
                "iid_steps": int(iid_steps),
                "chunks_per_step": int(iid_chunks_per_step),
            }
            iid_path = out_path / "iid" / reader.split
            _write_json_arrayrecord_dataset_or_empty(
                _iter_iid_records_from_phase_records(
                    raw_paths,
                    trim_counts=trim_counts,
                    allocation_order=allocation,
                    chunk_length=chunk_length,
                    eos_repair=eos_repair,
                ),
                iid_path,
                records_per_shard=records_per_shard,
                metadata=iid_metadata,
                num_records=iid_num_chunks,
            )
            split_metadata["iid"] = {
                "path": str(iid_path.relative_to(out_path)),
                "num_chunks": int(iid_num_chunks),
                "iid_steps": int(iid_steps),
                "chunks_per_step": int(iid_chunks_per_step),
            }
            root_metadata["splits"][reader.split] = split_metadata

    (out_path / COMPILED_METADATA_FILENAME).write_text(json.dumps(root_metadata, indent=2) + "\n")
    return out_path


def build_statepassing_fixed_c_indexes(
    root: str | Path,
    out_dir: str | Path,
    *,
    num_segments: int,
    trim_batch_size: int,
    trim_grad_accum_steps: int,
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
    eos_id: int | None = DEFAULT_EOS_ID,
    splits: Sequence[str] = (DEFAULT_DATA_SET_SPLIT,),
    records_per_shard: int = 100_000,
    overwrite: bool = False,
) -> Path:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be > 0")

    num_segments = int(num_segments)
    windows_per_step = _fixed_c_windows_per_step(
        num_segments=num_segments,
        batch_size=trim_batch_size,
        grad_accum_steps=trim_grad_accum_steps,
    )
    iid_chunks_per_step = int(trim_batch_size) * int(trim_grad_accum_steps)
    out_path = _prepare_curriculum_out_dir(out_dir, overwrite=overwrite)
    root_metadata: dict[str, Any] = {
        "format": STATEPASSING_FIXED_C_INDEX_FORMAT,
        "chunk_length": int(chunk_length),
        "num_segments": num_segments,
        "eos_id": eos_id,
        "trim_batch_size": int(trim_batch_size),
        "trim_grad_accum_steps": int(trim_grad_accum_steps),
        "windows_per_step": int(windows_per_step),
        "chunks_per_step": int(iid_chunks_per_step),
        "splits": {},
    }

    for split in splits:
        reader = DataSetReader(root, split=split)
        if "data_set_root" not in root_metadata:
            root_metadata["data_set_root"] = str(reader.root)
            root_metadata["bucket_names"] = list(reader.bucket_names)
        elif root_metadata["data_set_root"] != str(reader.root):
            raise ValueError("All fixed-C splits must use the same data_set_root")

        raw_count = 0
        num_residual_chunks = 0
        bucket_record_counts = [0 for _ in reader.bucket_paths]

        with tempfile.TemporaryDirectory(dir=out_path) as tmp_name:
            raw_path = Path(tmp_name) / "windows.jsonl"
            with raw_path.open("w") as writer:
                for bucket_idx, record_idx, doc in reader.iter_records():
                    bucket_record_counts[bucket_idx] += 1
                    if doc.doc_token_count <= 0:
                        continue
                    doc_num_chunks = (doc.doc_token_count + chunk_length - 1) // chunk_length
                    num_windows = doc_num_chunks // num_segments
                    num_residual_chunks += doc_num_chunks % num_segments
                    doc_ends_with_eos = (
                        eos_id is not None
                        and doc.token_ids.size > 0
                        and int(doc.token_ids[-1]) == int(eos_id)
                    )

                    for window_idx in range(num_windows):
                        window_metadata = WindowMetadata(
                            bucket_idx=bucket_idx,
                            record_idx=record_idx,
                            doc_id=doc.doc_id,
                            window_idx=window_idx,
                            start_chunk=window_idx * num_segments,
                            num_segments=num_segments,
                            doc_token_count=doc.doc_token_count,
                            doc_num_chunks=doc_num_chunks,
                            eos_token_idx=None,
                        )
                        record = window_metadata_to_record(window_metadata)
                        record["_doc_ends_with_eos"] = doc_ends_with_eos
                        writer.write(json.dumps(record, sort_keys=True) + "\n")
                    raw_count += num_windows

            trim_count = (raw_count // windows_per_step) * windows_per_step
            statepassing_steps = trim_count // windows_per_step
            iid_num_chunks = trim_count * num_segments
            iid_steps = iid_num_chunks // iid_chunks_per_step
            trimmed_window_chunks = (raw_count - trim_count) * num_segments
            raw_paths = {num_segments: raw_path}
            trim_counts = {num_segments: trim_count}
            eos_repair = _eos_repair_by_doc(
                raw_paths,
                trim_counts=trim_counts,
                allocation_order=[num_segments],
                chunk_length=chunk_length,
            )

            statepassing_metadata = {
                "format": STATEPASSING_WINDOW_INDEX_FORMAT,
                "data_set_root": str(reader.root),
                "split": reader.split,
                "bucket_names": list(reader.bucket_names),
                "chunk_length": int(chunk_length),
                "num_segments": num_segments,
                "eos_id": eos_id,
                "num_windows": int(trim_count),
                "raw_num_windows": int(raw_count),
                "num_residual_chunks": int(num_residual_chunks + trimmed_window_chunks),
                "num_bucket_records": sum(bucket_record_counts),
                "bucket_record_counts": bucket_record_counts,
                "fixed_c_format": STATEPASSING_FIXED_C_INDEX_FORMAT,
                "statepassing_steps": int(statepassing_steps),
                "windows_per_step": int(windows_per_step),
            }
            statepassing_path = out_path / reader.split
            _write_json_arrayrecord_dataset_or_empty(
                _iter_final_curriculum_records(
                    raw_path,
                    limit=trim_count,
                    chunk_length=chunk_length,
                    eos_repair=eos_repair,
                ),
                statepassing_path,
                records_per_shard=records_per_shard,
                metadata=statepassing_metadata,
                num_records=trim_count,
            )

            iid_metadata = {
                "format": STATEPASSING_WINDOW_INDEX_FORMAT,
                "data_set_root": str(reader.root),
                "split": reader.split,
                "bucket_names": list(reader.bucket_names),
                "chunk_length": int(chunk_length),
                "num_segments": 1,
                "eos_id": eos_id,
                "num_windows": int(iid_num_chunks),
                "num_residual_chunks": 0,
                "num_bucket_records": sum(bucket_record_counts),
                "bucket_record_counts": bucket_record_counts,
                "fixed_c_format": STATEPASSING_FIXED_C_INDEX_FORMAT,
                "iid_steps": int(iid_steps),
                "chunks_per_step": int(iid_chunks_per_step),
            }
            iid_path = out_path / "iid" / reader.split
            _write_json_arrayrecord_dataset_or_empty(
                _iter_iid_records_from_phase_records(
                    raw_paths,
                    trim_counts=trim_counts,
                    allocation_order=[num_segments],
                    chunk_length=chunk_length,
                    eos_repair=eos_repair,
                ),
                iid_path,
                records_per_shard=records_per_shard,
                metadata=iid_metadata,
                num_records=iid_num_chunks,
            )

            split_metadata = {
                "path": str(statepassing_path.relative_to(out_path)),
                "num_windows": int(trim_count),
                "raw_num_windows": int(raw_count),
                "statepassing_steps": int(statepassing_steps),
                "windows_per_step": int(windows_per_step),
                "iid": {
                    "path": str(iid_path.relative_to(out_path)),
                    "num_chunks": int(iid_num_chunks),
                    "iid_steps": int(iid_steps),
                    "chunks_per_step": int(iid_chunks_per_step),
                },
            }
            root_metadata["splits"][reader.split] = split_metadata
            if reader.split == DEFAULT_DATA_SET_SPLIT:
                root_metadata["statepassing_steps"] = int(statepassing_steps)
                root_metadata["iid_steps"] = int(iid_steps)

    (out_path / COMPILED_METADATA_FILENAME).write_text(json.dumps(root_metadata, indent=2) + "\n")
    return out_path


def _load_window_index_metadata(
    index_path: str | Path,
    chunk_length: int | None,
) -> dict[str, Any]:
    metadata = load_arrayrecord_metadata(index_path)
    fmt = metadata.get("format")
    if fmt != STATEPASSING_WINDOW_INDEX_FORMAT:
        raise ValueError(f"Expected {STATEPASSING_WINDOW_INDEX_FORMAT} dataset, got format={fmt}")
    index_chunk_length = int(metadata["chunk_length"])
    if chunk_length is not None and int(chunk_length) != index_chunk_length:
        raise ValueError(
            f"chunk_length mismatch: index has {index_chunk_length}, loader got {chunk_length}"
        )
    return metadata


def _is_statepassing_window_index(path: str | Path) -> bool:
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        return False
    try:
        return load_arrayrecord_metadata(path).get("format") == STATEPASSING_WINDOW_INDEX_FORMAT
    except ValueError:
        return False


def _single_statepassing_window_index_path(
    indexes: str | Path | Sequence[str | Path],
) -> Path | None:
    if isinstance(indexes, (str, Path)):
        path = Path(indexes).expanduser().resolve()
        return path if _is_statepassing_window_index(path) else None
    if isinstance(indexes, Sequence) and len(indexes) == 1:
        path = Path(indexes[0]).expanduser().resolve()
        return path if _is_statepassing_window_index(path) else None
    return None


def _make_window_batch(
    window_metadata_batch: Sequence[WindowMetadata],
    *,
    reader: DataSetReader,
    chunk_length: int,
    pad_id: int,
    eos_id: int | None,
) -> dict[str, Any]:
    token_ids = []
    attention_masks = []
    loss_masks = []
    chunk_indices = []
    reset_states = []
    doc_ids = []
    bucket_indices = []
    record_indices = []
    window_indices = []
    doc_cache = {}

    for window_metadata in window_metadata_batch:
        doc_key = (window_metadata.bucket_idx, window_metadata.record_idx)
        doc = doc_cache.get(doc_key)
        if doc is None:
            doc = reader.read(window_metadata.bucket_idx, window_metadata.record_idx)
            doc_cache[doc_key] = doc
        if (
            doc.doc_id != window_metadata.doc_id
            or doc.doc_token_count != window_metadata.doc_token_count
        ):
            raise ValueError(
                "Statepassing window index does not match bucket record "
                f"bucket_idx={window_metadata.bucket_idx}, record_idx={window_metadata.record_idx}"
            )
        arrays = build_window_arrays(
            doc.token_ids,
            window_metadata,
            chunk_length=chunk_length,
            pad_id=pad_id,
            eos_id=eos_id,
        )
        token_ids.append(arrays["token_ids_CT"])
        attention_masks.append(arrays["attention_mask_CT"])
        loss_masks.append(arrays["loss_mask_CT"])
        chunk_indices.append(arrays["chunk_idx_C"])
        reset_states.append(arrays["reset_state_C"])
        doc_ids.append(window_metadata.doc_id)
        bucket_indices.append(window_metadata.bucket_idx)
        record_indices.append(window_metadata.record_idx)
        window_indices.append(window_metadata.window_idx)

    return {
        "token_ids_BCT": np.stack(token_ids).astype(np.int32, copy=False),
        "attention_mask_BCT": np.stack(attention_masks).astype(np.int32, copy=False),
        "loss_mask_BCT": np.stack(loss_masks).astype(np.int32, copy=False),
        "chunk_idx_BC": np.stack(chunk_indices).astype(np.int32, copy=False),
        "reset_state_BC": np.stack(reset_states).astype(np.bool_, copy=False),
        BATCH_PRETRAIN_METADATA_KEY: {
            "doc_ids": doc_ids,
            "bucket_idx_B": np.asarray(bucket_indices, dtype=np.int32),
            "record_idx_B": np.asarray(record_indices, dtype=np.int32),
            "window_idx_B": np.asarray(window_indices, dtype=np.int32),
        },
    }


class _StatepassingWindowPretrainBatchBuilder:
    def __init__(
        self,
        *,
        data_set_root: str | Path,
        split: str,
        bucket_names: Sequence[str],
        chunk_length: int,
        pad_id: int,
        eos_id: int | None = DEFAULT_EOS_ID,
    ) -> None:
        self.data_set_root = str(data_set_root)
        self.split = str(split)
        self.bucket_names = list(bucket_names)
        self.chunk_length = int(chunk_length)
        self.pad_id = int(pad_id)
        self.eos_id = eos_id
        self.reader: DataSetReader | None = None

    def __call__(self, index_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if self.reader is None:
            reader = DataSetReader(self.data_set_root, split=self.split)
            if reader.bucket_names != self.bucket_names:
                raise ValueError(
                    "Statepassing window index bucket_names do not match data-set root"
                )
            self.reader = reader
        window_metadata_batch = [
            _window_metadata_from_index_record(record) for record in index_records
        ]
        return _make_window_batch(
            window_metadata_batch,
            reader=self.reader,
            chunk_length=self.chunk_length,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
        )


def make_statepassing_iterator(
    indexes: str | Path | Sequence[str | Path],
    *,
    batch_size: int,
    chunk_length: int | None = DEFAULT_CHUNK_LENGTH,
    pad_id: int = DEFAULT_PAD_ID,
    eos_id: int | None = DEFAULT_EOS_ID,
    shuffle: bool = True,
    seed: int = 0,
    num_epochs: int | None = None,
    dp_size: int,
    fsdp_size: int,
    dp_index: int | None = None,
    process_index: int | None = None,
    grain_workers: int = 8,
    grain_worker_buffer_size: int = 1,
    grain_read_threads: int = 2,
    grain_read_prefetch_buffer_size: int = 4,
) -> Iterator[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if grain_workers < 0:
        raise ValueError("grain_workers must be >= 0")
    if grain_worker_buffer_size <= 0:
        raise ValueError("grain_worker_buffer_size must be > 0")
    if grain_read_threads <= 0:
        raise ValueError("grain_read_threads must be > 0")
    if grain_read_prefetch_buffer_size <= 0:
        raise ValueError("grain_read_prefetch_buffer_size must be > 0")

    window_index_path = _single_statepassing_window_index_path(indexes)
    if window_index_path is None:
        raise ValueError(
            "Statepassing pretraining requires a statepassing window index; "
            "call build_statepassing_window_index first."
        )

    effective_dp_size, resolved_dp_index = resolve_pretrain_dp(
        dp_size=dp_size,
        fsdp_size=fsdp_size,
        process_index=process_index,
    )
    if dp_index is not None:
        resolved_dp_index = int(dp_index)
    if resolved_dp_index < 0 or resolved_dp_index >= effective_dp_size:
        raise ValueError(f"dp_index must be in [0, {effective_dp_size}), got {resolved_dp_index}")

    index_path = window_index_path
    index_metadata = _load_window_index_metadata(index_path, chunk_length)
    index_num_segments = int(index_metadata["num_segments"])
    if batch_size % index_num_segments:
        raise ValueError("batch_size must be divisible by num_segments")
    records_per_local_batch = batch_size // index_num_segments

    index_chunk_length = int(index_metadata["chunk_length"])
    index_eos_id = index_metadata.get("eos_id")
    if eos_id != index_eos_id:
        raise ValueError(f"eos_id mismatch: index has {index_eos_id}, loader got {eos_id}")

    raw_dataset_root = index_metadata.get("data_set_root")
    if raw_dataset_root is None:
        raise ValueError("Statepassing window index metadata is missing data_set_root")
    data_set_root = rewrite_data_set_root_path(raw_dataset_root)
    split = str(index_metadata["split"])
    bucket_names = [str(name) for name in index_metadata["bucket_names"]]
    index_shard_paths = resolve_arrayrecord_paths(index_path)
    index_source = grain.sources.ArrayRecordDataSource([str(path) for path in index_shard_paths])
    num_records = len(index_source)
    if num_records == 0:
        raise ValueError(f"Statepassing window index has no records: {index_path}")

    dataset_index, _ = make_dataset_index(
        index_shard_paths=index_shard_paths,
        num_records=num_records,
        num_epochs=num_epochs,
        dp_size=effective_dp_size,
        dp_index=resolved_dp_index,
        records_per_local_batch=records_per_local_batch,
        shuffle=shuffle,
        seed=seed,
        shuffle_rounds=STATEPASSING_INDEX_SHUFFLE_ROUNDS,
    )
    batched = dataset_index.batch(
        batch_size=records_per_local_batch,
        drop_remainder=True,
        batch_fn=_StatepassingWindowPretrainBatchBuilder(
            data_set_root=data_set_root,
            split=split,
            bucket_names=bucket_names,
            chunk_length=index_chunk_length,
            pad_id=pad_id,
            eos_id=eos_id,
        ),
    )
    iter_dataset = batched.to_iter_dataset(
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
