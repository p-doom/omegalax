"""Fixed-window pretraining iterators for state passing."""

from __future__ import annotations

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
    DataSetReader,
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
