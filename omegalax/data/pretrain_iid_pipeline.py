"""IID-style chunk iteration over fixed-window pretraining indexes."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import grain
import numpy as np

from omegalax.data.pretrain_data_set import (
    DEFAULT_EOS_ID,
    DEFAULT_PAD_ID,
    DEFAULT_CHUNK_LENGTH,
    BATCH_PRETRAIN_METADATA_KEY,
    DataSetReader,
    build_chunk_arrays,
    calculate_samples_per_process,
    load_arrayrecord_metadata,
    num_pretrain_positions,
    num_pretrain_records_usable,
    resolve_arrayrecord_paths,
    resolve_pretrain_dp,
    rewrite_data_set_root_path,
)
from omegalax.data.pretrain_statepassing import STATEPASSING_WINDOW_INDEX_FORMAT

IID_INDEX_SHUFFLE_ROUNDS = 4


def _load_window_index_metadata(index_path: str | Path, chunk_length: int | None) -> dict[str, Any]:
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


class _WindowChunkRecordLookup:
    def __init__(
        self,
        *,
        index_shard_paths: Sequence[str | Path],
        num_windows: int,
        num_segments: int,
        chunk_length: int,
        epoch_samples_per_process: int,
        dp_size: int,
        dp_index: int,
        shuffle: bool,
        seed: int,
        shuffle_rounds: int,
    ) -> None:
        self.index_shard_paths = [
            str(Path(path).expanduser().resolve()) for path in index_shard_paths
        ]
        self.num_windows = int(num_windows)
        self.num_segments = int(num_segments)
        self.chunk_length = int(chunk_length)
        self.num_records = self.num_windows * self.num_segments
        self.epoch_samples_per_process = int(epoch_samples_per_process)
        self.dp_size = int(dp_size)
        self.dp_index = int(dp_index)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.shuffle_rounds = int(shuffle_rounds)
        self._source: grain.sources.ArrayRecordDataSource | None = None

    def _index_source(self) -> grain.sources.ArrayRecordDataSource:
        if self._source is None:
            self._source = grain.sources.ArrayRecordDataSource(self.index_shard_paths)
        return self._source

    def __call__(self, absolute_pos: int) -> dict[str, Any]:
        epoch, local_pos = divmod(int(absolute_pos), self.epoch_samples_per_process)
        global_pos = self.dp_index + local_pos * self.dp_size
        if self.shuffle:
            virtual_idx = grain.experimental.index_shuffle(
                global_pos,
                self.num_records - 1,
                self.seed + epoch,
                self.shuffle_rounds,
            )
        else:
            virtual_idx = global_pos

        window_record_idx, chunk_offset = divmod(int(virtual_idx), self.num_segments)
        window_record = json.loads(self._index_source()[window_record_idx])
        chunk_idx = int(window_record["start_chunk"]) + chunk_offset
        start = chunk_idx * self.chunk_length
        end = min(start + self.chunk_length, int(window_record["doc_token_count"]))
        eos_token_idx = window_record.get("eos_token_idx")
        if eos_token_idx is not None:
            eos_token_idx = int(eos_token_idx)
            if eos_token_idx < start or eos_token_idx >= end:
                eos_token_idx = None

        return {
            "bucket_idx": int(window_record["bucket_idx"]),
            "record_idx": int(window_record["record_idx"]),
            "doc_id": str(window_record["doc_id"]),
            "window_idx": int(window_record["window_idx"]),
            "chunk_offset": int(chunk_offset),
            "chunk_idx": int(chunk_idx),
            "start": int(start),
            "end": int(end),
            "doc_token_count": int(window_record["doc_token_count"]),
            "eos_token_idx": eos_token_idx,
        }


def _make_window_chunk_dataset_index(
    *,
    index_shard_paths: Sequence[str | Path],
    num_windows: int,
    num_segments: int,
    chunk_length: int,
    num_epochs: int | None,
    dp_size: int,
    dp_index: int,
    shuffle: bool,
    seed: int,
    shuffle_rounds: int,
    records_per_local_batch: int,
) -> tuple[grain.MapDataset, int]:
    num_records = int(num_windows) * int(num_segments)
    usable_records = num_pretrain_records_usable(
        num_records=num_records,
        dp_size=dp_size,
        records_per_local_batch=records_per_local_batch,
    )
    epoch_samples_per_process = calculate_samples_per_process(
        num_records=usable_records,
        dp_size=dp_size,
        dp_index=dp_index,
    )
    if not epoch_samples_per_process:
        raise ValueError(
            f"No complete pretrain batch assigned to dp_index={dp_index} "
            f"with dp_size={dp_size} and records_per_local_batch={records_per_local_batch}"
        )
    total_samples_per_process = num_pretrain_positions(
        epoch_samples_per_process=epoch_samples_per_process,
        num_epochs=num_epochs,
    )
    dataset_index = grain.MapDataset.range(total_samples_per_process).map(
        _WindowChunkRecordLookup(
            index_shard_paths=index_shard_paths,
            num_windows=num_windows,
            num_segments=num_segments,
            chunk_length=chunk_length,
            epoch_samples_per_process=epoch_samples_per_process,
            dp_size=dp_size,
            dp_index=dp_index,
            shuffle=shuffle,
            seed=seed,
            shuffle_rounds=shuffle_rounds,
        )
    )
    return dataset_index, epoch_samples_per_process


def _make_batch(
    entries: Sequence[dict[str, Any]],
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
    doc_ids = []
    bucket_indices = []
    record_indices = []
    window_indices = []
    chunk_offsets = []
    doc_cache = {}

    for entry in entries:
        bucket_idx = int(entry["bucket_idx"])
        record_idx = int(entry["record_idx"])
        doc_key = (bucket_idx, record_idx)
        doc = doc_cache.get(doc_key)
        if doc is None:
            doc = reader.read(bucket_idx, record_idx)
            doc_cache[doc_key] = doc
        if doc.doc_id != str(entry["doc_id"]) or (
            "doc_token_count" in entry and doc.doc_token_count != int(entry["doc_token_count"])
        ):
            raise ValueError(
                "IID window index does not match bucket record "
                f"bucket_idx={bucket_idx}, record_idx={record_idx}"
            )
        arrays = build_chunk_arrays(
            doc.token_ids,
            start=int(entry["start"]),
            end=int(entry["end"]),
            chunk_length=chunk_length,
            pad_id=pad_id,
            eos_id=eos_id,
            eos_token_idx=entry.get("eos_token_idx"),
        )
        token_ids.append(arrays["token_ids_T"])
        attention_masks.append(arrays["attention_mask_T"])
        loss_masks.append(arrays["loss_mask_T"])
        chunk_indices.append(int(entry["chunk_idx"]))
        doc_ids.append(str(entry["doc_id"]))
        bucket_indices.append(bucket_idx)
        record_indices.append(record_idx)
        window_indices.append(int(entry["window_idx"]))
        chunk_offsets.append(int(entry["chunk_offset"]))

    metadata = {
        "doc_ids": doc_ids,
        "bucket_idx_B": np.asarray(bucket_indices, dtype=np.int32),
        "record_idx_B": np.asarray(record_indices, dtype=np.int32),
        "window_idx_B": np.asarray(window_indices, dtype=np.int32),
        "chunk_offset_B": np.asarray(chunk_offsets, dtype=np.int32),
    }

    return {
        "token_ids_BT": np.stack(token_ids).astype(np.int32, copy=False),
        "attention_mask_BT": np.stack(attention_masks).astype(np.int32, copy=False),
        "loss_mask_BT": np.stack(loss_masks).astype(np.int32, copy=False),
        "chunk_idx_B": np.asarray(chunk_indices, dtype=np.int32),
        BATCH_PRETRAIN_METADATA_KEY: metadata,
    }


class _IIDPretrainBatchBuilder:
    def __init__(
        self,
        *,
        data_set_root: str | Path,
        split: str,
        bucket_names: Sequence[str],
        chunk_length: int,
        pad_id: int,
        eos_id: int | None,
    ) -> None:
        self.data_set_root = str(data_set_root)
        self.split = str(split)
        self.bucket_names = list(bucket_names)
        self.chunk_length = int(chunk_length)
        self.pad_id = int(pad_id)
        self.eos_id = eos_id
        self.reader: DataSetReader | None = None

    def __call__(self, entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if self.reader is None:
            reader = DataSetReader(self.data_set_root, split=self.split)
            if reader.bucket_names != self.bucket_names:
                raise ValueError("IID window index bucket_names do not match data-set root")
            self.reader = reader
        return _make_batch(
            entries,
            reader=self.reader,
            chunk_length=self.chunk_length,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
        )


def make_iid_iterator(
    index_path: str | Path,
    *,
    batch_size: int,
    chunk_length: int | None = DEFAULT_CHUNK_LENGTH,
    pad_id: int = DEFAULT_PAD_ID,
    eos_id: int | None = DEFAULT_EOS_ID,
    shuffle: bool = True,
    seed: int = 0,
    num_epochs: int | None = None,
    dp_size: int = 1,
    fsdp_size: int = 1,
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

    effective_dp_size, resolved_dp_index = resolve_pretrain_dp(
        dp_size=dp_size,
        fsdp_size=fsdp_size,
        process_index=process_index,
    )
    if dp_index is not None:
        resolved_dp_index = int(dp_index)
    if resolved_dp_index < 0 or resolved_dp_index >= effective_dp_size:
        raise ValueError(f"dp_index must be in [0, {effective_dp_size}), got {resolved_dp_index}")

    index_path = Path(index_path).expanduser().resolve()
    metadata = _load_window_index_metadata(index_path, chunk_length)
    index_chunk_length = int(metadata["chunk_length"])
    index_eos_id = metadata.get("eos_id")
    if eos_id != index_eos_id:
        raise ValueError(f"eos_id mismatch: index has {index_eos_id}, loader got {eos_id}")

    raw_dataset_root = metadata.get("data_set_root")
    if raw_dataset_root is None:
        raise ValueError("IID window index metadata is missing data_set_root")
    data_set_root = rewrite_data_set_root_path(raw_dataset_root)
    split = str(metadata["split"])
    bucket_names = [str(name) for name in metadata["bucket_names"]]
    index_shard_paths = resolve_arrayrecord_paths(index_path)
    index_source = grain.sources.ArrayRecordDataSource([str(path) for path in index_shard_paths])
    num_records = len(index_source)
    if num_records == 0:
        raise ValueError(f"Pretrain index has no records: {index_path}")

    dataset_index, _ = _make_window_chunk_dataset_index(
        index_shard_paths=index_shard_paths,
        num_windows=num_records,
        num_segments=int(metadata["num_segments"]),
        chunk_length=index_chunk_length,
        num_epochs=num_epochs,
        dp_size=effective_dp_size,
        dp_index=resolved_dp_index,
        records_per_local_batch=batch_size,
        shuffle=shuffle,
        seed=seed,
        shuffle_rounds=IID_INDEX_SHUFFLE_ROUNDS,
    )
    batched = dataset_index.batch(
        batch_size=batch_size,
        drop_remainder=True,
        batch_fn=_IIDPretrainBatchBuilder(
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
