"""IID-style chunk iteration over fixed-window pretraining indexes."""

from __future__ import annotations

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
    load_arrayrecord_metadata,
    make_dataset_index,
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
        chunk_idx = int(entry["start_chunk"])
        start = chunk_idx * int(chunk_length)
        end = min(start + int(chunk_length), int(entry["doc_token_count"]))
        eos_token_idx = entry.get("eos_token_idx")
        if eos_token_idx is not None:
            eos_token_idx = int(eos_token_idx)
            if eos_token_idx < start or eos_token_idx >= end:
                eos_token_idx = None
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
            start=start,
            end=end,
            chunk_length=chunk_length,
            pad_id=pad_id,
            eos_id=eos_id,
            eos_token_idx=eos_token_idx,
        )
        token_ids.append(arrays["token_ids_T"])
        attention_masks.append(arrays["attention_mask_T"])
        loss_masks.append(arrays["loss_mask_T"])
        chunk_indices.append(chunk_idx)
        doc_ids.append(str(entry["doc_id"]))
        bucket_indices.append(bucket_idx)
        record_indices.append(record_idx)
        window_indices.append(int(entry["window_idx"]))
        chunk_offsets.append(0)

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
    index_num_segments = int(metadata["num_segments"])
    if index_num_segments != 1:
        raise ValueError(
            "IID pretraining requires a physical num_segments=1 index; "
            "use the bundle's iid/{split} child path instead of a C>1 statepassing "
            f"index (got num_segments={index_num_segments}). Bundle roots are resolved "
            "only by scripts/train_text_pretrain.py."
        )
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

    dataset_index, _ = make_dataset_index(
        index_shard_paths=index_shard_paths,
        num_records=num_records,
        num_epochs=num_epochs,
        dp_size=effective_dp_size,
        dp_index=resolved_dp_index,
        shuffle=shuffle,
        seed=seed,
        shuffle_rounds=IID_INDEX_SHUFFLE_ROUNDS,
        records_per_local_batch=batch_size,
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
