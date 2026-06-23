"""IID-style chunk indexing and iteration for pretraining documents."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import grain
import numpy as np

from omegalax.data.pretrain_data_set import (
    DEFAULT_EOS_ID,
    DEFAULT_PAD_ID,
    DEFAULT_DOC_CHAIN_SPLIT,
    DEFAULT_SEGMENT_LENGTH,
    BATCH_PRETRAIN_METADATA_KEY,
    DataSetReader,
    build_chunk_arrays,
    flatten_pair_ref,
    iter_document_pair_refs,
    load_arrayrecord_metadata,
    make_pretrain_index_record_dataset,
    resolve_arrayrecord_paths,
    resolve_pretrain_dp,
    rewrite_data_set_root_path,
    write_json_arrayrecord_dataset,
)

IID_CHUNK_INDEX_FORMAT = "omegalax_pretrain_iid_chunk_index_v1"
IID_INDEX_SHUFFLE_ROUNDS = 4


def build_iid_chunk_index(
    root: str | Path,
    out_dir: str | Path,
    *,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    eos_id: int | None = DEFAULT_EOS_ID,
    split: str = DEFAULT_DOC_CHAIN_SPLIT,
    records_per_shard: int = 100_000,
    overwrite: bool = False,
) -> Path:
    if segment_length <= 0:
        raise ValueError("segment_length must be > 0")

    reader = DataSetReader(root, split=split)
    dynamic_metadata: dict[str, Any] = {
        "format": IID_CHUNK_INDEX_FORMAT,
        "data_set_root": str(reader.root),
        "split": reader.split,
        "bucket_names": list(reader.bucket_names),
        "segment_length": int(segment_length),
        "eos_id": eos_id,
        "num_chunks": 0,
        "num_pairs": 0,
        "num_bucket_records": 0,
        "bucket_record_counts": [],
    }

    def _iter_index_records() -> Iterator[dict[str, Any]]:
        bucket_record_counts = [0 for _ in reader.bucket_paths]
        num_chunks = 0
        num_pairs = 0
        for bucket_idx, record_idx, doc in reader.iter_records():
            bucket_record_counts[bucket_idx] += 1
            for pair in iter_document_pair_refs(
                doc,
                segment_length=segment_length,
                bucket_idx=bucket_idx,
                record_idx=record_idx,
                eos_id=eos_id,
            ):
                num_pairs += 1
                for chunk in flatten_pair_ref(pair):
                    num_chunks += 1
                    yield chunk
        dynamic_metadata["num_chunks"] = num_chunks
        dynamic_metadata["num_pairs"] = num_pairs
        dynamic_metadata["num_bucket_records"] = sum(bucket_record_counts)
        dynamic_metadata["bucket_record_counts"] = bucket_record_counts

    return write_json_arrayrecord_dataset(
        _iter_index_records(),
        out_dir,
        records_per_shard=records_per_shard,
        overwrite=overwrite,
        metadata=dynamic_metadata,
    )


def _load_index_metadata(index_path: str | Path, segment_length: int | None) -> dict[str, Any]:
    metadata = load_arrayrecord_metadata(index_path)
    fmt = metadata.get("format")
    if fmt != IID_CHUNK_INDEX_FORMAT:
        raise ValueError(f"Expected {IID_CHUNK_INDEX_FORMAT} dataset, got format={fmt}")
    index_segment_length = int(metadata["segment_length"])
    if segment_length is not None and int(segment_length) != index_segment_length:
        raise ValueError(
            f"segment_length mismatch: index has {index_segment_length}, "
            f"loader got {segment_length}"
        )
    return metadata


def _make_batch(
    entries: Sequence[dict[str, Any]],
    *,
    reader: DataSetReader,
    segment_length: int,
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
    pair_indices = []
    segments_in_pair = []
    doc_cache = {}

    for entry in entries:
        bucket_idx = int(entry["bucket_idx"])
        record_idx = int(entry["record_idx"])
        doc_key = (bucket_idx, record_idx)
        doc = doc_cache.get(doc_key)
        if doc is None:
            doc = reader.read(bucket_idx, record_idx)
            doc_cache[doc_key] = doc
        if doc.doc_id != str(entry["doc_id"]):
            raise ValueError(
                "IID chunk index does not match bucket record "
                f"bucket_idx={bucket_idx}, record_idx={record_idx}"
            )
        arrays = build_chunk_arrays(
            doc.token_ids,
            start=int(entry["start"]),
            end=int(entry["end"]),
            segment_length=segment_length,
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
        pair_indices.append(int(entry["pair_idx"]))
        segments_in_pair.append(int(entry["segment_in_pair"]))

    return {
        "token_ids_BT": np.stack(token_ids).astype(np.int32, copy=False),
        "attention_mask_BT": np.stack(attention_masks).astype(np.int32, copy=False),
        "loss_mask_BT": np.stack(loss_masks).astype(np.int32, copy=False),
        "chunk_idx_B": np.asarray(chunk_indices, dtype=np.int32),
        BATCH_PRETRAIN_METADATA_KEY: {
            "doc_ids": doc_ids,
            "bucket_idx_B": np.asarray(bucket_indices, dtype=np.int32),
            "record_idx_B": np.asarray(record_indices, dtype=np.int32),
            "pair_idx_B": np.asarray(pair_indices, dtype=np.int32),
            "segment_in_pair_B": np.asarray(segments_in_pair, dtype=np.int32),
        },
    }


class _IIDPretrainBatchBuilder:
    def __init__(
        self,
        *,
        data_set_root: str | Path,
        split: str,
        bucket_names: Sequence[str],
        segment_length: int,
        pad_id: int,
        eos_id: int | None,
    ) -> None:
        self.data_set_root = str(data_set_root)
        self.split = str(split)
        self.bucket_names = list(bucket_names)
        self.segment_length = int(segment_length)
        self.pad_id = int(pad_id)
        self.eos_id = eos_id
        self.reader: DataSetReader | None = None

    def __call__(self, entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if self.reader is None:
            reader = DataSetReader(self.data_set_root, split=self.split)
            if reader.bucket_names != self.bucket_names:
                raise ValueError("IID chunk index bucket_names do not match data-set root")
            self.reader = reader
        return _make_batch(
            entries,
            reader=self.reader,
            segment_length=self.segment_length,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
        )


def make_iid_iterator(
    index_path: str | Path,
    *,
    batch_size: int,
    segment_length: int | None = DEFAULT_SEGMENT_LENGTH,
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
    grain_read_threads: int = 16,
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
    metadata = _load_index_metadata(index_path, segment_length)
    index_segment_length = int(metadata["segment_length"])
    index_eos_id = metadata.get("eos_id")
    if eos_id != index_eos_id:
        raise ValueError(f"eos_id mismatch: index has {index_eos_id}, loader got {eos_id}")

    metadata_root = metadata.get("data_set_root", metadata.get("doc_chain_root"))
    if metadata_root is None:
        raise ValueError("IID chunk index metadata is missing data_set_root")
    data_set_root = rewrite_data_set_root_path(metadata_root)
    split = str(metadata["split"])
    bucket_names = [str(name) for name in metadata["bucket_names"]]
    index_shard_paths = resolve_arrayrecord_paths(index_path)
    index_source = grain.sources.ArrayRecordDataSource([str(path) for path in index_shard_paths])
    num_records = len(index_source)
    if num_records == 0:
        raise ValueError(f"IID chunk index has no records: {index_path}")

    dataset, _ = make_pretrain_index_record_dataset(
        index_shard_paths=index_shard_paths,
        num_records=num_records,
        num_epochs=num_epochs,
        dp_size=effective_dp_size,
        dp_index=resolved_dp_index,
        records_per_local_batch=batch_size,
        shuffle=shuffle,
        seed=seed,
        shuffle_rounds=IID_INDEX_SHUFFLE_ROUNDS,
    )
    batched = dataset.batch(
        batch_size=batch_size,
        drop_remainder=True,
        batch_fn=_IIDPretrainBatchBuilder(
            data_set_root=data_set_root,
            split=split,
            bucket_names=bucket_names,
            segment_length=index_segment_length,
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
