"""Data-only helpers for ``omegalax_doc_chain_v1`` pretraining corpora."""

from __future__ import annotations

import json
import os
import shutil
import struct
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any

import grain
import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter

DOC_CHAIN_FORMAT = "omegalax_doc_chain_v1"
DOC_CHAIN_DATASET_VERSION = 1
DOC_CHAIN_BINARY_MAGIC = b"OMXDC01\n"
DOC_CHAIN_BINARY_HEADER = struct.Struct("<QQ")
COMPILED_METADATA_FILENAME = "metadata.json"
ARRAY_RECORD_SUFFIX = ".array_record"
DEFAULT_CHUNK_LENGTH = 2048
DEFAULT_PAD_ID = 0
DEFAULT_EOS_ID = 248046  # Qwen/Qwen3.5-0.8B <|im_end|>
BATCH_PRETRAIN_METADATA_KEY = "metadata"
DEFAULT_DATA_SET_SPLIT = "train"
PRETRAIN_SOURCE_ROOT_ENV = "OMEGALAX_PRETRAIN_SOURCE_ROOT"
PRETRAIN_LOCAL_ROOT_ENV = "OMEGALAX_PRETRAIN_LOCAL_ROOT"
MAX_PRETRAIN_POSITIONS = 2**63 - 1


@dataclass(frozen=True)
class DataSetRecord:
    doc_id: str
    token_ids: np.ndarray
    doc_token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairMetadata:
    bucket_idx: int
    record_idx: int
    doc_id: str
    pair_idx: int
    start: int
    mid: int
    end: int
    doc_token_count: int
    eos_token_idx: int | None = None


def _json_payload(payload: bytes | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, bytes):
        if payload.startswith(DOC_CHAIN_BINARY_MAGIC):
            return _binary_payload(payload)
        return json.loads(payload)
    if isinstance(payload, str):
        return json.loads(payload)
    if isinstance(payload, dict):
        return dict(payload)
    raise TypeError(f"Unsupported data-set payload type: {type(payload).__name__}")


def _binary_payload(payload: bytes) -> dict[str, Any]:
    pos = len(DOC_CHAIN_BINARY_MAGIC)
    header_end = pos + DOC_CHAIN_BINARY_HEADER.size
    if len(payload) < header_end:
        raise ValueError("truncated doc-chain binary header")

    header_len, token_count = DOC_CHAIN_BINARY_HEADER.unpack(payload[pos:header_end])
    pos = header_end
    header = json.loads(payload[pos : pos + header_len].decode("utf-8"))
    pos += header_len

    expected = int(token_count) * np.dtype(np.int32).itemsize
    token_bytes = payload[pos : pos + expected]
    if len(token_bytes) != expected:
        raise ValueError(
            f"truncated token payload: expected {expected} bytes, got {len(token_bytes)}"
        )

    header["token_ids"] = np.frombuffer(token_bytes, dtype=np.int32).copy()
    return header


def deserialize_data_set_record(
    payload: DataSetRecord | bytes | str | dict[str, Any],
) -> DataSetRecord:
    if isinstance(payload, DataSetRecord):
        return payload
    raw = _json_payload(payload)
    fmt = (
        raw.get("format") or raw.get("dataset_format") or raw.get("data_format") or raw.get("type")
    )
    if fmt is not None and fmt != DOC_CHAIN_FORMAT:
        raise ValueError(f"Unsupported doc-chain format: {fmt}")

    doc_id = raw.get("doc_id", raw.get("id"))
    if doc_id is None:
        raise ValueError("Data-set record is missing doc_id")

    raw_tokens = None
    for key in ("token_ids", "tokens", "input_ids"):
        if key in raw:
            raw_tokens = raw[key]
            break
    if raw_tokens is None:
        raise ValueError("Data-set record is missing token_ids")

    token_ids = np.asarray(raw_tokens, dtype=np.int32)
    doc_token_count = int(raw.get("doc_token_count", token_ids.shape[0]))

    metadata = dict(raw.get("metadata") or {})
    core_keys = {
        "format",
        "dataset_format",
        "data_format",
        "type",
        "doc_id",
        "id",
        "token_ids",
        "tokens",
        "input_ids",
        "doc_token_count",
        "metadata",
    }
    for key, value in raw.items():
        if key not in core_keys:
            metadata[key] = value

    return DataSetRecord(
        doc_id=str(doc_id),
        token_ids=token_ids,
        doc_token_count=doc_token_count,
        metadata=metadata,
    )


def load_arrayrecord_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    metadata_path = path / COMPILED_METADATA_FILENAME
    if not metadata_path.exists():
        raise ValueError(f"Compiled ArrayRecord dataset metadata does not exist: {metadata_path}")
    return json.loads(metadata_path.read_text())


def load_data_set_metadata(path: str | Path) -> dict[str, Any]:
    metadata = load_arrayrecord_metadata(path)
    fmt = metadata.get("format") or metadata.get("dataset_format")
    if fmt != DOC_CHAIN_FORMAT:
        raise ValueError(f"Expected {DOC_CHAIN_FORMAT} dataset, got format={fmt}")
    return metadata


def _bucket_sort_key(path: Path) -> tuple[int, int, str]:
    suffix = path.name.removeprefix("bucket_")
    unit = suffix[-1:].lower()
    value_text = suffix[:-1] if unit == "k" else suffix
    try:
        value = int(value_text)
    except ValueError:
        return (1, 0, path.name)
    if unit == "k":
        value *= 1024
    return (0, value, path.name)


def resolve_data_set_buckets(
    root: str | Path,
    *,
    split: str = DEFAULT_DATA_SET_SPLIT,
) -> list[Path]:
    root = Path(root).expanduser().resolve()
    split_path = root / split
    if not split_path.is_dir():
        raise ValueError(f"Data-set split directory does not exist: {split_path}")
    bucket_paths = sorted(
        (
            child.resolve()
            for child in split_path.iterdir()
            if child.is_dir() and child.name.startswith("bucket_")
        ),
        key=_bucket_sort_key,
    )
    if not bucket_paths:
        raise ValueError(f"No data-set buckets found under: {split_path}")
    return bucket_paths


def rewrite_data_set_root_path(
    root: str | Path,
    *,
    source_root: str | Path | None = None,
    local_root: str | Path | None = None,
) -> Path:
    source_root = source_root or os.environ.get(PRETRAIN_SOURCE_ROOT_ENV)
    local_root = local_root or os.environ.get(PRETRAIN_LOCAL_ROOT_ENV)
    root_path = Path(root).expanduser().resolve()

    if source_root is None and local_root is None:
        return root_path
    if source_root is None or local_root is None:
        raise ValueError(
            f"{PRETRAIN_SOURCE_ROOT_ENV} and {PRETRAIN_LOCAL_ROOT_ENV} must be set together"
        )

    resolved_source_root = Path(source_root).expanduser().resolve()
    resolved_local_root = Path(local_root).expanduser().resolve()
    try:
        rel_path = root_path.relative_to(resolved_source_root)
    except ValueError as exc:
        raise ValueError(
            f"Cannot rewrite data-set root path outside {PRETRAIN_SOURCE_ROOT_ENV}: "
            f"{root_path} is not under {resolved_source_root}"
        ) from exc

    rewritten_path = resolved_local_root / rel_path
    if not rewritten_path.exists():
        raise ValueError(f"Rewritten data-set root path does not exist: {rewritten_path}")
    return rewritten_path


def _resolve_data_set_bucket_shards(bucket_path: str | Path) -> list[Path]:
    bucket_path = Path(bucket_path).expanduser().resolve()
    metadata = load_data_set_metadata(bucket_path)
    shard_paths = [bucket_path / rel for rel in metadata["shard_paths"]]
    if not shard_paths:
        raise ValueError(f"No ArrayRecord shards found under: {bucket_path}")
    missing = [p for p in shard_paths if not p.exists()]
    if missing:
        raise ValueError(f"Missing ArrayRecord shard(s): {missing}")
    return shard_paths


def resolve_arrayrecord_paths(path: str | Path) -> list[Path]:
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix != ARRAY_RECORD_SUFFIX:
            raise ValueError(
                f"Expected an ArrayRecord shard ({ARRAY_RECORD_SUFFIX}) "
                f"or dataset directory, got: {path}"
            )
        return [path]

    metadata = load_arrayrecord_metadata(path)
    shard_paths = [path / rel for rel in metadata["shard_paths"]]
    if not shard_paths:
        raise ValueError(f"No ArrayRecord shards found under: {path}")
    missing = [p for p in shard_paths if not p.exists()]
    if missing:
        raise ValueError(f"Missing ArrayRecord shard(s): {missing}")
    return shard_paths


def iter_json_arrayrecord_records(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    source = grain.sources.ArrayRecordDataSource([str(p) for p in resolve_arrayrecord_paths(path)])
    for record_idx in range(len(source)):
        yield record_idx, json.loads(source[record_idx])


def calculate_samples_per_process(
    *,
    num_records: int,
    dp_size: int,
    dp_index: int,
) -> int:
    if num_records <= 0:
        raise ValueError("num_records must be > 0")
    if dp_size <= 0:
        raise ValueError("dp_size must be > 0")
    if dp_index < 0 or dp_index >= dp_size:
        raise ValueError(f"dp_index must be in [0, {dp_size}), got {dp_index}")
    if dp_index >= num_records:
        return 0
    return (num_records - 1 - dp_index) // dp_size + 1


def num_pretrain_records_usable(
    *,
    num_records: int,
    dp_size: int,
    records_per_local_batch: int,
) -> int:
    if num_records <= 0:
        raise ValueError("num_records must be > 0")
    if dp_size <= 0:
        raise ValueError("dp_size must be > 0")
    if records_per_local_batch <= 0:
        raise ValueError("records_per_local_batch must be > 0")
    records_per_global_step = int(dp_size) * int(records_per_local_batch)
    return (int(num_records) // records_per_global_step) * records_per_global_step


def num_pretrain_positions(
    *,
    epoch_samples_per_process: int,
    num_epochs: int | None,
) -> int:
    if epoch_samples_per_process <= 0:
        raise ValueError("epoch_samples_per_process must be > 0")
    if num_epochs is None:
        return MAX_PRETRAIN_POSITIONS
    if num_epochs <= 0:
        raise ValueError("num_epochs must be > 0")
    return int(epoch_samples_per_process) * int(num_epochs)


class IndexRecordLookup:
    def __init__(
        self,
        *,
        index_shard_paths: Sequence[str | Path],
        num_records: int,
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
        self.num_records = int(num_records)
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
            index_record_idx = grain.experimental.index_shuffle(
                global_pos,
                self.num_records - 1,
                self.seed + epoch,
                self.shuffle_rounds,
            )
        else:
            index_record_idx = global_pos
        return json.loads(self._index_source()[index_record_idx])


def make_dataset_index(
    *,
    index_shard_paths: Sequence[str | Path],
    num_records: int,
    num_epochs: int | None,
    dp_size: int,
    dp_index: int,
    shuffle: bool,
    seed: int,
    shuffle_rounds: int,
    records_per_local_batch: int = 1,
) -> tuple[grain.MapDataset, int]:
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
        IndexRecordLookup(
            index_shard_paths=index_shard_paths,
            num_records=num_records,
            epoch_samples_per_process=epoch_samples_per_process,
            dp_size=dp_size,
            dp_index=dp_index,
            shuffle=shuffle,
            seed=seed,
            shuffle_rounds=shuffle_rounds,
        )
    )
    return dataset_index, epoch_samples_per_process


def write_json_arrayrecord_dataset(
    records: Iterable[dict[str, Any]],
    out_dir: str | Path,
    *,
    records_per_shard: int,
    overwrite: bool,
    metadata: dict[str, Any],
) -> Path:
    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be > 0")

    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists():
        has_contents = any(out_dir.iterdir())
        if has_contents and not overwrite:
            raise ValueError(f"Refusing to overwrite non-empty output directory: {out_dir}")
        if has_contents and overwrite:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: list[str] = []
    records_in_shard = 0
    total_records = 0
    shard_idx = 0

    def _open_writer(next_idx: int) -> ArrayRecordWriter:
        shard_path = out_dir / f"part-{next_idx:05d}{ARRAY_RECORD_SUFFIX}"
        shard_paths.append(shard_path.name)
        return ArrayRecordWriter(str(shard_path), "group_size:1")

    records_iter = iter(records)
    try:
        first_record = next(records_iter)
    except StopIteration:
        raise ValueError("No records were written") from None

    writer = _open_writer(shard_idx)
    try:
        for record in chain((first_record,), records_iter):
            if records_in_shard >= records_per_shard:
                writer.close()
                shard_idx += 1
                writer = _open_writer(shard_idx)
                records_in_shard = 0
            writer.write(json.dumps(record, sort_keys=True).encode("utf-8"))
            records_in_shard += 1
            total_records += 1
    finally:
        writer.close()

    final_metadata = dict(metadata)
    final_metadata.update(
        {
            "version": DOC_CHAIN_DATASET_VERSION,
            "num_records": total_records,
            "num_shards": len(shard_paths),
            "shard_paths": shard_paths,
        }
    )
    (out_dir / COMPILED_METADATA_FILENAME).write_text(json.dumps(final_metadata, indent=2) + "\n")
    return out_dir


class DataSetReader:
    def __init__(
        self,
        root: str | Path,
        *,
        split: str = DEFAULT_DATA_SET_SPLIT,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
        self.bucket_paths = resolve_data_set_buckets(self.root, split=self.split)
        self.bucket_names = [path.name for path in self.bucket_paths]
        self._shard_paths = [_resolve_data_set_bucket_shards(path) for path in self.bucket_paths]
        self._buckets: list[grain.sources.ArrayRecordDataSource | None] = [
            None for _ in self.bucket_paths
        ]

    @property
    def num_buckets(self) -> int:
        return len(self.bucket_paths)

    def _bucket(self, bucket_idx: int) -> grain.sources.ArrayRecordDataSource:
        if bucket_idx < 0 or bucket_idx >= len(self.bucket_paths):
            raise IndexError(f"bucket_idx out of range: {bucket_idx}")
        bucket = self._buckets[bucket_idx]
        if bucket is None:
            bucket = grain.sources.ArrayRecordDataSource(
                [str(path) for path in self._shard_paths[bucket_idx]]
            )
            self._buckets[bucket_idx] = bucket
        return bucket

    def num_records(self, bucket_idx: int) -> int:
        return len(self._bucket(bucket_idx))

    def read(self, bucket_idx: int, record_idx: int) -> DataSetRecord:
        bucket = self._bucket(bucket_idx)
        return deserialize_data_set_record(bucket[record_idx])

    def iter_records(self) -> Iterator[tuple[int, int, DataSetRecord]]:
        for bucket_idx in range(self.num_buckets):
            bucket = self._bucket(bucket_idx)
            for record_idx in range(len(bucket)):
                yield bucket_idx, record_idx, deserialize_data_set_record(bucket[record_idx])


def iter_document_pair_metadata(
    doc: DataSetRecord,
    *,
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
    bucket_idx: int = 0,
    record_idx: int = 0,
    eos_id: int | None = DEFAULT_EOS_ID,
) -> Iterator[PairMetadata]:
    """Yield retained 2-chunk windows for statepassing pretraining."""

    if chunk_length <= 0:
        raise ValueError("chunk_length must be > 0")
    if doc.doc_token_count <= chunk_length:
        return

    pair_length = 2 * chunk_length
    num_full_pairs, tail = divmod(doc.doc_token_count, pair_length)
    keep_tail_pair = tail > chunk_length
    num_pairs = num_full_pairs + int(keep_tail_pair)
    if num_pairs == 0:
        return

    retained_end = num_full_pairs * pair_length
    if keep_tail_pair:
        retained_end += tail

    eos_token_idx = None
    if (
        eos_id is not None
        and tail > 0
        and tail <= chunk_length
        and retained_end > 0
        and int(doc.token_ids[-1]) == int(eos_id)
    ):
        eos_token_idx = retained_end - 1

    for pair_idx in range(num_pairs):
        start = pair_idx * pair_length
        end = min(start + pair_length, doc.doc_token_count)
        pair_eos_idx = (
            eos_token_idx if eos_token_idx is not None and start <= eos_token_idx < end else None
        )
        yield PairMetadata(
            bucket_idx=bucket_idx,
            record_idx=record_idx,
            doc_id=doc.doc_id,
            pair_idx=pair_idx,
            start=start,
            mid=start + chunk_length,
            end=end,
            doc_token_count=doc.doc_token_count,
            eos_token_idx=pair_eos_idx,
        )


def build_chunk_arrays(
    token_ids: Sequence[int] | np.ndarray,
    *,
    start: int = 0,
    end: int | None = None,
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
    pad_id: int = DEFAULT_PAD_ID,
    eos_id: int | None = DEFAULT_EOS_ID,
    eos_token_idx: int | None = None,
) -> dict[str, np.ndarray]:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be > 0")

    ids = np.asarray(token_ids, dtype=np.int32)
    end = int(ids.shape[0] if end is None else end)
    start = int(start)
    if start < 0 or end < start or end > ids.shape[0]:
        raise ValueError(f"Invalid chunk range start={start}, end={end}, length={ids.shape[0]}")

    real_length = end - start
    if real_length > chunk_length:
        raise ValueError(f"Chunk length {real_length} exceeds chunk_length={chunk_length}")

    token_ids_T = np.full((chunk_length,), int(pad_id), dtype=np.int32)
    attention_mask_T = np.zeros((chunk_length,), dtype=np.int32)
    if real_length:
        token_ids_T[:real_length] = ids[start:end]
        if eos_token_idx is not None:
            if eos_id is None:
                raise ValueError("eos_id is required when eos_token_idx is set")
            eos_token_idx = int(eos_token_idx)
            if eos_token_idx < start or eos_token_idx >= end:
                raise ValueError(
                    f"eos_token_idx={eos_token_idx} is outside chunk range [{start}, {end})"
                )
            token_ids_T[eos_token_idx - start] = int(eos_id)
        attention_mask_T[:real_length] = 1

    return {
        "token_ids_T": token_ids_T,
        "attention_mask_T": attention_mask_T,
        "loss_mask_T": attention_mask_T.copy(),
    }


def build_pair_arrays(
    token_ids: Sequence[int] | np.ndarray,
    pair_metadata: PairMetadata,
    *,
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
    pad_id: int = DEFAULT_PAD_ID,
    eos_id: int | None = DEFAULT_EOS_ID,
) -> dict[str, np.ndarray]:
    token_ids_CT = []
    attention_mask_CT = []
    loss_mask_CT = []
    chunk_idx_C = []
    is_last_chunk_C = []

    for chunk_in_pair, start, end, eos_token_idx in _iter_pair_metadata_chunks(pair_metadata):
        arrays = build_chunk_arrays(
            token_ids,
            start=start,
            end=end,
            chunk_length=chunk_length,
            pad_id=pad_id,
            eos_id=eos_id,
            eos_token_idx=eos_token_idx,
        )
        token_ids_CT.append(arrays["token_ids_T"])
        attention_mask_CT.append(arrays["attention_mask_T"])
        loss_mask_CT.append(arrays["loss_mask_T"])
        chunk_idx_C.append(pair_metadata.pair_idx * 2 + chunk_in_pair)
        is_last_chunk_C.append(end >= pair_metadata.doc_token_count)

    return {
        "token_ids_CT": np.stack(token_ids_CT).astype(np.int32),
        "attention_mask_CT": np.stack(attention_mask_CT).astype(np.int32),
        "loss_mask_CT": np.stack(loss_mask_CT).astype(np.int32),
        "chunk_idx_C": np.asarray(chunk_idx_C, dtype=np.int32),
        "reset_state_C": np.asarray([True, False], dtype=np.bool_),
        "is_last_chunk_C": np.asarray(is_last_chunk_C, dtype=np.bool_),
    }


def _iter_pair_metadata_chunks(
    pair_metadata: PairMetadata,
) -> Iterator[tuple[int, int, int, int | None]]:
    for chunk_in_pair, (start, end) in enumerate(
        (
            (pair_metadata.start, min(pair_metadata.mid, pair_metadata.end)),
            (pair_metadata.mid, pair_metadata.end),
        )
    ):
        eos_token_idx = (
            pair_metadata.eos_token_idx
            if pair_metadata.eos_token_idx is not None
            and start <= pair_metadata.eos_token_idx < end
            else None
        )
        yield chunk_in_pair, start, end, eos_token_idx


def flatten_pair_metadata(pair_metadata: PairMetadata) -> list[dict[str, Any]]:
    return [
        {
            "bucket_idx": pair_metadata.bucket_idx,
            "record_idx": pair_metadata.record_idx,
            "doc_id": pair_metadata.doc_id,
            "pair_idx": pair_metadata.pair_idx,
            "chunk_in_pair": chunk_in_pair,
            "chunk_idx": pair_metadata.pair_idx * 2 + chunk_in_pair,
            "start": start,
            "end": end,
            "eos_token_idx": eos_token_idx,
        }
        for chunk_in_pair, start, end, eos_token_idx in _iter_pair_metadata_chunks(pair_metadata)
    ]


def pair_metadata_to_record(pair_metadata: PairMetadata) -> dict[str, Any]:
    return {
        "bucket_idx": pair_metadata.bucket_idx,
        "record_idx": pair_metadata.record_idx,
        "doc_id": pair_metadata.doc_id,
        "pair_idx": pair_metadata.pair_idx,
        "start": pair_metadata.start,
        "mid": pair_metadata.mid,
        "end": pair_metadata.end,
        "doc_token_count": pair_metadata.doc_token_count,
        "eos_token_idx": pair_metadata.eos_token_idx,
    }


def pop_pretrain_metadata(batch: dict[str, Any]) -> dict[str, Any] | None:
    """Remove non-array pretraining metadata before batch sharding.

    IID and statepassing iterators surface document ids and bucket record
    positions under ``metadata`` for logging/debugging. The JAX sharding helpers
    expect every top-level batch value to be an array, so callers should pop this
    field before passing a batch to ``shard_batch_dict``.
    """

    raw = batch.pop(BATCH_PRETRAIN_METADATA_KEY, None)
    return dict(raw) if raw is not None else None


def resolve_pretrain_dp(
    *,
    dp_size: int,
    fsdp_size: int = 1,
    process_index: int | None = None,
) -> tuple[int, int]:
    """Resolve effective data-shard count and local shard index.

    This mirrors the existing Grain pipeline convention: data loading is
    sharded across the product of data-parallel and FSDP axes.
    """

    if dp_size <= 0:
        raise ValueError("dp_size must be > 0")
    if fsdp_size <= 0:
        raise ValueError("fsdp_size must be > 0")

    effective_dp_size = int(dp_size) * int(fsdp_size)
    if process_index is None:
        import jax

        process_index = int(jax.process_index())
    if process_index < 0:
        raise ValueError("process_index must be >= 0")
    return effective_dp_size, int(process_index) % effective_dp_size
