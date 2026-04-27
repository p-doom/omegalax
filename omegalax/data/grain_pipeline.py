"""Grain-backed SFT dataset compilation, chunk indexing, and iteration helpers."""

from __future__ import annotations

import json
import multiprocessing as mp
import numpy as np
import shutil
from collections.abc import Iterable
from itertools import chain
from pathlib import Path
from typing import Any

from tqdm import tqdm

from array_record.python.array_record_module import ArrayRecordWriter
import grain
import jax

COMPILED_DATASET_VERSION = 1
COMPILED_METADATA_FILENAME = "metadata.json"
TOKEN_STATS_FILENAME = "token_stats.json"
ARRAY_RECORD_SUFFIX = ".array_record"

_measure_fn = None


def _prepare_output_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists():
        has_contents = any(out_dir.iterdir())
        if has_contents and not overwrite:
            raise ValueError(f"Refusing to overwrite non-empty output directory: {out_dir}")
        if has_contents and overwrite:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _write_arrayrecord_dataset(
    records: Iterable[dict[str, Any]],
    out_dir: Path,
    *,
    records_per_shard: int,
    overwrite: bool,
    metadata: dict[str, Any],
) -> Path:
    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be > 0")

    _prepare_output_dir(out_dir, overwrite=overwrite)

    shard_paths: list[str] = []
    records_in_shard = 0
    total_records = 0
    shard_idx = 0

    def _open_writer(next_idx: int) -> ArrayRecordWriter:
        shard_path = out_dir / f"part-{next_idx:05d}.array_record"
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

            payload = json.dumps(record, sort_keys=True).encode("utf-8")
            writer.write(payload)
            records_in_shard += 1
            total_records += 1
    finally:
        writer.close()

    final_metadata = dict(metadata)
    final_metadata.update(
        {
            "version": COMPILED_DATASET_VERSION,
            "num_records": total_records,
            "num_shards": len(shard_paths),
            "shard_paths": shard_paths,
        }
    )
    (out_dir / COMPILED_METADATA_FILENAME).write_text(json.dumps(final_metadata, indent=2) + "\n")
    return out_dir

def _make_payload_block_record(
    *,
    session_id: str,
    source_line: int,
    block_idx: int,
    message_start: int,
    messages: list[dict[str, Any]],
    session_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "source_line": source_line,
        "block_idx": block_idx,
        "message_start": message_start,
        "message_end": message_start + len(messages),
        "session_meta": session_meta,
        "messages": messages,
    }


def _build_session_id(path: Path, line_num: int) -> str:
    return f"{path.stem}-{line_num:09d}"


def _iter_jsonl_message_blocks(
    path: Path,
    *,
    messages_per_record: int,
):
    with path.open() as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            messages = raw["messages"]
            assert isinstance(messages, list), f"Expected 'messages' to be a list at {path}:{line_num}"

            session_id = _build_session_id(path, line_num)
            session_meta = {k: v for k, v in raw.items() if k not in {"messages", "session_id"}}

            block_messages: list[dict[str, Any]] = []
            block_start = 0
            block_idx = 0
            for msg_idx, message in enumerate(messages):
                candidate_messages = block_messages + [message]
                would_exceed_count = len(candidate_messages) > messages_per_record
                if block_messages and would_exceed_count:
                    yield _make_payload_block_record(
                        session_id=session_id,
                        source_line=line_num,
                        block_idx=block_idx,
                        message_start=block_start,
                        messages=block_messages,
                        session_meta=session_meta,
                    )
                    block_idx += 1
                    block_start = msg_idx
                    block_messages = [message]
                else:
                    block_messages = candidate_messages

            if block_messages:
                yield _make_payload_block_record(
                    session_id=session_id,
                    source_line=line_num,
                    block_idx=block_idx,
                    message_start=block_start,
                    messages=block_messages,
                    session_meta=session_meta,
                )


def compile_jsonl_to_arrayrecord(
    src_path: str | Path,
    out_dir: str | Path,
    *,
    messages_per_record: int = 128,
    records_per_shard: int = 10_000,
    overwrite: bool = False,
) -> Path:
    """Compile raw JSONL sessions into canonical message-block ArrayRecord shards.

    Session ids are always synthesized from the source filename and line number.
    """

    if messages_per_record <= 0:
        raise ValueError("messages_per_record must be > 0")

    src_path = Path(src_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()

    num_sessions = 0
    with src_path.open() as f:
        for line in f:
            if line.strip():
                num_sessions += 1

    records = _iter_jsonl_message_blocks(
        src_path,
        messages_per_record=messages_per_record,
    )
    return _write_arrayrecord_dataset(
        records,
        out_dir,
        records_per_shard=records_per_shard,
        overwrite=overwrite,
        metadata={
            "source_path": str(src_path),
            "messages_per_record": messages_per_record,
            "num_sessions": num_sessions,
        },
    )


def resolve_arrayrecord_paths(path: str | Path) -> list[Path]:
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix != ARRAY_RECORD_SUFFIX:
            raise ValueError(
                f"Expected a compiled Grain shard ({ARRAY_RECORD_SUFFIX}) or dataset directory, got file: {path}"
            )
        return [path]
    metadata_path = path / COMPILED_METADATA_FILENAME
    assert metadata_path.is_file(), f"Compiled Grain dataset metadata does not exist: {metadata_path}"
    metadata = json.loads(metadata_path.read_text())
    shard_paths = [path / rel for rel in metadata["shard_paths"]]

    if not shard_paths:
        raise ValueError(f"No ArrayRecord shards found under: {path}")
    missing = [p for p in shard_paths if not p.exists()]
    if missing:
        raise ValueError(f"Missing ArrayRecord shard(s): {missing}")
    return shard_paths


def load_compiled_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    metadata_path = path / COMPILED_METADATA_FILENAME
    if not metadata_path.exists():
        raise ValueError(f"Compiled Grain dataset metadata does not exist: {metadata_path}")
    return json.loads(metadata_path.read_text())

def required_epochs_for_batches(
    path: str | Path,
    *,
    batch_size: int,
    num_batches: int,
    dp_size: int | None = None,
) -> int:
    if num_batches <= 0:
        return 1
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    from omegalax.distributed.mesh import data_parallel_size

    metadata = load_compiled_metadata(path)
    num_records = int(metadata["num_records"])
    dp = data_parallel_size(dp_size)
    records_per_epoch = num_records // dp
    if records_per_epoch <= 0:
        raise ValueError(
            f"Compiled Grain dataset has {num_records} records, which is too small to shard "
            f"across data_parallel_size={dp} with drop_remainder=True."
        )
    required_records = batch_size * num_batches
    return max(1, (required_records + records_per_epoch - 1) // records_per_epoch)

def _iter_indexed_records(path: str | Path):
    source = grain.sources.ArrayRecordDataSource([str(p) for p in resolve_arrayrecord_paths(path)])
    for record_idx in range(len(source)):
        yield record_idx, json.loads(source[record_idx])


def _measure_worker(keyed_message):
    key, message = keyed_message
    return key, _measure_fn(message)


def _precompute_message_lengths(payload_path, measure_message, num_workers):
    global _measure_fn
    _measure_fn = measure_message

    tasks: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for record_idx, block in _iter_indexed_records(payload_path):
        for msg_offset, message in enumerate(block["messages"]):
            tasks.append(((record_idx, msg_offset), message))

    ctx = mp.get_context("fork")
    chunksize = max(1, min(32, len(tasks) // num_workers))
    with ctx.Pool(num_workers) as pool:
        results = dict(tqdm(
            pool.imap_unordered(_measure_worker, tasks, chunksize=chunksize),
            total=len(tasks),
            desc=f"Measuring messages ({num_workers} workers)",
        ))
    return results


def _compute_distribution(values: list[int]) -> dict[str, int | float]:
    """Compute summary statistics for a list of integers."""
    if not values:
        return {"sum": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0,
                "std": 0.0, "p95": 0.0, "p99": 0.0}
    arr = np.array(values)
    return {
        "sum": int(arr.sum()),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "std": round(float(arr.std()), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
    }


def _frequency_table(values: list[int]) -> dict[str, int]:
    """Return a ``{value: count}`` mapping sorted by value."""
    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return {str(k): v for k, v in sorted(counts.items())}


def _emit_token_stats(
    out_dir: Path,
    *,
    msg_lengths: list[int],
    msg_vision_tokens: list[int],
    msg_num_images: list[int],
    chunk_lengths: list[int],
    chunk_vision_tokens: list[int],
    chunk_vision_patches: list[int],
    chunk_num_images: list[int],
    chunk_num_messages: list[int],
    image_shape_counts: dict[str, int],
) -> None:
    """Assemble per-message / per-chunk token statistics and write them to
    ``token_stats.json`` in ``out_dir``.
    """
    msg_text_tokens = [l - v for l, v in zip(msg_lengths, msg_vision_tokens)]
    chunk_text_tokens = [l - v for l, v in zip(chunk_lengths, chunk_vision_tokens)]
    stats = {
        "per_message": {
            "num_messages": len(msg_lengths),
            "length": _compute_distribution(msg_lengths),
            "text_tokens": _compute_distribution(msg_text_tokens),
            "vision_tokens": _compute_distribution(msg_vision_tokens),
            "num_images": _compute_distribution(msg_num_images),
        },
        "per_chunk": {
            "num_chunks": len(chunk_lengths),
            "measured_length": _compute_distribution(chunk_lengths),
            "text_tokens": _compute_distribution(chunk_text_tokens),
            "vision_tokens": _compute_distribution(chunk_vision_tokens),
            "vision_patches": _compute_distribution(chunk_vision_patches),
            "num_images": _compute_distribution(chunk_num_images),
            "num_messages": _compute_distribution(chunk_num_messages),
        },
        "image_shapes": dict(sorted(
            image_shape_counts.items(), key=lambda kv: -kv[1]
        )),
        "vision_variability": {
            "num_images_per_chunk": _frequency_table(chunk_num_images),
            "vision_tokens_per_chunk": _frequency_table(chunk_vision_tokens),
            "vision_patches_per_chunk": _frequency_table(chunk_vision_patches),
        },
    }
    stats_path = out_dir / TOKEN_STATS_FILENAME
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")


def build_chunk_index(
    payload_path: str | Path,
    out_dir: str | Path,
    *,
    max_length: int,
    measure_message,
    records_per_shard: int = 100_000,
    overwrite: bool = False,
    profile_metadata: dict[str, Any] | None = None,
    num_workers: int = 2,
) -> Path:
    """Build an offline chunk index over a canonical payload-block dataset.

    ``measure_message`` is called exactly once per message and must return either
    the number of tokens (``int``) or a dict containing at least a ``"length"``
    key.  When a dict is returned, extra fields (``vision_tokens``,
    ``num_images``, ``image_grid_thw``) are aggregated into per-chunk
    descriptors and a ``token_stats.json`` summary is written next to the index.
    """

    if max_length <= 0:
        raise ValueError("max_length must be > 0")

    payload_path = Path(payload_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    payload_metadata = load_compiled_metadata(payload_path)
    if "payload_path" in payload_metadata:
        raise ValueError(f"Chunk indices can only be built from payload datasets, got chunk index: {payload_path}")

    precomputed_lengths = _precompute_message_lengths(
        payload_path, measure_message, num_workers
    )

    # -- token stats accumulators (populated lazily by the generator) ----------
    _msg_lengths: list[int] = []
    _msg_vision_tokens: list[int] = []
    _msg_num_images: list[int] = []
    _chunk_lengths: list[int] = []
    _chunk_vision_tokens: list[int] = []
    _chunk_vision_patches: list[int] = []
    _chunk_num_images: list[int] = []
    _chunk_num_messages: list[int] = []
    _image_shape_counts: dict[str, int] = {}

    def _iter_chunk_descriptors():
        current_session_id: str | None = None
        current_messages: list[dict[str, Any]] = []
        current_length = 0
        current_vision_tokens = 0
        current_vision_patches = 0
        current_num_images = 0
        start_record_idx = 0
        start_message_offset = 0
        end_record_idx = 0
        end_message_offset = 0

        def emit_current() -> dict[str, Any] | None:
            if current_session_id is None or not current_messages:
                return None
            descriptor = {
                "session_id": current_session_id,
                "start_record_idx": start_record_idx,
                "start_message_offset": start_message_offset,
                "end_record_idx": end_record_idx,
                "end_message_offset": end_message_offset,
                "num_messages": len(current_messages),
                "measured_length": current_length,
            }
            if _msg_lengths:
                descriptor["vision_tokens"] = current_vision_tokens
                descriptor["vision_patches"] = current_vision_patches
                descriptor["num_images"] = current_num_images
                _chunk_lengths.append(current_length)
                _chunk_vision_tokens.append(current_vision_tokens)
                _chunk_vision_patches.append(current_vision_patches)
                _chunk_num_images.append(current_num_images)
                _chunk_num_messages.append(len(current_messages))
            return descriptor

        for record_idx, block in _iter_indexed_records(payload_path):
            block_session_id = str(block["session_id"])
            if current_session_id is None:
                current_session_id = block_session_id
            elif block_session_id != current_session_id:
                descriptor = emit_current()
                if descriptor is not None:
                    yield descriptor
                current_session_id = block_session_id
                current_messages = []
                current_length = 0
                current_vision_tokens = 0
                current_vision_patches = 0
                current_num_images = 0

            for msg_offset, message in enumerate(block["messages"]):
                result = precomputed_lengths[(record_idx, msg_offset)]

                if isinstance(result, dict):
                    msg_length = result["length"]
                    msg_vision_tokens = result["vision_tokens"]
                    msg_vision_patches = result["vision_patches"]
                    msg_num_images = result["num_images"]
                    _msg_lengths.append(msg_length)
                    _msg_vision_tokens.append(msg_vision_tokens)
                    _msg_num_images.append(msg_num_images)
                    for shape in result["image_grid_thw"]:
                        key = str(tuple(shape))
                        _image_shape_counts[key] = _image_shape_counts.get(key, 0) + 1
                else:
                    msg_length = int(result)
                    msg_vision_tokens = 0
                    msg_vision_patches = 0
                    msg_num_images = 0

                if msg_length > max_length:
                    raise ValueError(
                        f"Single message at session={block_session_id} record={record_idx} "
                        f"offset={msg_offset} exceeds max_length={max_length}"
                    )

                if not current_messages:
                    start_record_idx = record_idx
                    start_message_offset = msg_offset
                elif current_length + msg_length > max_length:
                    descriptor = emit_current()
                    if descriptor is None:
                        raise AssertionError("Expected a chunk descriptor before overflow split")
                    yield descriptor
                    current_messages = []
                    current_length = 0
                    current_vision_tokens = 0
                    current_vision_patches = 0
                    current_num_images = 0
                    start_record_idx = record_idx
                    start_message_offset = msg_offset

                current_messages.append(message)
                current_length += msg_length
                current_vision_tokens += msg_vision_tokens
                current_vision_patches += msg_vision_patches
                current_num_images += msg_num_images
                end_record_idx = record_idx
                end_message_offset = msg_offset + 1

        descriptor = emit_current()
        if descriptor is not None:
            yield descriptor

    out_path = _write_arrayrecord_dataset(
        _iter_chunk_descriptors(),
        out_dir,
        records_per_shard=records_per_shard,
        overwrite=overwrite,
        metadata={
            "payload_path": str(payload_path),
            "payload_num_records": int(payload_metadata["num_records"]),
            "max_length": max_length,
            "profile_metadata": profile_metadata or {},
        },
    )

    if _msg_lengths:
        _emit_token_stats(
            out_dir,
            msg_lengths=_msg_lengths,
            msg_vision_tokens=_msg_vision_tokens,
            msg_num_images=_msg_num_images,
            chunk_lengths=_chunk_lengths,
            chunk_vision_tokens=_chunk_vision_tokens,
            chunk_vision_patches=_chunk_vision_patches,
            chunk_num_images=_chunk_num_images,
            chunk_num_messages=_chunk_num_messages,
            image_shape_counts=_image_shape_counts,
        )

    return out_path

class _JsonLoadsMap(grain.transforms.Map):
    def map(self, element):
        return json.loads(element)


class _ChunkDescriptorResolver(grain.transforms.Map):
    def __init__(self, payload_path: str | Path) -> None:
        self._payload_shards = [str(path) for path in resolve_arrayrecord_paths(payload_path)]
        self._payload_source = None

    def _source(self):
        if self._payload_source is None:
            self._payload_source = grain.sources.ArrayRecordDataSource(self._payload_shards)
        return self._payload_source

    def map(self, descriptor: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        session_meta: dict[str, Any] = {}
        payload_source = self._source()

        start_record_idx = int(descriptor["start_record_idx"])
        end_record_idx = int(descriptor["end_record_idx"])
        start_message_offset = int(descriptor["start_message_offset"])
        end_message_offset = int(descriptor["end_message_offset"])

        for record_idx in range(start_record_idx, end_record_idx + 1):
            block = json.loads(payload_source[record_idx])
            if not session_meta:
                session_meta = dict(block.get("session_meta", {}))
            lo = start_message_offset if record_idx == start_record_idx else 0
            hi = end_message_offset if record_idx == end_record_idx else len(block["messages"])
            messages.extend(block["messages"][lo:hi])

        example = dict(session_meta)
        example["messages"] = messages
        example["_omegalax_session_id"] = descriptor["session_id"]
        example["_omegalax_start_record_idx"] = start_record_idx
        example["_omegalax_end_record_idx"] = end_record_idx
        example["_omegalax_measured_length"] = descriptor.get("measured_length")
        return example


def make_grain_read_options(
    *,
    num_threads: int = 16,
    prefetch_buffer_size: int = 500,
) -> grain.ReadOptions:
    return grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=prefetch_buffer_size)


def make_grain_multiprocessing_options(
    *,
    num_workers: int = 0,
    per_worker_buffer_size: int = 1,
    enable_profiling: bool = False,
) -> grain.MultiprocessingOptions:
    return grain.MultiprocessingOptions(
        num_workers=num_workers,
        per_worker_buffer_size=per_worker_buffer_size,
        enable_profiling=enable_profiling,
    )


def make_grain_iterator(
    compiled_path: str | Path,
    *,
    batch_size: int,
    batch_fn,
    shuffle: bool = True,
    seed: int = 0,
    num_epochs: int = 1,
    read_options: grain.ReadOptions | None = None,
    multiprocessing_options: grain.MultiprocessingOptions | None = None,
    dp_size: int | None = None,
):
    """Create a checkpointable Grain dataloader iterator over a chunk-index dataset."""
    from omegalax.distributed.mesh import data_parallel_index, data_parallel_size

    compiled_path = Path(compiled_path).expanduser().resolve()
    metadata = load_compiled_metadata(compiled_path)
    if "payload_path" not in metadata:
        raise ValueError(f"Expected compiled Grain chunk-index dataset, missing payload_path: {compiled_path}")

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    shard_paths = [str(path) for path in resolve_arrayrecord_paths(compiled_path)]
    payload_path = str(metadata["payload_path"])
    mp_options = multiprocessing_options or make_grain_multiprocessing_options()
    read_options = read_options or make_grain_read_options()

    dp = data_parallel_size(dp_size)
    dp_index = data_parallel_index(dp_size)
    source = grain.sources.ArrayRecordDataSource(shard_paths)
    shard_options = grain.sharding.NoSharding() if dp <= 1 else grain.sharding.ShardOptions(
        shard_index=dp_index, shard_count=dp, drop_remainder=True,
    )
    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=shard_options,
        shuffle=shuffle,
        num_epochs=num_epochs,
        seed=seed,
    )
    operations = [
        _JsonLoadsMap(),
        _ChunkDescriptorResolver(payload_path),
        grain.transforms.Batch(batch_size=batch_size, drop_remainder=True, batch_fn=batch_fn),
    ]
    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=mp_options.num_workers,
        worker_buffer_size=mp_options.per_worker_buffer_size,
        shard_options=shard_options,
        read_options=read_options,
        enable_profiling=mp_options.enable_profiling,
    )
    return iter(dataloader)
