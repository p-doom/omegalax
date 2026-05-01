"""Grain-backed SFT dataset compilation, chunk indexing, and iteration helpers."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

from array_record.python.array_record_module import ArrayRecordWriter
import grain
import jax
import numpy as np

COMPILED_DATASET_VERSION = 1
COMPILED_METADATA_FILENAME = "metadata.json"
ARRAY_RECORD_SUFFIX = ".array_record"

SOURCE_ID_KEY = "_omegalax_source_id"
BATCH_SOURCE_IDS_KEY = "source_ids"


@dataclass(frozen=True)
class MixSource:
    """One dataset in a (potentially mixed) training corpus.

    ``path`` is a compiled chunk-index dataset directory (with metadata.json
    pointing at the payload). ``weight`` is unnormalized — relative weights
    across sources determine the realized example mix (see
    ``grain.MapDataset.mix``).
    """

    path: str | Path
    weight: float = 1.0


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

def build_chunk_index(
    payload_path: str | Path,
    out_dir: str | Path,
    *,
    max_length: int,
    measure_message,
    records_per_shard: int = 100_000,
    overwrite: bool = False,
    profile_metadata: dict[str, Any] | None = None,
) -> Path:
    """Build an offline chunk index over a canonical payload-block dataset.

    ``measure_message`` is called exactly once per message and must return the
    number of tokens that message contributes to a sequence.  For chat templates
    where tokenization is exactly additive at message boundaries (e.g. ChatML
    with ``add_special_tokens=False``), the accumulated per-message sum equals
    the full-sequence length exactly.  Swap in a different ``measure_message``
    implementation for other chat templates.
    """

    if max_length <= 0:
        raise ValueError("max_length must be > 0")

    payload_path = Path(payload_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    payload_metadata = load_compiled_metadata(payload_path)
    if "payload_path" in payload_metadata:
        raise ValueError(f"Chunk indices can only be built from payload datasets, got chunk index: {payload_path}")

    def _iter_chunk_descriptors():
        current_session_id: str | None = None
        current_messages: list[dict[str, Any]] = []
        current_length = 0
        start_record_idx = 0
        start_message_offset = 0
        end_record_idx = 0
        end_message_offset = 0

        def emit_current() -> dict[str, Any] | None:
            if current_session_id is None or not current_messages:
                return None
            return {
                "session_id": current_session_id,
                "start_record_idx": start_record_idx,
                "start_message_offset": start_message_offset,
                "end_record_idx": end_record_idx,
                "end_message_offset": end_message_offset,
                "num_messages": len(current_messages),
                "measured_length": current_length,
            }

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

            for msg_offset, message in enumerate(block["messages"]):
                msg_length = int(measure_message(message))
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
                    start_record_idx = record_idx
                    start_message_offset = msg_offset

                current_messages.append(message)
                current_length += msg_length
                end_record_idx = record_idx
                end_message_offset = msg_offset + 1

        descriptor = emit_current()
        if descriptor is not None:
            yield descriptor

    return _write_arrayrecord_dataset(
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


class _TagSourceMap(grain.transforms.Map):
    """Tag each example with its source index so batches expose realized mix ratios."""

    def __init__(self, source_id: int) -> None:
        self._source_id = int(source_id)

    def map(self, example: dict[str, Any]) -> dict[str, Any]:
        example[SOURCE_ID_KEY] = self._source_id
        return example


class _SourceTaggingCollator:
    """Wrap a user-provided collator to surface per-example source ids in the batch."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        source_ids = np.asarray(
            [int(ex.get(SOURCE_ID_KEY, 0)) for ex in examples], dtype=np.int32,
        )
        result = self._inner(examples)
        result[BATCH_SOURCE_IDS_KEY] = source_ids
        return result


def pop_source_ids(batch: dict[str, Any]) -> np.ndarray | None:
    """Pop source-id metadata from a batch dict before sharding.

    Returns the per-example source ids attached by the mixing iterator
    (shape ``(B,)``, int32), or None if the batch was not produced by a
    source-tagging collator. Removing the key keeps it out of the JIT
    cache key for ``sft_train_step`` and out of distributed sharding.
    """
    raw = batch.pop(BATCH_SOURCE_IDS_KEY, None)
    return np.asarray(raw) if raw is not None else None


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


def _coerce_sources(
    sources: str | Path | MixSource | Sequence[str | Path | MixSource],
) -> list[MixSource]:
    """Normalize mixed scalar/list inputs into a list of ``MixSource``."""
    if isinstance(sources, (str, Path)):
        return [MixSource(path=sources, weight=1.0)]
    if isinstance(sources, MixSource):
        return [sources]
    out: list[MixSource] = []
    for s in sources:
        if isinstance(s, MixSource):
            out.append(s)
        elif isinstance(s, (str, Path)):
            out.append(MixSource(path=s, weight=1.0))
        else:
            raise TypeError(f"Unsupported source spec: {s!r} (type {type(s).__name__})")
    if not out:
        raise ValueError("make_grain_iterator: at least one source required")
    return out


def _validate_mix_compatibility(
    sources: list[MixSource], metadatas: list[dict[str, Any]],
) -> None:
    """Refuse mixes that would silently corrupt training (different tokenization, length, etc.)."""
    if len(sources) <= 1:
        return
    max_lengths = {int(m["max_length"]) for m in metadatas if "max_length" in m}
    if len(max_lengths) > 1:
        raise ValueError(
            f"Cannot mix datasets compiled with different max_length: {max_lengths}. "
            f"Rebuild chunk indices with a shared --max_length."
        )
    tokenizer_ids = {
        (m.get("profile_metadata") or {}).get("tokenizer_id")
        for m in metadatas
    }
    tokenizer_ids.discard(None)
    if len(tokenizer_ids) > 1:
        raise ValueError(
            f"Cannot mix datasets compiled with different tokenizers: {tokenizer_ids}."
        )


def make_grain_iterator(
    sources: str | Path | MixSource | Sequence[str | Path | MixSource],
    *,
    batch_size: int,
    batch_fn,
    shuffle: bool = True,
    seed: int = 0,
    num_epochs: int | None = 1,
    read_options: grain.ReadOptions | None = None,
    multiprocessing_options: grain.MultiprocessingOptions | None = None,
    dp_size: int | None = None,
):
    """Create a checkpointable Grain iterator over one or more chunk-index datasets.

    When more than one source is supplied, examples are interleaved at the
    configured ``MixSource.weight`` ratios via ``grain.MapDataset.mix`` —
    every batch is a stochastic mix at the configured ratio, not a per-batch
    round-robin. ``num_epochs=None`` repeats each source indefinitely; set a
    finite value (per source) only for validation-style finite iteration.
    """
    from omegalax.distributed.mesh import data_parallel_index, data_parallel_size

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    mix_sources = _coerce_sources(sources)
    if any(s.weight < 0.0 for s in mix_sources):
        raise ValueError(
            f"Source weights must be non-negative: {[s.weight for s in mix_sources]}"
        )
    total_w = sum(s.weight for s in mix_sources)
    if total_w <= 0.0:
        raise ValueError("Sum of source weights must be > 0")
    # Drop zero-weight entries before mixing — Grain rejects them, but ablation
    # configs commonly zero out a source to disable it without changing structure.
    # Source ids stay aligned with the user-provided list so metric tags remain stable.
    active_indices = [i for i, s in enumerate(mix_sources) if s.weight > 0.0]
    metadatas = [load_compiled_metadata(mix_sources[i].path) for i in active_indices]
    for i, m in zip(active_indices, metadatas):
        if "payload_path" not in m:
            raise ValueError(
                f"Expected compiled Grain chunk-index dataset, missing payload_path: {mix_sources[i].path}"
            )
    _validate_mix_compatibility(
        [mix_sources[i] for i in active_indices], metadatas,
    )
    norm_weights = [
        mix_sources[i].weight / total_w for i in active_indices
    ]

    mp_options = multiprocessing_options or make_grain_multiprocessing_options()
    read_options = read_options or make_grain_read_options()
    dp = data_parallel_size(dp_size)
    dp_index = data_parallel_index(dp_size)

    per_source: list[grain.MapDataset] = []
    for active_idx, original_idx in enumerate(active_indices):
        s = mix_sources[original_idx]
        m = metadatas[active_idx]
        shard_paths = [str(p) for p in resolve_arrayrecord_paths(s.path)]
        payload_path = str(m["payload_path"])
        ds = grain.MapDataset.source(grain.sources.ArrayRecordDataSource(shard_paths))
        if dp > 1:
            # Contiguous-block DP shards with drop_remainder, matching the
            # legacy IndexSampler(ShardOptions(drop_remainder=True)) behavior.
            per_rank = len(ds) // dp
            ds = ds[dp_index * per_rank : (dp_index + 1) * per_rank]
        if shuffle:
            ds = ds.shuffle(seed=seed + original_idx)
        ds = ds.repeat(num_epochs)
        ds = ds.map(_JsonLoadsMap())
        ds = ds.map(_ChunkDescriptorResolver(payload_path))
        # Tag with the user-facing source id (position in the original list),
        # not the active-only index, so metric labels are stable across
        # ablations that zero out individual sources.
        ds = ds.map(_TagSourceMap(source_id=original_idx))
        per_source.append(ds)

    mixed = per_source[0] if len(per_source) == 1 else grain.MapDataset.mix(per_source, weights=norm_weights)
    batched = mixed.batch(
        batch_size=batch_size, drop_remainder=True, batch_fn=_SourceTaggingCollator(batch_fn),
    )
    iter_ds = batched.to_iter_dataset(read_options)
    if mp_options.num_workers > 0:
        iter_ds = iter_ds.mp_prefetch(mp_options)
    return iter(iter_ds)
