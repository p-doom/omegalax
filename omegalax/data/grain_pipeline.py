"""Grain-backed SFT dataset compilation, chunk indexing, and iteration helpers."""

from __future__ import annotations

import json
import multiprocessing as mp
import numpy as np
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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
TRUNCATION_STATS_FILENAME = "truncation_stats.json"
SEQUENCE_LENGTHS_FILENAME = "sequence_lengths.jsonl"
# Per-message token-length cache. Keyed by (record_idx, msg_offset), it holds
# the exact measure_message() output for every message in a payload. This is
# the only tokenizer-bound product of chunk-index building and is independent
# of max_length / overflow_mode / system_message, so it can be measured once
# and reused across every chunk-index build over the same payload + tokenizer.
MESSAGE_LENGTHS_FILENAME = "message_lengths.jsonl"
ARRAY_RECORD_SUFFIX = ".array_record"

SOURCE_ID_KEY = "_omegalax_source_id"
BATCH_SOURCE_IDS_KEY = "source_ids"

# Worker-process global for the parallel chunk-index builder (origin/main).
# Initialized once per worker via the Pool initializer, then reused for every
# message-length call to avoid pickling the tokenizer per task.
_measure_fn = None


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
            assert isinstance(messages, list), (
                f"Expected 'messages' to be a list at {path}:{line_num}"
            )

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
    assert metadata_path.is_file(), (
        f"Compiled Grain dataset metadata does not exist: {metadata_path}"
    )
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
    fsdp_size: int | None = None,
) -> int:
    if num_batches <= 0:
        return 1
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    metadata = load_compiled_metadata(path)
    num_records = int(metadata["num_records"])
    dp = dp_size * fsdp_size
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


def _measure_init(measure_message):
    """Pool initializer: install the measure fn in the worker's module global.

    Required because the workers run under the ``spawn`` start method (see
    ``_compute_message_lengths``), which does not inherit the parent's globals;
    ``measure_message`` is pickled in via ``initargs`` instead.
    """
    global _measure_fn
    _measure_fn = measure_message


def _measure_worker(keyed_message):
    key, message = keyed_message
    return key, _measure_fn(message)


def _compute_message_lengths(payload_path, measure_message, num_workers):
    """Tokenize every message in ``payload_path`` once, in parallel.

    Returns ``{(record_idx, msg_offset): measurement}`` where ``measurement`` is
    whatever ``measure_message`` returns (an ``int`` or a dict with a
    ``"length"`` key). This is the expensive, tokenizer-bound pass; everything
    downstream (binning, truncation, stats) is a pure function of its output.

    Workers run under the ``spawn`` start method, not ``fork``. The dataset
    carries images by reference into a native ArrayRecord store; a forked
    worker inherits the parent's thread-tainted ArrayRecord runtime and
    segfaults on its first image read (deadlocking the pool). ``spawn`` starts
    each worker from a clean interpreter that opens its own readers. Because
    ``spawn`` does not inherit globals, ``measure_message`` (a picklable
    ``qwen3_encoding._MessageLengthFn``) is shipped to each worker via the pool
    initializer.
    """
    tasks: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for record_idx, block in _iter_indexed_records(payload_path):
        for msg_offset, message in enumerate(block["messages"]):
            tasks.append(((record_idx, msg_offset), message))

    ctx = mp.get_context("spawn")
    chunksize = max(1, min(32, len(tasks) // num_workers))
    with ctx.Pool(num_workers, initializer=_measure_init, initargs=(measure_message,)) as pool:
        results = dict(
            tqdm(
                pool.imap_unordered(_measure_worker, tasks, chunksize=chunksize),
                total=len(tasks),
                desc=f"Measuring messages ({num_workers} workers)",
            )
        )
    return results


def _write_message_lengths(path: str | Path, results: dict) -> None:
    """Persist ``_compute_message_lengths`` output as JSONL (one row per
    message), so it can be reloaded by :func:`_load_message_lengths` to skip
    re-tokenization on subsequent chunk-index builds."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for (record_idx, msg_offset), measurement in sorted(results.items()):
            f.write(
                json.dumps(
                    {
                        "record_idx": record_idx,
                        "msg_offset": msg_offset,
                        "measurement": measurement,
                    }
                )
                + "\n"
            )


def _load_message_lengths(path: str | Path) -> dict:
    """Inverse of :func:`_write_message_lengths`: reconstruct the
    ``{(record_idx, msg_offset): measurement}`` map from the JSONL cache."""
    results: dict[tuple[int, int], Any] = {}
    with Path(path).expanduser().open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            results[(int(row["record_idx"]), int(row["msg_offset"]))] = row["measurement"]
    return results


def _validate_message_lengths(payload_path, results: dict) -> None:
    """Fail loudly if a cached length map does not match the payload exactly.

    Guards against a stale cache (payload changed, tokenizer/processor changed
    upstream of the cache, partial write): every payload message must have an
    entry and the counts must match. Cheap relative to tokenization -- a single
    metadata pass over the payload, no encoding.
    """
    expected = 0
    missing: list[tuple[int, int]] = []
    for record_idx, block in _iter_indexed_records(payload_path):
        for msg_offset in range(len(block["messages"])):
            expected += 1
            if (record_idx, msg_offset) not in results:
                if len(missing) < 5:
                    missing.append((record_idx, msg_offset))
    if missing or len(results) != expected:
        raise ValueError(
            f"cached message lengths do not match payload {payload_path}: payload has "
            f"{expected} messages, cache has {len(results)} entries"
            + (f"; first missing keys: {missing}" if missing else "")
            + ". The cache is stale for this payload -- delete it and re-measure."
        )


def _resolve_message_lengths(payload_path, measure_message, num_workers, message_lengths_path):
    """Return the per-message length map, loading from / writing to a cache file
    when ``message_lengths_path`` is given.

    * cache present  -> load it (skips the tokenizer pass entirely) and validate;
    * cache requested but absent -> compute, then write it for next time;
    * no cache path   -> compute, don't persist (legacy behaviour).
    """
    if message_lengths_path is not None:
        cache_path = Path(message_lengths_path).expanduser()
        if cache_path.exists():
            print(f"[chunk_index] loading cached message lengths from {cache_path}", flush=True)
            results = _load_message_lengths(cache_path)
            _validate_message_lengths(payload_path, results)
            return results

    results = _compute_message_lengths(payload_path, measure_message, num_workers)

    if message_lengths_path is not None:
        _write_message_lengths(message_lengths_path, results)
        print(f"[chunk_index] wrote message-length cache to {message_lengths_path}", flush=True)
    return results


def measure_message_lengths(
    payload_path: str | Path,
    out_path: str | Path,
    *,
    measure_message,
    num_workers: int = 2,
) -> Path:
    """Tokenize every message in a payload once and write the length cache.

    Standalone entry point for the offline "measure" stage: produces the
    ``message_lengths.jsonl`` file that :func:`build_chunk_index` consumes via
    its ``message_lengths_path`` argument, so re-binning at a different
    ``max_length`` / ``overflow_mode`` never re-tokenizes.
    """
    payload_path = Path(payload_path).expanduser().resolve()
    out_path = Path(out_path).expanduser()
    results = _compute_message_lengths(payload_path, measure_message, num_workers)
    _write_message_lengths(out_path, results)
    print(f"[measure] wrote {len(results)} message lengths to {out_path}", flush=True)
    return out_path


def _compute_distribution(values: list[int]) -> dict[str, int | float]:
    """Compute summary statistics for a list of integers."""
    if not values:
        return {
            "sum": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
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
    msg_text_tokens = [total - vis for total, vis in zip(msg_lengths, msg_vision_tokens)]
    chunk_text_tokens = [total - vis for total, vis in zip(chunk_lengths, chunk_vision_tokens)]
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
        "image_shapes": dict(sorted(image_shape_counts.items(), key=lambda kv: -kv[1])),
        "vision_variability": {
            "num_images_per_chunk": _frequency_table(chunk_num_images),
            "vision_tokens_per_chunk": _frequency_table(chunk_vision_tokens),
            "vision_patches_per_chunk": _frequency_table(chunk_vision_patches),
        },
    }
    stats_path = out_dir / TOKEN_STATS_FILENAME
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")


def _emit_sequence_lengths(
    out_dir: Path,
    *,
    sequence_stats: dict[str, dict[str, int]],
    effective_max: int,
) -> None:
    """Write one JSON object per conversation (session) to
    ``sequence_lengths.jsonl`` in ``out_dir``: the exact measured token length
    of the full conversation plus its text/vision breakdown. These are raw
    per-sequence measurements, independent of ``overflow_mode``; only
    ``exceeds_max_length`` depends on ``max_length``.
    """
    path = out_dir / SEQUENCE_LENGTHS_FILENAME
    with path.open("w") as f:
        for session_id, agg in sequence_stats.items():
            total = agg["total_tokens"]
            record = {
                "session_id": session_id,
                "num_messages": agg["num_messages"],
                "total_tokens": total,
                "text_tokens": total - agg["vision_tokens"],
                "vision_tokens": agg["vision_tokens"],
                "num_images": agg["num_images"],
                "max_message_tokens": agg["max_message_tokens"],
                "exceeds_max_length": total > effective_max,
                "num_messages_over_budget": agg["num_messages_over_budget"],
            }
            f.write(json.dumps(record) + "\n")
    print(
        f"[chunk_index] wrote {len(sequence_stats)} per-sequence token lengths to {path.name}",
        flush=True,
    )


def _emit_truncation_stats(
    out_dir: Path,
    *,
    overflow_mode: str,
    max_length: int,
    system_message_length: int,
    effective_max: int,
    total_sessions: int,
    total_message_tokens: int,
    session_chunk_counts: dict[str, int],
    prefix_sessions: set[str],
    overflow_sessions: set[str],
    dropped_sessions: set[str],
    dropped_messages: int,
    dropped_tokens: int,
) -> None:
    """Summarise per-session truncation/splitting, print it, and persist it to
    ``truncation_stats.json`` in ``out_dir``.

    ``prefix_sessions`` lost their tail because a single turn exceeded the
    budget (``split``/``truncate``); ``overflow_sessions`` lost their tail
    because packing overflowed (``truncate`` only); ``dropped_sessions`` were
    discarded wholesale because they did not fit in a single chunk (``drop``
    only). ``split`` mode never drops accumulation overflow, so its only
    dropped tokens come from the single-turn-too-big case.
    """
    truncated_sessions = prefix_sessions | overflow_sessions
    num_chunks = sum(session_chunk_counts.values())
    sessions_with_chunks = len(session_chunk_counts)
    sessions_split = sum(1 for c in session_chunk_counts.values() if c > 1)
    sessions_dropped_entirely = total_sessions - sessions_with_chunks
    kept_tokens = total_message_tokens - dropped_tokens

    summary = {
        "overflow_mode": overflow_mode,
        "max_length": max_length,
        "system_message_length": system_message_length,
        "effective_max": effective_max,
        "sessions": {
            "total": total_sessions,
            "emitted_at_least_one_chunk": sessions_with_chunks,
            "split_into_multiple_chunks": sessions_split,
            "truncated_total": len(truncated_sessions),
            "truncated_overflow": len(overflow_sessions),
            "truncated_single_message": len(prefix_sessions),
            "dropped_whole_session": len(dropped_sessions),
            "dropped_entirely": sessions_dropped_entirely,
        },
        "chunks": {
            "emitted": num_chunks,
            "max_per_session": max(session_chunk_counts.values(), default=0),
        },
        "messages": {"dropped": dropped_messages},
        "tokens": {
            "total_measured": total_message_tokens,
            "kept": kept_tokens,
            "dropped": dropped_tokens,
            "dropped_fraction": (
                round(dropped_tokens / total_message_tokens, 6) if total_message_tokens else 0.0
            ),
        },
    }
    (out_dir / TRUNCATION_STATS_FILENAME).write_text(json.dumps(summary, indent=2) + "\n")

    pct = summary["tokens"]["dropped_fraction"] * 100
    print(
        "[chunk_index] truncation summary "
        f"(overflow_mode={overflow_mode}, effective_max={effective_max} "
        f"= max_length={max_length} - system_tokens={system_message_length}):\n"
        f"  sessions: total={total_sessions} emitted={sessions_with_chunks} "
        f"split={sessions_split} truncated={len(truncated_sessions)} "
        f"(overflow={len(overflow_sessions)}, single_msg={len(prefix_sessions)}) "
        f"dropped_whole={len(dropped_sessions)} "
        f"dropped_entirely={sessions_dropped_entirely}\n"
        f"  chunks_emitted={num_chunks}\n"
        f"  messages_dropped={dropped_messages}\n"
        f"  tokens: total={total_message_tokens} kept={kept_tokens} "
        f"dropped={dropped_tokens} ({pct:.3f}%)",
        flush=True,
    )


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
    system_message: dict[str, Any] | None = None,
    overflow_mode: str = "split",
    message_lengths_path: str | Path | None = None,
) -> Path:
    """Build an offline chunk index over a canonical payload-block dataset.

    ``measure_message`` is called exactly once per message and must return either
    the number of tokens (``int``) or a dict containing at least a ``"length"``
    key.  When a dict is returned, extra fields (``vision_tokens``,
    ``num_images``, ``image_grid_thw``) are aggregated into per-chunk
    descriptors and a ``token_stats.json`` summary is written next to the index.

    If ``system_message`` is provided, every emitted chunk has it prepended at
    iteration time (see :class:`_ChunkDescriptorResolver`). The system message
    is measured once with ``measure_message`` and its token count is subtracted
    from the per-chunk budget so ``system_tokens + chunk_content_tokens``
    stays within ``max_length``.

    ``overflow_mode`` controls what happens to a conversation (session) whose
    turns do not all fit within ``effective_max = max_length - system_tokens``:

    * ``"split"`` (default, legacy behaviour): the session is packed into as
      many consecutive ≤budget chunks as needed at turn boundaries. No turns
      are dropped (every chunk is a fresh training sample with no shared
      history across the split).
    * ``"truncate"``: only the first ≤budget chunk (the longest prefix of whole
      turns that fits) is kept; the overflowing turn and the rest of the
      session are dropped.
    * ``"drop"``: any conversation that does not fit entirely in a single chunk
      (``total_tokens > effective_max``) is dropped wholesale; nothing is
      emitted for it. Only conversations that fit within the budget survive.

    In ``split`` and ``truncate`` a single turn that alone exceeds
    ``effective_max`` triggers prefix-truncation (the over-length turn and the
    session tail are dropped); in ``drop`` that whole session is dropped too.

    Truncation accounting (sessions/messages/tokens dropped, sessions split) is
    printed to stdout and written to ``truncation_stats.json`` next to the
    index.

    ``message_lengths_path`` enables the per-message length cache. Tokenization
    (``measure_message`` over every message) is the only step that depends on
    the tokenizer/processor and not on ``max_length`` / ``overflow_mode`` /
    ``system_message``. When this path is given and exists, the cache is loaded
    and the tokenizer pass is skipped; when it is given but absent, lengths are
    measured and written there for reuse; when ``None`` (default), lengths are
    measured in-memory and not persisted. Build it once with
    :func:`measure_message_lengths`, then point every chunk-index build over the
    same payload at it to avoid re-tokenizing per sequence length.
    """

    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if overflow_mode not in ("split", "truncate", "drop"):
        raise ValueError(
            f"overflow_mode must be 'split', 'truncate', or 'drop', got {overflow_mode!r}"
        )

    payload_path = Path(payload_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    payload_metadata = load_compiled_metadata(payload_path)
    if "payload_path" in payload_metadata:
        raise ValueError(
            f"Chunk indices can only be built from payload datasets, got chunk index: {payload_path}"
        )

    system_message_length = 0
    if system_message is not None:
        system_result = measure_message(system_message)
        system_message_length = (
            system_result["length"] if isinstance(system_result, dict) else int(system_result)
        )
        if system_message_length >= max_length:
            raise ValueError(
                f"system_message ({system_message_length} tokens) leaves no room "
                f"for content under max_length={max_length}"
            )
    effective_max = max_length - system_message_length

    precomputed_lengths = _resolve_message_lengths(
        payload_path, measure_message, num_workers, message_lengths_path
    )

    # Single prescan pass over every message, accumulating:
    #  * session_truncate_at: earliest (record_idx, msg_offset) of a message
    #    exceeding the chunk budget (the binner emits chunks for the valid
    #    prefix and drops the over-length message + tail);
    #  * sequence_stats: per-session (= per-conversation) token totals, written
    #    out verbatim as ``sequence_lengths.jsonl``. A session may span several
    #    payload blocks; blocks of a session are contiguous, so we accumulate.
    session_truncate_at: dict[str, tuple[int, int]] = {}
    sequence_stats: dict[str, dict[str, int]] = {}
    total_message_tokens = 0
    for record_idx, block in _iter_indexed_records(payload_path):
        block_session_id = str(block["session_id"])
        agg = sequence_stats.get(block_session_id)
        if agg is None:
            agg = {
                "num_messages": 0,
                "total_tokens": 0,
                "vision_tokens": 0,
                "num_images": 0,
                "max_message_tokens": 0,
                "num_messages_over_budget": 0,
            }
            sequence_stats[block_session_id] = agg
        for msg_offset in range(len(block["messages"])):
            result = precomputed_lengths[(record_idx, msg_offset)]
            if isinstance(result, dict):
                msg_length = int(result["length"])
                msg_vision_tokens = int(result["vision_tokens"])
                msg_num_images = int(result["num_images"])
            else:
                msg_length = int(result)
                msg_vision_tokens = 0
                msg_num_images = 0
            total_message_tokens += msg_length
            agg["num_messages"] += 1
            agg["total_tokens"] += msg_length
            agg["vision_tokens"] += msg_vision_tokens
            agg["num_images"] += msg_num_images
            if msg_length > agg["max_message_tokens"]:
                agg["max_message_tokens"] = msg_length
            if msg_length > effective_max:
                agg["num_messages_over_budget"] += 1
                if block_session_id not in session_truncate_at:
                    session_truncate_at[block_session_id] = (record_idx, msg_offset)
    all_session_ids = set(sequence_stats)
    if session_truncate_at:
        print(
            f"[chunk_index] prefix-truncating {len(session_truncate_at)} session(s) "
            f"at the first message exceeding effective_max={effective_max} "
            f"(max_length={max_length}, system_tokens={system_message_length}; "
            f"valid prefix turns are preserved as chunks): "
            f"{sorted(session_truncate_at.keys())[:5]}"
            + (" ..." if len(session_truncate_at) > 5 else ""),
            flush=True,
        )

    # drop mode: any conversation that doesn't fit in a single chunk is dropped
    # wholesale (handled at the top of the binner loop). Empty in other modes.
    drop_sessions: set[str] = set()
    if overflow_mode == "drop":
        drop_sessions = {
            sid for sid, agg in sequence_stats.items() if agg["total_tokens"] > effective_max
        }
        if drop_sessions:
            print(
                f"[chunk_index] drop mode: dropping {len(drop_sessions)} session(s) whose "
                f"total length exceeds effective_max={effective_max} "
                f"(max_length={max_length}, system_tokens={system_message_length})",
                flush=True,
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

    # -- truncation accounting (populated lazily by the generator) -------------
    # prefix:   session lost its tail because a single turn exceeded the budget
    # overflow: session lost its tail because packing overflowed (truncate mode)
    _trunc: dict[str, Any] = {
        "prefix_sessions": set(),
        "overflow_sessions": set(),
        "dropped_sessions": set(),  # drop mode: whole conversation discarded
        "dropped_messages": 0,
        "dropped_tokens": 0,
    }
    _session_chunk_counts: dict[str, int] = {}

    def _dropped_range(record_idx: int, block: dict[str, Any], start_offset: int):
        """Count (messages, tokens) dropped from ``start_offset`` to block end."""
        n_msgs = 0
        n_toks = 0
        for off in range(start_offset, len(block["messages"])):
            result = precomputed_lengths[(record_idx, off)]
            n_toks += result["length"] if isinstance(result, dict) else int(result)
            n_msgs += 1
        return n_msgs, n_toks

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
            # Skip chunks whose loss mask would be all zeros (no assistant
            # tokens to supervise). Comes up after prefix-truncation on
            # single-turn data where the over-length message is the only
            # assistant turn.
            if not any(m.get("role") == "assistant" for m in current_messages):
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
            _session_chunk_counts[current_session_id] = (
                _session_chunk_counts.get(current_session_id, 0) + 1
            )
            return descriptor

        truncated_sessions: set[str] = set()
        for record_idx, block in _iter_indexed_records(payload_path):
            block_session_id = str(block["session_id"])
            if block_session_id in drop_sessions:
                # drop mode: discard the whole conversation, emit nothing. The
                # pending keeper session (if any) stays buffered and is emitted
                # when the next keeper arrives or at the final flush.
                dm, dt = _dropped_range(record_idx, block, 0)
                _trunc["dropped_messages"] += dm
                _trunc["dropped_tokens"] += dt
                _trunc["dropped_sessions"].add(block_session_id)
                continue
            if block_session_id in truncated_sessions:
                # whole remaining block of an already-truncated session is dropped
                dm, dt = _dropped_range(record_idx, block, 0)
                _trunc["dropped_messages"] += dm
                _trunc["dropped_tokens"] += dt
                continue
            truncate_pos = session_truncate_at.get(block_session_id)
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
                if truncate_pos is not None and (record_idx, msg_offset) >= truncate_pos:
                    descriptor = emit_current()
                    if descriptor is not None:
                        yield descriptor
                    current_messages = []
                    current_length = 0
                    current_vision_tokens = 0
                    current_vision_patches = 0
                    current_num_images = 0
                    truncated_sessions.add(block_session_id)
                    _trunc["prefix_sessions"].add(block_session_id)
                    dm, dt = _dropped_range(record_idx, block, msg_offset)
                    _trunc["dropped_messages"] += dm
                    _trunc["dropped_tokens"] += dt
                    break

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

                assert msg_length <= effective_max, (
                    f"prefix-truncation pre-scan missed session={block_session_id} "
                    f"record={record_idx} offset={msg_offset} "
                    f"(msg_length={msg_length} > effective_max={effective_max}, "
                    f"max_length={max_length}, system_tokens={system_message_length})"
                )

                if not current_messages:
                    start_record_idx = record_idx
                    start_message_offset = msg_offset
                elif current_length + msg_length > effective_max:
                    descriptor = emit_current()
                    if descriptor is not None:
                        yield descriptor
                    current_messages = []
                    current_length = 0
                    current_vision_tokens = 0
                    current_vision_patches = 0
                    current_num_images = 0
                    if overflow_mode == "truncate":
                        # Keep only the first chunk; drop the overflowing turn
                        # and the rest of the session.
                        truncated_sessions.add(block_session_id)
                        _trunc["overflow_sessions"].add(block_session_id)
                        dm, dt = _dropped_range(record_idx, block, msg_offset)
                        _trunc["dropped_messages"] += dm
                        _trunc["dropped_tokens"] += dt
                        break
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
            "overflow_mode": overflow_mode,
            "profile_metadata": profile_metadata or {},
            "system_message": system_message,
            "system_message_length": system_message_length,
        },
    )

    _emit_sequence_lengths(
        out_dir,
        sequence_stats=sequence_stats,
        effective_max=effective_max,
    )

    _emit_truncation_stats(
        out_dir,
        overflow_mode=overflow_mode,
        max_length=max_length,
        system_message_length=system_message_length,
        effective_max=effective_max,
        total_sessions=len(all_session_ids),
        total_message_tokens=total_message_tokens,
        session_chunk_counts=_session_chunk_counts,
        prefix_sessions=_trunc["prefix_sessions"],
        overflow_sessions=_trunc["overflow_sessions"],
        dropped_sessions=_trunc["dropped_sessions"],
        dropped_messages=_trunc["dropped_messages"],
        dropped_tokens=_trunc["dropped_tokens"],
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
    def __init__(
        self,
        payload_path: str | Path,
        *,
        system_message: dict[str, Any] | None = None,
    ) -> None:
        self._payload_shards = [str(path) for path in resolve_arrayrecord_paths(payload_path)]
        self._payload_source = None
        self._system_message = system_message

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

        if self._system_message is not None:
            messages = [self._system_message, *messages]

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
            [int(ex.get(SOURCE_ID_KEY, 0)) for ex in examples],
            dtype=np.int32,
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
    prefetch_buffer_size: int = 4,
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
    sources: list[MixSource],
    metadatas: list[dict[str, Any]],
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
    tokenizer_ids = {(m.get("profile_metadata") or {}).get("tokenizer_id") for m in metadatas}
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
    dp_size: int,
    fsdp_size: int,
):
    """Create a checkpointable Grain iterator over one or more chunk-index datasets.

    When more than one source is supplied, examples are interleaved at the
    configured ``MixSource.weight`` ratios via ``grain.MapDataset.mix`` —
    every batch is a stochastic mix at the configured ratio, not a per-batch
    round-robin. ``num_epochs=None`` repeats each source indefinitely; set a
    finite value (per source) only for validation-style finite iteration.

    Data-parallel sharding spans both axes: ``dp = dp_size * fsdp_size``. The
    process's slot is ``jax.process_index() % dp``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    mix_sources = _coerce_sources(sources)
    if any(s.weight < 0.0 for s in mix_sources):
        raise ValueError(f"Source weights must be non-negative: {[s.weight for s in mix_sources]}")
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
        [mix_sources[i] for i in active_indices],
        metadatas,
    )
    norm_weights = [mix_sources[i].weight / total_w for i in active_indices]

    mp_options = multiprocessing_options or make_grain_multiprocessing_options()
    read_options = read_options or make_grain_read_options()
    dp = dp_size * fsdp_size
    dp_index = jax.process_index() % dp

    per_source: list[grain.MapDataset] = []
    for active_idx, original_idx in enumerate(active_indices):
        s = mix_sources[original_idx]
        m = metadatas[active_idx]
        shard_paths = [str(p) for p in resolve_arrayrecord_paths(s.path)]
        payload_path = str(m["payload_path"])
        system_message = m.get("system_message")
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
        ds = ds.map(_ChunkDescriptorResolver(payload_path, system_message=system_message))
        # Tag with the user-facing source id (position in the original list),
        # not the active-only index, so metric labels are stable across
        # ablations that zero out individual sources.
        ds = ds.map(_TagSourceMap(source_id=original_idx))
        per_source.append(ds)

    mixed = (
        per_source[0]
        if len(per_source) == 1
        else grain.MapDataset.mix(per_source, weights=norm_weights)
    )
    batched = mixed.batch(
        batch_size=batch_size,
        drop_remainder=True,
        batch_fn=_SourceTaggingCollator(batch_fn),
    )
    iter_ds = batched.to_iter_dataset(read_options)
    if mp_options.num_workers > 0:
        iter_ds = iter_ds.mp_prefetch(mp_options)
    return iter(iter_ds)
