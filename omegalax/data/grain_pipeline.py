"""Grain-backed SFT dataset compilation, chunk indexing, and iteration helpers."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import grain
import jax
import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter
from tqdm import tqdm

COMPILED_DATASET_VERSION = 1
COMPILED_METADATA_FILENAME = "metadata.json"
TOKEN_STATS_FILENAME = "token_stats.json"
TRUNCATION_STATS_FILENAME = "truncation_stats.json"
SEQUENCE_LENGTHS_FILENAME = "sequence_lengths.jsonl"
ARRAY_RECORD_SUFFIX = ".array_record"

SOURCE_ID_KEY = "_omegalax_source_id"
BATCH_SOURCE_IDS_KEY = "source_ids"

_prepare_conversation_fn = None


@dataclass(frozen=True)
class MixSource:
    """One dataset in a (potentially mixed) training corpus.

    ``path`` is an inline-records dataset directory (:func:`build_records_from_chat`
    output, with metadata.json). ``weight`` is unnormalized — relative weights
    across sources determine the realized example mix (see
    ``grain.MapDataset.mix``).
    """

    path: str | Path
    weight: float


def parse_data_mix(spec: str) -> list[MixSource]:
    return [MixSource(**entry) for entry in json.loads(spec)]


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

    metadata = {
        "version": COMPILED_DATASET_VERSION,
        "num_records": total_records,
        "num_shards": len(shard_paths),
        "shard_paths": shard_paths,
    }
    (out_dir / COMPILED_METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n")
    return out_dir


def _build_session_id(path: Path, line_num: int) -> str:
    return f"{path.stem}-{line_num:09d}"


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


def _process_worker(task):
    conv_idx, session_id, session_meta, messages, effective_max, overflow_mode = task
    prepared = _prepare_conversation_fn(messages)
    return (
        conv_idx,
        session_id,
        _process_conversation(
            session_id,
            session_meta,
            messages,
            prepared,
            effective_max=effective_max,
            overflow_mode=overflow_mode,
        ),
    )


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


def _iter_chat_conversations(path: Path):
    """Yield ``(conv_idx, session_id, session_meta, messages)`` per non-empty row."""
    with path.open() as f:
        conv_idx = 0
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
            yield conv_idx, session_id, session_meta, messages
            conv_idx += 1


def _prepare_init(prepare_conversation) -> None:
    global _prepare_conversation_fn
    _prepare_conversation_fn = prepare_conversation


def _emit_sequence_lengths(out_dir: Path, *, sequence_stats: dict, effective_max: int) -> None:
    """Write one JSON object per conversation to ``sequence_lengths.jsonl``: the
    exact measured token length of the full conversation plus its text/vision
    breakdown. Raw per-sequence measurements, independent of ``overflow_mode``;
    only ``exceeds_max_length`` depends on ``max_length``."""
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
        f"[records] wrote {len(sequence_stats)} per-sequence token lengths to {path.name}",
        flush=True,
    )


def _emit_truncation_stats(
    out_dir: Path,
    *,
    overflow_mode: str,
    max_length: int,
    effective_max: int,
    total_sessions: int,
    total_message_tokens: int,
    total_supervised_tokens: int,
    session_chunk_counts: dict[str, int],
    prefix_sessions: set[str],
    overflow_sessions: set[str],
    dropped_sessions: set[str],
    dropped_messages: int,
    dropped_tokens: int,
    dropped_supervised_tokens: int,
    emitted_tokens: int,
    emitted_supervised_tokens: int,
    supervision_basis: str,
) -> None:
    """Summarise per-session truncation/splitting, print it, and persist it to
    ``truncation_stats.json``."""
    truncated_sessions = prefix_sessions | overflow_sessions
    num_chunks = sum(session_chunk_counts.values())
    sessions_with_chunks = len(session_chunk_counts)
    sessions_split = sum(1 for c in session_chunk_counts.values() if c > 1)
    sessions_dropped_entirely = total_sessions - sessions_with_chunks
    kept_tokens = total_message_tokens - dropped_tokens
    kept_supervised = total_supervised_tokens - dropped_supervised_tokens
    if total_supervised_tokens != kept_supervised + dropped_supervised_tokens:
        raise RuntimeError("supervision accounting invariant failed")
    boundary_tokens = emitted_tokens - kept_tokens
    boundary_supervised = emitted_supervised_tokens - kept_supervised

    summary = {
        "overflow_mode": overflow_mode,
        "max_length": max_length,
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
            "chunk_boundary_adjustment": boundary_tokens,
            "dropped_fraction": (
                round(dropped_tokens / total_message_tokens, 6) if total_message_tokens else 0.0
            ),
        },
        "supervision": {
            "basis": supervision_basis,
            "total_measured": total_supervised_tokens,
            "kept": kept_supervised,
            "dropped": dropped_supervised_tokens,
            "chunk_boundary_adjustment": boundary_supervised,
            "emitted": emitted_supervised_tokens,
            "dropped_fraction": (
                round(dropped_supervised_tokens / total_supervised_tokens, 6)
                if total_supervised_tokens
                else 0.0
            ),
        },
    }
    (out_dir / TRUNCATION_STATS_FILENAME).write_text(json.dumps(summary, indent=2) + "\n")

    pct = summary["tokens"]["dropped_fraction"] * 100
    print(
        "[records] truncation summary "
        f"(overflow_mode={overflow_mode}, max_length={max_length}, "
        f"effective_max={effective_max}):\n"
        f"  sessions: total={total_sessions} emitted={sessions_with_chunks} "
        f"split={sessions_split} truncated={len(truncated_sessions)} "
        f"(overflow={len(overflow_sessions)}, single_msg={len(prefix_sessions)}) "
        f"dropped_whole={len(dropped_sessions)} "
        f"dropped_entirely={sessions_dropped_entirely}\n"
        f"  chunks_emitted={num_chunks}\n"
        f"  messages_dropped={dropped_messages}\n"
        f"  tokens: total={total_message_tokens} kept={kept_tokens} "
        f"dropped={dropped_tokens} ({pct:.3f}%)\n"
        f"  supervision: total={total_supervised_tokens} kept={kept_supervised} "
        f"dropped={dropped_supervised_tokens} "
        f"boundary_adjustment={boundary_supervised} emitted={emitted_supervised_tokens}",
        flush=True,
    )


def _process_conversation(
    session_id: str,
    session_meta: dict[str, Any],
    messages: list[dict[str, Any]],
    prepared,
    *,
    effective_max: int,
    overflow_mode: str,
) -> dict[str, Any]:
    """Build exact, independently renderable conversation chunks."""
    examples: list[dict[str, Any]] = []
    chunk_lengths: list[int] = []
    chunk_supervised_tokens: list[int] = []
    chunk_vision_tokens: list[int] = []
    chunk_vision_patches: list[int] = []
    chunk_num_images: list[int] = []
    chunk_num_messages: list[int] = []
    full = prepared(0, len(messages))
    message_measurements = prepared.message_measurements
    if sum(item["length"] for item in message_measurements) != full["length"]:
        raise RuntimeError("conversation message lengths do not sum to the rendered sequence")

    truncate_offset = next(
        (
            index
            for index, measurement in enumerate(message_measurements)
            if measurement["length"] > effective_max
        ),
        None,
    )
    limit = truncate_offset if truncate_offset is not None else len(messages)
    prefix_truncated = truncate_offset is not None
    overflow_truncated = False
    dropped_whole = False
    dropped_messages = 0
    dropped_tokens = 0
    dropped_supervised = 0

    def _drop(start: int, end: int) -> None:
        nonlocal dropped_messages, dropped_tokens, dropped_supervised
        for measurement in message_measurements[start:end]:
            dropped_messages += 1
            dropped_tokens += measurement["length"]
            dropped_supervised += measurement["supervised_tokens"]

    def _record(start: int, end: int, measurement: dict[str, Any]) -> None:
        example = dict(session_meta)
        example["messages"] = messages[start:end]
        example["_omegalax_session_id"] = session_id
        example["_omegalax_measured_length"] = measurement["length"]
        examples.append(example)
        chunk_lengths.append(measurement["length"])
        chunk_supervised_tokens.append(measurement["supervised_tokens"])
        chunk_vision_tokens.append(measurement["vision_tokens"])
        chunk_vision_patches.append(measurement["vision_patches"])
        chunk_num_images.append(measurement["num_images"])
        chunk_num_messages.append(end - start)

    def _longest_fitting(start: int, end_limit: int):
        best = None
        for end in range(start + 1, end_limit + 1):
            try:
                measurement = prepared(start, end)
            except ValueError:
                continue
            if (
                measurement["length"] <= effective_max
                and measurement["supervised_tokens"] > 0
            ):
                best = end, measurement
        return best

    if overflow_mode == "drop":
        if full["length"] <= effective_max and full["supervised_tokens"] > 0:
            _record(0, len(messages), full)
        else:
            _drop(0, len(messages))
            dropped_whole = True
    elif overflow_mode == "truncate":
        best = _longest_fitting(0, limit)
        if best is None:
            _drop(0, len(messages))
            dropped_whole = True
        else:
            end, measurement = best
            _record(0, end, measurement)
            _drop(end, len(messages))
            overflow_truncated = end < limit
    else:
        start = 0
        while start < limit:
            best = _longest_fitting(start, limit)
            if best is None:
                _drop(start, start + 1)
                start += 1
                continue
            end, measurement = best
            _record(start, end, measurement)
            start = end
        _drop(limit, len(messages))

    return {
        "examples": examples,
        "sequence_stats": {
            "num_messages": len(messages),
            "total_tokens": full["length"],
            "vision_tokens": full["vision_tokens"],
            "num_images": full["num_images"],
            "max_message_tokens": max(
                (item["length"] for item in message_measurements), default=0
            ),
            "num_messages_over_budget": sum(
                item["length"] > effective_max for item in message_measurements
            ),
            "total_supervised_tokens": full["supervised_tokens"],
        },
        "msg_lengths": [item["length"] for item in message_measurements],
        "msg_vision_tokens": [item["vision_tokens"] for item in message_measurements],
        "msg_num_images": [item["num_images"] for item in message_measurements],
        "chunk_lengths": chunk_lengths,
        "chunk_supervised_tokens": chunk_supervised_tokens,
        "chunk_vision_tokens": chunk_vision_tokens,
        "chunk_vision_patches": chunk_vision_patches,
        "chunk_num_images": chunk_num_images,
        "chunk_num_messages": chunk_num_messages,
        "image_shapes": [
            str(tuple(grid)) for item in message_measurements for grid in item["image_grid_thw"]
        ],
        "prefix_truncated": prefix_truncated,
        "overflow_truncated": overflow_truncated,
        "dropped_whole": dropped_whole,
        "dropped_messages": dropped_messages,
        "dropped_tokens": dropped_tokens,
        "dropped_supervised": dropped_supervised,
    }


def recording_split(recording_id: Any, val_fraction: float) -> str:
    """Deterministic recording-level train/val split (whole recording -> one side,
    so a recording never leaks across the split). No RNG/seed."""
    if val_fraction <= 0.0 or not recording_id:
        return "train"
    bucket = int(hashlib.sha1(str(recording_id).encode()).hexdigest(), 16) % 1000
    return "val" if bucket < round(val_fraction * 1000) else "train"


def build_records_from_chat(
    chat_path: str | Path,
    out_dir: str | Path,
    *,
    max_length: int,
    prepare_conversation,
    records_per_shard: int = 100_000,
    overwrite: bool = False,
    num_workers: int = 2,
    overflow_mode: str = "drop",
    val_fraction: float = 0.0,
    split: str | None = None,
    split_key: str = "recording_id",
) -> Path:
    """Build self-contained inline training records straight from a chat.jsonl.

    Reads chat.jsonl (one row per conversation), bins each conversation's turns
    into ``<= max_length`` token chunks, and writes ArrayRecord shards whose
    records ARE the training examples (message slices with ar:// image refs
    preserved) -- not pointers into a shared payload. The stage 01 master image
    store is unchanged; records reference it by ar:// exactly as chat.jsonl does.
    The trainer reads these records directly via :func:`make_grain_iterator`.

    The system prompt is NOT injected here: it is part of the conversation (the
    upstream chat.jsonl builder emits it as the first turn), so it is measured
    and budgeted as a normal message. ``overflow_mode`` is "drop" (default),
    "split", or "truncate".

    Train/val split (optional): when ``split`` is set, only conversations whose
    ``recording_split(row[split_key], val_fraction)`` equals ``split`` are emitted.
    ``split=None`` emits all conversations.
    """
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if num_workers <= 0:
        raise ValueError("num_workers must be > 0")
    if overflow_mode not in ("split", "truncate", "drop"):
        raise ValueError(
            f"overflow_mode must be 'split', 'truncate', or 'drop', got {overflow_mode!r}"
        )
    chat_path = Path(chat_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    effective_max = max_length

    def _in_split(session_meta: dict[str, Any]) -> bool:
        return split is None or recording_split(session_meta.get(split_key), val_fraction) == split

    tasks = [
        (conv_idx, session_id, session_meta, messages, effective_max, overflow_mode)
        for conv_idx, session_id, session_meta, messages in _iter_chat_conversations(chat_path)
        if _in_split(session_meta)
    ]
    sequence_stats: dict[str, dict[str, int]] = {}
    totals = {"tokens": 0, "supervised": 0}

    _msg_lengths: list[int] = []
    _msg_vision_tokens: list[int] = []
    _msg_num_images: list[int] = []
    _chunk_lengths: list[int] = []
    _chunk_supervised_tokens: list[int] = []
    _chunk_vision_tokens: list[int] = []
    _chunk_vision_patches: list[int] = []
    _chunk_num_images: list[int] = []
    _chunk_num_messages: list[int] = []
    _image_shape_counts: dict[str, int] = {}
    _trunc: dict[str, Any] = {
        "prefix_sessions": set(),
        "overflow_sessions": set(),
        "dropped_sessions": set(),
        "dropped_messages": 0,
        "dropped_tokens": 0,
        "dropped_supervised": 0,
    }
    _session_chunk_counts: dict[str, int] = {}

    def _iter_records(results):
        for _conv_idx, session_id, res in results:
            sequence_stats[session_id] = res["sequence_stats"]
            totals["tokens"] += res["sequence_stats"]["total_tokens"]
            totals["supervised"] += res["sequence_stats"]["total_supervised_tokens"]
            _msg_lengths.extend(res["msg_lengths"])
            _msg_vision_tokens.extend(res["msg_vision_tokens"])
            _msg_num_images.extend(res["msg_num_images"])
            _chunk_lengths.extend(res["chunk_lengths"])
            _chunk_supervised_tokens.extend(res["chunk_supervised_tokens"])
            _chunk_vision_tokens.extend(res["chunk_vision_tokens"])
            _chunk_vision_patches.extend(res["chunk_vision_patches"])
            _chunk_num_images.extend(res["chunk_num_images"])
            _chunk_num_messages.extend(res["chunk_num_messages"])
            for shape in res["image_shapes"]:
                _image_shape_counts[shape] = _image_shape_counts.get(shape, 0) + 1
            if res["prefix_truncated"]:
                _trunc["prefix_sessions"].add(session_id)
            if res["overflow_truncated"]:
                _trunc["overflow_sessions"].add(session_id)
            if res["dropped_whole"]:
                _trunc["dropped_sessions"].add(session_id)
            _trunc["dropped_messages"] += res["dropped_messages"]
            _trunc["dropped_tokens"] += res["dropped_tokens"]
            _trunc["dropped_supervised"] += res["dropped_supervised"]
            if res["examples"]:
                _session_chunk_counts[session_id] = len(res["examples"])
            yield from res["examples"]

    ctx = mp.get_context("spawn")
    chunksize = max(1, min(32, len(tasks) // num_workers))
    with ctx.Pool(
        num_workers,
        initializer=_prepare_init,
        initargs=(prepare_conversation,),
    ) as pool:
        results = tqdm(
            pool.imap(_process_worker, tasks, chunksize=chunksize),
            total=len(tasks),
            desc=f"Preparing conversations ({num_workers} workers)",
        )
        out_path = _write_arrayrecord_dataset(
            _iter_records(results),
            out_dir,
            records_per_shard=records_per_shard,
            overwrite=overwrite,
        )

    _emit_sequence_lengths(out_dir, sequence_stats=sequence_stats, effective_max=effective_max)
    _emit_truncation_stats(
        out_dir,
        overflow_mode=overflow_mode,
        max_length=max_length,
        effective_max=effective_max,
        total_sessions=len(sequence_stats),
        total_message_tokens=totals["tokens"],
        total_supervised_tokens=totals["supervised"],
        session_chunk_counts=_session_chunk_counts,
        prefix_sessions=_trunc["prefix_sessions"],
        overflow_sessions=_trunc["overflow_sessions"],
        dropped_sessions=_trunc["dropped_sessions"],
        dropped_messages=_trunc["dropped_messages"],
        dropped_tokens=_trunc["dropped_tokens"],
        dropped_supervised_tokens=_trunc["dropped_supervised"],
        emitted_tokens=sum(_chunk_lengths),
        emitted_supervised_tokens=sum(_chunk_supervised_tokens),
        supervision_basis="loss_mask",
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
    extra_transform: grain.transforms.RandomMap | None = None,
):
    """Create a checkpointable Grain iterator over one or more inline-records datasets.

    Each source is a :func:`build_records_from_chat` dataset whose records ARE
    the training examples, so there is no payload to resolve -- records are just
    JSON-decoded and used directly.

    When more than one source is supplied, examples are interleaved at the
    configured ``MixSource.weight`` ratios via ``grain.MapDataset.mix`` —
    every batch is a stochastic mix at the configured ratio, not a per-batch
    round-robin. ``num_epochs=None`` repeats each source indefinitely; set a
    finite value (per source) only for validation-style finite iteration.

    Data-parallel sharding spans both axes: ``dp = dp_size * fsdp_size``. The
    process's slot is ``jax.process_index() % dp``.

    ``extra_transform`` is an optional dataset-specific augmentation
    (e.g. action-magnitude scaling): a ``grain.transforms.RandomMap`` applied
    per-source via ``ds.random_map`` after content resolution and source
    tagging, before mixing and batching, so it sees fully-materialized chat
    content and may inspect the source-id tag to filter. It gets a
    deterministic per-source seed derived from ``seed``, so
    resume-from-checkpoint reproduces the same augmentation. Caller must
    restrict this to the train iterator — augmenting a val iterator silently
    corrupts metrics.
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
    norm_weights = [mix_sources[i].weight / total_w for i in active_indices]

    mp_options = multiprocessing_options or make_grain_multiprocessing_options()
    read_options = read_options or make_grain_read_options()
    dp = dp_size * fsdp_size
    dp_index = jax.process_index() % dp

    per_source: list[grain.MapDataset] = []
    for original_idx in active_indices:
        s = mix_sources[original_idx]
        shard_paths = [str(p) for p in resolve_arrayrecord_paths(s.path)]
        ds = grain.MapDataset.source(grain.sources.ArrayRecordDataSource(shard_paths))
        if dp > 1:
            # Contiguous-block DP shards with drop_remainder, matching the
            # legacy IndexSampler(ShardOptions(drop_remainder=True)) behavior.
            per_rank = len(ds) // dp
            ds = ds[dp_index * per_rank : (dp_index + 1) * per_rank]
        if shuffle:
            ds = ds.shuffle(seed=seed + original_idx)
        ds = ds.repeat(num_epochs)
        # Inline records ARE the examples: decode, then tag with the user-facing
        # source id (position in the original list, stable across ablations that
        # zero out individual sources).
        ds = ds.map(_JsonLoadsMap())
        ds = ds.map(_TagSourceMap(source_id=original_idx))
        # Optional user-supplied augmentation (a RandomMap), with a deterministic
        # per-source seed so resume-from-checkpoint reproduces it.
        if extra_transform is not None:
            ds = ds.random_map(extra_transform, seed=seed + 10_000 * (original_idx + 1))
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
