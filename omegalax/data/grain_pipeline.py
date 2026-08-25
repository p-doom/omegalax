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

from omegalax.data.artifact_contract import (
    file_identity,
    validate_measurement_contract,
)

COMPILED_DATASET_VERSION = 2
COMPILED_METADATA_FILENAME = "metadata.json"
TOKEN_STATS_FILENAME = "token_stats.json"
TRUNCATION_STATS_FILENAME = "truncation_stats.json"
SEQUENCE_LENGTHS_FILENAME = "sequence_lengths.jsonl"
# Per-message token-length cache for the payload-free inline path. Keyed by
# (conv_idx, msg_offset) -- conv_idx is the 0-based index over non-empty
# chat.jsonl rows (the no-payload analog of the payload record_idx). Independent
# of max_length / overflow_mode / split, so it is measured once
# (measure_message_lengths_from_chat) and reused across every seq-length build.
MESSAGE_LENGTHS_FILENAME = "message_lengths.jsonl"
MESSAGE_LENGTHS_VERSION = 2
ARRAY_RECORD_SUFFIX = ".array_record"

SOURCE_ID_KEY = "_omegalax_source_id"
BATCH_SOURCE_IDS_KEY = "source_ids"
CARRY_KEY = "_omegalax_carry_messages"
SPLIT_UNIT_ENDS_KEY = "_omegalax_split_unit_ends"

# Worker-process global for the parallel measure pass. Installed once per worker
# via the Pool initializer (_measure_init), then reused for every message-length
# call so the (picklable) measure fn is shipped once, not per task.
_measure_fn = None


@dataclass(frozen=True)
class MixSource:
    """One dataset in a (potentially mixed) training corpus.

    ``path`` is an inline-records dataset directory (:func:`build_records_from_chat`
    output, with metadata.json). ``weight`` is unnormalized — relative weights
    across sources determine the realized example mix (see
    ``grain.MapDataset.mix``).
    """

    path: str | Path
    weight: float = 1.0


def parse_data_mix(spec: str) -> list[MixSource]:
    """Parse a ``--data_mix`` JSON spec into a list of MixSource.

    ``weight`` is as required as ``path``: a defaulted weight turns a typo'd key
    into a uniform mixture that trains and reports as the recipe's own, which is
    the one failure mode a mixing run cannot detect from its metrics.
    """
    raw = json.loads(spec)
    if not isinstance(raw, list) or not raw:
        raise ValueError("--data_mix must be a non-empty JSON list of {path, weight} objects")
    out: list[MixSource] = []
    for entry in raw:
        if not isinstance(entry, dict) or not {"path", "weight"} <= entry.keys():
            raise ValueError(
                f"--data_mix entry must be an object with 'path' and 'weight' fields: {entry!r}"
            )
        out.append(MixSource(path=str(entry["path"]), weight=float(entry["weight"])))
    return out


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
            "complete": False,
            "num_records": total_records,
            "num_shards": len(shard_paths),
            "shard_paths": shard_paths,
        }
    )
    (out_dir / COMPILED_METADATA_FILENAME).write_text(json.dumps(final_metadata, indent=2) + "\n")
    return out_dir


def _build_session_id(path: Path, line_num: int) -> str:
    return f"{path.stem}-{line_num:09d}"


def resolve_arrayrecord_paths(path: str | Path) -> list[Path]:
    path = Path(path).expanduser().resolve()
    metadata = load_compiled_metadata(path)
    shard_paths = [path / rel for rel in metadata["shard_paths"]]

    invalid = [candidate for candidate in shard_paths if not candidate.is_file()]
    if invalid:
        raise ValueError(f"Compiled dataset has missing or non-file shard(s): {invalid}")
    return shard_paths


def load_compiled_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    metadata_path = path / COMPILED_METADATA_FILENAME
    if not metadata_path.is_file():
        raise ValueError(f"Compiled Grain dataset metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise TypeError("Compiled Grain dataset metadata must be an object")
    if metadata.get("version") != COMPILED_DATASET_VERSION or metadata.get("complete") is not True:
        raise ValueError(
            f"Compiled Grain dataset is incomplete or has unsupported version at {metadata_path}"
        )
    shard_paths = metadata.get("shard_paths")
    if (
        not isinstance(shard_paths, list)
        or not shard_paths
        or not all(
            isinstance(item, str) and Path(item).name == item and item.endswith(ARRAY_RECORD_SUFFIX)
            for item in shard_paths
        )
        or shard_paths != sorted(set(shard_paths))
        or metadata.get("num_shards") != len(shard_paths)
        or not isinstance(metadata.get("num_records"), int)
        or metadata["num_records"] <= 0
    ):
        raise ValueError(f"Compiled Grain dataset metadata is invalid: {metadata_path}")
    validate_measurement_contract(metadata.get("measurement_contract"))
    return metadata


def _mark_compiled_dataset_complete(out_dir: Path) -> None:
    metadata_path = out_dir / COMPILED_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("complete") is not False:
        raise ValueError(f"Compiled dataset completion state is invalid: {metadata_path}")
    metadata["complete"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


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


def _measure_worker(keyed_message):
    key, message = keyed_message
    return key, _measure_fn(message)


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
    """Yield ``(conv_idx, session_id, session_meta, messages)`` per non-empty row.

    ``conv_idx`` is a 0-based index over non-empty JSONL rows -- the no-payload
    analog of the payload ``record_idx`` and the key the length cache is built
    on. ``session_id`` mirrors the payload path's synthesis so provenance is
    identical (filename stem + 1-based file line).
    """
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
            marker = raw.get(CARRY_KEY, [])
            valid = (
                isinstance(marker, list)
                and all(type(i) is int and 0 <= i < len(messages) for i in marker)
                and marker == sorted(set(marker))
            )
            if not valid:
                raise ValueError(
                    f"{CARRY_KEY} must be strictly increasing in-range message indices, got "
                    f"{marker!r} for {len(messages)} messages at {path}:{line_num}"
                )
            unit_ends = raw.get(SPLIT_UNIT_ENDS_KEY, list(range(1, len(messages) + 1)))
            valid = (
                isinstance(unit_ends, list)
                and all(type(i) is int and 0 < i <= len(messages) for i in unit_ends)
                and unit_ends == sorted(set(unit_ends))
                and unit_ends[-1:] == ([len(messages)] if messages else [])
            )
            if not valid:
                raise ValueError(
                    f"{SPLIT_UNIT_ENDS_KEY} must be strictly increasing exclusive message "
                    f"offsets ending at {len(messages)}, got {unit_ends!r} at {path}:{line_num}"
                )
            session_id = _build_session_id(path, line_num)
            session_meta = {k: v for k, v in raw.items() if k not in {"messages", "session_id"}}
            session_meta[SPLIT_UNIT_ENDS_KEY] = unit_ends
            yield conv_idx, session_id, session_meta, messages
            conv_idx += 1


def _extract_measurement(result) -> tuple[int, int, int, int, list]:
    """Normalize a measure_message() result to (length, vision_tokens,
    vision_patches, num_images, image_grid_thw)."""
    if isinstance(result, dict):
        return (
            int(result["length"]),
            int(result["vision_tokens"]),
            int(result["vision_patches"]),
            int(result["num_images"]),
            result.get("image_grid_thw", []),
        )
    return (int(result), 0, 0, 0, [])


def _supervised_tokens(result, message: dict[str, Any]) -> int:
    if message.get("role") != "assistant":
        return 0
    if isinstance(result, dict) and "supervised_tokens" in result:
        return int(result["supervised_tokens"])
    return _extract_measurement(result)[0]


def _measure_init(measure_message) -> None:
    """Pool initializer: install the measure fn in the worker's module global.

    Required because the from-chat workers run under ``spawn`` (see
    ``_compute_message_lengths_from_chat``), which does not inherit the parent's
    globals; ``measure_message`` is pickled in via ``initargs`` instead.
    """
    global _measure_fn
    _measure_fn = measure_message


def _preflight_measure_fn(measure_message, tasks, chat_path) -> None:
    """Reject a whole-run misconfiguration in the PARENT, before spawning workers.

    A measure fn that cannot handle a message raises on the first one — but that
    happens inside a pool worker, so the operator sees a child traceback with the
    pool machinery stacked on top, after the job has started, instead of the flag
    they forgot. The task list is already fully enumerated here, so the same
    check is affordable up front.

    The condition itself lives on the measure fn (``reject_unmeasurable``), not
    here, so the two boundaries cannot disagree about what is measurable. A
    caller's own measure fn that does not declare one is left alone: an arbitrary
    callable has no contract this could check.
    """
    reject_unmeasurable = getattr(measure_message, "reject_unmeasurable", None)
    if reject_unmeasurable is None:
        return
    for (conv_idx, msg_offset), message in tasks:
        try:
            reject_unmeasurable(message)
        except ValueError as exc:
            raise ValueError(
                f"{Path(chat_path).name} conversation {conv_idx} message "
                f"{msg_offset}: {exc} Every worker would fail on it."
            ) from exc


def _compute_message_lengths_from_chat(chat_path, measure_message, num_workers) -> dict:
    """Tokenize every message in a chat.jsonl once, in parallel, under ``spawn``.

    Returns ``{(conv_idx, msg_offset): measurement}``. Uses the ``spawn`` start
    method, not ``fork``: ar:// image refs are read from a native ArrayRecord
    store, and a forked worker inherits a thread-tainted reader that segfaults on
    its first image read (deadlocking the pool). ``spawn`` starts each worker
    clean; ``measure_message`` is shipped to each worker via the pool
    initializer since spawn does not inherit globals.
    """
    tasks: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for conv_idx, _sid, _meta, messages in _iter_chat_conversations(chat_path):
        for msg_offset, message in enumerate(messages):
            tasks.append(((conv_idx, msg_offset), message))
    if not tasks:
        return {}
    _preflight_measure_fn(measure_message, tasks, chat_path)
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


def _message_lengths_header(chat_path: str | Path, measurement_contract: dict) -> dict:
    validate_measurement_contract(measurement_contract)
    return {
        "type": "omegalax_message_lengths",
        "version": MESSAGE_LENGTHS_VERSION,
        "source_chat": file_identity(chat_path),
        "measurement_contract": measurement_contract,
    }


def _write_chat_message_lengths(
    path: str | Path,
    results: dict,
    chat_path: str | Path,
    measurement_contract: dict,
) -> None:
    """Persist a source- and processor-bound message-length cache."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _message_lengths_header(chat_path, measurement_contract)
    with path.open("x") as file:
        file.write(json.dumps({"header": header}, sort_keys=True) + "\n")
        for (conv_idx, msg_offset), measurement in sorted(results.items()):
            file.write(
                json.dumps(
                    {
                        "conv_idx": conv_idx,
                        "msg_offset": msg_offset,
                        "measurement": measurement,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _load_chat_message_lengths(path: str | Path) -> tuple[dict, dict]:
    """Inverse of :func:`_write_chat_message_lengths`."""
    results: dict[tuple[int, int], Any] = {}
    with Path(path).expanduser().open() as f:
        first_line = f.readline()
        if not first_line:
            raise ValueError("message-length cache is empty")
        first_row = json.loads(first_line)
        header = first_row.get("header") if isinstance(first_row, dict) else None
        if not isinstance(header, dict):
            raise TypeError("message-length cache is missing its versioned header")
        if (
            header.get("type") != "omegalax_message_lengths"
            or header.get("version") != MESSAGE_LENGTHS_VERSION
        ):
            raise ValueError(
                f"unsupported message-length cache header: expected version {MESSAGE_LENGTHS_VERSION}"
            )
        validate_measurement_contract(header.get("measurement_contract"))
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (int(row["conv_idx"]), int(row["msg_offset"]))
            if key in results:
                raise ValueError(f"message-length cache contains duplicate key {key}")
            results[key] = row["measurement"]
    return header, results


def _validate_chat_message_lengths(
    chat_path,
    header: dict,
    results: dict,
    measurement_contract: dict,
) -> None:
    """Fail loudly if a cached length map does not match chat_path exactly."""
    validate_measurement_contract(measurement_contract)
    if header["source_chat"] != file_identity(chat_path):
        raise ValueError(
            f"cached message lengths do not match source chat identity for {chat_path}"
        )
    if header["measurement_contract"] != measurement_contract:
        raise ValueError("cached message lengths do not match the measurement contract")
    expected = 0
    missing: list[tuple[int, int]] = []
    for conv_idx, _sid, _meta, messages in _iter_chat_conversations(chat_path):
        for msg_offset in range(len(messages)):
            expected += 1
            if (conv_idx, msg_offset) not in results and len(missing) < 5:
                missing.append((conv_idx, msg_offset))
    if missing or len(results) != expected:
        raise ValueError(
            f"cached message lengths do not match chat dataset {chat_path}: chat has "
            f"{expected} messages, cache has {len(results)} entries"
            + (f"; first missing keys: {missing}" if missing else "")
            + ". The cache is stale for this chat.jsonl -- delete it and re-measure."
        )


def _resolve_chat_message_lengths(
    chat_path,
    measure_message,
    num_workers,
    message_lengths_path,
    measurement_contract,
):
    """Load-or-compute the per-message length map for the inline path.

    Mirrors the payload path's cache semantics: cache present -> load + validate;
    requested but absent -> compute then write; None -> compute in-memory.
    """
    if message_lengths_path is not None:
        if not measurement_contract:
            raise ValueError("message_lengths_path requires an explicit measurement_contract")
        cache_path = Path(message_lengths_path).expanduser()
        if cache_path.exists():
            print(f"[records] loading cached message lengths from {cache_path}", flush=True)
            header, results = _load_chat_message_lengths(cache_path)
            _validate_chat_message_lengths(chat_path, header, results, measurement_contract)
            return results

    results = _compute_message_lengths_from_chat(chat_path, measure_message, num_workers)

    if message_lengths_path is not None:
        _write_chat_message_lengths(message_lengths_path, results, chat_path, measurement_contract)
        print(f"[records] wrote message-length cache to {message_lengths_path}", flush=True)
    return results


def measure_message_lengths_from_chat(
    chat_path: str | Path,
    out_path: str | Path,
    *,
    measure_message,
    measurement_contract: dict,
    num_workers: int = 2,
) -> Path:
    """Tokenize every message in a chat.jsonl once and write the length cache.

    Standalone entry point for the payload-free "measure" stage: produces the
    ``message_lengths.jsonl`` that :func:`build_records_from_chat` consumes via
    its ``message_lengths_path``, so re-binning at a different ``max_length`` /
    ``overflow_mode`` never re-tokenizes. Reads chat.jsonl directly -- no
    intermediate grain payload.
    """
    chat_path = Path(chat_path).expanduser().resolve()
    out_path = Path(out_path).expanduser()
    results = _compute_message_lengths_from_chat(chat_path, measure_message, num_workers)
    _write_chat_message_lengths(out_path, results, chat_path, measurement_contract)
    print(f"[measure] wrote {len(results)} message lengths to {out_path}", flush=True)
    return out_path


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
    repeated_supervised_tokens: int,
    emitted_tokens: int,
    carried_tokens: int,
    carried_messages: int,
    chunks_with_carry: int,
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
    emitted_supervised = kept_supervised + repeated_supervised_tokens
    assert emitted_tokens == kept_tokens + carried_tokens
    assert total_supervised_tokens == kept_supervised + dropped_supervised_tokens

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
            "dropped_fraction": (
                round(dropped_tokens / total_message_tokens, 6) if total_message_tokens else 0.0
            ),
        },
        "supervision": {
            "basis": supervision_basis,
            "total_measured": total_supervised_tokens,
            "kept": kept_supervised,
            "dropped": dropped_supervised_tokens,
            "repeated": repeated_supervised_tokens,
            "emitted": emitted_supervised,
            "dropped_fraction": (
                round(dropped_supervised_tokens / total_supervised_tokens, 6)
                if total_supervised_tokens
                else 0.0
            ),
        },
        "carry": {
            "chunks_with_carry": chunks_with_carry,
            "carried_messages": carried_messages,
            "carried_tokens": carried_tokens,
            "emitted_tokens": emitted_tokens,
            "respent_fraction": (
                round(carried_tokens / emitted_tokens, 6) if emitted_tokens else 0.0
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
        f"dropped={dropped_supervised_tokens} repeated={repeated_supervised_tokens} "
        f"emitted={emitted_supervised}\n"
        f"  carry: chunks={chunks_with_carry} messages={carried_messages} "
        f"tokens={carried_tokens} of emitted={emitted_tokens}",
        flush=True,
    )


def _process_conversation(
    conv_idx: int,
    session_id: str,
    session_meta: dict[str, Any],
    messages: list[dict[str, Any]],
    precomputed: dict,
    *,
    effective_max: int,
    overflow_mode: str,
    truncate_offset: int | None,
    carry: tuple[int, ...],
    split_unit_ends: tuple[int, ...],
) -> dict[str, Any]:
    """Bin one conversation's messages into <=effective_max token chunks and
    build the self-contained inline example records for it.

    Pure function of the precomputed lengths; the caller handles drop-mode
    (whole-conversation) drops before calling this.

    Marked messages preceding a continuation are re-prepended within its budget.
    """
    examples: list[dict[str, Any]] = []
    msg_lengths: list[int] = []
    msg_vision_tokens: list[int] = []
    msg_num_images: list[int] = []
    chunk_lengths: list[int] = []
    chunk_vision_tokens: list[int] = []
    chunk_vision_patches: list[int] = []
    chunk_num_images: list[int] = []
    chunk_num_messages: list[int] = []
    image_shapes: list[str] = []
    prefix_truncated = False
    overflow_truncated = False
    dropped_messages = 0
    dropped_tokens = 0
    dropped_supervised = 0
    carried_messages = 0
    carried_tokens = 0
    repeated_supervised = 0
    chunks_with_carry = 0

    def _dropped(start: int) -> tuple[int, int, int]:
        dm = 0
        dt = 0
        ds = 0
        for off in range(start, len(messages)):
            length, *_ = _extract_measurement(precomputed[(conv_idx, off)])
            dm += 1
            dt += length
            ds += _supervised_tokens(precomputed[(conv_idx, off)], messages[off])
        return dm, dt, ds

    def _make_example(cur_msgs, cur_len, cur_vt, cur_vp, cur_ni):
        """Return an example, or charge an assistant-free slice as dropped."""
        nonlocal dropped_messages, dropped_tokens
        if not cur_msgs:
            return None
        if not any(m.get("role") == "assistant" for m in cur_msgs):
            dropped_messages += len(cur_msgs)
            dropped_tokens += cur_len
            return None
        chunk_msgs = carry_msgs + list(cur_msgs)
        chunk_len = carry_len + cur_len
        assert chunk_len <= effective_max, (
            f"emitted chunk over budget session={session_id} "
            f"(carried={carry_len} + slice={cur_len} > effective_max={effective_max})"
        )
        example = dict(session_meta)
        example["messages"] = chunk_msgs
        example["_omegalax_session_id"] = session_id
        example["_omegalax_measured_length"] = chunk_len
        return example, (
            chunk_len,
            carry_vt + cur_vt,
            carry_vp + cur_vp,
            carry_ni + cur_ni,
            len(chunk_msgs),
        )

    def _record(pair) -> None:
        nonlocal carried_messages, carried_tokens, repeated_supervised, chunks_with_carry
        example, chunk_stat = pair
        examples.append(example)
        length, vt, vp, ni, nm = chunk_stat
        chunk_lengths.append(length)
        chunk_vision_tokens.append(vt)
        chunk_vision_patches.append(vp)
        chunk_num_images.append(ni)
        chunk_num_messages.append(nm)
        if carry_msgs:
            chunks_with_carry += 1
            carried_messages += len(carry_msgs)
            carried_tokens += carry_len
            repeated_supervised += carry_supervised

    cur_msgs: list[dict[str, Any]] = []
    cur_len = 0
    cur_vt = 0
    cur_vp = 0
    cur_ni = 0
    pending_msgs: list[dict[str, Any]] = []
    pending_len = 0
    pending_vt = 0
    pending_vp = 0
    pending_ni = 0
    pending_supervised = 0
    carry_msgs: list[dict[str, Any]] = []
    carry_len = 0
    carry_vt = 0
    carry_vp = 0
    carry_ni = 0
    carry_supervised = 0

    unit_start = 0
    for unit_end in split_unit_ends:
        if truncate_offset is not None and unit_start >= truncate_offset:
            pair = _make_example(cur_msgs, cur_len, cur_vt, cur_vp, cur_ni)
            if pair is not None:
                _record(pair)
            prefix_truncated = True
            dm, dt, ds = _dropped(unit_start)
            dropped_messages += dm
            dropped_tokens += dt
            dropped_supervised += ds
            break

        unit_len = 0
        unit_vt = 0
        unit_vp = 0
        unit_ni = 0
        for msg_offset in range(unit_start, unit_end):
            result = precomputed[(conv_idx, msg_offset)]
            length, vt, vp, ni, grid = _extract_measurement(result)
            unit_len += length
            unit_vt += vt
            unit_vp += vp
            unit_ni += ni
            if isinstance(result, dict):
                msg_lengths.append(length)
                msg_vision_tokens.append(vt)
                msg_num_images.append(ni)
                for shape in grid:
                    image_shapes.append(str(tuple(shape)))

        assert pending_len + unit_len <= effective_max, (
            f"prefix-truncation pre-scan missed session={session_id} "
            f"offset={unit_start} (unit_length={unit_len} + carried prefix {pending_len} "
            f"> effective_max={effective_max})"
        )

        if not cur_msgs:
            pass
        elif carry_len + cur_len + unit_len > effective_max:
            pair = _make_example(cur_msgs, cur_len, cur_vt, cur_vp, cur_ni)
            if pair is not None:
                _record(pair)
            cur_msgs, cur_len, cur_vt, cur_vp, cur_ni = [], 0, 0, 0, 0
            if overflow_mode == "truncate":
                overflow_truncated = True
                dm, dt, ds = _dropped(unit_start)
                dropped_messages += dm
                dropped_tokens += dt
                dropped_supervised += ds
                break
            carry_msgs = list(pending_msgs)
            carry_len, carry_vt, carry_vp, carry_ni, carry_supervised = (
                pending_len,
                pending_vt,
                pending_vp,
                pending_ni,
                pending_supervised,
            )

        cur_msgs.extend(messages[unit_start:unit_end])
        cur_len += unit_len
        cur_vt += unit_vt
        cur_vp += unit_vp
        cur_ni += unit_ni
        for msg_offset in range(unit_start, unit_end):
            if msg_offset not in carry:
                continue
            result = precomputed[(conv_idx, msg_offset)]
            length, vt, vp, ni, _grid = _extract_measurement(result)
            pending_msgs.append(messages[msg_offset])
            pending_len += length
            pending_vt += vt
            pending_vp += vp
            pending_ni += ni
            pending_supervised += _supervised_tokens(result, messages[msg_offset])
        unit_start = unit_end
    else:
        pair = _make_example(cur_msgs, cur_len, cur_vt, cur_vp, cur_ni)
        if pair is not None:
            _record(pair)

    return {
        "examples": examples,
        "msg_lengths": msg_lengths,
        "msg_vision_tokens": msg_vision_tokens,
        "msg_num_images": msg_num_images,
        "chunk_lengths": chunk_lengths,
        "chunk_vision_tokens": chunk_vision_tokens,
        "chunk_vision_patches": chunk_vision_patches,
        "chunk_num_images": chunk_num_images,
        "chunk_num_messages": chunk_num_messages,
        "image_shapes": image_shapes,
        "prefix_truncated": prefix_truncated,
        "overflow_truncated": overflow_truncated,
        "dropped_messages": dropped_messages,
        "dropped_tokens": dropped_tokens,
        "dropped_supervised": dropped_supervised,
        "carried_messages": carried_messages,
        "carried_tokens": carried_tokens,
        "repeated_supervised": repeated_supervised,
        "chunks_with_carry": chunks_with_carry,
    }


def recording_split(recording_id: Any, val_fraction: float) -> str:
    """Deterministic recording-level train/val split (whole recording -> one side,
    so a recording never leaks across the split). No RNG/seed.

    Mirrors the stage-04 builder's ``_split_of`` byte-for-byte so a split applied
    here (records stage) matches one baked upstream. Lets the split move out of the
    measure stage: the per-message length cache is split-agnostic and reused."""
    if val_fraction <= 0.0 or not recording_id:
        return "train"
    bucket = int(hashlib.sha1(str(recording_id).encode()).hexdigest(), 16) % 1000
    return "val" if bucket < round(val_fraction * 1000) else "train"


def build_records_from_chat(
    chat_path: str | Path,
    out_dir: str | Path,
    *,
    max_length: int,
    measure_message,
    records_per_shard: int = 100_000,
    overwrite: bool = False,
    profile_metadata: dict[str, Any] | None = None,
    num_workers: int = 2,
    overflow_mode: str = "drop",
    message_lengths_path: str | Path | None = None,
    measurement_contract: dict[str, Any] | None = None,
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
    and budgeted as a normal message. ``overflow_mode`` ("drop" (default) |
    "split" | "truncate") and ``message_lengths_path`` (measure-once cache, keyed to chat.jsonl
    positions, reused across every max_length / overflow_mode) are the only
    binning knobs. Under "split", messages named by ``CARRY_KEY`` are prepended
    to continuation chunks within the same token budget. Exclusive offsets in
    ``SPLIT_UNIT_ENDS_KEY`` make contiguous message groups indivisible.

    Train/val split (optional): when ``split`` is set, only conversations whose
    ``recording_split(row[split_key], val_fraction)`` equals ``split`` are emitted.
    The message-length cache is keyed by position over the FULL chat.jsonl and is
    resolved/validated against it in full, so ``conv_idx`` stays aligned no matter
    which split is being built -- the split changes only which conversations reach
    the output, never the cache. That is what lets a single (split-agnostic) cache
    serve every split and every val_fraction without re-tokenizing. ``split=None``
    (default) emits all conversations (single-split / pre-split input).
    """
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if overflow_mode not in ("split", "truncate", "drop"):
        raise ValueError(
            f"overflow_mode must be 'split', 'truncate', or 'drop', got {overflow_mode!r}"
        )
    validate_measurement_contract(measurement_contract)

    chat_path = Path(chat_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    effective_max = max_length

    def _carry_of(carry: tuple[int, ...]) -> tuple[int, ...]:
        return carry if overflow_mode == "split" else ()

    def _split_units_of(unit_ends: tuple[int, ...], message_count: int) -> tuple[int, ...]:
        return unit_ends if overflow_mode == "split" else tuple(range(1, message_count + 1))

    def _in_split(session_meta: dict[str, Any]) -> bool:
        # conv_idx (and thus the cache key) is always over the full chat; this only
        # decides whether a conversation's records are emitted for this split.
        return split is None or recording_split(session_meta.get(split_key), val_fraction) == split

    precomputed = _resolve_chat_message_lengths(
        chat_path,
        measure_message,
        num_workers,
        message_lengths_path,
        measurement_contract,
    )
    supervision_fields = {
        isinstance(result, dict) and "supervised_tokens" in result
        for result in precomputed.values()
    }
    if len(supervision_fields) > 1:
        raise ValueError("message-length cache mixes supervision measurement schemas")
    supervision_basis = (
        "loss_mask" if supervision_fields == {True} else "assistant_message_length_estimate"
    )

    # Prescan: per-conversation token totals (-> sequence_lengths.jsonl) and the
    # earliest over-budget split unit per session (prefix-truncation point).
    session_truncate_at: dict[str, int] = {}
    session_carry: dict[str, tuple[int, ...]] = {}
    session_split_unit_ends: dict[str, tuple[int, ...]] = {}
    sequence_stats: dict[str, dict[str, int]] = {}
    total_message_tokens = 0
    total_supervised_tokens = 0
    for conv_idx, session_id, session_meta, messages in _iter_chat_conversations(chat_path):
        if not _in_split(session_meta):
            continue
        carried = _carry_of(tuple(session_meta.pop(CARRY_KEY, [])))
        session_carry[session_id] = carried
        split_unit_ends = _split_units_of(
            tuple(session_meta.pop(SPLIT_UNIT_ENDS_KEY)), len(messages)
        )
        session_split_unit_ends[session_id] = split_unit_ends
        reserve = 0
        agg = {
            "num_messages": 0,
            "total_tokens": 0,
            "vision_tokens": 0,
            "num_images": 0,
            "max_message_tokens": 0,
            "num_messages_over_budget": 0,
        }
        sequence_stats[session_id] = agg
        unit_start = 0
        for unit_end in split_unit_ends:
            unit_length = 0
            added_reserve = 0
            for msg_offset in range(unit_start, unit_end):
                result = precomputed[(conv_idx, msg_offset)]
                length, vt, _vp, ni, _grid = _extract_measurement(result)
                unit_length += length
                if msg_offset in carried:
                    added_reserve += length
                total_message_tokens += length
                total_supervised_tokens += _supervised_tokens(result, messages[msg_offset])
                agg["num_messages"] += 1
                agg["total_tokens"] += length
                agg["vision_tokens"] += vt
                agg["num_images"] += ni
                agg["max_message_tokens"] = max(agg["max_message_tokens"], length)
                if length > effective_max:
                    agg["num_messages_over_budget"] += 1
            if reserve + unit_length > effective_max and session_id not in session_truncate_at:
                session_truncate_at[session_id] = unit_start
            reserve += added_reserve
            unit_start = unit_end
    all_session_ids = set(sequence_stats)
    if session_truncate_at:
        print(
            f"[records] prefix-truncating {len(session_truncate_at)} session(s) at the first "
            f"split unit plus carry prefix exceeding max_length={max_length}: "
            f"{sorted(session_truncate_at)[:5]}" + (" ..." if len(session_truncate_at) > 5 else ""),
            flush=True,
        )

    drop_sessions: set[str] = set()
    if overflow_mode == "drop":
        drop_sessions = {
            sid for sid, agg in sequence_stats.items() if agg["total_tokens"] > effective_max
        }
        if drop_sessions:
            print(
                f"[records] drop mode: dropping {len(drop_sessions)} session(s) whose total "
                f"length exceeds effective_max={effective_max}",
                flush=True,
            )

    _msg_lengths: list[int] = []
    _msg_vision_tokens: list[int] = []
    _msg_num_images: list[int] = []
    _chunk_lengths: list[int] = []
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
        "carried_messages": 0,
        "carried_tokens": 0,
        "repeated_supervised": 0,
        "chunks_with_carry": 0,
    }
    _session_chunk_counts: dict[str, int] = {}

    def _iter_records():
        for conv_idx, session_id, session_meta, messages in _iter_chat_conversations(chat_path):
            if not _in_split(session_meta):
                continue
            session_meta.pop(CARRY_KEY, None)
            session_meta.pop(SPLIT_UNIT_ENDS_KEY, None)
            if session_id in drop_sessions:
                for off in range(len(messages)):
                    length, *_ = _extract_measurement(precomputed[(conv_idx, off)])
                    _trunc["dropped_tokens"] += length
                    _trunc["dropped_supervised"] += _supervised_tokens(
                        precomputed[(conv_idx, off)], messages[off]
                    )
                _trunc["dropped_messages"] += len(messages)
                _trunc["dropped_sessions"].add(session_id)
                continue

            res = _process_conversation(
                conv_idx,
                session_id,
                session_meta,
                messages,
                precomputed,
                effective_max=effective_max,
                overflow_mode=overflow_mode,
                truncate_offset=session_truncate_at.get(session_id),
                carry=session_carry[session_id],
                split_unit_ends=session_split_unit_ends[session_id],
            )

            _msg_lengths.extend(res["msg_lengths"])
            _msg_vision_tokens.extend(res["msg_vision_tokens"])
            _msg_num_images.extend(res["msg_num_images"])
            _chunk_lengths.extend(res["chunk_lengths"])
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
            _trunc["dropped_messages"] += res["dropped_messages"]
            _trunc["dropped_tokens"] += res["dropped_tokens"]
            _trunc["dropped_supervised"] += res["dropped_supervised"]
            _trunc["carried_messages"] += res["carried_messages"]
            _trunc["carried_tokens"] += res["carried_tokens"]
            _trunc["repeated_supervised"] += res["repeated_supervised"]
            _trunc["chunks_with_carry"] += res["chunks_with_carry"]
            if res["examples"]:
                _session_chunk_counts[session_id] = len(res["examples"])
            yield from res["examples"]

    out_path = _write_arrayrecord_dataset(
        _iter_records(),
        out_dir,
        records_per_shard=records_per_shard,
        overwrite=overwrite,
        metadata={
            "inline_records": True,
            "source_chat_path": str(chat_path),
            "max_length": max_length,
            "overflow_mode": overflow_mode,
            "split": split,
            "val_fraction": val_fraction,
            "profile_metadata": profile_metadata or {},
            "source_chat": file_identity(chat_path),
            "measurement_contract": measurement_contract,
        },
    )

    _emit_sequence_lengths(out_dir, sequence_stats=sequence_stats, effective_max=effective_max)
    _emit_truncation_stats(
        out_dir,
        overflow_mode=overflow_mode,
        max_length=max_length,
        effective_max=effective_max,
        total_sessions=len(all_session_ids),
        total_message_tokens=total_message_tokens,
        total_supervised_tokens=total_supervised_tokens,
        session_chunk_counts=_session_chunk_counts,
        prefix_sessions=_trunc["prefix_sessions"],
        overflow_sessions=_trunc["overflow_sessions"],
        dropped_sessions=_trunc["dropped_sessions"],
        dropped_messages=_trunc["dropped_messages"],
        dropped_tokens=_trunc["dropped_tokens"],
        dropped_supervised_tokens=_trunc["dropped_supervised"],
        repeated_supervised_tokens=_trunc["repeated_supervised"],
        emitted_tokens=sum(_chunk_lengths),
        carried_tokens=_trunc["carried_tokens"],
        carried_messages=_trunc["carried_messages"],
        chunks_with_carry=_trunc["chunks_with_carry"],
        supervision_basis=supervision_basis,
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

    _mark_compiled_dataset_complete(out_dir)
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
    metadatas = [load_compiled_metadata(mix_sources[i].path) for i in active_indices]
    for i, m in zip(active_indices, metadatas):
        if not m.get("inline_records"):
            raise ValueError(
                f"Expected an inline-records dataset (build_records_from_chat): {mix_sources[i].path}"
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
