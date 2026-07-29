"""Build a fixed-C tail bundle from historical curriculum iterator positions."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import grain

from omegalax.data.pretrain_data_set import (
    COMPILED_METADATA_FILENAME,
    load_arrayrecord_metadata,
    resolve_arrayrecord_paths,
    write_json_arrayrecord_dataset,
)
from omegalax.data.pretrain_statepassing import (
    STATEPASSING_CURRICULUM_INDEX_FORMAT,
    STATEPASSING_FIXED_C_INDEX_FORMAT,
    STATEPASSING_INDEX_SHUFFLE_ROUNDS,
    STATEPASSING_WINDOW_INDEX_FORMAT,
)


def _json_hash(path: Path) -> str:
    value = json.loads(path.read_text())
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _iterator_batches_emitted(path: Path) -> int:
    state = json.loads(path.read_text())
    states = state.get("iterators_in_use_states")
    if not isinstance(states, list) or not states:
        raise ValueError(f"Unsupported Grain iterator state: {path}")
    if any(int(value) for value in state.get("exhausted", ())):
        raise ValueError(f"Cannot extract from an exhausted iterator state: {path}")
    return sum(int(worker["next_index"]) for worker in states)


def _source(index_path: Path):
    return grain.sources.ArrayRecordDataSource(
        [str(path) for path in resolve_arrayrecord_paths(index_path)]
    )


def _record_at_shuffled_position(source, absolute_position: int, *, seed: int) -> dict[str, Any]:
    epoch, local_position = divmod(int(absolute_position), len(source))
    record_index = grain.experimental.index_shuffle(
        local_position,
        len(source) - 1,
        int(seed) + epoch,
        STATEPASSING_INDEX_SHUFFLE_ROUNDS,
    )
    return json.loads(source[record_index])


def _split_record(
    source_record: dict[str, Any],
    *,
    start_chunk: int,
    window_idx: int,
    chunk_length: int,
    source_phase: str,
) -> dict[str, Any]:
    record = dict(source_record)
    record["source_phase"] = source_phase
    record["source_window_idx"] = int(source_record["window_idx"])
    record["window_idx"] = int(window_idx)
    record["start_chunk"] = int(start_chunk)
    record["num_segments"] = 4
    eos_token_idx = source_record.get("eos_token_idx")
    start = int(start_chunk) * int(chunk_length)
    end = min(start + 4 * int(chunk_length), int(record["doc_token_count"]))
    record["eos_token_idx"] = (
        int(eos_token_idx)
        if eos_token_idx is not None and start <= int(eos_token_idx) < end
        else None
    )
    return record


def _iter_tail_records(
    *,
    c4_source,
    c8_source,
    c4_start_position: int,
    c8_start_position: int,
    c4_num_records: int,
    c8_num_records: int,
    chunk_length: int,
    seed: int,
) -> Iterator[dict[str, Any]]:
    window_idx = 0
    for position in range(c4_start_position, c4_start_position + c4_num_records):
        source_record = _record_at_shuffled_position(c4_source, position, seed=seed)
        if int(source_record["num_segments"]) != 4:
            raise ValueError("C4 source record does not use num_segments=4")
        yield _split_record(
            source_record,
            start_chunk=int(source_record["start_chunk"]),
            window_idx=window_idx,
            chunk_length=chunk_length,
            source_phase="c4",
        )
        window_idx += 1
    for position in range(c8_start_position, c8_start_position + c8_num_records):
        source_record = _record_at_shuffled_position(c8_source, position, seed=seed)
        if int(source_record["num_segments"]) != 8:
            raise ValueError("C8 source record does not use num_segments=8")
        source_start = int(source_record["start_chunk"])
        for offset in (0, 4):
            yield _split_record(
                source_record,
                start_chunk=source_start + offset,
                window_idx=window_idx,
                chunk_length=chunk_length,
                source_phase="c8",
            )
            window_idx += 1


def _iter_iid_records(tail_index_path: Path, *, chunk_length: int) -> Iterator[dict[str, Any]]:
    source = _source(tail_index_path)
    iid_window_idx = 0
    for record_idx in range(len(source)):
        record = json.loads(source[record_idx])
        for chunk_offset in range(4):
            chunk_idx = int(record["start_chunk"]) + chunk_offset
            start = chunk_idx * int(chunk_length)
            end = min(start + int(chunk_length), int(record["doc_token_count"]))
            eos_token_idx = record.get("eos_token_idx")
            iid_record = dict(record)
            iid_record["window_idx"] = iid_window_idx
            iid_record["start_chunk"] = chunk_idx
            iid_record["num_segments"] = 1
            iid_record["eos_token_idx"] = (
                int(eos_token_idx)
                if eos_token_idx is not None and start <= int(eos_token_idx) < end
                else None
            )
            yield iid_record
            iid_window_idx += 1


def build_curriculum_c4_tail_bundle(
    curriculum_index_root: str | Path,
    out_dir: str | Path,
    *,
    c4_iterator_state: str | Path,
    c8_iterator_state: str | Path,
    c4_steps: int,
    c8_steps: int,
    trim_batch_size: int = 192,
    trim_grad_accum_steps: int = 5,
    seed: int = 0,
    records_per_shard: int = 100_000,
    overwrite: bool = False,
) -> Path:
    root = Path(curriculum_index_root).expanduser().resolve()
    root_metadata = load_arrayrecord_metadata(root)
    if root_metadata.get("format") != STATEPASSING_CURRICULUM_INDEX_FORMAT:
        raise ValueError("Source must be a curriculum bundle")
    if int(root_metadata["chunk_length"]) <= 0:
        raise ValueError("Source chunk_length must be positive")
    if int(root_metadata["trim_batch_size"]) != int(trim_batch_size):
        raise ValueError("Source and tail trim_batch_size differ")
    if int(root_metadata["trim_grad_accum_steps"]) != int(trim_grad_accum_steps):
        raise ValueError("Source and tail trim_grad_accum_steps differ")
    if c4_steps <= 0 or c8_steps < 0:
        raise ValueError("c4_steps must be positive and c8_steps non-negative")

    phases = dict(dict(dict(root_metadata["splits"])["train"])["phases"])
    c4_path = root / str(dict(phases["4"])["path"])
    c8_path = root / str(dict(phases["8"])["path"])
    c4_metadata = load_arrayrecord_metadata(c4_path)
    c8_metadata = load_arrayrecord_metadata(c8_path)
    for source_metadata, expected_c in ((c4_metadata, 4), (c8_metadata, 8)):
        if source_metadata.get("format") != STATEPASSING_WINDOW_INDEX_FORMAT:
            raise ValueError("Curriculum phase is not a physical window index")
        if int(source_metadata["num_segments"]) != expected_c:
            raise ValueError(f"Expected C{expected_c} source phase")

    c4_state_path = Path(c4_iterator_state).expanduser().resolve()
    c8_state_path = Path(c8_iterator_state).expanduser().resolve()
    c4_records_per_batch = int(trim_batch_size) // 4
    c8_records_per_batch = int(trim_batch_size) // 8
    c4_start_position = _iterator_batches_emitted(c4_state_path) * c4_records_per_batch
    c8_start_position = _iterator_batches_emitted(c8_state_path) * c8_records_per_batch
    c4_windows_per_step = int(trim_batch_size) * int(trim_grad_accum_steps) // 4
    c8_windows_per_step = int(trim_batch_size) * int(trim_grad_accum_steps) // 8
    c4_num_records = int(c4_steps) * c4_windows_per_step
    c8_num_records = int(c8_steps) * c8_windows_per_step
    c4_source = _source(c4_path)
    c8_source = _source(c8_path)
    if c4_start_position + c4_num_records > len(c4_source):
        raise ValueError("Requested C4 tail exceeds the source phase")
    if c8_start_position + c8_num_records > len(c8_source):
        raise ValueError("Requested C8 tail exceeds the source phase")

    out_path = Path(out_dir).expanduser().resolve()
    if out_path.exists() and any(out_path.iterdir()):
        if not overwrite:
            raise ValueError(f"Refusing to overwrite non-empty output directory: {out_path}")
        import shutil

        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    chunk_length = int(root_metadata["chunk_length"])
    num_tail_windows = c4_num_records + 2 * c8_num_records
    windows_per_step = int(trim_batch_size) * int(trim_grad_accum_steps) // 4
    statepassing_steps = num_tail_windows // windows_per_step
    if num_tail_windows % windows_per_step:
        raise ValueError("Tail windows do not form complete optimizer steps")

    common_metadata = {
        "data_set_root": str(c4_metadata["data_set_root"]),
        "split": "train",
        "bucket_names": list(c4_metadata["bucket_names"]),
        "chunk_length": chunk_length,
        "eos_id": root_metadata.get("eos_id"),
        "num_bucket_records": int(c4_metadata["num_bucket_records"]),
        "bucket_record_counts": list(c4_metadata["bucket_record_counts"]),
        "fixed_c_format": STATEPASSING_FIXED_C_INDEX_FORMAT,
    }
    tail_path = out_path / "train"
    write_json_arrayrecord_dataset(
        _iter_tail_records(
            c4_source=c4_source,
            c8_source=c8_source,
            c4_start_position=c4_start_position,
            c8_start_position=c8_start_position,
            c4_num_records=c4_num_records,
            c8_num_records=c8_num_records,
            chunk_length=chunk_length,
            seed=seed,
        ),
        tail_path,
        records_per_shard=records_per_shard,
        overwrite=False,
        metadata=common_metadata
        | {
            "format": STATEPASSING_WINDOW_INDEX_FORMAT,
            "num_segments": 4,
            "num_windows": num_tail_windows,
            "raw_num_windows": num_tail_windows,
            "num_residual_chunks": 0,
            "statepassing_steps": statepassing_steps,
            "windows_per_step": windows_per_step,
        },
    )
    iid_num_chunks = num_tail_windows * 4
    chunks_per_step = int(trim_batch_size) * int(trim_grad_accum_steps)
    iid_steps = iid_num_chunks // chunks_per_step
    iid_path = out_path / "iid" / "train"
    write_json_arrayrecord_dataset(
        _iter_iid_records(tail_path, chunk_length=chunk_length),
        iid_path,
        records_per_shard=records_per_shard,
        overwrite=False,
        metadata=common_metadata
        | {
            "format": STATEPASSING_WINDOW_INDEX_FORMAT,
            "num_segments": 1,
            "num_windows": iid_num_chunks,
            "num_residual_chunks": 0,
            "source_index_path": str(tail_path),
            "source_num_segments": 4,
            "source_num_windows": num_tail_windows,
            "iid_steps": iid_steps,
            "chunks_per_step": chunks_per_step,
        },
    )

    bundle_metadata = {
        "format": STATEPASSING_FIXED_C_INDEX_FORMAT,
        "data_set_root": str(c4_metadata["data_set_root"]),
        "bucket_names": list(c4_metadata["bucket_names"]),
        "chunk_length": chunk_length,
        "num_segments": 4,
        "eos_id": root_metadata.get("eos_id"),
        "trim_batch_size": int(trim_batch_size),
        "trim_grad_accum_steps": int(trim_grad_accum_steps),
        "windows_per_step": windows_per_step,
        "chunks_per_step": chunks_per_step,
        "statepassing_steps": statepassing_steps,
        "iid_steps": iid_steps,
        "source": {
            "curriculum_index_root": str(root),
            "seed": int(seed),
            "c4_iterator_state": str(c4_state_path),
            "c4_iterator_state_hash": _json_hash(c4_state_path),
            "c4_start_position": c4_start_position,
            "c4_steps": int(c4_steps),
            "c8_iterator_state": str(c8_state_path),
            "c8_iterator_state_hash": _json_hash(c8_state_path),
            "c8_start_position": c8_start_position,
            "c8_steps": int(c8_steps),
        },
        "splits": {
            "train": {
                "path": "train",
                "num_windows": num_tail_windows,
                "raw_num_windows": num_tail_windows,
                "statepassing_steps": statepassing_steps,
                "windows_per_step": windows_per_step,
                "iid": {
                    "path": "iid/train",
                    "num_chunks": iid_num_chunks,
                    "iid_steps": iid_steps,
                    "chunks_per_step": chunks_per_step,
                },
            }
        },
    }
    (out_path / COMPILED_METADATA_FILENAME).write_text(json.dumps(bundle_metadata, indent=2) + "\n")
    return out_path
