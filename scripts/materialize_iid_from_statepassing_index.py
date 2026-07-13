"""Materialize physical IID indexes from retained Statepassing windows."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from absl import app, flags

from omegalax.data.pretrain_data_set import (
    iter_json_arrayrecord_records,
    load_arrayrecord_metadata,
    write_json_arrayrecord_dataset,
)
from omegalax.data.pretrain_statepassing import STATEPASSING_WINDOW_INDEX_FORMAT

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "index_root", None, "Root containing {split}/ Statepassing indexes.", required=True
)
flags.DEFINE_multi_string("split", ["train", "val"], "Splits to materialize.")
flags.DEFINE_integer("records_per_shard", 100_000, "Records per output shard.")
flags.DEFINE_bool("overwrite", False, "Overwrite existing iid/{split} directories.")
flags.DEFINE_integer("batch_size", None, "Optional global IID chunk batch size for metadata.")
flags.DEFINE_integer("grad_accum_steps", None, "Optional gradient accumulation steps for metadata.")


def _iid_records(source_index: Path, *, chunk_length: int) -> Iterator[dict[str, Any]]:
    iid_window_idx = 0
    for _, record in iter_json_arrayrecord_records(source_index):
        source_num_segments = int(record["num_segments"])
        for chunk_offset in range(source_num_segments):
            chunk_idx = int(record["start_chunk"]) + chunk_offset
            start = chunk_idx * int(chunk_length)
            end = min(start + int(chunk_length), int(record["doc_token_count"]))
            eos_token_idx = record.get("eos_token_idx")
            if eos_token_idx is not None:
                eos_token_idx = int(eos_token_idx)
                if eos_token_idx < start or eos_token_idx >= end:
                    eos_token_idx = None
            yield {
                "bucket_idx": int(record["bucket_idx"]),
                "record_idx": int(record["record_idx"]),
                "doc_id": str(record["doc_id"]),
                "window_idx": iid_window_idx,
                "start_chunk": chunk_idx,
                "num_segments": 1,
                "doc_token_count": int(record["doc_token_count"]),
                "doc_num_chunks": int(record["doc_num_chunks"]),
                "eos_token_idx": eos_token_idx,
            }
            iid_window_idx += 1


def _materialize_split(index_root: Path, split: str) -> Path:
    source_index = index_root / split
    source_metadata = load_arrayrecord_metadata(source_index)
    if source_metadata.get("format") != STATEPASSING_WINDOW_INDEX_FORMAT:
        raise ValueError(
            f"Expected {STATEPASSING_WINDOW_INDEX_FORMAT} at {source_index}, "
            f"got {source_metadata.get('format')}"
        )
    source_num_segments = int(source_metadata["num_segments"])
    if source_num_segments <= 1:
        raise ValueError(f"Expected a C>1 Statepassing index, got C={source_num_segments}")

    chunk_length = int(source_metadata["chunk_length"])
    num_source_windows = int(source_metadata["num_records"])
    num_iid_chunks = num_source_windows * source_num_segments
    metadata = {
        "format": STATEPASSING_WINDOW_INDEX_FORMAT,
        "data_set_root": source_metadata["data_set_root"],
        "split": source_metadata["split"],
        "bucket_names": list(source_metadata["bucket_names"]),
        "chunk_length": chunk_length,
        "num_segments": 1,
        "eos_id": source_metadata.get("eos_id"),
        "num_windows": num_iid_chunks,
        "num_residual_chunks": 0,
        "num_bucket_records": source_metadata.get("num_bucket_records"),
        "bucket_record_counts": source_metadata.get("bucket_record_counts"),
        "source_index_path": str(source_index),
        "source_num_segments": source_num_segments,
        "source_num_windows": num_source_windows,
    }
    if FLAGS.batch_size is not None and FLAGS.grad_accum_steps is not None:
        chunks_per_step = int(FLAGS.batch_size) * int(FLAGS.grad_accum_steps)
        if chunks_per_step <= 0:
            raise ValueError("--batch_size * --grad_accum_steps must be > 0")
        metadata["chunks_per_step"] = chunks_per_step
        metadata["iid_steps"] = num_iid_chunks // chunks_per_step

    return write_json_arrayrecord_dataset(
        _iid_records(source_index, chunk_length=chunk_length),
        index_root / "iid" / split,
        records_per_shard=FLAGS.records_per_shard,
        overwrite=FLAGS.overwrite,
        metadata=metadata,
    )


def main(_) -> None:
    index_root = Path(FLAGS.index_root).expanduser().resolve()
    for split in FLAGS.split:
        out_path = _materialize_split(index_root, split)
        print(out_path, flush=True)


if __name__ == "__main__":
    app.run(main)
