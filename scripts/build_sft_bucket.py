#!/usr/bin/env python3
"""Build a length-band SFT artifact from precomputed omegalax token counts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Callable

from omegalax.data.grain_pipeline import build_chunk_index, compile_jsonl_to_arrayrecord


MeasureResult = int | dict[str, Any]
MeasureFn = Callable[[dict[str, Any]], MeasureResult]

VALID_SPLITS = {"train", "val"}
REQUIRED_MEASURE_KEYS = (
    "length",
    "vision_tokens",
    "vision_patches",
    "num_images",
    "image_grid_thw",
)


def ensure_empty_dir(path: Path, *, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Refusing to overwrite non-empty output dir: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_num}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def load_token_count_artifact(
    token_count_root: Path,
    records_jsonl: Path | None = None,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    token_count_root = token_count_root.expanduser().resolve()
    manifest_path = token_count_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"token count manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("artifact_type") != "omegalax_sft_token_counts":
        raise ValueError(
            "expected omegalax_sft_token_counts artifact, "
            f"got {manifest.get('artifact_type')!r}"
        )

    if records_jsonl is None:
        records_jsonl = token_count_root / "records.jsonl"
    else:
        records_jsonl = records_jsonl.expanduser()
        if not records_jsonl.is_absolute():
            records_jsonl = token_count_root / records_jsonl
        records_jsonl = records_jsonl.resolve()
    return manifest, records_jsonl, read_jsonl(records_jsonl)


def normalize_measure_result(result: MeasureResult) -> dict[str, Any]:
    if isinstance(result, int):
        return {
            "length": int(result),
            "vision_tokens": 0,
            "vision_patches": 0,
            "num_images": 0,
            "image_grid_thw": [],
        }
    if not isinstance(result, dict):
        raise ValueError(f"expected measure dict or int, got {type(result).__name__}")
    missing = [key for key in REQUIRED_MEASURE_KEYS if key not in result]
    if missing:
        raise ValueError(f"precomputed measure missing keys: {missing}")
    return {
        "length": int(result["length"]),
        "vision_tokens": int(result["vision_tokens"]),
        "vision_patches": int(result["vision_patches"]),
        "num_images": int(result["num_images"]),
        "image_grid_thw": result["image_grid_thw"],
    }


def make_precomputed_measure_fn(
    *,
    system_message: dict[str, Any] | None,
    system_message_measure: dict[str, Any] | None,
) -> MeasureFn:
    normalized_system_measure = (
        normalize_measure_result(system_message_measure) if system_message_measure else None
    )

    def measure(message: dict[str, Any]) -> dict[str, Any]:
        if system_message is not None and message == system_message:
            if normalized_system_measure is None:
                raise ValueError("token count artifact has a system_message without its measure")
            return normalized_system_measure

        result = message.get("_omegalax_token_measure")
        if result is None:
            raise ValueError(
                "message is missing _omegalax_token_measure; run count_sft_tokens.py first"
            )
        return normalize_measure_result(result)

    return measure


def split_by_length_band(
    records: list[dict[str, Any]],
    *,
    min_length: int,
    max_length: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    if min_length < 0:
        raise ValueError("min_length must be >= 0")
    if max_length <= min_length:
        raise ValueError("max_length must be > min_length")

    accepted: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    token_lengths: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        sample_id = record.get("sample_id")
        split = record.get("split")
        length = record.get("_omegalax_token_length")
        row = {
            "source_index": index,
            "sample_id": sample_id,
            "split": split,
            "length": length,
            "min_length_exclusive": min_length,
            "max_length_inclusive": max_length,
        }
        token_lengths.append(row)

        if not isinstance(length, int):
            rejected.append({**row, "reason": "invalid_sample", "detail": "missing token length"})
            continue
        if length <= min_length:
            rejected.append({**row, "reason": "too_short_for_bucket"})
            continue
        if length > max_length:
            rejected.append({**row, "reason": "too_long_for_bucket"})
            continue
        if split not in VALID_SPLITS:
            rejected.append({**row, "reason": "missing_split"})
            continue
        accepted[str(split)].append({**record, "_omegalax_bucket_length": length})

    return accepted, token_lengths, rejected


def build_sft_bucket(
    *,
    token_count_root: Path,
    out_dir: Path,
    min_length: int,
    max_length: int,
    records_jsonl: Path | None = None,
    messages_per_record: int = 128,
    records_per_shard_payload: int = 10_000,
    records_per_shard_index: int = 100_000,
    num_workers: int = 2,
    overwrite: bool = False,
) -> dict[str, Any]:
    out_dir = ensure_empty_dir(out_dir, overwrite=overwrite)
    token_manifest, records_path, records = load_token_count_artifact(
        token_count_root,
        records_jsonl,
    )

    accepted, token_lengths, rejected = split_by_length_band(
        records,
        min_length=min_length,
        max_length=max_length,
    )

    source_dir = out_dir / "source"
    write_jsonl(source_dir / "token_lengths.jsonl", token_lengths)
    write_jsonl(source_dir / "rejected.jsonl", rejected)
    for split in ("train", "val"):
        if not accepted[split]:
            raise RuntimeError(
                f"No {split} records survived bucket filter "
                f"({min_length} < tokens <= {max_length})"
            )
        write_jsonl(source_dir / f"{split}.jsonl", accepted[split])

    system_message = token_manifest.get("system_message")
    system_message_measure = token_manifest.get("system_message_measure")
    measure = make_precomputed_measure_fn(
        system_message=system_message,
        system_message_measure=system_message_measure,
    )

    payload_root = out_dir / "payload"
    for split in ("train", "val"):
        compile_jsonl_to_arrayrecord(
            source_dir / f"{split}.jsonl",
            payload_root / split,
            messages_per_record=messages_per_record,
            records_per_shard=records_per_shard_payload,
            overwrite=True,
        )
        build_chunk_index(
            payload_root / split,
            out_dir / split,
            max_length=max_length,
            measure_message=measure,
            records_per_shard=records_per_shard_index,
            overwrite=True,
            num_workers=num_workers,
            system_message=system_message,
            profile_metadata={
                "model_id": token_manifest.get("model_id"),
                "tokenizer": token_manifest.get("tokenizer"),
                "processor": token_manifest.get("processor"),
                "preprocessor_config": token_manifest.get("preprocessor_config"),
                "token_count_root": str(token_count_root.expanduser().resolve()),
                "records_jsonl": str(records_path),
                "min_length_exclusive": min_length,
                "max_length_inclusive": max_length,
            },
        )

    manifest = {
        "artifact_type": "omegalax_sft_bucket",
        "schema_version": 1,
        "token_count_root": str(token_count_root.expanduser().resolve()),
        "records_jsonl": str(records_path),
        "model_id": token_manifest.get("model_id"),
        "tokenizer": token_manifest.get("tokenizer"),
        "processor": token_manifest.get("processor"),
        "preprocessor_config": token_manifest.get("preprocessor_config"),
        "bucket": {
            "min_length_exclusive": min_length,
            "max_length_inclusive": max_length,
        },
        "messages_per_record": messages_per_record,
        "records_per_shard_payload": records_per_shard_payload,
        "records_per_shard_index": records_per_shard_index,
        "num_workers": num_workers,
        "n_input": len(records),
        "n_train": len(accepted["train"]),
        "n_val": len(accepted["val"]),
        "n_rejected": len(rejected),
        "files": {
            "source_train": "source/train.jsonl",
            "source_val": "source/val.jsonl",
            "token_lengths": "source/token_lengths.jsonl",
            "rejected": "source/rejected.jsonl",
            "payload_train": "payload/train/",
            "payload_val": "payload/val/",
            "train": "train/",
            "val": "val/",
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token-count-root",
        "--token_count_root",
        dest="token_count_root",
        type=Path,
        required=True,
    )
    parser.add_argument("--records-jsonl", "--records_jsonl", dest="records_jsonl", type=Path)
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", type=Path, required=True)
    parser.add_argument("--min-length", "--min_length", dest="min_length", type=int, required=True)
    parser.add_argument("--max-length", "--max_length", dest="max_length", type=int, required=True)
    parser.add_argument(
        "--messages-per-record",
        "--messages_per_record",
        dest="messages_per_record",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--records-per-shard-payload",
        "--records_per_shard_payload",
        dest="records_per_shard_payload",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--records-per-shard-index",
        "--records_per_shard_index",
        dest="records_per_shard_index",
        type=int,
        default=100_000,
    )
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_sft_bucket(
        token_count_root=args.token_count_root,
        records_jsonl=args.records_jsonl,
        out_dir=args.out_dir,
        min_length=args.min_length,
        max_length=args.max_length,
        messages_per_record=args.messages_per_record,
        records_per_shard_payload=args.records_per_shard_payload,
        records_per_shard_index=args.records_per_shard_index,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote bucket {manifest['bucket']['min_length_exclusive']} < tokens <= "
        f"{manifest['bucket']['max_length_inclusive']} to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
