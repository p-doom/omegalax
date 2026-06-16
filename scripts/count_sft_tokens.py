#!/usr/bin/env python3
"""Precompute omegalax SFT token counts once for later bucket materialization."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm


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

_MEASURE_FN: MeasureFn | None = None
_SYSTEM_MESSAGE_MEASURE: dict[str, Any] | None = None


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


def resolve_image(path_value: str, *, canonical_root: Path) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = canonical_root / path
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    return str(path.resolve())


def normalize_record(record: dict[str, Any], *, canonical_root: Path) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record missing messages list")
    split = record.get("split")
    if split not in VALID_SPLITS:
        raise ValueError("record split must be 'train' or 'val'")

    normalized = dict(record)
    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("message is not an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {role!r}")
        content = message.get("content", [])
        if isinstance(content, str):
            normalized_messages.append({**message, "role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise ValueError("message content must be string or list")

        normalized_blocks: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("content block is not an object")
            if block.get("type") == "image":
                image_value = block.get("image") or block.get("url") or block.get("path")
                if not image_value:
                    raise ValueError("image block missing image/url/path")
                new_block = dict(block)
                new_block.pop("url", None)
                new_block.pop("path", None)
                new_block["image"] = resolve_image(str(image_value), canonical_root=canonical_root)
                normalized_blocks.append(new_block)
            else:
                normalized_blocks.append(dict(block))
        normalized_messages.append({**message, "role": role, "content": normalized_blocks})

    normalized["messages"] = normalized_messages
    return normalized


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
        raise ValueError(f"measure result missing keys: {missing}")
    return {
        "length": int(result["length"]),
        "vision_tokens": int(result["vision_tokens"]),
        "vision_patches": int(result["vision_patches"]),
        "num_images": int(result["num_images"]),
        "image_grid_thw": result["image_grid_thw"],
    }


def _set_measure_globals(
    measure_message: MeasureFn,
    system_message_measure: dict[str, Any] | None,
) -> None:
    global _MEASURE_FN, _SYSTEM_MESSAGE_MEASURE
    _MEASURE_FN = measure_message
    _SYSTEM_MESSAGE_MEASURE = system_message_measure


def _count_images(messages: list[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            count += sum(1 for block in content if block.get("type") == "image")
    return count


def _measure_record_idx(
    item: tuple[int, dict[str, Any]],
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None, str | None]:
    idx, record = item
    if _MEASURE_FN is None:
        raise RuntimeError("measure function is not initialized")

    try:
        total = 0
        measured_messages: list[dict[str, Any]] = []
        for message in record["messages"]:
            result = normalize_measure_result(_MEASURE_FN(message))
            measured_messages.append({**message, "_omegalax_token_measure": result})
            total += int(result["length"])
        if _SYSTEM_MESSAGE_MEASURE is not None:
            total += int(_SYSTEM_MESSAGE_MEASURE["length"])

        measured_record = {
            **record,
            "messages": measured_messages,
            "_omegalax_token_length": total,
        }
        row = {
            "source_index": idx,
            "sample_id": record.get("sample_id"),
            "split": record.get("split"),
            "length": total,
            "n_messages": len(measured_messages),
            "n_images": _count_images(measured_messages),
        }
        return idx, measured_record, row, None
    except Exception as exc:  # noqa: BLE001 - failed samples should stay auditable.
        return idx, None, None, f"{type(exc).__name__}: {exc}"


def measure_records(
    records: list[dict[str, Any]],
    *,
    measure_message: MeasureFn,
    system_message_measure: dict[str, Any] | None,
    num_workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], [], []

    _set_measure_globals(measure_message, system_message_measure)
    measured_by_index: list[dict[str, Any] | None] = [None] * len(records)
    lengths_by_index: list[dict[str, Any] | None] = [None] * len(records)
    rejected: list[dict[str, Any]] = []

    n_workers = max(1, min(num_workers, len(records)))
    pool: Any | None = None
    if n_workers == 1:
        iterator = (
            _measure_record_idx((idx, record))
            for idx, record in enumerate(records)
        )
        results = tqdm(iterator, total=len(records), desc="[count] measure records", unit="rec")
    else:
        ctx = mp.get_context("fork")
        chunksize = max(1, min(16, len(records) // n_workers))
        pool = ctx.Pool(n_workers)
        results = pool.imap_unordered(
            _measure_record_idx,
            enumerate(records),
            chunksize=chunksize,
        )
        results = tqdm(
            results,
            total=len(records),
            desc=f"[count] measure records ({n_workers} workers)",
            unit="rec",
        )

    try:
        for idx, measured_record, length_row, error in results:
            if error is not None:
                rejected.append(
                    {
                        "source_index": idx,
                        "sample_id": records[idx].get("sample_id"),
                        "split": records[idx].get("split"),
                        "reason": "tokenization_failed",
                        "detail": error,
                    }
                )
                continue
            assert measured_record is not None
            assert length_row is not None
            measured_by_index[idx] = measured_record
            lengths_by_index[idx] = length_row
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    measured = [record for record in measured_by_index if record is not None]
    token_lengths = [row for row in lengths_by_index if row is not None]
    return measured, token_lengths, rejected


def make_hf_measure_fn(
    *,
    model_id: str,
    tokenizer_name: str | None,
    processor_name: str | None,
    preprocessor_config: Path | None,
) -> MeasureFn:
    from transformers import AutoImageProcessor, AutoTokenizer

    from omegalax.data.qwen3_encoding import make_message_length_fn
    from omegalax.registry import resolve_hf_repo_id

    resolved_tokenizer = tokenizer_name or resolve_hf_repo_id(model_id)
    tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer)

    image_processor = None
    if processor_name:
        ip_kwargs: dict[str, Any] = {}
        if preprocessor_config:
            ip_kwargs = json.loads(preprocessor_config.read_text())
        image_processor = AutoImageProcessor.from_pretrained(
            processor_name,
            use_fast=False,
            **ip_kwargs,
        )
    return make_message_length_fn(tokenizer, image_processor)


def count_sft_tokens(
    *,
    canonical_root: Path,
    out_dir: Path,
    model_id: str,
    chat_jsonl: Path | None = None,
    tokenizer: str | None = None,
    processor: str | None = None,
    preprocessor_config: Path | None = None,
    system_message_text: str = "",
    num_workers: int = 2,
    measure_message: MeasureFn | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    canonical_root = canonical_root.expanduser().resolve()
    out_dir = ensure_empty_dir(out_dir, overwrite=overwrite)
    if chat_jsonl is None:
        chat_path = canonical_root / "chat.jsonl"
    else:
        chat_path = chat_jsonl.expanduser()
        if not chat_path.is_absolute():
            chat_path = canonical_root / chat_path
        chat_path = chat_path.resolve()
    raw_records = read_jsonl(chat_path)

    normalized_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, record in enumerate(raw_records):
        try:
            normalized_records.append(normalize_record(record, canonical_root=canonical_root))
        except Exception as exc:  # noqa: BLE001 - keep bad records auditable.
            rejected.append(
                {
                    "source_index": index,
                    "sample_id": record.get("sample_id"),
                    "split": record.get("split"),
                    "reason": "invalid_sample",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

    measure = measure_message or make_hf_measure_fn(
        model_id=model_id,
        tokenizer_name=tokenizer,
        processor_name=processor,
        preprocessor_config=preprocessor_config,
    )

    system_message = None
    system_message_measure = None
    if system_message_text:
        system_message = {
            "role": "system",
            "content": [{"type": "text", "text": system_message_text}],
        }
        system_message_measure = normalize_measure_result(measure(system_message))

    measured, token_lengths, measurement_rejected = measure_records(
        normalized_records,
        measure_message=measure,
        system_message_measure=system_message_measure,
        num_workers=num_workers,
    )
    rejected.extend(measurement_rejected)

    write_jsonl(out_dir / "records.jsonl", measured)
    write_jsonl(out_dir / "token_lengths.jsonl", token_lengths)
    write_jsonl(out_dir / "rejected.jsonl", rejected)

    manifest = {
        "artifact_type": "omegalax_sft_token_counts",
        "schema_version": 1,
        "canonical_root": str(canonical_root),
        "chat_jsonl": str(chat_path),
        "model_id": model_id,
        "tokenizer": tokenizer,
        "processor": processor,
        "preprocessor_config": str(preprocessor_config) if preprocessor_config else None,
        "system_message": system_message,
        "system_message_measure": system_message_measure,
        "n_input": len(raw_records),
        "n_valid": len(measured),
        "n_rejected": len(rejected),
        "num_workers": num_workers,
        "files": {
            "records": "records.jsonl",
            "token_lengths": "token_lengths.jsonl",
            "rejected": "rejected.jsonl",
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-root",
        "--canonical_root",
        dest="canonical_root",
        type=Path,
        required=True,
    )
    parser.add_argument("--chat-jsonl", "--chat_jsonl", dest="chat_jsonl", type=Path)
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", type=Path, required=True)
    parser.add_argument("--model-id", "--model_id", dest="model_id", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--processor")
    parser.add_argument(
        "--preprocessor-config", "--preprocessor_config", dest="preprocessor_config", type=Path
    )
    parser.add_argument(
        "--system-message-text", "--system_message_text", dest="system_message_text", default=""
    )
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = count_sft_tokens(
        canonical_root=args.canonical_root,
        chat_jsonl=args.chat_jsonl,
        out_dir=args.out_dir,
        model_id=args.model_id,
        tokenizer=args.tokenizer,
        processor=args.processor,
        preprocessor_config=args.preprocessor_config,
        system_message_text=args.system_message_text,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote token counts for {manifest['n_valid']} samples "
        f"({manifest['n_rejected']} rejected) to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
