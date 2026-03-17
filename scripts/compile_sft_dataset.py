"""Compile a JSONL SFT dataset into Grain-friendly ArrayRecord shards."""

from __future__ import annotations

import argparse

from omegalax.data.grain_pipeline import compile_jsonl_to_arrayrecord


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile a JSONL SFT dataset into canonical ArrayRecord message blocks.")
    p.add_argument("--data-path", type=str, required=True, help="Path to the raw JSONL dataset.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for ArrayRecord shards.")
    p.add_argument("--messages-per-record", type=int, default=128, help="Maximum contiguous messages to store in one payload block.")
    p.add_argument("--records-per-shard", type=int, default=10_000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = compile_jsonl_to_arrayrecord(
        args.data_path,
        args.out_dir,
        messages_per_record=args.messages_per_record,
        records_per_shard=args.records_per_shard,
        overwrite=args.overwrite,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
