"""Build an offline chunk index for a compiled canonical SFT dataset."""

from __future__ import annotations

import argparse
import json

from transformers import AutoImageProcessor, AutoTokenizer

from omegalax.data.grain_pipeline import build_chunk_index
from omegalax.data.qwen3_encoding import encode_qwen_messages
from omegalax.registry import resolve_hf_repo_id


def _messages_have_images(messages) -> bool:
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            continue
        for block in content:
            if block.get("type") == "image":
                return True
    return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an offline chunk index for a compiled SFT payload dataset.")
    p.add_argument("--data-path", type=str, required=True, help="Path to a canonical compiled payload-block dataset.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for the chunk-index dataset.")
    p.add_argument("--model-id", type=str, required=True, help="Model id used to resolve the default tokenizer.")
    p.add_argument("--tokenizer", type=str, default=None, help="HF tokenizer name/path (defaults to --model-id).")
    p.add_argument("--processor", type=str, default=None, help="HF repo to read image config from when the dataset contains images.")
    p.add_argument("--preprocessor-config", type=str, default=None, help="Path to a JSON file whose keys override the default image processor config.")
    p.add_argument("--max-length", type=int, required=True)
    p.add_argument("--records-per-shard", type=int, default=100_000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer_name = args.tokenizer or resolve_hf_repo_id(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    image_processor = None
    processor_name = None
    if args.processor:
        processor_name = args.processor
        ip_kwargs: dict = {}
        if args.preprocessor_config:
            with open(args.preprocessor_config) as f:
                ip_kwargs = json.load(f)
        image_processor = AutoImageProcessor.from_pretrained(processor_name, use_fast=False, **ip_kwargs)

    def measure_messages(messages) -> int:
        if image_processor is None and _messages_have_images(messages):
            raise ValueError("Chunk indexing encountered image content but no --processor was provided.")
        encoded = encode_qwen_messages(
            messages,
            tokenizer=tokenizer,
            image_processor=image_processor,
            include_pixels=False,
        )
        return int(len(encoded["input_ids"]))

    out_dir = build_chunk_index(
        args.data_path,
        args.out_dir,
        max_length=args.max_length,
        measure_messages=measure_messages,
        records_per_shard=args.records_per_shard,
        overwrite=args.overwrite,
        profile_metadata={
            "model_id": args.model_id,
            "tokenizer": tokenizer_name,
            "processor": processor_name,
            "preprocessor_config": args.preprocessor_config,
        },
    )
    print(out_dir)


if __name__ == "__main__":
    main()
