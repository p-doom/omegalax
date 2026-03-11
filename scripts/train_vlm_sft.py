"""VLM SFT training from a JSONL dataset (text-only or multimodal)."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import jax
import numpy as np
from transformers import AutoImageProcessor, AutoTokenizer

from omegalax.data.collator_qwen3 import VLMSFTCollator
from omegalax.data.jsonl import JSONLDataset
from omegalax.trainers import vlm as vlm_trainer
from omegalax.registry import resolve_hf_repo_id
from omegalax.trainers.perf import resolve_peak_tflops


def _default_save_dir(model_id: str) -> Path:
    safe_name = model_id.replace("/", "_")
    return Path("runs") / "vlm_sft" / safe_name


def _batched_iter(
    dataset: JSONLDataset,
    collator: VLMSFTCollator,
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield collated batches from the dataset forever."""
    buf: list[dict] = []
    for ex in dataset.iter_examples(shuffle=shuffle, seed=seed, num_epochs=None):
        buf.append(ex)
        if len(buf) == batch_size:
            yield collator(buf)
            buf = []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT a VLM from a JSONL dataset.")
    p.add_argument("--model-id", type=str, required=True)
    p.add_argument("--data-path", type=str, required=True, help="Path to JSONL training data.")
    p.add_argument("--processor", type=str, default=None, help="HF repo to read tokenizer and image config from (defaults to --model-id).")
    p.add_argument("--min-pixels", type=int, default=None, help="Override image processor min_pixels before preprocessing.")
    p.add_argument("--max-pixels", type=int, default=None, help="Override image processor max_pixels before preprocessing.")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tp-size", type=int, default=None)
    p.add_argument("--fsdp-size", type=int, default=None)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--log-jsonl", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--pad-id", type=int, default=0)
    p.add_argument("--peak-tflops", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jax.distributed.initialize()

    repo_id = args.processor or resolve_hf_repo_id(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    assert args.max_length <= tokenizer.model_max_length, f"--max-length={args.max_length} exceeds tokenizer.model_max_length={tokenizer.model_max_length}"
    image_processor = AutoImageProcessor.from_pretrained(repo_id, use_fast=False)
    if args.min_pixels is not None:
        image_processor.min_pixels = args.min_pixels
        if hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
            image_processor.size["shortest_edge"] = args.min_pixels
    if args.max_pixels is not None:
        image_processor.max_pixels = args.max_pixels
        if hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
            image_processor.size["longest_edge"] = args.max_pixels
    collator = VLMSFTCollator(tokenizer, max_length=args.max_length, image_processor=image_processor)

    dataset = JSONLDataset(args.data_path)
    data_iter = _batched_iter(dataset, collator, args.batch_size, shuffle=True, seed=args.seed)

    train_cfg = vlm_trainer.TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.max_length,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        print_every=args.log_every,
    )
    save_dir = Path(args.save_dir) if args.save_dir else _default_save_dir(args.model_id)
    peak_tflops = resolve_peak_tflops(args.peak_tflops)

    _, last_metrics = vlm_trainer.run_sft(
        args.model_id,
        train_cfg,
        data_iter,
        save_dir=save_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        log_jsonl=args.log_jsonl,
        resume=args.resume,
        pad_id=args.pad_id,
        peak_tflops=peak_tflops,
        tp_size=args.tp_size,
        fsdp_size=args.fsdp_size,
    )
    if last_metrics:
        print(f"finished step={int(last_metrics['step'])} loss={last_metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
