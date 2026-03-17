"""VLM SFT training from a JSONL dataset (text-only or multimodal)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import jax
import numpy as np
from tensorboardX import SummaryWriter
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
    p.add_argument("--preprocessor-config", type=str, default=None, help="Path to a JSON file whose keys override the default image processor config.")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "cos", "wsd"], help="LR schedule type.")
    p.add_argument("--warmup-steps", type=int, default=0, help="Linear warmup steps.")
    p.add_argument("--min-lr-ratio", type=float, default=0.0, help="Min LR as a fraction of --learning-rate (used by cos and wsd).")
    p.add_argument("--wsd-decay-fraction", type=float, default=0.1, help="Fraction of total steps for the WSD linear decay phase.")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient global norm for clipping. Set to 0 to disable.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tp-size", type=int, default=None)
    p.add_argument("--fsdp-size", type=int, default=None)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--profile-dir", type=str, default=None)
    p.add_argument("--profile-start", type=int, default=3, help="Step to start profiling (after JIT warmup).")
    p.add_argument("--profile-end", type=int, default=8, help="Step to stop profiling.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--pad-id", type=int, default=0)
    p.add_argument("--peak-tflops", type=str, default=None)
    p.add_argument("--tensorboard-dir", type=str, default=None, help="Directory for TensorBoard event files.")
    p.add_argument("--max-turns", type=int, default=None, help="Max messages per conversation; longer chats are split into chunks.")
    p.add_argument("--val-data-path", type=str, default=None, help="Path to JSONL validation data.")
    p.add_argument("--val-every", type=int, default=None, help="Run validation every N training steps.")
    p.add_argument("--val-steps", type=int, default=10, help="Number of batches per validation run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jax.distributed.initialize()

    repo_id = args.processor or resolve_hf_repo_id(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    assert args.max_length <= tokenizer.model_max_length, f"--max-length={args.max_length} exceeds tokenizer.model_max_length={tokenizer.model_max_length}"

    ip_kwargs: dict = {}
    if args.preprocessor_config:
        with open(args.preprocessor_config) as f:
            ip_kwargs = json.load(f)
    image_processor = AutoImageProcessor.from_pretrained(repo_id, use_fast=False, **ip_kwargs)
    collator = VLMSFTCollator(tokenizer, max_length=args.max_length, image_processor=image_processor)

    dataset = JSONLDataset(args.data_path, max_turns=args.max_turns)
    data_iter = _batched_iter(dataset, collator, args.batch_size, shuffle=True, seed=args.seed)

    val_data_iter = None
    if args.val_data_path:
        val_dataset = JSONLDataset(args.val_data_path, max_turns=args.max_turns)
        val_data_iter = _batched_iter(val_dataset, collator, args.batch_size, shuffle=False, seed=args.seed)

    train_cfg = vlm_trainer.TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.max_length,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        print_every=args.log_every,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        wsd_decay_fraction=args.wsd_decay_fraction,
        max_grad_norm=args.max_grad_norm or None,
    )
    save_dir = Path(args.save_dir) if args.save_dir else _default_save_dir(args.model_id)
    peak_tflops = resolve_peak_tflops(args.peak_tflops)

    tb_writer = None
    if args.tensorboard_dir and jax.process_index() == 0:
        tb_dir = Path(args.tensorboard_dir)
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(str(tb_dir))
        tb_writer.add_hparams(args.__dict__, {}, name="hparams")
    try:
        _, last_metrics = vlm_trainer.run_sft(
            args.model_id,
            train_cfg,
            data_iter,
            save_dir=save_dir,
            save_every=args.save_every,
            log_every=args.log_every,
            resume=args.resume,
            pad_id=args.pad_id,
            peak_tflops=peak_tflops,
            tp_size=args.tp_size,
            fsdp_size=args.fsdp_size,
            profile_dir=args.profile_dir,
            profile_steps=(args.profile_start, args.profile_end),
            tb_writer=tb_writer,
            val_data_iter=val_data_iter,
            val_every=args.val_every,
            val_steps=args.val_steps,
        )
    finally:
        if tb_writer is not None:
            tb_writer.close()

    if last_metrics:
        print(f"finished step={int(last_metrics['step'])} loss={last_metrics['loss']:.4f}")

if __name__ == "__main__":
    main()
