"""Simple text-model training loop with synthetic data, logging, and checkpointing."""
# FIXME(f.srambical): Change this such that it supports real data

from __future__ import annotations

import argparse
from pathlib import Path

from omegalax.trainers import text as text_trainer


def _default_save_dir(model_id: str) -> Path:
    safe_name = model_id.replace("/", "_")
    return Path("runs") / "text" / safe_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a text model on synthetic data.")
    parser.add_argument("--model-id", type=str, default="qwen3-smoke")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to store checkpoints/logs.")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--log-jsonl", type=str, default=None, help="Optional JSONL metrics file.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if present.")
    parser.add_argument("--pad-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = text_trainer.TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        print_every=args.log_every,
    )
    save_dir = Path(args.save_dir) if args.save_dir else _default_save_dir(args.model_id)

    _, last_metrics = text_trainer.run_training(
        args.model_id,
        train_cfg,
        save_dir=save_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        log_jsonl=args.log_jsonl,
        resume=args.resume,
        pad_id=args.pad_id,
    )
    if last_metrics:
        print(
            f"finished step={int(last_metrics['step'])} loss={last_metrics['loss']:.4f}"
        )


if __name__ == "__main__":
    main()
