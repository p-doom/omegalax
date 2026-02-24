"""Export a Qwen3 dense JAX model to HuggingFace safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

from omegalax.text import api as text_api
from omegalax.trainers import text as text_trainer
from omegalax.models.qwen3.dense.params_dense import export_qwen3_dense_to_safetensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Qwen3 dense model to HF safetensors.")
    parser.add_argument("--model-id", type=str, default="qwen3-smoke", help="Model id to export.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional orbax checkpoint dir from training.")
    parser.add_argument("--out-dir", type=str, required=True, help="Destination directory for safetensors+config.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed used when initializing the model.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer LR used if restoring a checkpoint.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Optimizer WD used if restoring a checkpoint.")
    parser.add_argument("--pad-id", type=int, default=0, help="Padding token id (for cache creation).")
    return parser.parse_args()


def load_model(args) -> tuple[object, object]:
    model_cfg = text_api.registry.build_config(args.model_id)
    if model_cfg.variant != "dense":
        raise NotImplementedError("export_to_hf currently supports dense Qwen3 models only.")

    train_cfg = text_trainer.TrainConfig(
        seed=args.seed,
        batch_size=1,
        seq_len=8,
        num_steps=1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        print_every=1,
    )

    rng = jax.random.key(train_cfg.seed)
    rng, init_rng = jax.random.split(rng)
    model = text_trainer.init_model(model_cfg, init_rng)
    optimizer = text_trainer.build_optimizer(model, train_cfg)

    if args.checkpoint:
        ckpt_dir = Path(args.checkpoint)
        checkpoint_manager = text_trainer._make_checkpoint_manager(ckpt_dir, save_interval=None)  # type: ignore
        optimizer, _, _ = text_trainer._restore_checkpoint(checkpoint_manager, optimizer, rng)  # type: ignore
        model = optimizer.model

    return model, model_cfg


def main() -> None:
    args = parse_args()
    model, cfg = load_model(args)
    out_dir = Path(args.out_dir)
    path = export_qwen3_dense_to_safetensors(model, cfg, out_dir)
    print(f"Exported safetensors to {path}")


if __name__ == "__main__":
    main()
