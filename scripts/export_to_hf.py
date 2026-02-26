"""Export any supported omegalax model to HuggingFace safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path

from flax import nnx
import jax

from omegalax import export as export_lib
from omegalax import registry
from omegalax.text import api as text_api
from omegalax.trainers import text as text_trainer
from omegalax.vlm import api as vlm_api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a model to HF safetensors.")
    parser.add_argument("--model-id", type=str, required=True, help="Model id to export.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional orbax checkpoint dir from training.")
    parser.add_argument("--out-dir", type=str, required=True, help="Destination directory for safetensors+config.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed used when initializing the model.")
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--fsdp-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer LR used if restoring a checkpoint.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Optimizer WD used if restoring a checkpoint.")
    parser.add_argument("--pad-id", type=int, default=0, help="Padding token id (for cache creation).")
    return parser.parse_args()


def _load_text_model(args):
    model_cfg = text_api.registry.build_config(args.model_id)
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
    model = text_trainer.init_model(
        model_cfg,
        init_rng,
        tp_size=args.tp_size,
        fsdp_size=args.fsdp_size,
    )
    optimizer = text_trainer.build_optimizer(model, train_cfg)

    if args.checkpoint:
        ckpt_dir = Path(args.checkpoint)
        checkpoint_manager = text_trainer._make_checkpoint_manager(ckpt_dir, save_interval=None)  # type: ignore
        optimizer_state, _, _ = text_trainer._restore_checkpoint(checkpoint_manager, nnx.state(optimizer), rng)  # type: ignore
        optimizer = nnx.merge(nnx.graphdef(optimizer), optimizer_state)
        model = optimizer.model

    return model, model_cfg


def _load_vlm_model(args):
    rng = jax.random.key(args.seed)
    model, cfg = vlm_api.init_model(
        args.model_id,
        rng,
        tp_size=args.tp_size,
        fsdp_size=args.fsdp_size,
    )
    if args.checkpoint:
        raise NotImplementedError("Checkpoint restoration for VLM exports is not implemented yet.")
    return model, cfg


def load_model(args):
    arch = registry.resolve(args.model_id).arch
    if arch == registry.Arch.TEXT:
        return _load_text_model(args)
    if arch == registry.Arch.VLM:
        return _load_vlm_model(args)
    raise ValueError(f"Unsupported architecture for model id '{args.model_id}'")


def main() -> None:
    args = parse_args()
    jax.distributed.initialize()
    model, cfg = load_model(args)
    out_dir = Path(args.out_dir)
    path = export_lib.export_model_to_hf(model, cfg, out_dir)
    print(f"Exported safetensors to {path}")


if __name__ == "__main__":
    main()
