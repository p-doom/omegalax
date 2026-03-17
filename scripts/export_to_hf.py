"""Export any supported omegalax model to HuggingFace safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path
import jax

from omegalax import export as export_lib
from omegalax import registry
from omegalax.text import api as text_api
from omegalax.trainers import text as text_trainer
from omegalax.vlm import api as vlm_api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a model to HF safetensors.")
    parser.add_argument("--model-id", type=str, required=True, help="Model id to export.")
    parser.add_argument("--out-dir", type=str, required=True, help="Destination directory for safetensors+config.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed used when initializing the model.")
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--fsdp-size", type=int, default=None)
    parser.add_argument("--pad-id", type=int, default=0, help="Padding token id (for cache creation).")
    return parser.parse_args()


def _load_text_model(args):
    model_cfg = text_api.registry.build_config(args.model_id)
    rng = jax.random.key(args.seed)
    rng, init_rng = jax.random.split(rng)
    model, model_cfg = text_trainer.init_model(
        model_cfg,
        init_rng,
        tp_size=args.tp_size,
        fsdp_size=args.fsdp_size,
    )
    return model, model_cfg


def _load_vlm_model(args):
    rng = jax.random.key(args.seed)
    model, cfg = vlm_api.init_model(
        args.model_id,
        rng,
        tp_size=args.tp_size,
        fsdp_size=args.fsdp_size,
    )
    return model, cfg


def load_model(args):
    arch = registry.resolve(args.model_id)
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
