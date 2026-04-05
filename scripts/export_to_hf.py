"""Export any supported omegalax model to HuggingFace safetensors."""

from __future__ import annotations

from pathlib import Path

from absl import app, flags
import jax

from omegalax import export as export_lib
from omegalax import registry
from omegalax.text import api as text_api
from omegalax.trainers import text as text_trainer
from omegalax.vlm import api as vlm_api

FLAGS = flags.FLAGS

flags.DEFINE_string("model_id", None, "Model id to export.", required=True)
flags.DEFINE_string("out_dir", None, "Destination directory for safetensors+config.", required=True)
flags.DEFINE_integer("seed", 0, "RNG seed used when initializing the model.")
flags.DEFINE_integer("tp_size", None, "Tensor parallelism size.")
flags.DEFINE_integer("fsdp_size", None, "FSDP parallelism size.")
flags.DEFINE_integer("dp_size", None, "Data parallelism size.")
flags.DEFINE_integer("pad_id", 0, "Padding token id (for cache creation).")


def _load_text_model():
    model_cfg = text_api.registry.build_config(FLAGS.model_id)
    rng = jax.random.key(FLAGS.seed)
    rng, init_rng = jax.random.split(rng)
    model, model_cfg = text_trainer.init_model(
        model_cfg,
        init_rng,
        tp_size=FLAGS.tp_size,
        fsdp_size=FLAGS.fsdp_size,
        dp_size=FLAGS.dp_size,
    )
    return model, model_cfg


def _load_vlm_model():
    rng = jax.random.key(FLAGS.seed)
    model, cfg = vlm_api.init_model(
        FLAGS.model_id,
        rng,
        tp_size=FLAGS.tp_size,
        fsdp_size=FLAGS.fsdp_size,
        dp_size=FLAGS.dp_size,
    )
    return model, cfg


def load_model():
    arch = registry.resolve(FLAGS.model_id)
    if arch == registry.Arch.TEXT:
        return _load_text_model()
    if arch == registry.Arch.VLM:
        return _load_vlm_model()
    raise ValueError(f"Unsupported architecture for model id '{FLAGS.model_id}'")


def main(_) -> None:
    jax.distributed.initialize()
    model, cfg = load_model()
    out_dir = Path(FLAGS.out_dir)
    path = export_lib.export_model_to_hf(model, cfg, out_dir)
    print(f"Exported safetensors to {path}")


if __name__ == "__main__":
    app.run(main)
