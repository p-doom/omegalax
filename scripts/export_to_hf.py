"""Export any supported omegalax model to HuggingFace safetensors.

Two modes:
  * Default: export the off-the-shelf pretrained weights for ``--model_id``.
  * With ``--checkpoint_path``: load architecture from ``--model_id``, then
    restore trained weights from an orbax checkpoint directory (one of the
    step subdirs written by ``omegalax.trainers.vlm`` during SFT) and export
    those.

The optimizer-build flags (``--max_grad_norm``, ``--grad_accum_steps``, ...)
must match the training run's settings closely enough that the optimizer
pytree shape matches what was saved. Specifically: ``max_grad_norm > 0``
toggles a clip step in the chain, and ``grad_accum_steps > 1`` toggles an
``optax.MultiSteps`` wrapper. Numeric values themselves are not stored in
the checkpoint and don't have to match.
"""

from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path

from absl import app, flags
import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import orbax.checkpoint as ocp
from flax import nnx

from omegalax import export as export_lib
from omegalax import registry
from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.text import api as text_api
from omegalax.trainers import text as text_trainer
from omegalax.trainers import vlm as vlm_trainer
from omegalax.trainers.lr_schedule import build_lr_schedule
from omegalax.vlm import api as vlm_api

FLAGS = flags.FLAGS

flags.DEFINE_string("model_id", None, "Model id to export.", required=True)
flags.DEFINE_string("out_dir", None, "Destination directory for safetensors+config.", required=True)
flags.DEFINE_integer("seed", 0, "RNG seed used when initializing the model.")
flags.DEFINE_integer("tp_size", None, "Tensor parallelism size.")
flags.DEFINE_integer("fsdp_size", None, "FSDP parallelism size.")
flags.DEFINE_integer("dp_size", None, "Data parallelism size.")
flags.DEFINE_integer("pad_id", 0, "Padding token id (for cache creation).")

# Trained-checkpoint mode: set to a step dir like /.../first_training_run_*/010000/
flags.DEFINE_string("checkpoint_path", None,
                    "If set, restore weights from this orbax step directory "
                    "before exporting. Parent dir is treated as the save_dir.")
# Optimizer-shape flags; defaults match a typical full-finetune (max_grad_norm>0
# and grad_accum_steps>1). Override if the saved checkpoint used different
# wiring.
flags.DEFINE_float("max_grad_norm", 0.5,
                   "Affects optimizer state shape: >0 includes optax.clip_by_global_norm.")
flags.DEFINE_integer("grad_accum_steps", 8,
                     "Affects optimizer state shape: >1 wraps with optax.MultiSteps.")
flags.DEFINE_float("learning_rate", 1e-5, "LR (numeric value not saved; needed only for build_optimizer).")
flags.DEFINE_float("weight_decay", 0.01, "WD (numeric value not saved; needed only for build_optimizer).")
flags.DEFINE_integer("warmup_steps", 1000, "LR-schedule warmup steps (not saved).")
flags.DEFINE_integer("num_steps", 200000, "LR-schedule total steps (not saved).")
flags.DEFINE_string("lr_schedule", "wsd", "LR schedule kind (not saved).")
flags.DEFINE_float("lr_stable_fraction", 0.9, "LR-schedule stable fraction (not saved).")
flags.DEFINE_float("lr_end_factor", 0.0, "LR-schedule end factor (not saved).")


def _load_text_model():
    # NB: text_api currently has no load_pretrained; re-using init_model here
    # would silently export random weights (cf. the VLM-path fix below).
    # If text export becomes needed, mirror vlm_api.load_pretrained: snapshot
    # download + create_qwen3{,_5}_from_safetensors.
    raise NotImplementedError(
        "Text export not yet wired to load_pretrained; would silently export "
        "random weights. Add text_api.load_pretrained first."
    )


def _load_vlm_model():
    # IMPORTANT: vlm_api.init_model() does *random* sharded init, not weight
    # loading. Calling it here produced syntactically-valid safetensors with
    # untrained weights — a silent corruption. Use load_pretrained instead.
    model, cfg = vlm_api.load_pretrained(
        FLAGS.model_id,
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


def _restore_trained_weights(model, cfg, checkpoint_path: Path):
    """Restore trained weights from an orbax step directory into ``model``.

    Builds an optimizer mirroring training-time wiring (so the saved pytree
    shape matches), constructs a CheckpointManager handling only the
    train_state subkey, and updates ``model`` from the restored optimizer
    state.
    """
    train_cfg = vlm_trainer.TrainConfig(
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay,
        max_grad_norm=FLAGS.max_grad_norm,
        grad_accum_steps=FLAGS.grad_accum_steps,
        num_steps=FLAGS.num_steps,
        warmup_steps=FLAGS.warmup_steps,
        lr_schedule=FLAGS.lr_schedule,
        lr_stable_fraction=FLAGS.lr_stable_fraction,
        lr_end_factor=FLAGS.lr_end_factor,
    )
    lr_schedule_fn = build_lr_schedule(
        peak_lr=train_cfg.learning_rate,
        num_steps=train_cfg.num_steps,
        warmup_steps=train_cfg.warmup_steps,
        schedule=train_cfg.lr_schedule,
        end_factor=train_cfg.lr_end_factor,
        stable_fraction=train_cfg.lr_stable_fraction,
    )
    save_dir = checkpoint_path.parent.resolve()
    step = int(checkpoint_path.name)

    # build_optimizer (and the abstract-state sharding it produces) needs
    # the logical->physical axis-rules context the trainer sets up.
    mesh = ensure_mesh(tp_size=FLAGS.tp_size, fsdp_size=FLAGS.fsdp_size, dp_size=FLAGS.dp_size)
    # Replicated scalar rng placed on the mesh; otherwise _abstract_train_state
    # produces P(None,) sharding for the rank-0 key which orbax rejects.
    rng = jax.device_put(jax.random.key(FLAGS.seed), NamedSharding(mesh, P()))
    with mesh_rules(mesh):
        optimizer = vlm_trainer.build_optimizer(model, lr_schedule_fn, train_cfg)

        handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
        handler_registry.add("train_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
        options = ocp.CheckpointManagerOptions(step_format_fixed_length=6)
        cm = ocp.CheckpointManager(save_dir, options=options, handler_registry=handler_registry)

        abstract_state = vlm_trainer._abstract_train_state(optimizer, rng)
        # partial_restore=True: only restore keys present in both saved and
        # abstract trees. The opt_state subtree may differ (different optax
        # composition between save-time and now) but model weights match,
        # which is all the exporter needs.
        restored = cm.restore(
            step,
            args=ocp.args.Composite(
                train_state=ocp.args.PyTreeRestore(abstract_state, partial_restore=True),
            ),
        )
        nnx.update(optimizer, restored["train_state"]["optimizer"])
    print(f"Restored train_state from step {step} at {save_dir}")
    return optimizer.model


def main(_) -> None:
    jax.distributed.initialize()
    model, cfg = load_model()
    if FLAGS.checkpoint_path:
        ckpt = Path(FLAGS.checkpoint_path).expanduser()
        model = _restore_trained_weights(model, cfg, ckpt)
    out_dir = Path(FLAGS.out_dir)
    path = export_lib.export_model_to_hf(model, cfg, out_dir)
    print(f"Exported safetensors to {path}")


if __name__ == "__main__":
    app.run(main)
