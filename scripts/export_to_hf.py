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
flags.DEFINE_string(
    "checkpoint_path",
    None,
    "If set, restore weights from this orbax step directory "
    "before exporting. Parent dir is treated as the save_dir.",
)
# Optimizer-shape flags; defaults match a typical full-finetune (max_grad_norm>0
# and grad_accum_steps>1). Override if the saved checkpoint used different
# wiring.
flags.DEFINE_float(
    "max_grad_norm", 0.5, "Affects optimizer state shape: >0 includes optax.clip_by_global_norm."
)
flags.DEFINE_integer(
    "grad_accum_steps", 8, "Affects optimizer state shape: >1 wraps with optax.MultiSteps."
)
flags.DEFINE_float(
    "learning_rate", 1e-5, "LR (numeric value not saved; needed only for build_optimizer)."
)
flags.DEFINE_float(
    "weight_decay", 0.01, "WD (numeric value not saved; needed only for build_optimizer)."
)
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


def _read_lora_metadata(save_dir: Path) -> dict:
    """Return the LoRA settings the trainer persisted next to the orbax tree.

    Never infers. ``vlm.run_sft`` writes this file for every checkpoint, full-FT
    included, so an absent or incomplete one means we cannot know whether to
    ``inject_lora`` before deriving the restore template -- and guessing "full-FT"
    is the silent-corruption path: ``partial_restore=True`` then matches the base
    subtree, drops every LoRA leaf without a word, and exports the base model.
    """
    import json

    p = save_dir / "lora_metadata.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no lora_metadata.json next to the checkpoint at {save_dir}. Every "
            "checkpoint written by omegalax.trainers.vlm has one; without it an "
            "adapter checkpoint exports as the base model with no error. Write the "
            "file from the training run's recipe (enable_lora, lora_rank, lora_alpha)."
        )
    meta = json.loads(p.read_text())
    missing = {"enable_lora", "lora_rank", "lora_alpha"} - meta.keys()
    if missing:
        raise ValueError(f"{p} is missing {sorted(missing)}; refusing to guess a LoRA rank")
    return meta


def _restore_trained_weights(model, cfg, checkpoint_path: Path):
    """Restore trained weights from an orbax step directory into ``model``.

    Restores ONLY the model parameter subtree (``train_state/optimizer/model``)
    from the saved checkpoint — never the optimizer state or step counters.
    Optimizer state contains replicated scalar leaves stored as
    ``(num_devices,)`` arrays by orbax; orbax doesn't collapse those to ``()``
    on a smaller-device restore topology, so any export that asks for the
    full train_state (model + opt_state + step + rng) fails with
    ``Requested shape: () is not compatible with the stored shape: (N,)``
    whenever training was done on more devices than the export node has. The
    exporter only needs model weights for the HF safetensors write, so the
    optimizer subtree is genuinely unneeded: skipping it sidesteps the bug
    AND meaningfully reduces work / memory.

    The restore uses explicit ``ocp.ArrayRestoreArgs(sharding=...,
    global_shape=..., dtype=...)`` per leaf so orbax cross-topology-restores
    cleanly without falling back to the on-disk sharding file (which still
    references the original training devices).

    For LoRA-trained checkpoints, the trainer wrote a
    ``lora_metadata.json`` next to the orbax tree. We read it and inject
    LoRA into the freshly-loaded base model BEFORE deriving the abstract
    template so the model state's tree shape matches the saved subtree.
    """
    save_dir = checkpoint_path.parent.resolve()
    lora_meta = _read_lora_metadata(save_dir)
    step = int(checkpoint_path.name)

    mesh = ensure_mesh(tp_size=FLAGS.tp_size, fsdp_size=FLAGS.fsdp_size, dp_size=FLAGS.dp_size)
    default_sharding = NamedSharding(mesh, P())

    with mesh_rules(mesh):
        if bool(lora_meta["enable_lora"]):
            from omegalax.trainers.lora import inject_lora

            rank = int(lora_meta["lora_rank"])
            n_wrapped = inject_lora(
                model,
                r=rank,
                alpha=float(lora_meta["lora_alpha"]),
                rngs=nnx.Rngs(FLAGS.seed),
            )
            print(f"[export] re-injected LoRA into base for restore: r={rank} wrapped={n_wrapped}")

        # Build abstract template for the model-params subtree only. The
        # saved tree stored it under ``train_state/optimizer/model``; we
        # mirror that nesting so orbax's path matching finds it.
        model_state = nnx.state(model)
        model_abstract = jax.tree.map(
            lambda v: jax.ShapeDtypeStruct(
                v.shape,
                v.dtype,
                sharding=getattr(v, "sharding", None) or default_sharding,
            ),
            model_state,
        )
        params_only_abstract = {"optimizer": {"model": model_abstract}}

        # Explicit ArrayRestoreArgs per leaf — bypasses orbax's disk-sharding
        # fallback path which fails when the original training topology's
        # devices aren't present on the export node.
        def _to_restore_args(s):
            if isinstance(s, jax.ShapeDtypeStruct):
                return ocp.ArrayRestoreArgs(
                    sharding=s.sharding if s.sharding is not None else default_sharding,
                    global_shape=s.shape,
                    dtype=s.dtype,
                )
            return s

        params_only_restore_args = jax.tree.map(
            _to_restore_args,
            params_only_abstract,
            is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
        )

        handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
        handler_registry.add(
            "train_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler
        )
        options = ocp.CheckpointManagerOptions(step_format_fixed_length=6)
        cm = ocp.CheckpointManager(save_dir, options=options, handler_registry=handler_registry)

        restored = cm.restore(
            step,
            args=ocp.args.Composite(
                train_state=ocp.args.PyTreeRestore(
                    params_only_abstract,
                    restore_args=params_only_restore_args,
                    partial_restore=True,
                ),
            ),
        )
        restored_model_state = restored["train_state"]["optimizer"]["model"]
        nnx.update(model, restored_model_state)
    print(f"Restored model params from step {step} at {save_dir}")
    return model


def main(_) -> None:
    jax.distributed.initialize()
    model, cfg = load_model()
    if not FLAGS.checkpoint_path:
        out_dir = Path(FLAGS.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Exported safetensors to {export_lib.export_model_to_hf(model, cfg, out_dir)}")
        return

    base_fingerprint = export_lib.param_fingerprint(model)
    model = _restore_trained_weights(model, cfg, Path(FLAGS.checkpoint_path).expanduser())
    out_dir = Path(FLAGS.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = export_lib.export_model_to_hf(model, cfg, out_dir)

    # After the write, because export_model_to_hf owns the LoRA merge and a LoRA
    # run trains no base leaf: pre-merge, a correct adapter export is legitimately
    # base-identical on every shared key.
    exported = export_lib.param_fingerprint(model)
    if exported.keys() != base_fingerprint.keys():
        raise ValueError(
            f"{path} was written from a parameter tree that is not the base's: "
            f"{sorted(exported.keys() - base_fingerprint.keys())[:3]}. An unmerged LoRA "
            f"adapter exports its BASE kernel -- LoRALinear.kernel forwards to "
            f"base.kernel -- so the trained delta is silently absent. Do NOT use it."
        )
    changed = [k for k, v in base_fingerprint.items() if exported[k] != v]
    if not changed:
        raise ValueError(
            f"{path} is identical to the pretrained {FLAGS.model_id} on all "
            f"{len(base_fingerprint)} parameter leaves. The restore matched nothing -- "
            f"orbax partial_restore drops leaves without raising -- so this export is the "
            f"base model. Do NOT use it. Check that "
            f"{Path(FLAGS.checkpoint_path).expanduser().parent}/lora_metadata.json "
            f"matches the training run."
        )
    print(f"Exported safetensors to {path}")
    print(f"[export] {len(changed)}/{len(base_fingerprint)} parameter leaves differ from the base")


if __name__ == "__main__":
    app.run(main)
