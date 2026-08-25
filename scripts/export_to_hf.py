"""Export a sealed local VLM snapshot or one exact trained checkpoint."""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from absl import app, flags
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from omegalax import export as export_lib
from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.vlm import api as vlm_api
from omegalax.vlm.local_snapshot import LocalVLMSnapshot, open_local_vlm_snapshot

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_snapshot", None, "Absolute sealed local VLM snapshot directory.", required=True
)
flags.DEFINE_string("out_dir", None, "Destination directory for safetensors+config.", required=True)
flags.DEFINE_integer("seed", 0, "RNG seed used when initializing the model.")
flags.DEFINE_integer("tp_size", None, "Tensor parallelism size.")
flags.DEFINE_integer("fsdp_size", None, "FSDP parallelism size.")
flags.DEFINE_integer("dp_size", None, "Data parallelism size.")

# Trained-checkpoint mode: set to a step dir like /.../first_training_run_*/010000/
flags.DEFINE_string(
    "checkpoint_path",
    None,
    "If set, restore weights from this orbax step directory "
    "before exporting. Parent dir is treated as the save_dir.",
)


def _load_vlm_model(model_snapshot: LocalVLMSnapshot):
    model, cfg = vlm_api.load_pretrained(
        model_snapshot,
        tp_size=FLAGS.tp_size,
        fsdp_size=FLAGS.fsdp_size,
        dp_size=FLAGS.dp_size,
    )
    return model, cfg


def _restore_trained_weights(
    model,
    checkpoint_path: Path,
    model_snapshot: LocalVLMSnapshot,
):
    """Restore only trained model weights and their bound snapshot identity."""
    if (
        not checkpoint_path.is_absolute()
        or checkpoint_path != Path(os.path.normpath(checkpoint_path))
        or not checkpoint_path.name.isdigit()
    ):
        raise ValueError("--checkpoint_path must be a canonical absolute step directory")
    save_dir = checkpoint_path.parent.resolve()
    lora_meta = export_lib.read_lora_metadata(save_dir)
    step = int(checkpoint_path.name)
    if step <= 0 or checkpoint_path.name != f"{step:06d}":
        raise ValueError("--checkpoint_path must name a six-digit positive generation")

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

        model_state = nnx.state(model)
        model_abstract = jax.tree.map(
            lambda v: jax.ShapeDtypeStruct(
                v.shape,
                v.dtype,
                sharding=getattr(v, "sharding", None) or default_sharding,
            ),
            model_state,
        )
        params_only_abstract = {
            "optimizer": {"model": model_abstract},
            "model_identity": jax.ShapeDtypeStruct(
                (32,),
                jnp.uint8,
                sharding=default_sharding,
            ),
        }

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

        active_error = None
        try:
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
        except BaseException as error:
            active_error = error
            raise
        finally:
            try:
                cm.close()
            except BaseException as cleanup_error:
                if active_error is None:
                    raise
                active_error.add_note(f"Checkpoint cleanup also failed: {cleanup_error!r}")
        train_state = restored["train_state"]
        if bytes(jax.device_get(train_state["model_identity"])) != bytes.fromhex(
            model_snapshot.sha256
        ):
            raise ValueError("Checkpoint model snapshot does not match --model_snapshot")
        restored_model_state = train_state["optimizer"]["model"]
        nnx.update(model, restored_model_state)
    print(f"Restored model params from step {step} at {save_dir}")
    return model


_DESCRIBES_BASE_WEIGHTS = ("quantization_config",)


def _write_servable_config(
    out_dir: Path,
    cfg,
    model_snapshot: LocalVLMSnapshot,
) -> None:
    with open(model_snapshot.files()["config.json"], encoding="utf-8") as stream:
        base = json.load(stream)
    for key in _DESCRIBES_BASE_WEIGHTS:
        base.pop(key, None)

    owned = export_lib.model_config_to_hf_dict(cfg)
    merged = {**base, **owned}
    for sub in ("text_config", "vision_config"):
        if isinstance(base.get(sub), dict) and isinstance(owned.get(sub), dict):
            merged[sub] = {**base[sub], **owned[sub]}

    (out_dir / "config.json").write_text(json.dumps(merged, indent=2) + "\n")


def main(_) -> None:
    jax.distributed.initialize()
    with open_local_vlm_snapshot(FLAGS.model_snapshot) as model_snapshot:
        model, cfg = _load_vlm_model(model_snapshot)
        out_dir = Path(FLAGS.out_dir)
        out_dir.mkdir(parents=True, exist_ok=False)
        model_snapshot.copy_identity_assets(out_dir)
        if FLAGS.checkpoint_path:
            model = _restore_trained_weights(
                model,
                Path(FLAGS.checkpoint_path),
                model_snapshot,
            )
        path = export_lib.export_model_to_hf(model, cfg, out_dir)
        _write_servable_config(out_dir, cfg, model_snapshot)
        print(f"Exported safetensors to {path}")


if __name__ == "__main__":
    app.run(main)
