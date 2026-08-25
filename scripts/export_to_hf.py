"""Export an Omegalax checkpoint to a HuggingFace model directory."""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import shutil
from pathlib import Path

import jax
import orbax.checkpoint as ocp
from absl import app, flags
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import AutoTokenizer

from omegalax import export as export_lib
from omegalax import registry
from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.vlm import api as vlm_api

FLAGS = flags.FLAGS

flags.DEFINE_string("model_id", None, "Model id to export.", required=True)
flags.DEFINE_string(
    "model_revision",
    None,
    "Exact HuggingFace commit for a remote --model_id; omit when --model_id is a local path.",
)
flags.DEFINE_string("out_dir", None, "Destination directory for safetensors+config.", required=True)
flags.DEFINE_integer("seed", 0, "RNG seed used when initializing the model.")
flags.DEFINE_integer("tp_size", None, "Tensor parallelism size.")
flags.DEFINE_integer("fsdp_size", None, "FSDP parallelism size.")
flags.DEFINE_integer("dp_size", None, "Data parallelism size.")
flags.DEFINE_string(
    "checkpoint_path",
    None,
    "Exact numeric Orbax checkpoint step directory to export.",
    required=True,
)


def _load_text_model():
    raise NotImplementedError("Text checkpoint export is not supported")


def _load_vlm_model(model_source: Path):
    model, cfg = vlm_api.load_pretrained(
        model_source,
        tp_size=FLAGS.tp_size,
        fsdp_size=FLAGS.fsdp_size,
        dp_size=FLAGS.dp_size,
    )
    return model, cfg


def load_model(model_source: Path):
    arch = registry.resolve(str(model_source))
    if arch == registry.Arch.TEXT:
        return _load_text_model()
    if arch == registry.Arch.VLM:
        return _load_vlm_model(model_source)
    raise ValueError(f"Unsupported architecture for model source '{model_source}'")


def _restore_trained_weights(model, checkpoint_path: Path):
    """Restore the model subtree from one exact Orbax checkpoint step."""
    save_dir = checkpoint_path.parent.resolve()
    lora_meta = export_lib.read_lora_metadata(save_dir)
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
        finally:
            cm.close()
        restored_model_state = restored["train_state"]["optimizer"]["model"]
        if jax.tree.structure(restored_model_state) != jax.tree.structure(model_state):
            raise ValueError("checkpoint model tree does not match the export model")
        for expected, restored_leaf in zip(
            jax.tree.leaves(model_state),
            jax.tree.leaves(restored_model_state),
        ):
            if expected.shape != restored_leaf.shape or expected.dtype != restored_leaf.dtype:
                raise ValueError("checkpoint model leaf does not match the export model")
        nnx.update(model, restored_model_state)
    print(f"Restored model params from step {step} at {save_dir}")
    return model


_DESCRIBES_BASE_WEIGHTS = ("quantization_config",)


def _save_identity_assets(out_dir: Path, model_source: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
    tokenizer.save_pretrained(out_dir)
    processor_config = model_source / "preprocessor_config.json"
    if not processor_config.is_file():
        raise ValueError(f"model source has no preprocessor_config.json: {model_source}")
    shutil.copyfile(processor_config, out_dir / processor_config.name)


def _write_servable_config(out_dir: Path, cfg, model_source: Path) -> None:
    _save_identity_assets(out_dir, model_source)
    base = json.loads((model_source / "config.json").read_text())
    for key in _DESCRIBES_BASE_WEIGHTS:
        base.pop(key, None)

    owned = export_lib.model_config_to_hf_dict(cfg)
    merged = {**base, **owned}
    for sub in ("text_config", "vision_config"):
        if isinstance(base.get(sub), dict) and isinstance(owned.get(sub), dict):
            merged[sub] = {**base[sub], **owned[sub]}

    (out_dir / "config.json").write_text(json.dumps(merged, indent=2) + "\n")
    added = sorted(set(merged) - set(owned))
    print(f"[export] config.json retained {len(added)} base fields")


def main(_) -> None:
    model_source = registry.resolve_hf_model_source(FLAGS.model_id, FLAGS.model_revision)
    jax.distributed.initialize()
    model, cfg = load_model(model_source)
    model = _restore_trained_weights(model, Path(FLAGS.checkpoint_path).expanduser())
    out_dir = Path(FLAGS.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=False)
    path = export_lib.export_model_to_hf(model, cfg, out_dir)
    _write_servable_config(out_dir, cfg, model_source)
    print(f"Exported safetensors to {path}")


if __name__ == "__main__":
    app.run(main)
