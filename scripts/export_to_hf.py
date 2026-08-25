"""Export a supported Omegalax VLM to Hugging Face safetensors.

The entrypoint runs as one task in a one-node Slurm step, creating that step
itself when called directly from an sbatch script.

Two modes:
  * Default: export the weights from ``--model_snapshot``.
  * With ``--checkpoint_path``: load architecture from ``--model_snapshot``, then
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
import socket
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
from pathlib import Path

from omegalax.export_entry import resolve_export_step


def _ensure_export_step() -> None:
    launch = resolve_export_step(
        os.environ,
        sys.argv,
        sys.executable,
        str(Path(__file__).resolve()),
        socket.gethostname(),
    )
    if launch is None:
        return
    command, env = launch
    print(f"[export] launching Slurm step: {' '.join(command[:6])}", flush=True)
    try:
        os.execvpe(command[0], command, env)
    except FileNotFoundError as exc:
        raise ValueError("srun is unavailable inside this Slurm allocation") from exc


if __name__ == "__main__":
    try:
        _ensure_export_step()
    except ValueError as exc:
        raise SystemExit(f"export topology error: {exc}") from exc

import jax
import orbax.checkpoint as ocp
from absl import app, flags
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from omegalax import export as export_lib
from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.vlm import api as vlm_api
from omegalax.vlm.checkpoint_identity import require_checkpoint_path_snapshot
from omegalax.vlm.local_snapshot import LocalVLMSnapshot, open_local_vlm_snapshot

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_snapshot",
    None,
    "Absolute sealed local Hugging Face VLM snapshot.",
    required=True,
)
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


def load_model(model_snapshot: LocalVLMSnapshot):
    model, cfg = vlm_api.load_pretrained(
        model_snapshot,
        tp_size=FLAGS.tp_size,
        fsdp_size=FLAGS.fsdp_size,
        dp_size=FLAGS.dp_size,
    )
    return model, cfg


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


# A base config field that describes how the *weights* are encoded, which this
# export does not reproduce: Qwen3.5-35B-A3B-FP8 carries
# quantization_config={"quant_method": "fp8", ...}, and inheriting it would label a
# bf16 export as FP8.
_DESCRIBES_BASE_WEIGHTS = ("quantization_config",)


def _copy_base_identity_assets(out_dir: Path, model_snapshot: LocalVLMSnapshot) -> None:
    copied = []
    for name in model_snapshot.identity_assets:
        if name == "config.json":
            continue
        output_fd = os.open(
            out_dir / name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o644,
        )
        try:
            model_snapshot.copy_identity_asset_to(name, output_fd)
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
        copied.append(name)
    print(f"[export] copied {len(copied)} identity assets from the base: {copied}")


def _write_servable_config(
    out_dir: Path,
    cfg,
    model_snapshot: LocalVLMSnapshot,
) -> None:
    """Overlay owned fields without dropping serving fields absent from the runtime config."""
    with model_snapshot.files() as files:
        _copy_base_identity_assets(out_dir, model_snapshot)
        with open(files["config.json"], encoding="utf-8") as stream:
            base = json.load(stream)
    for key in _DESCRIBES_BASE_WEIGHTS:
        base.pop(key, None)

    owned = export_lib.model_config_to_hf_dict(cfg)
    merged = {**base, **owned}
    # One level deep: omegalax owns the shapes inside these, the base owns the rest.
    for sub in ("text_config", "vision_config"):
        if isinstance(base.get(sub), dict) and isinstance(owned.get(sub), dict):
            merged[sub] = {**base[sub], **owned[sub]}

    (out_dir / "config.json").write_text(json.dumps(merged, indent=2) + "\n")
    added = sorted(set(merged) - set(owned))
    print(
        f"[export] config.json overlaid on {model_snapshot.path!s}: +{len(added)} base keys {added}"
    )


def _run(model_snapshot: LocalVLMSnapshot) -> None:
    jax.distributed.initialize()
    model, cfg = load_model(model_snapshot)
    if not FLAGS.checkpoint_path:
        out_dir = Path(FLAGS.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Exported safetensors to {export_lib.export_model_to_hf(model, cfg, out_dir)}")
        _write_servable_config(out_dir, cfg, model_snapshot)
        return

    base_fingerprint = export_lib.param_fingerprint(model)
    model = _restore_trained_weights(model, cfg, Path(FLAGS.checkpoint_path).expanduser())
    out_dir = Path(FLAGS.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = export_lib.export_model_to_hf(model, cfg, out_dir)
    _write_servable_config(out_dir, cfg, model_snapshot)

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
            f"{path} is identical to the pretrained snapshot on all "
            f"{len(base_fingerprint)} parameter leaves. The restore matched nothing -- "
            f"orbax partial_restore drops leaves without raising -- so this export is the "
            f"base model. Do NOT use it. Check that "
            f"{Path(FLAGS.checkpoint_path).expanduser().parent}/lora_metadata.json "
            f"matches the training run."
        )
    print(f"Exported safetensors to {path}")
    print(
        f"[export] {len(changed)}/{len(base_fingerprint)} parameter-leaf checksums "
        "differ from the base"
    )


def main(_) -> None:
    with open_local_vlm_snapshot(FLAGS.model_snapshot) as model_snapshot:
        vlm_api.validate_pretrained(model_snapshot)
        if FLAGS.checkpoint_path:
            require_checkpoint_path_snapshot(
                FLAGS.checkpoint_path,
                model_snapshot.sha256,
            )
        _run(model_snapshot)


if __name__ == "__main__":
    app.run(main)
