"""Params-only checkpoint loading for text inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flax import nnx
import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import orbax.checkpoint as ocp

from omegalax.distributed.mesh import ensure_mesh, mesh_rules


@dataclass(frozen=True)
class ResolvedCheckpoint:
    root: Path
    step: int
    step_path: Path
    config_path: Path


def _checkpoint_manager(root: Path) -> ocp.CheckpointManager:
    registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    registry.add("train_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    return ocp.CheckpointManager(
        root,
        options=ocp.CheckpointManagerOptions(
            step_format_fixed_length=6,
            create=False,
        ),
        handler_registry=registry,
    )


def resolve_checkpoint(path: str | Path) -> ResolvedCheckpoint:
    """Resolve a checkpoint root or numeric step directory."""
    requested = Path(path).expanduser().resolve()
    if (requested / "config.json").is_file():
        root = requested
        requested_step = None
    elif requested.name.isdigit():
        root = requested.parent
        requested_step = int(requested.name)
    else:
        root = requested
        requested_step = None

    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Checkpoint root has no config.json: {root}")

    manager = _checkpoint_manager(root)
    try:
        complete_steps = tuple(
            int(step)
            for step in manager.all_steps()
            if (root / f"{int(step):06d}" / "_CHECKPOINT_METADATA").is_file()
        )
    finally:
        manager.close()

    if requested_step is None:
        if not complete_steps:
            raise ValueError(f"Checkpoint root has no complete checkpoint steps: {root}")
        step = max(complete_steps)
    else:
        if requested_step not in complete_steps:
            raise ValueError(f"Path is not a complete checkpoint step: {requested}")
        step = requested_step

    step_path = root / f"{step:06d}"
    return ResolvedCheckpoint(
        root=root,
        step=step,
        step_path=step_path,
        config_path=config_path,
    )


def restore_model_params(model: nnx.Module, checkpoint: ResolvedCheckpoint) -> nnx.Module:
    """Restore only ``train_state/optimizer/model`` into an initialized model."""
    mesh = ensure_mesh()
    default_sharding = NamedSharding(mesh, P())

    with mesh_rules(mesh):
        model_abstract = jax.tree.map(
            lambda value: jax.ShapeDtypeStruct(
                value.shape,
                value.dtype,
                sharding=getattr(value, "sharding", None) or default_sharding,
            ),
            nnx.state(model),
        )
        params_abstract = {"optimizer": {"model": model_abstract}}

        def restore_arg(value):
            if isinstance(value, jax.ShapeDtypeStruct):
                return ocp.ArrayRestoreArgs(
                    sharding=value.sharding or default_sharding,
                    global_shape=value.shape,
                    dtype=value.dtype,
                )
            return value

        restore_args = jax.tree.map(
            restore_arg,
            params_abstract,
            is_leaf=lambda value: isinstance(value, jax.ShapeDtypeStruct),
        )
        manager = _checkpoint_manager(checkpoint.root)
        try:
            restored = manager.restore(
                checkpoint.step,
                args=ocp.args.Composite(
                    train_state=ocp.args.PyTreeRestore(
                        params_abstract,
                        restore_args=restore_args,
                        partial_restore=True,
                    )
                ),
            )
        finally:
            manager.close()

        nnx.update(model, restored["train_state"]["optimizer"]["model"])
    return model
