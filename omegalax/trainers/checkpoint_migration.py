"""Offline checkpoint migrations for trainer state-schema changes."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp

_COMPOSITE_HANDLERS = {
    "input_iter": "grain._src.python.checkpoint.handler.CheckpointHandler",
    "train_state": (
        "orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler"
    ),
}
_MIGRATION_SCHEMA = "omegalax.multisteps_k1_to_direct.v1"


def _require_mapping(value: Any, where: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(value).__name__}.")
    return value


def _require_keys(value: Any, expected: set[Any], where: str) -> Mapping:
    mapping = _require_mapping(value, where)
    actual = set(mapping)
    if actual != expected:
        raise ValueError(f"{where} keys must be {expected}, got {actual}.")
    return mapping


def _scalar_int(value: Any, where: str) -> int:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"{where} must be a scalar integer, got shape={array.shape} dtype={array.dtype}."
        )
    return int(array)


def _named_leaves(value: Any, name: str) -> list[Any]:
    leaves: list[Any] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                if key == name:
                    leaves.append(child)
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)
    return leaves


def _adam_states(value: Any) -> list[Mapping]:
    states: list[Mapping] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            if set(node) == {"count", "mu", "nu"}:
                states.append(node)
            for child in node.values():
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)
    return states


def migrate_multisteps_k1_train_state(train_state: Any, checkpoint_step: int) -> dict[str, Any]:
    """Convert the exact legacy MultiSteps(k=1) state into direct optimizer state."""
    if checkpoint_step <= 0:
        raise ValueError(f"checkpoint_step must be positive, got {checkpoint_step}.")

    state = _require_keys(train_state, {"optimizer", "rng"}, "train_state")
    optimizer = _require_keys(
        state["optimizer"], {"model", "opt_state", "step"}, "train_state.optimizer"
    )
    legacy = _require_keys(
        optimizer["opt_state"],
        {"acc_grads", "gradient_step", "inner_opt_state", "mini_step"},
        "train_state.optimizer.opt_state",
    )

    optimizer_step = _scalar_int(optimizer["step"], "train_state.optimizer.step")
    gradient_step = _scalar_int(
        legacy["gradient_step"], "train_state.optimizer.opt_state.gradient_step"
    )
    mini_step = _scalar_int(legacy["mini_step"], "train_state.optimizer.opt_state.mini_step")
    if (optimizer_step, gradient_step, mini_step) != (checkpoint_step, checkpoint_step, 0):
        raise ValueError(
            "Checkpoint is not a completed MultiSteps(k=1) boundary: "
            f"checkpoint_step={checkpoint_step}, optimizer_step={optimizer_step}, "
            f"gradient_step={gradient_step}, mini_step={mini_step}."
        )

    acc_grads = legacy["acc_grads"]
    inner_state = legacy["inner_opt_state"]
    adam_states = _adam_states(inner_state)
    if len(adam_states) != 1:
        raise ValueError(
            f"inner_opt_state must contain exactly one Adam state, got {len(adam_states)}."
        )
    adam_state = adam_states[0]
    if not (
        jax.tree.structure(acc_grads)
        == jax.tree.structure(adam_state["mu"])
        == jax.tree.structure(adam_state["nu"])
    ):
        raise ValueError("acc_grads, Adam mu, and Adam nu must have the same tree structure.")
    for leaf in jax.tree.leaves(acc_grads):
        if np.count_nonzero(np.asarray(leaf)):
            raise ValueError("acc_grads must be zero at a completed MultiSteps(k=1) boundary.")

    counts = _named_leaves(inner_state, "count")
    if not counts:
        raise ValueError("inner_opt_state must contain at least one optimizer count.")
    count_values = [_scalar_int(value, "inner_opt_state count") for value in counts]
    if any(value != checkpoint_step for value in count_values):
        raise ValueError(
            f"Every inner optimizer count must equal checkpoint_step={checkpoint_step}, "
            f"got {count_values}."
        )

    return {
        "optimizer": {
            "model": optimizer["model"],
            "opt_state": inner_state,
            "step": optimizer["step"],
        },
        "rng": state["rng"],
    }


def _validate_composite_metadata(step_dir: Path) -> None:
    metadata_path = step_dir / "_CHECKPOINT_METADATA"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid composite checkpoint metadata at {metadata_path}.") from exc
    if metadata.get("item_handlers") != _COMPOSITE_HANDLERS:
        raise ValueError(
            f"Unsupported checkpoint item handlers: {metadata.get('item_handlers')!r}."
        )
    actual_items = {path.name for path in step_dir.iterdir()}
    expected_items = {"_CHECKPOINT_METADATA", "input_iter", "train_state"}
    if actual_items != expected_items:
        raise ValueError(f"Checkpoint step items must be {expected_items}, got {actual_items}.")


def migrate_multisteps_k1_checkpoint(
    source_root: str | Path,
    destination_root: str | Path,
    checkpoint_step: int,
) -> Path:
    """Write a new checkpoint root with one migrated, strictly restorable step."""
    source_root = Path(source_root).expanduser().resolve()
    destination_root = Path(destination_root).expanduser().resolve()
    source_step = source_root / f"{checkpoint_step:06d}"
    if not source_step.is_dir():
        raise ValueError(f"Source checkpoint step does not exist: {source_step}.")
    if destination_root.exists():
        raise FileExistsError(f"Destination checkpoint root already exists: {destination_root}.")
    config_path = source_root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Source checkpoint root is missing config.json: {source_root}.")
    _validate_composite_metadata(source_step)

    checkpointer = ocp.PyTreeCheckpointer()
    try:
        train_state_path = source_step / "train_state"
        item_metadata = checkpointer.metadata(train_state_path).item_metadata
        if item_metadata is None:
            raise ValueError(f"Checkpoint has no PyTree item metadata: {train_state_path}.")
        restore_args = ocp.checkpoint_utils.construct_restore_args(item_metadata)
        train_state = checkpointer.restore(train_state_path, restore_args=restore_args)
        migrated = migrate_multisteps_k1_train_state(train_state, checkpoint_step)

        destination_root.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix=f".{destination_root.name}.tmp-", dir=destination_root.parent)
        )
        try:
            destination_step = temporary_root / source_step.name
            destination_step.mkdir()
            checkpointer.save(destination_step / "train_state", migrated)
            shutil.copytree(source_step / "input_iter", destination_step / "input_iter")
            shutil.copy2(
                source_step / "_CHECKPOINT_METADATA",
                destination_step / "_CHECKPOINT_METADATA",
            )
            shutil.copy2(config_path, temporary_root / "config.json")
            lora_metadata = source_root / "lora_metadata.json"
            if lora_metadata.is_file():
                shutil.copy2(lora_metadata, temporary_root / "lora_metadata.json")
            (temporary_root / "checkpoint_migration.json").write_text(
                json.dumps(
                    {
                        "schema": _MIGRATION_SCHEMA,
                        "checkpoint_step": checkpoint_step,
                    },
                    indent=2,
                )
                + "\n"
            )
            temporary_root.rename(destination_root)
        except BaseException:
            shutil.rmtree(temporary_root, ignore_errors=True)
            raise
    finally:
        checkpointer.close()
    return destination_root / source_step.name
