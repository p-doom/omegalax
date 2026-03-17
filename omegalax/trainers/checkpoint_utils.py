"""Shared checkpoint helpers for train-state and Grain iterators."""

from __future__ import annotations

from typing import Any

import grain.python as grain
import orbax.checkpoint as ocp

type GrainIterator = grain.DatasetIterator | grain.PyGrainDatasetIterator


def register_grain_iterator_handler(handler_registry: ocp.handlers.DefaultCheckpointHandlerRegistry) -> None:
    handler_registry.add("input_iter", grain.PyGrainCheckpointSave, grain.PyGrainCheckpointHandler)
    handler_registry.add("input_iter", grain.PyGrainCheckpointRestore, grain.PyGrainCheckpointHandler)


def make_grain_save_args(train_state: Any, input_iter: GrainIterator) -> ocp.args.Composite:
    items: dict[str, Any] = {
        "train_state": ocp.args.PyTreeSave(train_state),
        "input_iter": grain.PyGrainCheckpointSave(input_iter),
    }
    return ocp.args.Composite(**items)


def make_grain_restore_args(abstract_train_state: Any, input_iter: GrainIterator) -> ocp.args.Composite:
    items: dict[str, Any] = {
        "train_state": ocp.args.PyTreeRestore(abstract_train_state),
        "input_iter": grain.PyGrainCheckpointRestore(input_iter),
    }
    return ocp.args.Composite(**items)


def restored_input_iter(restored: Any) -> GrainIterator:
    return restored["input_iter"]
