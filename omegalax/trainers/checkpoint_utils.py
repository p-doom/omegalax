"""Shared checkpoint helpers for train-state and Grain iterators."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeAlias, cast

import grain
import orbax.checkpoint as ocp

# Plain `TypeAlias` (not the PEP 695 `type X = ...` statement) so the module
# imports on Python 3.11, matching `requires-python = ">=3.11"`.
GrainIterator: TypeAlias = grain.DataLoaderIterator | grain.DatasetIterator


class ResumeMode(StrEnum):
    """Trainer resume policy.

    NEVER      — always start fresh; do not consult any existing checkpoints.
    IF_PRESENT — resume from the latest checkpoint at ``save_dir`` if one exists,
                 otherwise start fresh. Right mode for SLURM time-limit recovery,
                 where the same recipe may be submitted with no checkpoint yet
                 (first run) or with checkpoints from a previous timed-out attempt.
    REQUIRED   — resume; error if no usable checkpoint is found. Right mode for
                 explicit "this must be a continuation" workflows.
    """

    NEVER = "never"
    IF_PRESENT = "if_present"
    REQUIRED = "required"


def register_grain_iterator_handler(
    handler_registry: ocp.handlers.DefaultCheckpointHandlerRegistry,
) -> None:
    handler_registry.add(
        "input_iter",
        grain.checkpoint.CheckpointSave,
        cast(ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler),
    )
    handler_registry.add(
        "input_iter",
        grain.checkpoint.CheckpointRestore,
        cast(ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler),
    )


def make_grain_save_args(train_state: Any, input_iter: GrainIterator) -> ocp.args.Composite:
    items: dict[str, Any] = {
        "train_state": ocp.args.PyTreeSave(train_state),
        "input_iter": grain.checkpoint.CheckpointSave(input_iter),
    }
    return ocp.args.Composite(**items)


def make_grain_restore_args(
    abstract_train_state: Any, input_iter: GrainIterator
) -> ocp.args.Composite:
    items: dict[str, Any] = {
        "train_state": ocp.args.PyTreeRestore(abstract_train_state),
        "input_iter": grain.checkpoint.CheckpointRestore(input_iter),
    }
    return ocp.args.Composite(**items)


def restored_input_iter(restored: Any) -> GrainIterator:
    return restored["input_iter"]
