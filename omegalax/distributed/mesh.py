"""Mesh construction and global mesh setup."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import jax
from flax import nnx
from jax.sharding import Mesh, PartitionSpec, get_abstract_mesh, get_mesh

from omegalax.models.shard_config import axis_rules_for_mesh

_AXES = ("tp", "fsdp", "dp")


def _resolve_mesh_shape(tp_size: int, fsdp_size: int, dp_size: int) -> tuple[int, int, int]:
    ndev = jax.device_count()
    if tp_size <= 0 or fsdp_size <= 0 or dp_size <= 0:
        raise ValueError(f"Mesh axes must be > 0, got tp={tp_size}, fsdp={fsdp_size}, dp={dp_size}.")
    if tp_size * fsdp_size * dp_size != ndev:
        raise ValueError(
            f"Mesh shape ({tp_size}, {fsdp_size}, {dp_size}) does not match device_count={ndev}."
        )
    return tp_size, fsdp_size, dp_size


def required_batch_multiple(batch_spec: PartitionSpec, mesh: Mesh) -> int:
    axis = batch_spec[0]
    if axis is None:
        return 1
    return int(mesh.shape[axis])


def data_parallel_size(dp_size: int | None = None, fsdp_size: int | None = None) -> int:
    """Return the effective number of batch shards across processes.

    The batch logical axis is sharded over both the ``dp`` and ``fsdp`` mesh
    axes (see ``DEFAULT_AXIS_RULES``), so the effective batch-shard count is
    ``dp_size * fsdp_size``. When neither is provided, fall back to
    ``jax.process_count()`` (the typical multi-host case with one device per
    process).
    """
    if dp_size is None and fsdp_size is None:
        return jax.process_count()
    return (dp_size or 1) * (fsdp_size or 1)


def data_parallel_index(dp_size: int | None = None, fsdp_size: int | None = None) -> int:
    """Return this process's index along the batch-shard axis."""
    dp = data_parallel_size(dp_size, fsdp_size)
    return jax.process_index() % dp


def process_local_batch_size(
    global_batch_size: int,
    dp_size: int | None = None,
    fsdp_size: int | None = None,
) -> int:
    dp = data_parallel_size(dp_size, fsdp_size)
    if global_batch_size <= 0:
        raise ValueError(f"Global batch size must be > 0, got {global_batch_size}.")
    if global_batch_size % dp != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"data_parallel_size={dp}."
        )
    return global_batch_size // dp


def make_mesh(tp_size: int, fsdp_size: int, dp_size: int) -> Mesh:
    tp, fsdp, dp = _resolve_mesh_shape(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    return jax.make_mesh((tp, fsdp, dp), _AXES)


def set_default_mesh(tp_size: int, fsdp_size: int, dp_size: int) -> Mesh:
    mesh = make_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    jax.set_mesh(mesh)
    return mesh


def ensure_mesh(tp_size: int | None = None, fsdp_size: int | None = None, dp_size: int | None = None) -> Mesh:
    current_mesh = get_mesh()
    abstract_mesh = get_abstract_mesh()
    has_active_mesh = not abstract_mesh.empty
    has_active_3axis_mesh = has_active_mesh and tuple(abstract_mesh.axis_names) == _AXES

    if tp_size is None and fsdp_size is None and dp_size is None:
        if has_active_3axis_mesh:
            return current_mesh
        if has_active_mesh:
            raise ValueError(
                f"Active mesh axes are {tuple(abstract_mesh.axis_names)}; expected {_AXES}. "
                "Refusing to override active mesh implicitly."
            )
        raise ValueError(
            f"No active {_AXES} mesh found. Please provide tp_size, fsdp_size, and dp_size explicitly."
        )

    if tp_size is None or fsdp_size is None or dp_size is None:
        raise ValueError(
            f"No active {_AXES} mesh found. Please provide tp_size, fsdp_size, and dp_size explicitly."
        )

    if has_active_3axis_mesh:
        active_tp = int(abstract_mesh.shape["tp"])
        active_fsdp = int(abstract_mesh.shape["fsdp"])
        active_dp = int(abstract_mesh.shape["dp"])
        if tp_size != active_tp or fsdp_size != active_fsdp or dp_size != active_dp:
            raise ValueError(
                f"Requested mesh ({tp_size}, {fsdp_size}, {dp_size}) conflicts with active mesh "
                f"({active_tp}, {active_fsdp}, {active_dp}). Refusing to override active mesh."
            )
        return current_mesh

    if has_active_mesh:
        raise ValueError(
            f"Active mesh axes are {tuple(abstract_mesh.axis_names)}; expected {_AXES}. "
            "Refusing to override active mesh."
        )

    return set_default_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)


@contextmanager
def mesh_rules(mesh: Mesh) -> Iterator[Mesh]:
    """Activate mesh + logical axis rules for a scoped block."""
    with jax.set_mesh(mesh), nnx.logical_axis_rules(axis_rules_for_mesh(mesh)):
        yield mesh


@contextmanager
def mesh_rules_for(tp_size: int, fsdp_size: int, dp_size: int) -> Iterator[Mesh]:
    """Resolve a mesh and activate mesh + logical axis rules for a scoped block."""
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    with mesh_rules(mesh):
        yield mesh
