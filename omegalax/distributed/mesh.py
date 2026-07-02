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
        raise ValueError(
            f"Mesh axes must be > 0, got tp={tp_size}, fsdp={fsdp_size}, dp={dp_size}."
        )
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


def process_local_batch_size(global_batch_size: int, dp_size: int, fsdp_size: int) -> int:
    dp = dp_size * fsdp_size
    if global_batch_size <= 0:
        raise ValueError(f"Global batch size must be > 0, got {global_batch_size}.")
    if global_batch_size % dp != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by data_parallel_size={dp}."
        )
    return global_batch_size // dp


def make_mesh(tp_size: int, fsdp_size: int, dp_size: int) -> Mesh:
    tp, fsdp, dp = _resolve_mesh_shape(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    return jax.make_mesh((tp, fsdp, dp), _AXES)


def set_default_mesh(tp_size: int, fsdp_size: int, dp_size: int) -> Mesh:
    mesh = make_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    jax.set_mesh(mesh)
    return mesh


def ensure_mesh(
    tp_size: int | None = None, fsdp_size: int | None = None, dp_size: int | None = None
) -> Mesh:
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


# --- Expert parallelism (MoE grouped-GEMM + ragged all-to-all) ---------------
# Self-contained helper added for the `feat/moe-ep-grouped-gemm` branch. It builds
# a dedicated 1-D `('expert',)` mesh for the expert-parallel MoE path (used by
# omegalax.models.moe_grouped.grouped_moe_ep), independent of the flat
# `('tp','fsdp','dp')` training mesh above.
#
# MERGE NOTE: the separate `feat/topology-aware-mesh` branch is reworking mesh
# construction. At merge, the `'expert'` axis should be folded into the unified
# mesh factory / DEFAULT_AXIS_RULES (see shard_config.py, where the "experts"
# logical rule is currently mapped to None) rather than living as this standalone
# builder. Kept minimal here to avoid conflicting with that rework.
_EXPERT_AXIS = "expert"


def make_expert_mesh(ep_size: int, axis_name: str = _EXPERT_AXIS) -> Mesh:
    """Build a 1-D expert-parallel mesh over ``ep_size`` devices.

    The stacked expert weights are sharded on ``axis_name`` (each device owns
    ``E / ep_size`` experts) and the token axis is likewise sharded, so
    ``grouped_moe_ep`` can dispatch each token to the device owning its expert via
    ``ragged_all_to_all``. Requires ``jax.device_count() % ep_size == 0``; uses the
    first ``ep_size`` devices.
    """
    ndev = jax.device_count()
    if ep_size <= 0 or ndev % ep_size != 0:
        raise ValueError(
            f"expert-parallel size {ep_size} must divide device_count={ndev} and be > 0."
        )
    import numpy as np
    from jax.sharding import AxisType

    devices = np.array(jax.devices()[:ep_size]).reshape(ep_size)
    return Mesh(devices, (axis_name,), axis_types=(AxisType.Explicit,))


@contextmanager
def mesh_rules_for(tp_size: int, fsdp_size: int, dp_size: int) -> Iterator[Mesh]:
    """Resolve a mesh and activate mesh + logical axis rules for a scoped block."""
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    with mesh_rules(mesh):
        yield mesh
