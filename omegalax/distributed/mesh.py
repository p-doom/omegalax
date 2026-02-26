"""Mesh construction and global mesh setup."""

from __future__ import annotations

import jax
from jax.sharding import Mesh, PartitionSpec, get_abstract_mesh, get_mesh

_AXES = ("tp", "fsdp")


def _resolve_mesh_shape(tp_size: int, fsdp_size: int) -> tuple[int, int]:
    ndev = jax.device_count()
    if tp_size <= 0 or fsdp_size <= 0:
        raise ValueError(f"Mesh axes must be > 0, got tp={tp_size}, fsdp={fsdp_size}.")
    if tp_size * fsdp_size != ndev:
        raise ValueError(
            f"Mesh shape ({tp_size}, {fsdp_size}) does not match device_count={ndev}."
        )
    return tp_size, fsdp_size


def required_batch_multiple(batch_spec: PartitionSpec, mesh: Mesh) -> int:
    axis = batch_spec[0]
    if axis is None:
        return 1
    return int(mesh.shape[axis])


def make_mesh(tp_size: int, fsdp_size: int) -> Mesh:
    tp, fsdp = _resolve_mesh_shape(tp_size=tp_size, fsdp_size=fsdp_size)
    return jax.make_mesh((tp, fsdp), _AXES)


def set_default_mesh(tp_size: int, fsdp_size: int) -> Mesh:
    mesh = make_mesh(tp_size=tp_size, fsdp_size=fsdp_size)
    jax.set_mesh(mesh)
    return mesh


def ensure_mesh(tp_size: int | None = None, fsdp_size: int | None = None) -> Mesh:
    current_mesh = get_mesh()
    abstract_mesh = get_abstract_mesh()
    has_active_mesh = not abstract_mesh.empty
    has_active_tp_fsdp_mesh = has_active_mesh and tuple(abstract_mesh.axis_names) == _AXES

    if tp_size is None and fsdp_size is None:
        if has_active_tp_fsdp_mesh:
            return current_mesh
        if has_active_mesh:
            raise ValueError(
                f"Active mesh axes are {tuple(abstract_mesh.axis_names)}; expected {_AXES}. "
                "Refusing to override active mesh implicitly. "
                "Please provide tp_size and fsdp_size explicitly."
            )
        raise ValueError(
            "No active ('tp', 'fsdp') mesh found. Please provide both tp_size and fsdp_size explicitly."
        )

    if tp_size is None or fsdp_size is None:
        raise ValueError(
            "No active ('tp', 'fsdp') mesh found. Please provide both tp_size and fsdp_size explicitly."
        )

    if has_active_tp_fsdp_mesh:
        active_tp = int(abstract_mesh.shape["tp"])
        active_fsdp = int(abstract_mesh.shape["fsdp"])
        if tp_size != active_tp or fsdp_size != active_fsdp:
            raise ValueError(
                f"Requested mesh ({tp_size}, {fsdp_size}) conflicts with active mesh "
                f"({active_tp}, {active_fsdp}). Refusing to override active mesh."
            )
        return current_mesh

    if has_active_mesh:
        raise ValueError(
            f"Active mesh axes are {tuple(abstract_mesh.axis_names)}; expected {_AXES}. "
            "Refusing to override active mesh. Clear or replace the active mesh explicitly first."
        )

    return set_default_mesh(tp_size=tp_size, fsdp_size=fsdp_size)
