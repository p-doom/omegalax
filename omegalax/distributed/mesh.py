"""Mesh construction and global mesh setup.

Topology-aware ``('tp', 'cp', 'fsdp', 'dp')`` mesh (comm-heaviest axis first).
The comm-heavy axes (``tp``, ``cp``, ``fsdp`` up to a node) ride the intra-node
NVLink domain (ICI); ``dp`` and any ``fsdp`` spill ride the data-center network
(DCN). Without this a flat multi-node mesh scatters the per-layer TP collective
across nodes onto InfiniBand. Assumes ONE process per node, so
``local_device_count`` == GPUs/node == ICI size and ``process_count`` == nodes
== DCN granules; :func:`make_hierarchical_mesh` warns otherwise.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import jax
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import AxisType, Mesh, PartitionSpec, get_abstract_mesh, get_mesh

from omegalax.models.shard_config import axis_rules_for_mesh

logger = logging.getLogger(__name__)

# Comm-heaviest-first. tp and cp ride the intra-node NVLink domain (both are
# comm-heavy per-layer collectives); cp_size==1 is a strict no-op (size-1 axis
# dropped downstream by shard_config._filter_axis).
_AXES = ("tp", "cp", "fsdp", "dp")

# Per-axis-type parallelism quadruple, comm-heaviest-first: (tp, cp, fsdp, dp).
Quad = tuple[int, int, int, int]


@dataclasses.dataclass(frozen=True, slots=True)
class ParallelismConfig:
    """Explicit ICI/DCN parallelism degrees per axis type (comm-heaviest-first).

    ``ici_*`` ride the NVLink domain (one node), ``dcn_*`` the data-center network
    (across nodes). Derived from the legacy (tp, cp, fsdp, dp) sizes by
    :func:`derive_ici_dcn`; per-type products (``tp == ici_tp * dcn_tp`` etc.)
    equal those sizes.
    """

    ici_tp: int
    ici_cp: int
    ici_fsdp: int
    ici_dp: int
    dcn_tp: int
    dcn_cp: int
    dcn_fsdp: int
    dcn_dp: int

    @property
    def ici_shape(self) -> Quad:
        return (self.ici_tp, self.ici_cp, self.ici_fsdp, self.ici_dp)

    @property
    def dcn_shape(self) -> Quad:
        return (self.dcn_tp, self.dcn_cp, self.dcn_fsdp, self.dcn_dp)

    @property
    def tp_size(self) -> int:
        return self.ici_tp * self.dcn_tp

    @property
    def cp_size(self) -> int:
        return self.ici_cp * self.dcn_cp

    @property
    def fsdp_size(self) -> int:
        return self.ici_fsdp * self.dcn_fsdp

    @property
    def dp_size(self) -> int:
        return self.ici_dp * self.dcn_dp


def local_device_count() -> int:
    """GPUs per node == NVLink/ICI domain size (assuming one process per node)."""
    return jax.local_device_count()


def num_processes() -> int:
    """Number of processes == nodes == DCN granules (assuming one process per node)."""
    return jax.process_count()


def _resolve_mesh_shape(
    tp_size: int, cp_size: int, fsdp_size: int, dp_size: int
) -> tuple[int, int, int, int]:
    ndev = jax.device_count()
    if tp_size <= 0 or cp_size <= 0 or fsdp_size <= 0 or dp_size <= 0:
        raise ValueError(
            f"Mesh axes must be > 0, got tp={tp_size}, cp={cp_size}, fsdp={fsdp_size}, dp={dp_size}."
        )
    if tp_size * cp_size * fsdp_size * dp_size != ndev:
        raise ValueError(
            f"Mesh shape (tp={tp_size}, cp={cp_size}, fsdp={fsdp_size}, dp={dp_size}) "
            f"does not match device_count={ndev}."
        )
    return tp_size, cp_size, fsdp_size, dp_size


def derive_ici_dcn(
    tp_size: int,
    fsdp_size: int,
    dp_size: int,
    local_device_count: int,
    num_processes: int,
    cp_size: int = 1,
) -> ParallelismConfig:
    """Map legacy (tp, cp, fsdp, dp) sizes onto a hierarchical ICI/DCN split.

    MaxText-style capacity-filling: fill the ICI domain (one node) comm-heaviest
    first -- TP, then CP, then FSDP, then DP -- spilling the remainder to the DCN
    (``num_processes`` granules). TP and CP are ICI-only and MUST jointly fit and
    tile one node (the correctness guardrail asserted below); on a single node
    (``num_processes == 1``) everything stays ICI.

    Pure (no JAX calls) so it is CPU-unit-testable; ``cp_size`` is keyword-optional
    (default 1, a strict no-op) to preserve legacy call sites.
    """
    if tp_size <= 0 or cp_size <= 0 or fsdp_size <= 0 or dp_size <= 0:
        raise ValueError(
            f"Parallelism sizes must be > 0, got tp={tp_size}, cp={cp_size}, "
            f"fsdp={fsdp_size}, dp={dp_size}."
        )
    if local_device_count <= 0 or num_processes <= 0:
        raise ValueError(
            f"local_device_count and num_processes must be > 0, got "
            f"local_device_count={local_device_count}, num_processes={num_processes}."
        )

    # TP*CP -> ICI only, and MUST fit + tile one NVLink domain (both are
    # comm-heavy per-layer collectives that must stay off the DCN).
    tpcp = tp_size * cp_size
    if tpcp > local_device_count:
        raise ValueError(
            f"tp_size*cp_size={tpcp} (tp={tp_size}, cp={cp_size}) exceeds "
            f"local_device_count={local_device_count}: tensor- and context-parallel groups "
            "must jointly fit within a single NVLink domain (node). "
            "Reduce tp_size/cp_size so that TP*CP <= GPUs-per-node."
        )
    if local_device_count % tpcp != 0:
        raise ValueError(
            f"local_device_count={local_device_count} is not divisible by tp_size*cp_size={tpcp} "
            f"(tp={tp_size}, cp={cp_size}); TP*CP must tile the NVLink domain evenly."
        )
    ici_tp, dcn_tp = tp_size, 1
    ici_cp, dcn_cp = cp_size, 1

    # FSDP fills the rest of the node, then spills to DCN.
    ici_slots = local_device_count // (ici_tp * ici_cp)
    ici_fsdp = min(fsdp_size, ici_slots)
    if fsdp_size % ici_fsdp != 0:
        raise ValueError(
            f"fsdp_size={fsdp_size} is not divisible by its intra-node share "
            f"ici_fsdp={ici_fsdp} (local_device_count={local_device_count}, tp_size={tp_size}, "
            f"cp_size={cp_size}); FSDP must split evenly between the NVLink domain and the "
            "data-center network."
        )
    dcn_fsdp = fsdp_size // ici_fsdp

    # DP fills any leftover node room, then spills to DCN.
    ici_slots_left = ici_slots // ici_fsdp  # == ldc // (ici_tp*ici_cp*ici_fsdp)
    ici_dp = min(dp_size, ici_slots_left)
    if dp_size % ici_dp != 0:
        raise ValueError(
            f"dp_size={dp_size} is not divisible by its intra-node share "
            f"ici_dp={ici_dp} (local_device_count={local_device_count}, tp_size={tp_size}, "
            f"cp_size={cp_size}, fsdp_size={fsdp_size}); DP must split evenly between the "
            "NVLink domain and the data-center network."
        )
    dcn_dp = dp_size // ici_dp

    cfg = ParallelismConfig(
        ici_tp=ici_tp,
        ici_cp=ici_cp,
        ici_fsdp=ici_fsdp,
        ici_dp=ici_dp,
        dcn_tp=dcn_tp,
        dcn_cp=dcn_cp,
        dcn_fsdp=dcn_fsdp,
        dcn_dp=dcn_dp,
    )

    ici_prod = ici_tp * ici_cp * ici_fsdp * ici_dp
    dcn_prod = dcn_tp * dcn_cp * dcn_fsdp * dcn_dp
    if ici_prod != local_device_count:
        raise ValueError(
            f"ICI shape {cfg.ici_shape} product {ici_prod} != local_device_count="
            f"{local_device_count}. Requested (tp={tp_size}, cp={cp_size}, fsdp={fsdp_size}, "
            f"dp={dp_size}) cannot be laid out with one process per node; the intra-node axes "
            "(tp * cp * fsdp-share) do not fill the NVLink domain exactly."
        )
    if dcn_prod != num_processes:
        raise ValueError(
            f"DCN shape {cfg.dcn_shape} product {dcn_prod} != num_processes="
            f"{num_processes}. Requested (tp={tp_size}, cp={cp_size}, fsdp={fsdp_size}, "
            f"dp={dp_size}) cannot be laid out across the available nodes; the inter-node axes "
            "(fsdp-spill * dp) do not tile the node count exactly."
        )
    assert cfg.tp_size == tp_size
    assert cfg.cp_size == cp_size
    assert cfg.fsdp_size == fsdp_size
    assert cfg.dp_size == dp_size
    return cfg


def make_hierarchical_mesh(ici_shape: Sequence[int], dcn_shape: Sequence[int]) -> Mesh:
    """Build a topology-aware ``('tp', 'cp', 'fsdp', 'dp')`` mesh from ICI/DCN
    ``(tp, cp, fsdp, dp)`` quadruples (ICI rides NVLink, DCN the data-center net).

    Single process: plain row-major reshape (the whole node is one NVLink domain).
    Multi process: ``create_hybrid_device_mesh`` with ``process_is_granule=True``;
    the returned ndarray is wrapped in :class:`Mesh` directly (routing through
    ``jax.make_mesh`` would re-reshape and destroy the hybrid layout).
    """
    ici_shape = tuple(int(x) for x in ici_shape)
    dcn_shape = tuple(int(x) for x in dcn_shape)
    if len(ici_shape) != len(_AXES) or len(dcn_shape) != len(_AXES):
        raise ValueError(
            f"ici_shape and dcn_shape must be (tp, cp, fsdp, dp) quadruples, got "
            f"ici_shape={ici_shape}, dcn_shape={dcn_shape}."
        )

    devices = jax.devices()
    ldc = local_device_count()
    nproc = num_processes()

    ici_prod = math.prod(ici_shape)
    dcn_prod = math.prod(dcn_shape)

    # Per-axis validation lives in derive_ici_dcn (the sole caller); here just
    # guard that the shapes tile all devices.
    if ici_prod * dcn_prod != len(devices):
        raise ValueError(
            f"prod(ici_shape)*prod(dcn_shape)={ici_prod * dcn_prod} != device_count="
            f"{len(devices)} (ici_shape={ici_shape}, dcn_shape={dcn_shape})."
        )

    if ldc == 1 and nproc > 1:
        logger.warning(
            "local_device_count == 1 with num_processes=%d: the NVLink/ICI domain has "
            "collapsed. omegalax must launch ONE process per node (not one per GPU) for "
            "process_is_granule=True to mean one granule per node and for ICI axes to ride "
            "NVLink. TP/intra-node axes will be forced onto the data-center network.",
            nproc,
        )

    # Explicit axis types (matching jax.make_mesh): downstream out_sharding= code
    # relies on them; a bare Mesh(...) defaults to AxisType.Auto and regresses.
    axis_types = (AxisType.Explicit,) * len(_AXES)

    if nproc == 1:
        device_grid = mesh_utils.create_device_mesh(ici_shape, devices)
        return Mesh(device_grid, _AXES, axis_types=axis_types)

    # process_is_granule=True is mandatory: under init_distributed()'s multi-node
    # jax.distributed.initialize() every device reports a uniform slice_index, so
    # granules must be grouped by process_index (node) to keep ICI axes on NVLink.
    device_grid = mesh_utils.create_hybrid_device_mesh(
        ici_shape,
        dcn_shape,
        devices,
        process_is_granule=True,
        allow_split_physical_axes=False,
    )
    return Mesh(device_grid, _AXES, axis_types=axis_types)


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


def make_mesh(
    tp_size: int,
    fsdp_size: int,
    dp_size: int,
    *,
    cp_size: int = 1,
) -> Mesh:
    """Build the ``('tp', 'cp', 'fsdp', 'dp')`` mesh for the given sizes (``cp_size``
    default 1 == no-op), deriving the ICI/DCN split via :func:`derive_ici_dcn`."""
    tp, cp, fsdp, dp = _resolve_mesh_shape(
        tp_size=tp_size, cp_size=cp_size, fsdp_size=fsdp_size, dp_size=dp_size
    )
    parallelism = derive_ici_dcn(
        tp_size=tp,
        cp_size=cp,
        fsdp_size=fsdp,
        dp_size=dp,
        local_device_count=local_device_count(),
        num_processes=num_processes(),
    )
    return make_hierarchical_mesh(parallelism.ici_shape, parallelism.dcn_shape)


def set_default_mesh(
    tp_size: int,
    fsdp_size: int,
    dp_size: int,
    *,
    cp_size: int = 1,
) -> Mesh:
    mesh = make_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size, cp_size=cp_size)
    jax.set_mesh(mesh)
    return mesh


def ensure_mesh(
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    dp_size: int | None = None,
    *,
    cp_size: int = 1,
) -> Mesh:
    current_mesh = get_mesh()
    abstract_mesh = get_abstract_mesh()
    has_active_mesh = not abstract_mesh.empty
    has_active_cp_mesh = has_active_mesh and tuple(abstract_mesh.axis_names) == _AXES

    if tp_size is None and fsdp_size is None and dp_size is None:
        if has_active_cp_mesh:
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

    if has_active_cp_mesh:
        active_tp = int(abstract_mesh.shape["tp"])
        active_cp = int(abstract_mesh.shape["cp"])
        active_fsdp = int(abstract_mesh.shape["fsdp"])
        active_dp = int(abstract_mesh.shape["dp"])
        if (
            tp_size != active_tp
            or cp_size != active_cp
            or fsdp_size != active_fsdp
            or dp_size != active_dp
        ):
            raise ValueError(
                f"Requested mesh (tp={tp_size}, cp={cp_size}, fsdp={fsdp_size}, dp={dp_size}) "
                f"conflicts with active mesh (tp={active_tp}, cp={active_cp}, "
                f"fsdp={active_fsdp}, dp={active_dp}). Refusing to override active mesh."
            )
        return current_mesh

    if has_active_mesh:
        raise ValueError(
            f"Active mesh axes are {tuple(abstract_mesh.axis_names)}; expected {_AXES}. "
            "Refusing to override active mesh."
        )

    return set_default_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size, cp_size=cp_size)


@contextmanager
def mesh_rules(mesh: Mesh) -> Iterator[Mesh]:
    """Activate mesh + logical axis rules for a scoped block."""
    with jax.set_mesh(mesh), nnx.logical_axis_rules(axis_rules_for_mesh(mesh)):
        yield mesh


# Dedicated 1-D ``('expert',)`` mesh for grouped_moe_ep, SEPARATE from the
# ``('tp','cp','fsdp','dp')`` training mesh: no caller composes EP with training.
_EXPERT_AXIS = "expert"


def make_expert_mesh(ep_size: int, axis_name: str = _EXPERT_AXIS) -> Mesh:
    """Build a 1-D expert-parallel mesh over the first ``ep_size`` devices.

    ``AxisType.Explicit`` (matching the ``_AXES`` mesh) so Explicit-sharding code
    behaves the same here. Requires ``jax.device_count() % ep_size == 0``.
    """
    ndev = jax.device_count()
    if ep_size <= 0 or ndev % ep_size != 0:
        raise ValueError(
            f"expert-parallel size {ep_size} must divide device_count={ndev} and be > 0."
        )
    device_grid = mesh_utils.create_device_mesh((ep_size,), jax.devices()[:ep_size])
    return Mesh(device_grid, (axis_name,), axis_types=(AxisType.Explicit,))


@contextmanager
def mesh_rules_for(
    tp_size: int, fsdp_size: int, dp_size: int, *, cp_size: int = 1
) -> Iterator[Mesh]:
    """Resolve a mesh and activate mesh + logical axis rules for a scoped block."""
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size, cp_size=cp_size)
    with mesh_rules(mesh):
        yield mesh
