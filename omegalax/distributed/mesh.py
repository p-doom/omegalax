"""Mesh construction and global mesh setup.

Topology-aware device placement
================================
The device mesh keeps the logical axis names ``('tp', 'fsdp', 'dp')`` (ordered
comm-heaviest-first) so that :mod:`omegalax.models.shard_config`,
:mod:`omegalax.distributed.sharding_runtime`, :func:`ensure_mesh` validation and
Orbax ``NamedSharding`` are all unaffected. Only the *physical device
placement* underneath those axis names changes.

On GPU, ``jax.make_mesh`` -> ``mesh_utils.create_device_mesh`` takes a plain
row-major reshape (topology-aware reordering is TPU-only). Because
``jax.devices()`` is process-major (each node's GPUs are contiguous) and ``tp``
is the *major* (most-strided) axis, a flat ``(tp, fsdp, dp)`` mesh scatters the
TP communication group across nodes (one GPU per node) for any multi-node run
with ``tp_size > 1``. That forces the highest-frequency collective (per-layer TP
all-reduce/all-gather) onto InfiniBand instead of NVLink.

The fix is a *hierarchical* (hybrid ICI/DCN) mesh:

* **ICI** (Inter-Chip Interconnect = NVLink domain = one node) carries the
  comm-heavy ``tp`` axis (and ``fsdp`` up to the size of the node).
* **DCN** (Data-Center Network = InfiniBand) carries the comm-light ``dp`` axis
  (and any ``fsdp`` that spills past a single node).

Assumptions (asserted / documented below):

* **One process per node.** ``jax.local_device_count()`` is then the number of
  GPUs per node == the NVLink/ICI domain size, and ``jax.process_count()`` is
  the number of nodes == the number of DCN granules. If omegalax is instead
  launched one-process-per-GPU, ``local_device_count == 1`` and the ICI domain
  collapses; :func:`make_hierarchical_mesh` warns in that case.
* ``process_is_granule=True`` is **mandatory** on this GPU cluster: we launch
  via :func:`omegalax.distributed.launch.init_distributed`, which for the
  multi-node path calls ``jax.distributed.initialize()`` without a partition
  index, so every device reports a uniform ``slice_index`` and the slice-based
  granule detection would collapse to a single granule.
  ``process_is_granule=True`` groups DCN granules by ``process_index`` (== node),
  keeping ICI axes on NVLink.
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

_AXES = ("tp", "fsdp", "dp")

# A per-axis-type parallelism triple, always in comm-heaviest-first order
# matching ``_AXES``: (tp, fsdp, dp).
Triple = tuple[int, int, int]


@dataclasses.dataclass(frozen=True, slots=True)
class ParallelismConfig:
    """Explicit ICI/DCN parallelism degrees, comm-heaviest-first per axis type.

    The six degrees fully specify the hierarchical mesh. ``ici_*`` degrees ride
    the NVLink domain (one node); ``dcn_*`` degrees ride the data-center network
    (across nodes). The legacy three-arg interface (``tp_size``, ``fsdp_size``,
    ``dp_size``) is derived into this via :func:`derive_ici_dcn`; callers may
    also pass an explicit :class:`ParallelismConfig` to override the split.

    Invariants (checked by :func:`make_hierarchical_mesh`):
      * ``ici_tp * ici_fsdp * ici_dp == local_device_count``
      * ``dcn_tp * dcn_fsdp * dcn_dp == num_processes``
      * per-type products (``tp = ici_tp * dcn_tp`` etc.) equal the legacy sizes,
        preserving ``tp * fsdp * dp == device_count`` (see :func:`_resolve_mesh_shape`)
        and the data-pipeline invariant ``dp = dp_size * fsdp_size``.
    """

    ici_tp: int
    ici_fsdp: int
    ici_dp: int
    dcn_tp: int
    dcn_fsdp: int
    dcn_dp: int

    @property
    def ici_shape(self) -> Triple:
        return (self.ici_tp, self.ici_fsdp, self.ici_dp)

    @property
    def dcn_shape(self) -> Triple:
        return (self.dcn_tp, self.dcn_fsdp, self.dcn_dp)

    @property
    def tp_size(self) -> int:
        return self.ici_tp * self.dcn_tp

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


def derive_ici_dcn(
    tp_size: int,
    fsdp_size: int,
    dp_size: int,
    local_device_count: int,
    num_processes: int,
) -> ParallelismConfig:
    """Map legacy (tp, fsdp, dp) sizes onto a hierarchical ICI/DCN split.

    MaxText-style capacity-filling placement. The ICI domain (one node, size
    ``local_device_count``) is filled in comm-heaviest-first priority order --
    TP, then FSDP, then DP -- and whatever does not fit spills to the DCN
    (``num_processes`` granules). Concretely:

    * **TP -> ICI only.** ``ici_tp = tp_size``, ``dcn_tp = 1``. TP is the
      comm-heaviest collective and MUST fit inside the NVLink domain, so we
      require ``tp_size <= local_device_count`` and
      ``local_device_count % tp_size == 0`` (the correctness guardrail).
    * **FSDP fills the rest of the node first, then spills to DCN.**
      ``ici_fsdp = min(fsdp_size, local_device_count // ici_tp)``;
      ``dcn_fsdp = fsdp_size // ici_fsdp`` (must divide evenly).
    * **DP fills any node room left after TP+FSDP, then spills to DCN.** In the
      multi-node target case a full node is already consumed by TP (+FSDP), so
      DP lands entirely in the DCN (``dcn_dp = dp_size``, ``ici_dp = 1``) -- the
      intended layout that keeps the DP all-reduce off NVLink and the TP
      collective on it. On a *single node* (``num_processes == 1``) there is no
      DCN, so DP (and FSDP) must fit inside the ICI; this branch handles that so
      single-node runs keep working unchanged.

    Pure function (no JAX calls); ``local_device_count`` and ``num_processes``
    are passed in so it is unit-testable on CPU.
    """
    if tp_size <= 0 or fsdp_size <= 0 or dp_size <= 0:
        raise ValueError(
            f"Parallelism sizes must be > 0, got tp={tp_size}, fsdp={fsdp_size}, dp={dp_size}."
        )
    if local_device_count <= 0 or num_processes <= 0:
        raise ValueError(
            f"local_device_count and num_processes must be > 0, got "
            f"local_device_count={local_device_count}, num_processes={num_processes}."
        )

    # TP -> ICI only. Guardrail: TP must fit within the NVLink domain.
    if tp_size > local_device_count:
        raise ValueError(
            f"tp_size={tp_size} exceeds local_device_count={local_device_count}: "
            "tensor-parallel groups must fit within a single NVLink domain (node). "
            "Reduce tp_size or use fewer/larger axes so that TP <= GPUs-per-node."
        )
    if local_device_count % tp_size != 0:
        raise ValueError(
            f"local_device_count={local_device_count} is not divisible by tp_size={tp_size}; "
            "TP must tile the NVLink domain evenly."
        )
    ici_tp, dcn_tp = tp_size, 1

    # FSDP fills the remaining GPUs of the node first, then spills to DCN.
    ici_slots = local_device_count // ici_tp
    ici_fsdp = min(fsdp_size, ici_slots)
    if fsdp_size % ici_fsdp != 0:
        raise ValueError(
            f"fsdp_size={fsdp_size} is not divisible by its intra-node share "
            f"ici_fsdp={ici_fsdp} (local_device_count={local_device_count}, tp_size={tp_size}); "
            "FSDP must split evenly between the NVLink domain and the data-center network."
        )
    dcn_fsdp = fsdp_size // ici_fsdp

    # DP fills any leftover node room (only nonzero once FSDP did NOT spill, i.e.
    # dcn_fsdp == 1), then spills to DCN. In the multi-node target case TP(+FSDP)
    # already fill the node so ici_dp == 1 and DP is DCN-only.
    ici_slots_left = ici_slots // ici_fsdp  # == local_device_count // (ici_tp*ici_fsdp)
    ici_dp = min(dp_size, ici_slots_left)
    if dp_size % ici_dp != 0:
        raise ValueError(
            f"dp_size={dp_size} is not divisible by its intra-node share "
            f"ici_dp={ici_dp} (local_device_count={local_device_count}, tp_size={tp_size}, "
            f"fsdp_size={fsdp_size}); DP must split evenly between the NVLink domain and the "
            "data-center network."
        )
    dcn_dp = dp_size // ici_dp

    cfg = ParallelismConfig(
        ici_tp=ici_tp,
        ici_fsdp=ici_fsdp,
        ici_dp=ici_dp,
        dcn_tp=dcn_tp,
        dcn_fsdp=dcn_fsdp,
        dcn_dp=dcn_dp,
    )

    # ICI axes must exactly tile one node; DCN axes must exactly tile the nodes.
    ici_prod = ici_tp * ici_fsdp * ici_dp
    dcn_prod = dcn_tp * dcn_fsdp * dcn_dp
    if ici_prod != local_device_count:
        raise ValueError(
            f"ICI shape {cfg.ici_shape} product {ici_prod} != local_device_count="
            f"{local_device_count}. Requested (tp={tp_size}, fsdp={fsdp_size}, dp={dp_size}) "
            "cannot be laid out with one process per node; the intra-node axes "
            "(tp * fsdp-share) do not fill the NVLink domain exactly."
        )
    if dcn_prod != num_processes:
        raise ValueError(
            f"DCN shape {cfg.dcn_shape} product {dcn_prod} != num_processes="
            f"{num_processes}. Requested (tp={tp_size}, fsdp={fsdp_size}, dp={dp_size}) "
            "cannot be laid out across the available nodes; the inter-node axes "
            "(fsdp-spill * dp) do not tile the node count exactly."
        )
    # Per-type products preserved -> tp*fsdp*dp and dp=dp_size*fsdp_size unchanged.
    assert cfg.tp_size == tp_size
    assert cfg.fsdp_size == fsdp_size
    assert cfg.dp_size == dp_size
    return cfg


def make_hierarchical_mesh(ici_shape: Sequence[int], dcn_shape: Sequence[int]) -> Mesh:
    """Build a topology-aware ``('tp', 'fsdp', 'dp')`` mesh.

    ``ici_shape`` and ``dcn_shape`` are ``(tp, fsdp, dp)`` triples in
    comm-heaviest-first order. ICI axes ride the NVLink domain (one node); DCN
    axes ride the data-center network (across nodes).

    * Single process (``num_processes == 1``): a plain row-major reshape of the
      node's devices is fine because the whole node is one NVLink domain. We use
      ``mesh_utils.create_device_mesh(ici_shape, devices)``.
    * Multi process: ``mesh_utils.create_hybrid_device_mesh`` with
      ``process_is_granule=True`` groups granules by ``process_index`` (node) so
      ICI axes stay within a node. We wrap the returned ndarray in
      :class:`jax.sharding.Mesh` directly -- routing back through
      ``jax.make_mesh`` would re-reshape and destroy the hybrid layout.
    """
    ici_shape = tuple(int(x) for x in ici_shape)
    dcn_shape = tuple(int(x) for x in dcn_shape)
    if len(ici_shape) != 3 or len(dcn_shape) != 3:
        raise ValueError(
            f"ici_shape and dcn_shape must be (tp, fsdp, dp) triples, got "
            f"ici_shape={ici_shape}, dcn_shape={dcn_shape}."
        )

    devices = jax.devices()
    ldc = local_device_count()
    nproc = num_processes()

    ici_prod = math.prod(ici_shape)
    dcn_prod = math.prod(dcn_shape)

    # Validation: ICI tiles one node, DCN tiles the nodes, total tiles all devices.
    if ici_prod != ldc:
        raise ValueError(
            f"prod(ici_shape)={ici_prod} (ici_shape={ici_shape}) != local_device_count={ldc}. "
            "The ICI axes must exactly fill one NVLink domain (GPUs per node)."
        )
    if dcn_prod != nproc:
        raise ValueError(
            f"prod(dcn_shape)={dcn_prod} (dcn_shape={dcn_shape}) != num_processes={nproc}. "
            "The DCN axes must exactly tile the process (node) count."
        )
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

    # Match jax.make_mesh's default axis types (Explicit sharding) so downstream
    # code that relies on Explicit-typed axes -- e.g. out_sharding= in
    # .at[...].get() (omegalax/models/qwen3/model.py) -- keeps working. A bare
    # jax.sharding.Mesh(...) would default axes to AxisType.Auto and regress.
    axis_types = (AxisType.Explicit,) * len(_AXES)

    if nproc == 1:
        # Single node: plain reshape is fine intra-node (whole node is NVLink).
        device_grid = mesh_utils.create_device_mesh(ici_shape, devices)
        return Mesh(device_grid, _AXES, axis_types=axis_types)

    # Multi node: hybrid ICI/DCN placement. process_is_granule=True is mandatory
    # here (uniform slice_index under init_distributed()'s multi-node
    # jax.distributed.initialize(), which sets no partition index).
    device_grid = mesh_utils.create_hybrid_device_mesh(
        ici_shape,
        dcn_shape,
        devices,
        process_is_granule=True,
        allow_split_physical_axes=False,
    )
    # Wrap the hybrid ndarray directly -- do NOT route through jax.make_mesh.
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
    parallelism: ParallelismConfig | None = None,
) -> Mesh:
    """Build a topology-aware ``('tp', 'fsdp', 'dp')`` mesh.

    Public 3-arg interface preserved: given legacy (tp, fsdp, dp) sizes, the
    ICI/DCN split is derived via :func:`derive_ici_dcn` and materialized with
    :func:`make_hierarchical_mesh`. Callers may optionally pass an explicit
    :class:`ParallelismConfig` via ``parallelism`` to override the split; when
    given, its per-type products must match the three sizes.
    """
    tp, fsdp, dp = _resolve_mesh_shape(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    if parallelism is None:
        parallelism = derive_ici_dcn(
            tp_size=tp,
            fsdp_size=fsdp,
            dp_size=dp,
            local_device_count=local_device_count(),
            num_processes=num_processes(),
        )
    elif (parallelism.tp_size, parallelism.fsdp_size, parallelism.dp_size) != (tp, fsdp, dp):
        raise ValueError(
            f"Explicit ParallelismConfig per-type products "
            f"(tp={parallelism.tp_size}, fsdp={parallelism.fsdp_size}, dp={parallelism.dp_size}) "
            f"conflict with requested sizes (tp={tp}, fsdp={fsdp}, dp={dp})."
        )
    return make_hierarchical_mesh(parallelism.ici_shape, parallelism.dcn_shape)


def set_default_mesh(
    tp_size: int,
    fsdp_size: int,
    dp_size: int,
    *,
    parallelism: ParallelismConfig | None = None,
) -> Mesh:
    mesh = make_mesh(
        tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size, parallelism=parallelism
    )
    jax.set_mesh(mesh)
    return mesh


def ensure_mesh(
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    dp_size: int | None = None,
    *,
    parallelism: ParallelismConfig | None = None,
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

    return set_default_mesh(
        tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size, parallelism=parallelism
    )


@contextmanager
def mesh_rules(mesh: Mesh) -> Iterator[Mesh]:
    """Activate mesh + logical axis rules for a scoped block."""
    with jax.set_mesh(mesh), nnx.logical_axis_rules(axis_rules_for_mesh(mesh)):
        yield mesh


# --- Expert parallelism (MoE grouped-GEMM + ragged all-to-all) ---------------
# Dedicated 1-D ``('expert',)`` mesh for the expert-parallel MoE path (consumed by
# omegalax.models.moe_grouped.grouped_moe_ep), deliberately SEPARATE from the flat
# ``('tp', 'fsdp', 'dp')`` hierarchical training mesh built above.
#
# MERGE RECONCILIATION (feat/moe-ep-grouped-gemm x feat/topology-aware-mesh):
# The moe-ep branch's original MERGE NOTE suggested folding ``'expert'`` into the
# unified mesh factory / axis rules. That fold was evaluated and deliberately NOT
# done, because it would be semantically wrong here:
#   * The topology-aware factory (make_mesh/make_hierarchical_mesh) is a fixed
#     3-axis ``_AXES = ('tp','fsdp','dp')`` design whose ICI/DCN invariants
#     (ici_tp*ici_fsdp*ici_dp == local_device_count, the ensure_mesh _AXES
#     equality check, and all of test_topology_mesh) assume exactly those 3 axes.
#     Adding a 4th ``'expert'`` axis to _AXES/ParallelismConfig would break those
#     invariants and the topology test suite.
#   * grouped_moe_ep does not COMPOSE expert parallelism with the training mesh:
#     it reads a standalone ``'expert'`` axis off the active abstract mesh and runs
#     its own shard_map. No caller in-tree wires EP into the (tp,fsdp,dp) mesh.
# So ``make_expert_mesh`` is kept as a separate helper, made CONSISTENT with the
# hierarchical factory: it builds its device grid via ``mesh_utils.create_device_mesh``
# (same primitive make_hierarchical_mesh uses) and stamps ``AxisType.Explicit``
# (matching _AXES's axis types) rather than hand-rolling a numpy reshape + bare Mesh.
# A future unified expert+training mesh (composing 'expert' onto the ICI domain
# alongside 'tp') is left for later, when a caller actually needs the composition.
_EXPERT_AXIS = "expert"


def make_expert_mesh(ep_size: int, axis_name: str = _EXPERT_AXIS) -> Mesh:
    """Build a 1-D expert-parallel mesh over ``ep_size`` devices.

    The stacked expert weights are sharded on ``axis_name`` (each device owns
    ``E / ep_size`` experts) and the token axis is likewise sharded, so
    ``grouped_moe_ep`` can dispatch each token to the device owning its expert via
    ``ragged_all_to_all``. Requires ``jax.device_count() % ep_size == 0``; uses the
    first ``ep_size`` devices.

    Consistent with :func:`make_hierarchical_mesh`: the device grid is built with
    ``mesh_utils.create_device_mesh`` and the axis is stamped ``AxisType.Explicit``
    (matching the ``_AXES`` mesh), so downstream Explicit-sharding code behaves the
    same on the expert mesh as on the training mesh.
    """
    ndev = jax.device_count()
    if ep_size <= 0 or ndev % ep_size != 0:
        raise ValueError(
            f"expert-parallel size {ep_size} must divide device_count={ndev} and be > 0."
        )
    device_grid = mesh_utils.create_device_mesh((ep_size,), jax.devices()[:ep_size])
    return Mesh(device_grid, (axis_name,), axis_types=(AxisType.Explicit,))


@contextmanager
def mesh_rules_for(tp_size: int, fsdp_size: int, dp_size: int) -> Iterator[Mesh]:
    """Resolve a mesh and activate mesh + logical axis rules for a scoped block."""
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    with mesh_rules(mesh):
        yield mesh
