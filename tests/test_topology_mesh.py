"""Unit tests for topology-aware (NVLink-domain) hierarchical device meshes.

Two layers of coverage:

1. Pure-function tests of :func:`derive_ici_dcn` (no JAX devices needed): the
   TP->ICI / FSDP-spill / DP->DCN split, per-type product invariants, and the
   error cases (``tp_size > local_device_count``, non-divisible FSDP).

2. Faked multi-device CPU tests (8 host devices via ``XLA_FLAGS``) that build
   real ``Mesh`` objects and assert axis names, shape and single-process device
   placement. True DCN placement needs multiple processes and is covered by the
   multi-node probe script (scripts/probe_topology.py), not here.
"""

import os

# Must be set before jax is imported so the host-platform device count sticks.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest, parameterized  # noqa: E402
from jax.sharding import AxisType, Mesh  # noqa: E402

from omegalax.distributed.mesh import (  # noqa: E402
    ParallelismConfig,
    derive_ici_dcn,
    make_hierarchical_mesh,
    make_mesh,
)

_AXES = ("tp", "fsdp", "dp")


class DeriveIciDcnTest(parameterized.TestCase):
    """Pure-function tests of the legacy (tp, fsdp, dp) -> ICI/DCN mapping."""

    @parameterized.named_parameters(
        # name, tp, fsdp, dp, ldc, nproc, expected (ici_tp,ici_fsdp,ici_dp,dcn_tp,dcn_fsdp,dcn_dp)
        # --- single node (nproc=1): everything lands in ICI ---
        ("single_node_tp_only", 8, 1, 1, 8, 1, (8, 1, 1, 1, 1, 1)),
        ("single_node_tp_fsdp", 2, 4, 1, 8, 1, (2, 4, 1, 1, 1, 1)),
        ("single_node_no_parallel", 1, 1, 1, 1, 1, (1, 1, 1, 1, 1, 1)),
        # --- TP -> ICI only, DP -> DCN only ---
        ("tp_ici_dp_dcn", 8, 1, 2, 8, 2, (8, 1, 1, 1, 1, 2)),
        ("tp2_dp8", 2, 1, 8, 2, 8, (2, 1, 1, 1, 1, 8)),
        # --- FSDP fills the node first (ici), no spill ---
        ("fsdp_fills_node", 2, 4, 2, 8, 2, (2, 4, 1, 1, 1, 2)),
        # --- FSDP spills to DCN once the node is full ---
        # tp=2 -> ici_tp=2; node has 8 gpus -> ici_slots=4; fsdp=8 -> ici_fsdp=4, dcn_fsdp=2
        ("fsdp_spills", 2, 8, 1, 8, 2, (2, 4, 1, 1, 2, 1)),
        # tp=1, fsdp=16 across 4 nodes of 4 gpus: ici_fsdp=4, dcn_fsdp=4
        ("fsdp_spills_4nodes", 1, 16, 1, 4, 4, (1, 4, 1, 1, 4, 1)),
        # combined: tp=2, fsdp=8, dp=2 over 4 nodes of 8 gpus
        # ici_tp=2, ici_slots=4, ici_fsdp=4, dcn_fsdp=2, dcn_dp=2 -> nproc=4
        ("combined_spill", 2, 8, 2, 8, 4, (2, 4, 1, 1, 2, 2)),
    )
    def test_split(self, tp, fsdp, dp, ldc, nproc, expected):
        cfg = derive_ici_dcn(tp, fsdp, dp, local_device_count=ldc, num_processes=nproc)
        self.assertEqual(
            (cfg.ici_tp, cfg.ici_fsdp, cfg.ici_dp, cfg.dcn_tp, cfg.dcn_fsdp, cfg.dcn_dp),
            expected,
        )
        # TP is ICI-only; DP is DCN-only.
        self.assertEqual(cfg.dcn_tp, 1)
        self.assertEqual(cfg.ici_dp, 1)
        # Per-type products preserved -> tp*fsdp*dp and dp=dp_size*fsdp_size unchanged.
        self.assertEqual(cfg.tp_size, tp)
        self.assertEqual(cfg.fsdp_size, fsdp)
        self.assertEqual(cfg.dp_size, dp)
        # ICI tiles one node exactly; DCN tiles the node count exactly.
        self.assertEqual(cfg.ici_tp * cfg.ici_fsdp * cfg.ici_dp, ldc)
        self.assertEqual(cfg.dcn_tp * cfg.dcn_fsdp * cfg.dcn_dp, nproc)

    def test_tp_exceeds_local_device_count_raises(self):
        # The correctness guardrail: TP must fit within the NVLink domain.
        with self.assertRaisesRegex(ValueError, "exceeds local_device_count"):
            derive_ici_dcn(8, 1, 1, local_device_count=4, num_processes=2)

    def test_tp_not_dividing_node_raises(self):
        with self.assertRaisesRegex(ValueError, "not divisible by tp_size"):
            derive_ici_dcn(3, 1, 1, local_device_count=8, num_processes=1)

    def test_fsdp_not_dividing_raises(self):
        # tp=2 -> ici_slots=4; fsdp=6 not divisible by ici_fsdp=4.
        with self.assertRaisesRegex(ValueError, "not divisible by its intra-node share"):
            derive_ici_dcn(2, 6, 1, local_device_count=8, num_processes=1)

    def test_dcn_product_mismatch_raises(self):
        # fsdp spill + dp don't tile the node count: dcn_fsdp=2, dcn_dp=2 -> 4 != nproc=8.
        with self.assertRaisesRegex(ValueError, "!= num_processes"):
            derive_ici_dcn(2, 8, 2, local_device_count=8, num_processes=8)

    def test_nonpositive_sizes_raise(self):
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            derive_ici_dcn(0, 1, 1, local_device_count=8, num_processes=1)
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            derive_ici_dcn(1, 1, 1, local_device_count=0, num_processes=1)

    def test_parallelism_config_properties(self):
        cfg = ParallelismConfig(
            ici_tp=2, ici_fsdp=4, ici_dp=1, dcn_tp=1, dcn_fsdp=2, dcn_dp=3
        )
        self.assertEqual(cfg.ici_shape, (2, 4, 1))
        self.assertEqual(cfg.dcn_shape, (1, 2, 3))
        self.assertEqual(cfg.tp_size, 2)
        self.assertEqual(cfg.fsdp_size, 8)
        self.assertEqual(cfg.dp_size, 3)


class FakedDeviceMeshTest(parameterized.TestCase):
    """Build real meshes on 8 faked CPU devices (single process)."""

    def test_eight_devices_available(self):
        self.assertEqual(jax.device_count(), 8)
        self.assertEqual(jax.process_count(), 1)

    @parameterized.named_parameters(
        ("tp8", 8, 1, 1),
        ("fsdp8", 1, 8, 1),
        ("dp8", 1, 1, 8),
        ("tp2_fsdp4", 2, 4, 1),
        ("tp2_fsdp2_dp2", 2, 2, 2),
        ("tp4_dp2", 4, 1, 2),
    )
    def test_make_mesh_shape_and_axes(self, tp, fsdp, dp):
        mesh = make_mesh(tp_size=tp, fsdp_size=fsdp, dp_size=dp)
        self.assertIsInstance(mesh, Mesh)
        self.assertEqual(tuple(mesh.axis_names), _AXES)
        self.assertEqual(mesh.shape["tp"], tp)
        self.assertEqual(mesh.shape["fsdp"], fsdp)
        self.assertEqual(mesh.shape["dp"], dp)
        # Axis types must be Explicit to match jax.make_mesh's default; the model
        # code relies on Explicit-typed axes for out_sharding= in .at[].get().
        self.assertEqual(tuple(mesh.axis_types), (AxisType.Explicit,) * 3)
        # All 8 devices used exactly once.
        ids = sorted(d.id for d in mesh.devices.flatten())
        self.assertEqual(ids, list(range(8)))

    def test_single_process_placement_uses_all_devices(self):
        # Single-process path: create_device_mesh reshape; every device present once.
        mesh = make_mesh(tp_size=2, fsdp_size=2, dp_size=2)
        grid = np.vectorize(lambda d: d.id)(mesh.devices)
        self.assertEqual(grid.shape, (2, 2, 2))
        self.assertEqual(sorted(grid.flatten().tolist()), list(range(8)))

    def test_make_hierarchical_mesh_single_process(self):
        # nproc==1 so dcn_shape must be all-ones and prod(ici)==device_count.
        mesh = make_hierarchical_mesh((2, 2, 2), (1, 1, 1))
        self.assertEqual(tuple(mesh.axis_names), _AXES)
        self.assertEqual(mesh.shape["tp"], 2)

    def test_hierarchical_ici_product_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "!= local_device_count"):
            make_hierarchical_mesh((2, 2, 1), (1, 1, 1))  # prod ici = 4 != 8

    def test_hierarchical_dcn_product_mismatch_raises(self):
        # prod(dcn)=2 but nproc==1.
        with self.assertRaisesRegex(ValueError, "!= num_processes"):
            make_hierarchical_mesh((2, 2, 2), (1, 1, 2))

    def test_make_mesh_bad_device_count_raises(self):
        with self.assertRaisesRegex(ValueError, "does not match device_count"):
            make_mesh(tp_size=3, fsdp_size=1, dp_size=1)

    def test_make_mesh_tp_exceeds_node_raises(self):
        # 8 devices, single process -> ldc=8; tp=16 impossible anyway (device_count),
        # but tp=8,fsdp=2 would need 16 devices; test the guardrail via device_count.
        with self.assertRaisesRegex(ValueError, "does not match device_count"):
            make_mesh(tp_size=8, fsdp_size=2, dp_size=1)

    def test_explicit_parallelism_config_override(self):
        cfg = ParallelismConfig(
            ici_tp=2, ici_fsdp=2, ici_dp=2, dcn_tp=1, dcn_fsdp=1, dcn_dp=1
        )
        mesh = make_mesh(tp_size=2, fsdp_size=2, dp_size=2, parallelism=cfg)
        self.assertEqual(mesh.shape["tp"], 2)
        self.assertEqual(mesh.shape["fsdp"], 2)
        self.assertEqual(mesh.shape["dp"], 2)

    def test_explicit_parallelism_config_conflict_raises(self):
        cfg = ParallelismConfig(
            ici_tp=2, ici_fsdp=2, ici_dp=2, dcn_tp=1, dcn_fsdp=1, dcn_dp=1
        )
        with self.assertRaisesRegex(ValueError, "conflict with requested sizes"):
            make_mesh(tp_size=4, fsdp_size=1, dp_size=2, parallelism=cfg)


if __name__ == "__main__":
    absltest.main()
