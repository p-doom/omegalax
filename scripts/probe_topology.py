"""Multi-node probe: verify TP peers are co-located within an NVLink domain.

Compares the CURRENT flat mesh (``jax.make_mesh((tp, fsdp, dp), _AXES)``) against
the PROPOSED hierarchical mesh (``omegalax.distributed.mesh.make_mesh``) and,
for each, prints the TP communication groups and asserts whether every TP peer
shares a physical node (== process, one process per node). Also runs a small TP
all-reduce microbench so the InfiniBand-vs-NVLink difference is measurable.

Why this is a script and not a unit test
-----------------------------------------
True DCN placement only differs from the flat layout when
``jax.process_count() > 1``. On a single host (CPU or one GPU node) both meshes
are identical, so the bug and the fix are only observable under a real
multi-node ``srun`` with ONE process per node. The 16-node reservations were
pending when this was written; run this later in a healthy GPU env.

How to run (once a multi-node allocation is available)
------------------------------------------------------
Launch ONE process per node (``--ntasks-per-node=1``) so that
``jax.local_device_count()`` == GPUs-per-node == the NVLink/ICI domain and
``jax.process_count()`` == number of nodes == DCN granules. Example for a
16-node x N-GPU reservation, from the worktree root::

    srun --jobid=<JOBID> --overlap \
         --nodes=16 --ntasks-per-node=1 --gpus-per-task=<GPUS_PER_NODE> \
         --chdir=/hkfs/work/workspace/scratch/tum_cte0515-crowd-cast/pdoom_shared/franz/omegalax-wt-topology-mesh \
         bash -c 'source ~/.bashrc >/dev/null 2>&1; \
                  uv run --no-sync python scripts/probe_topology.py \
                      --tp_size=<GPUS_PER_NODE> --fsdp_size=16 --dp_size=1'

Pick sizes so that ``tp_size <= GPUs-per-node`` and
``tp*fsdp*dp == total GPUs``. A clean demonstration of the bug: set ``tp_size``
> 1 with ``fsdp``/``dp`` spanning nodes; the CURRENT mesh will place TP peers on
distinct nodes (all-reduce -> InfiniBand) while the PROPOSED mesh keeps them on
one node (all-reduce -> NVLink).

NOTE: there is a known cuDNN 9.5.1-vs-9.8.0 mismatch that can break GPU XLA
compile on some dev nodes; the microbench must run in a healthy env. The
placement checks (which do not compile anything heavy) work regardless.
"""

from __future__ import annotations

import time

import jax
import numpy as np
from absl import app, flags
from jax.sharding import Mesh, NamedSharding, PartitionSpec

# Import lazily-safe: this module only touches placement logic.
from omegalax.distributed.launch import init_distributed
from omegalax.distributed.mesh import derive_ici_dcn, make_mesh

FLAGS = flags.FLAGS
flags.DEFINE_integer("tp_size", 2, "Tensor-parallel degree.")
flags.DEFINE_integer("fsdp_size", 1, "FSDP degree.")
flags.DEFINE_integer("dp_size", 1, "Data-parallel degree.")
flags.DEFINE_integer("bench_iters", 50, "All-reduce microbench iterations.")
flags.DEFINE_integer("bench_mib", 64, "All-reduce payload size in MiB (per device).")

_AXES = ("tp", "fsdp", "dp")


def _current_flat_mesh(tp: int, fsdp: int, dp: int) -> Mesh:
    """The mesh omegalax built BEFORE this change (the buggy flat layout)."""
    return jax.make_mesh((tp, fsdp, dp), _AXES)


def _device_node(dev) -> int:
    """Node identity of a device == its owning process (one process per node)."""
    return dev.process_index


def _tp_groups(mesh: Mesh) -> list[list]:
    """Return the list of TP communication groups (devices sharing (fsdp, dp))."""
    grid = mesh.devices  # ndarray shape (tp, fsdp, dp)
    tp, fsdp, dp = grid.shape
    groups = []
    for f in range(fsdp):
        for d in range(dp):
            groups.append([grid[t, f, d] for t in range(tp)])
    return groups


def _report(name: str, mesh: Mesh) -> bool:
    print(f"\n=== {name} mesh ===")
    print(f"axis_names={mesh.axis_names} shape={dict(mesh.shape)}")
    grid_ids = np.vectorize(lambda x: x.id)(mesh.devices)
    grid_nodes = np.vectorize(_device_node)(mesh.devices)
    print(f"device-id grid (tp,fsdp,dp):\n{grid_ids}")
    print(f"node/process grid (tp,fsdp,dp):\n{grid_nodes}")

    all_colocated = True
    for i, group in enumerate(_tp_groups(mesh)):
        nodes = {_device_node(d) for d in group}
        ids = [d.id for d in group]
        colocated = len(nodes) == 1
        all_colocated = all_colocated and colocated
        flag = "NVLink (co-located)" if colocated else "InfiniBand (STRIDED across nodes!)"
        print(f"  TP group {i}: ids={ids} nodes={sorted(nodes)} -> {flag}")
    print(f"  => all TP groups co-located within a node: {all_colocated}")
    return all_colocated


def _bench_tp_allreduce(mesh: Mesh, mib: int, iters: int) -> float:
    """Time a TP-axis all-reduce; returns median seconds per iteration."""
    n = (mib * 1024 * 1024) // 4  # float32 elements per shard
    tp = int(mesh.shape["tp"])
    spec = PartitionSpec("tp")
    sharding = NamedSharding(mesh, spec)
    x = jax.device_put(np.ones((tp * n,), dtype=np.float32), sharding)

    @jax.jit
    def allreduce(v):
        # psum over the tp axis via a mean/sum along the sharded axis.
        return jax.lax.with_sharding_constraint(v, NamedSharding(mesh, PartitionSpec("tp"))) * 1.0

    # Warmup + a real collective: sum a tp-sharded array to a replicated scalar.
    @jax.jit
    def reduce_sum(v):
        return jax.numpy.sum(v)

    reduce_sum(x).block_until_ready()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        reduce_sum(x).block_until_ready()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main(_) -> None:
    # Robust launch: init_distributed() detects single- vs multi-process from the
    # SLURM env and either skips initialize() (single process -> all local GPUs)
    # or calls it with an explicitly-derived coordinator/num_processes/process_id
    # (multi-node -> one process per node). The placement report is meaningful in
    # both cases (trivially so on one process).
    init_distributed()
    tp, fsdp, dp = FLAGS.tp_size, FLAGS.fsdp_size, FLAGS.dp_size

    ndev = jax.device_count()
    ldc = jax.local_device_count()
    nproc = jax.process_count()
    if jax.process_index() == 0:
        print("=" * 72)
        print(f"device_count={ndev} local_device_count={ldc} process_count={nproc}")
        print(f"requested tp={tp} fsdp={fsdp} dp={dp}")
        if ldc == 1:
            print(
                "WARNING: local_device_count==1 -> one process per GPU. For "
                "process_is_granule=True to mean one granule per node, launch "
                "ONE process per node (--ntasks-per-node=1)."
            )
        try:
            cfg = derive_ici_dcn(tp, fsdp, dp, local_device_count=ldc, num_processes=nproc)
            print(f"derived ici_shape={cfg.ici_shape} dcn_shape={cfg.dcn_shape}")
        except ValueError as e:
            print(f"derive_ici_dcn error (expected for bad configs): {e}")

    if jax.process_index() != 0:
        # Only rank 0 prints the placement report to avoid interleaving.
        current_ok = None
        proposed_ok = None
    else:
        current = _current_flat_mesh(tp, fsdp, dp)
        current_ok = _report("CURRENT (flat jax.make_mesh)", current)
        proposed = make_mesh(tp_size=tp, fsdp_size=fsdp, dp_size=dp)
        proposed_ok = _report("PROPOSED (hierarchical)", proposed)

        print("\n=== SUMMARY ===")
        print(f"CURRENT  TP peers co-located: {current_ok}")
        print(f"PROPOSED TP peers co-located: {proposed_ok}")
        if nproc > 1 and tp > 1:
            assert proposed_ok, (
                "PROPOSED hierarchical mesh FAILED to co-locate TP peers within a node!"
            )
            if not current_ok:
                print(
                    "As expected: the CURRENT flat mesh scatters TP peers across "
                    "nodes (InfiniBand); the PROPOSED mesh keeps them on NVLink."
                )

    # Microbench on both meshes (all ranks participate in the collective).
    try:
        current = _current_flat_mesh(tp, fsdp, dp)
        proposed = make_mesh(tp_size=tp, fsdp_size=fsdp, dp_size=dp)
        cur_t = _bench_tp_allreduce(current, FLAGS.bench_mib, FLAGS.bench_iters)
        prop_t = _bench_tp_allreduce(proposed, FLAGS.bench_mib, FLAGS.bench_iters)
        if jax.process_index() == 0:
            print("\n=== TP all-reduce microbench (median s/iter) ===")
            print(f"CURRENT  flat mesh : {cur_t * 1e3:.3f} ms")
            print(f"PROPOSED hierarch. : {prop_t * 1e3:.3f} ms")
            if cur_t > 0:
                print(f"speedup (current/proposed): {cur_t / prop_t:.2f}x")
    except Exception as e:  # noqa: BLE001 - microbench is best-effort (cuDNN env, etc.)
        if jax.process_index() == 0:
            print(f"\n[microbench skipped: {type(e).__name__}: {e}]")


if __name__ == "__main__":
    app.run(main)
