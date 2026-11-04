"""Sharding tests for the packed vision-attention shard_map wrap (Qwen3-VL).

``_cudnn_packed_vision_attention`` must enter a manual-mode ``jax.shard_map``
on ANY non-empty mesh, not just when the packed dim is dp-sharded: cuDNN fused
attention is a ``custom_partitioning`` op, and when the SPMD/Shardy partitioner
sees it on a multi-device mesh its sharding rule maps the softmax-stat output's
leading dim to ``CompoundFactor('batch', 'n')`` -- with our packed layout's
batch == 1 Shardy rejects the size-1 factor ("dim mapping can't have a factor
of size 1 if there are multiple factors") at lowering, even for fully
replicated q/k/v. The wrap hides the op from the partitioner entirely (same
technique as the Qwen3.5 vision fix).

These tests are torch-free and run on CPU: in fp32 the local kernel routes to
the pure-jax reference, so the wrap's routing/specs/numerics are exercised on a
faked 4-device mesh without a GPU. The cuDNN kernel itself is covered by the
GPU suite.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Fake multiple devices so we can exercise a real (sharded) mesh on CPU.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
)

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax.sharding import NamedSharding, PartitionSpec

from omegalax.distributed.mesh import make_mesh
from omegalax.models.qwen3_vl.vision import (
    _cudnn_packed_vision_attention,
    _cudnn_packed_vision_attention_local,
)

P = PartitionSpec

H, K = 4, 8
SCALE = 0.35


def _make_inputs(rng, segs):
    n = int(sum(segs))
    q = jax.random.normal(rng, (n, H, K), dtype=jnp.float32)
    k = jax.random.normal(jax.random.fold_in(rng, 1), (n, H, K), dtype=jnp.float32)
    v = jax.random.normal(jax.random.fold_in(rng, 2), (n, H, K), dtype=jnp.float32)
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(segs)]), dtype=jnp.int32)
    sq = jnp.asarray(segs, dtype=jnp.int32)
    return q, k, v, cu, sq


def _wrapped(q, k, v, cu, sq):
    return _cudnn_packed_vision_attention(q, k, v, cu, sq, SCALE)


class PackedVisionAttentionShardingTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.rng = jax.random.key(0)

    def test_no_mesh_guard_matches_local(self):
        """Without a mesh the wrap must fall through to the local kernel."""
        q, k, v, cu, sq = _make_inputs(self.rng, [3, 5])
        ref = _cudnn_packed_vision_attention_local(q, k, v, cu, sq, SCALE)
        out = jax.jit(_wrapped)(q, k, v, cu, sq)
        np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)

    def test_replicated_on_multi_device_mesh(self):
        """Replicated q/k/v on a real multi-device mesh: the regression case.

        Pre-fix, spec[0] is None here so the cuDNN op leaked to the
        partitioner (the vision.py:75 Shardy crash on GPU). Post-fix the wrap
        is entered with replicated specs and every shard computes the full
        packed sequence.
        """
        q, k, v, cu, sq = _make_inputs(self.rng, [3, 5])
        ref = _cudnn_packed_vision_attention_local(q, k, v, cu, sq, SCALE)
        mesh = make_mesh(tp_size=1, fsdp_size=4, dp_size=1)
        with jax.set_mesh(mesh):
            rep = NamedSharding(mesh, P(None, None, None))
            rep1 = NamedSharding(mesh, P(None))
            args = [jax.device_put(x, rep) for x in (q, k, v)]
            args += [jax.device_put(cu, rep1), jax.device_put(sq, rep1)]
            out = jax.jit(_wrapped)(*args)
        np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-6, atol=1e-6)

    def test_dim0_sharded_per_device_segments(self):
        """Packed dim sharded on fsdp: each device attends over its own segments."""
        n_local, n_dev = 6, 4
        segs_local = [6]
        cu_local = np.concatenate([[0], np.cumsum(segs_local)]).astype(np.int32)
        sq_local = np.asarray(segs_local, dtype=np.int32)
        q, k, v, _, _ = _make_inputs(self.rng, [n_local * n_dev])
        cug = jnp.asarray(np.tile(cu_local, n_dev))
        sqg = jnp.asarray(np.tile(sq_local, n_dev))

        mesh = make_mesh(tp_size=1, fsdp_size=4, dp_size=1)
        with jax.set_mesh(mesh):
            shd = NamedSharding(mesh, P("fsdp", None, None))
            shd1 = NamedSharding(mesh, P("fsdp"))
            args = [jax.device_put(x, shd) for x in (q, k, v)]
            args += [jax.device_put(cug, shd1), jax.device_put(sqg, shd1)]
            out = jax.jit(_wrapped)(*args)

        ref = np.concatenate(
            [
                np.asarray(
                    _cudnn_packed_vision_attention_local(
                        q[i * n_local : (i + 1) * n_local],
                        k[i * n_local : (i + 1) * n_local],
                        v[i * n_local : (i + 1) * n_local],
                        jnp.asarray(cu_local),
                        jnp.asarray(sq_local),
                        SCALE,
                    )
                )
                for i in range(n_dev)
            ]
        )
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-6, atol=1e-6)

    def test_heads_sharded_on_tp(self):
        """Heads dim tp-sharded (the vision heads_shd layout): heads are
        independent, so the wrapped result must match the unsharded local one."""
        q, k, v, cu, sq = _make_inputs(self.rng, [3, 5])
        ref = _cudnn_packed_vision_attention_local(q, k, v, cu, sq, SCALE)
        mesh = make_mesh(tp_size=4, fsdp_size=1, dp_size=1)
        with jax.set_mesh(mesh):
            hshd = NamedSharding(mesh, P(None, "tp", None))
            rep1 = NamedSharding(mesh, P(None))
            args = [jax.device_put(x, hshd) for x in (q, k, v)]
            args += [jax.device_put(cu, rep1), jax.device_put(sq, rep1)]
            out = jax.jit(_wrapped)(*args)
        np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-6, atol=1e-6)

    def test_grad_matches_local_reference(self):
        """Gradients flow through the wrap and match the unwrapped computation."""
        q, k, v, cu, sq = _make_inputs(self.rng, [3, 5])

        def loss_local(qq):
            return _cudnn_packed_vision_attention_local(qq, k, v, cu, sq, SCALE).sum()

        g_ref = jax.grad(loss_local)(q)

        mesh = make_mesh(tp_size=1, fsdp_size=4, dp_size=1)
        with jax.set_mesh(mesh):
            rep = NamedSharding(mesh, P(None, None, None))
            rep1 = NamedSharding(mesh, P(None))
            qr, kr, vr = (jax.device_put(x, rep) for x in (q, k, v))
            cur, sqr = jax.device_put(cu, rep1), jax.device_put(sq, rep1)

            def loss_wrapped(qq):
                return _cudnn_packed_vision_attention(qq, kr, vr, cur, sqr, SCALE).sum()

            g = jax.jit(jax.grad(loss_wrapped))(qr)
        np.testing.assert_allclose(np.asarray(g), np.asarray(g_ref), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    absltest.main()
