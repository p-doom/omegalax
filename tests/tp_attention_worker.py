"""Tensor-parallel Mosaic attention forward and backward against the XLA oracle."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

try:
    jax.distributed.initialize()
except ValueError as exc:
    print(f"requires a distributed JAX launch: {exc}")
    sys.exit(77)

if jax.device_count() < 2:
    print(f"requires >=2 devices for a tp mesh, got {jax.device_count()}")
    sys.exit(77)

from tokamax import dot_product_attention

MESH = jax.make_mesh((jax.device_count(),), ("tp",))
jax.set_mesh(MESH)

Q_SHARDING = NamedSharding(MESH, P(None, None, "tp", None))
B, T, H, K = 1, 128, jax.device_count() * 2, 64
SCALE = K**-0.5
BACKEND = "mosaic_gpu"


def _qkv(sharding):
    return tuple(
        jax.device_put(jax.random.normal(key, (B, T, H, K), dtype=jnp.bfloat16), sharding)
        for key in jax.random.split(jax.random.key(0), 3)
    )


def _attend(q, k, v, *, implementation, q_sharding):
    return dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
        scale=SCALE,
        implementation=implementation,
        q_sharding=q_sharding,
    )


def _gathered(x):
    return np.asarray(multihost_utils.process_allgather(x, tiled=True), dtype=np.float32)


class TpAttentionTest(absltest.TestCase):
    MAX_ATOL = 2.5e-2
    MEDIAN_ATOL = 2e-4

    def setUp(self):
        super().setUp()
        replicated = NamedSharding(MESH, P())
        self.sharded = _qkv(Q_SHARDING)
        self.replicated = _qkv(replicated)
        np.testing.assert_array_equal(_gathered(self.sharded[0]), _gathered(self.replicated[0]))

    def _assert_matches_oracle(self, got, oracle):
        self.assertEqual(got.sharding.spec, Q_SHARDING.spec)
        abs_diff = np.abs(_gathered(got) - _gathered(oracle))
        self.assertLess(np.median(abs_diff), self.MEDIAN_ATOL, f"max={np.max(abs_diff)}")
        self.assertLess(np.max(abs_diff), self.MAX_ATOL)

    def test_sharded_forward_matches_the_xla_oracle(self):
        out = jax.jit(
            lambda q, k, v: _attend(q, k, v, implementation=BACKEND, q_sharding=Q_SHARDING)
        )(*self.sharded)
        oracle = jax.jit(lambda q, k, v: _attend(q, k, v, implementation="xla", q_sharding=None))(
            *self.replicated
        )
        self._assert_matches_oracle(out, oracle)

    def test_sharded_backward_matches_the_xla_oracle(self):
        def loss(q, k, v, *, implementation, q_sharding):
            return _attend(q, k, v, implementation=implementation, q_sharding=q_sharding).sum()

        dq = jax.jit(
            jax.grad(lambda q, k, v: loss(q, k, v, implementation=BACKEND, q_sharding=Q_SHARDING))
        )(*self.sharded)
        oracle = jax.jit(
            jax.grad(lambda q, k, v: loss(q, k, v, implementation="xla", q_sharding=None))
        )(*self.replicated)
        self._assert_matches_oracle(dq, oracle)


if __name__ == "__main__":
    absltest.main()
