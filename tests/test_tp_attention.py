"""Tensor-parallel tokamax attention: sharded forward and backward vs the XLA oracle.

Every decoder layer calls ``dot_product_attention(q, k, v, is_causal=True,
scale=..., implementation=<backend>, q_sharding=<head-axis NamedSharding>)``
(``models/qwen3/attention.py``, ``models/qwen3_5/attention.py``). A mosaic kernel
that reshards wrongly, or a backward that drops the cross-device contribution,
is a silent wrong-numerics bug that needs >1 device and the multi-process launch
the trainers use, so no other test in this suite can see it.

Skipped, loudly, without that launch. Before this was a test it was four
``try/except Exception: print(...)`` probes at module scope with zero ``assert``
and zero ``def test``: pytest reported ``collected 0 items``, which reads as
green, and the probes named ``implementation="mosaic"`` -- not a key of
``tokamax``'s ``IMPLEMENTATIONS`` (that is ``mosaic_gpu``), so all four printed a
failure that nothing ever read.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

try:
    jax.distributed.initialize()
except ValueError as exc:
    pytest.skip(f"requires a distributed JAX launch: {exc}", allow_module_level=True)

if jax.device_count() < 2:
    pytest.skip(
        f"requires >=2 devices for a tp mesh, got {jax.device_count()}",
        allow_module_level=True,
    )

from tokamax import dot_product_attention  # noqa: E402  (after jax.distributed.initialize)

MESH = jax.make_mesh((jax.device_count(),), ("tp",))
jax.set_mesh(MESH)

# The production shapes and dtype: bf16, head count divisible by tp, same kv
# heads, sharded over the head axis exactly as ``shd_cfg.act_btnh`` does.
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
    """Global fp32 numpy view. A tp-sharded array spans non-addressable devices,
    so ``np.asarray`` on it raises rather than returning a partial result."""
    return np.asarray(multihost_utils.process_allgather(x, tiled=True), dtype=np.float32)


class TpAttentionTest(absltest.TestCase):
    """``mosaic_gpu`` on a tp-sharded head axis vs ``xla`` on replicated inputs.

    Two independent tokamax implementations of the same math, so a wrong reshard
    shows up as wrong numbers and not as a shape error.

    Both tolerances are measured on this shape, because the max-abs one alone is
    a poor discriminator here: clean bf16 disagreement is already 0.0156 fwd /
    0.0078 bwd (the bf16 representation floor through the xla path alone is
    0.0122), while an injected 1% error in ``scale`` reaches only 0.0313. The
    MEDIAN separates them by orders of magnitude -- exactly 0 when clean, 1e-3
    under that same 1% error -- so it is the assertion that bites.
    """

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
