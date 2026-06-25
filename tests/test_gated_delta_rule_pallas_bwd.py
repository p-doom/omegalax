"""Backward correctness for the Pallas chunked gated delta rule.

Uses ``jax.grad`` against a scalar L = sum(out**2). Compares pure-JAX
reference gradients to Pallas-backend gradients across q, k, v, g, beta.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from omegalax.models.qwen3_5.kernels.pallas_triton import (
    chunk_gated_delta_rule_pallas,
)
from omegalax.models.qwen3_5.kernels.xla_reference import (
    chunk_gated_delta_rule_xla,
)


def _has_cuda_device() -> bool:
    try:
        return any(device.platform == "gpu" for device in jax.devices())
    except Exception:
        return False


def _make_inputs(B, T, H, A, U, dtype=jnp.bfloat16, seed=0):
    rng = np.random.RandomState(seed)
    q = jnp.asarray(rng.randn(B, T, H, A).astype(np.float32) * 0.1, dtype=dtype)
    k = jnp.asarray(rng.randn(B, T, H, A).astype(np.float32) * 0.1, dtype=dtype)
    v = jnp.asarray(rng.randn(B, T, H, U).astype(np.float32) * 0.1, dtype=dtype)
    a = jnp.asarray(rng.randn(B, T, H).astype(np.float32) * 0.5)
    g = -jnp.exp(a) * jax.nn.softplus(a)
    beta = jax.nn.sigmoid(jnp.asarray(rng.randn(B, T, H).astype(np.float32) * 0.5))
    return q, k, v, g, beta


def _loss(fn):
    def loss_fn(q, k, v, g, beta):
        out = fn(q, k, v, g, beta)
        return jnp.sum(out.astype(jnp.float32) ** 2)

    return loss_fn


def _stateful_loss(fn):
    def loss_fn(q, k, v, g, beta, initial_state):
        out, final_state = fn(
            q,
            k,
            v,
            g,
            beta,
            initial_state_BHAU=initial_state,
            return_final_state=True,
        )
        return jnp.sum(out.astype(jnp.float32) ** 2) + jnp.sum(final_state**2)

    return loss_fn


@absltest.skipIf(not _has_cuda_device(), "requires CUDA GPU")
class BackwardEquivalenceTest(parameterized.TestCase):
    @parameterized.parameters(
        dict(B=1, T=128, H=2, A=64, U=64, name="tiny"),
        dict(B=2, T=512, H=4, A=128, U=128, name="medium"),
    )
    def test_grad_matches_xla(self, B, T, H, A, U, name):
        q, k, v, g, beta = _make_inputs(B, T, H, A, U)
        grads_ref = jax.grad(_loss(chunk_gated_delta_rule_xla), argnums=(0, 1, 2, 3, 4))(
            q, k, v, g, beta
        )
        grads_pal = jax.grad(_loss(chunk_gated_delta_rule_pallas), argnums=(0, 1, 2, 3, 4))(
            q, k, v, g, beta
        )
        names = ("dq", "dk", "dv", "dg", "dbeta")
        for n, gr, gp in zip(names, grads_ref, grads_pal):
            gr_np = np.asarray(gr, dtype=np.float32)
            gp_np = np.asarray(gp, dtype=np.float32)
            abs_err = np.abs(gr_np - gp_np).max()
            rel = abs_err / max(np.abs(gr_np).max(), 1e-6)
            np.testing.assert_array_less(
                abs_err,
                0.5,
                err_msg=f"[{name}] {n}: max|err|={abs_err:.3e} rel={rel:.3e}",
            )


@absltest.skipIf(not _has_cuda_device(), "requires CUDA GPU")
class StatefulBackwardEquivalenceTest(absltest.TestCase):
    def test_stateful_grad_matches_xla(self):
        q, k, v, g, beta = _make_inputs(1, 128, 2, 64, 32)
        initial_state = jnp.ones((1, 2, 64, 32), dtype=jnp.float32) * 0.01
        grads_ref = jax.grad(
            _stateful_loss(chunk_gated_delta_rule_xla), argnums=(0, 1, 2, 3, 4, 5)
        )(q, k, v, g, beta, initial_state)
        grads_pal = jax.grad(
            _stateful_loss(chunk_gated_delta_rule_pallas), argnums=(0, 1, 2, 3, 4, 5)
        )(q, k, v, g, beta, initial_state)
        names = ("dq", "dk", "dv", "dg", "dbeta", "dinitial_state")
        for n, gr, gp in zip(names, grads_ref, grads_pal):
            gr_np = np.asarray(gr, dtype=np.float32)
            gp_np = np.asarray(gp, dtype=np.float32)
            self.assertGreater(np.linalg.norm(gr_np), 0.0, msg=f"{n} reference gradient is zero")
            abs_err = np.abs(gr_np - gp_np).max()
            rel = abs_err / max(np.abs(gr_np).max(), 1e-6)
            np.testing.assert_array_less(
                abs_err,
                0.5,
                err_msg=f"{n}: max|err|={abs_err:.3e} rel={rel:.3e}",
            )


if __name__ == "__main__":
    absltest.main()
