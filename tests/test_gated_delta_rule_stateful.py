"""Stateful Gated DeltaNet kernel contracts used by statepassing pretraining."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from omegalax.models.qwen3_5.kernels.xla_reference import chunk_gated_delta_rule_xla


def _make_inputs(B=2, T=128, H=3, A=4, U=5, seed=0):
    rng = np.random.RandomState(seed)
    q = jnp.asarray(rng.randn(B, T, H, A).astype(np.float32) * 0.1)
    k = jnp.asarray(rng.randn(B, T, H, A).astype(np.float32) * 0.1)
    v = jnp.asarray(rng.randn(B, T, H, U).astype(np.float32) * 0.1)
    a = jnp.asarray(rng.randn(B, T, H).astype(np.float32) * 0.5)
    g = -jnp.exp(a) * jax.nn.softplus(a)
    beta = jax.nn.sigmoid(jnp.asarray(rng.randn(B, T, H).astype(np.float32) * 0.5))
    return q, k, v, g, beta


class StatefulXlaKernelTest(absltest.TestCase):
    def test_zero_initial_state_matches_stateless_output(self):
        q, k, v, g, beta = _make_inputs(T=96)
        zero_state = jnp.zeros((2, 3, 4, 5), dtype=jnp.float32)

        stateless = chunk_gated_delta_rule_xla(q, k, v, g, beta, chunk_size=32)
        stateful, final_state = chunk_gated_delta_rule_xla(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=32,
            initial_state_BHAU=zero_state,
            return_final_state=True,
        )

        np.testing.assert_allclose(stateful, stateless, rtol=1e-5, atol=1e-5)
        self.assertEqual(final_state.shape, zero_state.shape)

    def test_carried_state_matches_full_sequence_at_chunk_boundary(self):
        q, k, v, g, beta = _make_inputs(T=128)
        full_out, full_state = chunk_gated_delta_rule_xla(
            q, k, v, g, beta, chunk_size=64, return_final_state=True
        )

        out0, state0 = chunk_gated_delta_rule_xla(
            q[:, :64],
            k[:, :64],
            v[:, :64],
            g[:, :64],
            beta[:, :64],
            chunk_size=64,
            return_final_state=True,
        )
        out1, state1 = chunk_gated_delta_rule_xla(
            q[:, 64:],
            k[:, 64:],
            v[:, 64:],
            g[:, 64:],
            beta[:, 64:],
            chunk_size=64,
            initial_state_BHAU=state0,
            return_final_state=True,
        )

        np.testing.assert_allclose(
            jnp.concatenate([out0, out1], axis=1), full_out, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(state1, full_state, rtol=1e-5, atol=1e-5)

    def test_loss_can_backpropagate_to_initial_state(self):
        q, k, v, g, beta = _make_inputs(T=64)
        initial_state = jnp.ones((2, 3, 4, 5), dtype=jnp.float32) * 0.01

        def loss_fn(state):
            out, final_state = chunk_gated_delta_rule_xla(
                q,
                k,
                v,
                g,
                beta,
                chunk_size=64,
                initial_state_BHAU=state,
                return_final_state=True,
            )
            return jnp.sum(out.astype(jnp.float32) ** 2) + jnp.sum(final_state**2)

        grad = jax.grad(loss_fn)(initial_state)
        self.assertGreater(float(jnp.linalg.norm(grad)), 0.0)


if __name__ == "__main__":
    absltest.main()
