"""Policy loss and token log-probabilities against hand-computed values (CPU)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from omegalax.rl.loss import policy_loss, token_logprobs
from omegalax.rl.rollout import group_advantages


class PolicyLossTest(absltest.TestCase):
    def test_clipped_surrogate_by_hand(self):
        # ratios exp(log_ratio): 1.0, 1.5, 0.5, 1.5, 0.5, masked
        logp = jnp.log(jnp.array([[0.5, 0.6, 0.2, 0.6, 0.2, 0.9]]))
        behavior = jnp.log(jnp.array([[0.5, 0.4, 0.4, 0.4, 0.4, 0.1]]))
        adv = jnp.array([[2.0, 2.0, 2.0, -1.0, -1.0, 5.0]])
        mask = jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
        loss, log_ratio, clipped = policy_loss(logp, behavior, adv, mask, eps_low=0.2, eps_high=0.28)
        # token: -min(r*A, clip(r, 0.8, 1.28)*A)
        #  r=1.0, A=2 : -2.0
        #  r=1.5, A=2 : -min(3.0, 2.56) = -2.56   (clipped)
        #  r=0.5, A=2 : -min(1.0, 1.6)  = -1.0
        #  r=1.5, A=-1: -min(-1.5, -1.28) = 1.5
        #  r=0.5, A=-1: -min(-0.5, -0.8)  = 0.8   (clipped)
        self.assertAlmostEqual(float(loss), -2.0 - 2.56 - 1.0 + 1.5 + 0.8, places=5)
        np.testing.assert_array_equal(np.asarray(clipped), [[0, 1, 0, 0, 1, 0]])
        self.assertEqual(float(log_ratio[0, 5]), 0.0)

    def test_gradient_is_advantage_weighted_ratio_inside_the_clip(self):
        behavior = jnp.log(jnp.array([[0.5, 0.4]]))
        adv = jnp.array([[2.0, -1.0]])
        mask = jnp.ones((1, 2))
        grad = jax.grad(lambda lp: policy_loss(lp, behavior, adv, mask, 0.2, 0.28)[0])(behavior)
        # d/dlogp of -r*A at r=1 is -A.
        np.testing.assert_allclose(np.asarray(grad), [[-2.0, 1.0]], rtol=1e-6)

    def test_group_advantages_match_unbiased_group_std(self):
        r = [1.0, 0.0, 0.0, 1.0]
        mean, std = 0.5, np.std(r, ddof=1)
        np.testing.assert_allclose(group_advantages(r), [(x - mean) / (std + 1e-6) for x in r])
        self.assertEqual(group_advantages([1.0, 1.0, 1.0]), [0.0, 0.0, 0.0])


class TokenLogprobsTest(absltest.TestCase):
    def test_matches_dense_log_softmax_for_every_tiling(self):
        rng = np.random.default_rng(0)
        B, T, D, V = 2, 11, 8, 32
        hidden = jnp.asarray(rng.normal(size=(B, T, D)), jnp.float32)
        head = jnp.asarray(rng.normal(size=(D, V)), jnp.float32)
        tokens = jnp.asarray(rng.integers(0, V, size=(B, T)), jnp.int32)
        logits = jnp.einsum("BTD,DV->BTV", hidden, head)
        ref = jnp.take_along_axis(jax.nn.log_softmax(logits[:, :-1]), tokens[:, 1:, None], -1)[..., 0]
        p = jax.nn.softmax(logits[:, :-1])
        ref_entropy = -jnp.sum(p * jnp.log(p), -1)
        for tiles in (1, 3, 10):
            logp, entropy = token_logprobs(hidden, head, tokens, tiles)
            np.testing.assert_allclose(np.asarray(logp[:, 1:]), np.asarray(ref), atol=1e-5)
            np.testing.assert_allclose(np.asarray(entropy[:, 1:]), np.asarray(ref_entropy), atol=1e-5)
            np.testing.assert_array_equal(np.asarray(logp[:, 0]), 0.0)


if __name__ == "__main__":
    absltest.main()
