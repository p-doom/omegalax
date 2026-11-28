"""Policy loss and token log-probabilities against hand-computed values (CPU)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from omegalax.engine.steps import _topk
from omegalax.rl.loss import policy_loss, score_centering, token_logprobs
from omegalax.rl.rollout import group_advantages


class PolicyLossTest(absltest.TestCase):
    def test_clipped_surrogate_by_hand(self):
        # ratios exp(log_ratio): 1.0, 1.5, 0.5, 1.5, 0.5, masked
        logp = jnp.log(jnp.array([[0.5, 0.6, 0.2, 0.6, 0.2, 0.9]]))
        behavior = jnp.log(jnp.array([[0.5, 0.4, 0.4, 0.4, 0.4, 0.1]]))
        adv = jnp.array([[2.0, 2.0, 2.0, -1.0, -1.0, 5.0]])
        mask = jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
        loss, log_ratio, clipped = policy_loss(logp, behavior, adv, mask, eps_low=0.2, eps_high=0.28, objective="ppo")
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
        grad = jax.grad(lambda lp: policy_loss(lp, behavior, adv, mask, 0.2, 0.28, "ppo")[0])(behavior)
        # d/dlogp of -r*A at r=1 is -A.
        np.testing.assert_allclose(np.asarray(grad), [[-2.0, 1.0]], rtol=1e-6)

    def test_group_advantages_match_unbiased_group_std(self):
        r = [1.0, 0.0, 0.0, 1.0]
        mean, std = 0.5, np.std(r, ddof=1)
        np.testing.assert_allclose(group_advantages(r), [(x - mean) / (std + 1e-6) for x in r])
        self.assertEqual(group_advantages([1.0, 1.0, 1.0]), [0.0, 0.0, 0.0])


class ScoreCenteringTest(absltest.TestCase):
    """Expected gradient under the sampler q, at one position, of (policy loss + centering):
    zero with a full-vocabulary head, for both objectives; the uncentered gradient is not."""

    def _expected_grad(self, objective, k, logits_p, logits_q, adv, centered=True):
        V = logits_p.shape[-1]
        logq = jax.nn.log_softmax(logits_q)
        head = jnp.argsort(-logq)[:k]

        def loss(theta, y):
            logp_V = jax.nn.log_softmax(theta)
            shape = lambda x: x.reshape(1, 1, *x.shape)
            pg, _, _ = policy_loss(shape(logp_V[y]), shape(logq[y]), shape(adv), jnp.ones((1, 1)), 0.2, 0.28,
                                   "pg_sc" if objective == "pg_sc" else "ppo")
            if not centered:
                return pg
            return pg + score_centering(shape(logp_V[head]), shape(logq[head]), shape(adv), jnp.ones((1, 1)),
                                        objective, 0.2, 0.28)

        grads = jax.vmap(lambda y: jax.grad(loss)(logits_p, y))(jnp.arange(V))
        return jnp.sum(jnp.exp(logq)[:, None] * grads, 0)

    def test_centered_gradient_has_zero_mean_under_the_sampler(self):
        rng = np.random.default_rng(0)
        V = 40
        logits_q = jnp.asarray(rng.normal(size=V) * 2, jnp.float32)
        logits_p = logits_q + jnp.asarray(rng.normal(size=V) * 0.5, jnp.float32)  # strong mismatch
        for objective in ("pg_sc", "ppo_sc"):
            for adv in (1.0, -1.0):
                adv = jnp.float32(adv)
                exact = self._expected_grad(objective, V, logits_p, logits_q, adv)
                raw = self._expected_grad(objective, V, logits_p, logits_q, adv, centered=False)
                approx = self._expected_grad(objective, 8, logits_p, logits_q, adv)
                print(f"{objective} A={float(adv):+.0f}: |E grad| uncentered {float(jnp.abs(raw).max()):.3e}, "
                      f"centered full head {float(jnp.abs(exact).max()):.1e}, top-8 head {float(jnp.abs(approx).max()):.1e}")
                self.assertLess(float(jnp.abs(exact).max()), 1e-5)
                self.assertGreater(float(jnp.abs(raw).max()), 1e-2)
                self.assertLess(float(jnp.abs(approx).max()), float(jnp.abs(raw).max()))

    def test_no_correction_when_sampler_and_trainer_agree(self):
        logits = jnp.asarray(np.random.default_rng(1).normal(size=30), jnp.float32)
        for objective in ("pg_sc", "ppo_sc"):
            g = self._expected_grad(objective, 30, logits, logits, jnp.float32(1.0))
            self.assertLess(float(jnp.abs(g).max()), 1e-5)


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
            cand = jnp.asarray(rng.integers(0, V, size=(B, T, 3)), jnp.int32)
            logp, entropy, cand_logp = token_logprobs(hidden, head, tokens, tiles, None, cand)
            ref_cand = jnp.take_along_axis(jax.nn.log_softmax(logits[:, :-1]), cand[:, 1:], -1)
            np.testing.assert_allclose(np.asarray(cand_logp[:, 1:]), np.asarray(ref_cand), atol=1e-5)
            np.testing.assert_allclose(np.asarray(logp[:, 1:]), np.asarray(ref), atol=1e-5)
            np.testing.assert_allclose(np.asarray(entropy[:, 1:]), np.asarray(ref_entropy), atol=1e-5)
            np.testing.assert_array_equal(np.asarray(logp[:, 0]), 0.0)


class TrainerTest(absltest.TestCase):
    def test_zero_advantage_samples_only_count_in_the_normalizer(self):
        from omegalax.rl.rollout import Sample
        from omegalax.rl.trainer import TrainConfig, Trainer
        from omegalax.models.sharding_runtime import set_attn_backend
        from omegalax.text import api as text_api

        model, cfg = text_api.init_model("qwen3-smoke", jax.random.key(0), tp_size=1, fsdp_size=1, dp_size=1)
        set_attn_backend(model, "xla")
        trainer = Trainer(model, cfg, TrainConfig(
            lr=1e-3, weight_decay=0.0, adam_b1=0.9, adam_b2=0.98, adam_eps=1e-8, max_grad_norm=1e9,
            eps_low=0.2, eps_high=0.28, tokens_per_microbatch=32, length_buckets=(32, 64),
            logit_tile_tokens=64, pad_id=0, host_optimizer=False, objective="ppo", topk=1, drift_every=0,
        ))
        rng = np.random.default_rng(0)
        samples = []
        for L in (20, 40, 30, 50):
            toks = rng.integers(1, 100, L).tolist()
            mask = [0] * 8 + [1] * (L - 8)
            samples.append(Sample(0, toks, 8, mask, list(rng.normal(-2, 0.5, L)), [0] * L,
                                  heads=[(np.zeros((L - 8, 1), np.int32), np.zeros((L - 8, 1), np.float32))]))
        adv = [1.0, 0.0, -1.0, 0.0]

        # One sample per microbatch: the active-only sum is the full sum minus exact zeros.
        def grads(subset):
            with jax.set_mesh(trainer.mesh):
                acc = trainer._zeros(trainer.params)
                for mb in trainer.microbatches([samples[i] for i in subset], [adv[i] for i in subset]):
                    acc, _ = trainer._accumulate(trainer.params, acc, trainer._put(mb), jnp.float32(1 / 100))
            return jax.tree.leaves(acc)

        full, active = grads([0, 1, 2, 3]), grads([0, 2])
        self.assertGreater(max(float(jnp.abs(g).max()) for g in active), 1e-4)
        for f, a in zip(full, active):
            np.testing.assert_array_equal(np.asarray(f), np.asarray(a))

    def test_drift_is_the_centering_gradient(self):
        """A ppo_sc step's gradient minus its measured drift is the plain ppo gradient."""
        from omegalax.rl.rollout import Sample
        from omegalax.rl.trainer import TrainConfig, Trainer
        from omegalax.models.sharding_runtime import set_attn_backend
        from omegalax.text import api as text_api

        rng = np.random.default_rng(1)
        samples = []
        for L in (20, 40, 30, 50):
            n = L - 8
            cand = np.stack([rng.choice(100, 4, replace=False) for _ in range(n)]).astype(np.int32)
            q = rng.dirichlet(np.ones(5), n)[:, :4] * 0.9
            samples.append(Sample(0, rng.integers(1, 100, L).tolist(), 8, [0] * 8 + [1] * n,
                                  list(rng.normal(-2, 0.5, L)), [0] * L, heads=[(cand, np.log(q).astype(np.float32))]))
        adv = [1.0, -0.5, -1.0, 0.5]
        out = {}
        for objective in ("ppo", "ppo_sc"):
            model, cfg = text_api.init_model("qwen3-smoke", jax.random.key(0), tp_size=1, fsdp_size=1, dp_size=1)
            set_attn_backend(model, "xla")
            trainer = Trainer(model, cfg, TrainConfig(
                lr=1e-3, weight_decay=0.0, adam_b1=0.9, adam_b2=0.98, adam_eps=1e-8, max_grad_norm=1e9,
                eps_low=0.2, eps_high=0.28, tokens_per_microbatch=64, length_buckets=(32, 64),
                logit_tile_tokens=64, pad_id=0, host_optimizer=False, objective=objective, topk=4, drift_every=1,
            ))
            out[objective] = trainer.step(samples, adv)
        ppo, sc = out["ppo"], out["ppo_sc"]
        self.assertGreater(sc["drift/norm"], 1e-6)
        self.assertAlmostEqual(sc["drift/coherence"], 1.0, places=5)
        np.testing.assert_allclose(ppo["drift/norm"], sc["drift/norm"], rtol=1e-5)
        np.testing.assert_allclose(ppo["drift/base_norm"], ppo["train/grad_norm"], rtol=1e-5)
        # Separate programs sum in different orders; a drift that were not the centering gradient
        # would be off by ~|drift| / |base| ~ 0.3.
        np.testing.assert_allclose(sc["drift/base_norm"], ppo["train/grad_norm"], rtol=1e-3)
        self.assertGreater(abs(sc["train/grad_norm"] - ppo["train/grad_norm"]), 1e-6)


class EngineTopkTest(absltest.TestCase):
    def test_blockwise_topk_is_exact(self):
        rng = np.random.default_rng(0)
        for V, k in ((1000, 32), (4099, 7), (300, 1)):
            x = jnp.asarray(rng.normal(size=(5, V)) * 3, jnp.float32)
            ref_v, ref_i = jax.lax.top_k(x, k)
            got_v, got_i = _topk(x, k, block=16)
            np.testing.assert_array_equal(got_v, ref_v)
            np.testing.assert_array_equal(np.sort(got_i, 1), np.sort(ref_i, 1))


if __name__ == "__main__":
    absltest.main()
