"""RL oracles on GPU with real weights: trainer recompute vs engine logprobs, weight sync."""

import dataclasses
import logging
import os

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")  # cpu: the host-optimizer placement

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from omegalax.engine import Engine, EngineConfig, Request
from omegalax.rl.rollout import Sample
from omegalax.rl.trainer import TrainConfig, Trainer
from omegalax.rl.weights import WeightSync, engine_replica, load_causal_lm, sub_mesh
from tests.real_weights import requires_real_weights

# TOLERANCES.md (1): engine vs trainer forward, fixed before any engine optimization.
LOGP_MEAN, LOGP_P99, LOGP_P999 = 0.03, 0.30, 0.70
PROMPT = "Solve: what is the sum of the first 50 odd numbers? Think step by step."


class _CompileLog(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.compiles = []

    def emit(self, record):
        if "Compiling" in record.getMessage():
            self.compiles.append(record.getMessage()[:200])


@requires_real_weights
class RLOracleTest(absltest.TestCase):
    """Trainer on an FSDP mesh of two devices, one engine on a third."""

    MODEL_ID = "Qwen/Qwen3.5-0.8B"

    @classmethod
    def setUpClass(cls):
        devices = jax.devices()
        assert len(devices) >= 3, "needs 3 devices"
        path = snapshot_download(cls.MODEL_ID)
        cls.tok = AutoTokenizer.from_pretrained(path)
        eos = cls.tok.convert_tokens_to_ids("<|im_end|>")
        cls.pad = cls.tok.convert_tokens_to_ids("<|endoftext|>")
        model, cfg = load_causal_lm(path, sub_mesh(devices[:2], 1, 2, 1), jnp.bfloat16)
        cls.trainer = Trainer(
            model,
            cfg,
            TrainConfig(
                lr=1e-4, weight_decay=0.1, adam_b1=0.9, adam_b2=0.98, adam_eps=1e-8,
                max_grad_norm=1.0, eps_low=0.2, eps_high=0.28, tokens_per_microbatch=2048,
                length_buckets=(1024, 2048, 4096), logit_tile_tokens=4096, pad_id=cls.pad, host_optimizer=False, loss_scale=1.0,
            ),
        )
        cls.sync = WeightSync(cls.trainer.graphdef, cls.trainer.params)
        replica, shardings = engine_replica(cls.sync.gather(cls.trainer.params), cfg, sub_mesh(devices[2:3], 1, 1, 1))
        cls.engine = Engine(
            replica,
            EngineConfig(
                num_pages=256, max_context=4096, decode_steps=8, decode_buckets=(4, 8),
                extend_buckets=(512, 1024, 2048), eos_token_ids=(eos, cls.pad), pad_id=cls.pad,
                max_group=4, max_sessions=8,
            ),
        )
        del replica, model
        cls.engine.start()
        cls.shardings = shardings
        cls.sync.add(cls.engine, shardings)

    @classmethod
    def tearDownClass(cls):
        cls.engine.stop()

    def _rollouts(self, seed, turns=2):
        """Four T=1 trajectories sharing a prompt, each continued with tool-style observations."""
        rng = np.random.default_rng(seed)
        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False
        )
        prompt = self.tok(prompt, add_special_tokens=False)["input_ids"]
        keys = rng.integers(0, 2**32, size=(4, 2), dtype=np.uint32)
        futures = self.engine.generate(Request(prompt, keys, [300, 200, 250, 150], [1.0] * 4, stop_at_eos=False))
        samples = []
        for f, key in zip(futures, keys):
            c = f.result()
            s = Sample(0, list(prompt), len(prompt), [0] * len(prompt), [0.0] * len(prompt), [-1] * len(prompt))
            s.append(c.tokens, c.logprobs, c.weight_versions)
            for t in range(turns):
                obs = self.tok(
                    f"<|im_end|>\n<|im_start|>user\n<tool_response>\nresult {t}: "
                    + "ok " * int(rng.integers(10, 100))
                    + "</tool_response><|im_end|>\n<|im_start|>assistant\n",
                    add_special_tokens=False,
                )["input_ids"]
                s.append(obs, None, None)
                c = self.engine.generate(Request(obs, key[None], [int(rng.integers(50, 200))], [1.0], session=c.session, stop_at_eos=False))[0].result()
                s.append(c.tokens, c.logprobs, c.weight_versions)
            self.engine.release(c.session)
            samples.append(s)
        return samples

    def _errors(self, samples):
        ref = self.trainer.logprobs([s.tokens for s in samples])
        return np.concatenate(
            [np.abs(np.asarray(s.logprobs) - r)[np.asarray(s.mask) == 1] for s, r in zip(samples, ref)]
        )

    def _assert_close(self, errs, what):
        mean, p99, p999 = errs.mean(), np.quantile(errs, 0.99), np.quantile(errs, 0.999)
        print(f"{what}: |engine - trainer| over {errs.size} tokens: mean {mean:.4f} p99 {p99:.4f} "
              f"p99.9 {p999:.4f} max {errs.max():.4f}")
        self.assertLess(mean, LOGP_MEAN)
        self.assertLess(p99, LOGP_P99)
        self.assertLess(p999, LOGP_P999)

    def test_1_engine_logprobs_match_trainer_recompute(self):
        self._assert_close(self._errors(self._rollouts(0)), "oracle 1")

    def test_2_weight_sync_no_compile_and_matching_logprobs(self):
        old = jax.tree.map(lambda x: x.copy(), self.sync.gather(self.trainer.params))
        samples = self._rollouts(1)
        adv = [1.0, -1.0, 0.5, -0.5]
        print("trainer warmup s", self.trainer.warmup())
        counter = _CompileLog()
        logging.getLogger("jax").addHandler(counter)
        jax.config.update("jax_log_compiles", True)
        try:
            first = self.trainer.step(samples, adv)
            first_compiles = list(counter.compiles)
            print("warmup push", self.sync.push(self.trainer.params, self.trainer.version))
            counter.compiles.clear()
            metrics = self.trainer.step(samples, adv)
            push = self.sync.push(self.trainer.params, self.trainer.version)
            fresh = self._rollouts(2)
        finally:
            jax.config.update("jax_log_compiles", False)
            logging.getLogger("jax").removeHandler(counter)
        print("step", {k: round(v, 4) for k, v in metrics.items()}, "push", push)
        self.assertGreater(first["train/microbatches"], 1)
        # After warmup the first step may only load the AOT-compiled update.
        self.assertTrue(all("_apply_fn" in c for c in first_compiles), first_compiles)
        self.assertEqual(counter.compiles, [])
        self.assertTrue(all(v == 2 for s in fresh for v, m in zip(s.versions, s.mask) if m))
        new_errs = self._errors(fresh)
        self._assert_close(new_errs, "oracle 2 after sync")
        self.engine.update_weights(jax.device_put(old, self.shardings), version=0).result()
        # score() rejects pad, which these stop_at_eos=False rollouts can sample anywhere: score the prefix before it.
        scored = [s.tokens[: s.tokens.index(self.pad)] if self.pad in s.tokens else s.tokens for s in fresh]
        old_ref = self.engine.score(scored).result()
        old_errs = np.concatenate([
            np.abs(np.asarray(s.logprobs)[1 : len(t)] - r)[np.asarray(s.mask)[1 : len(t)] == 1]
            for s, t, r in zip(fresh, scored, old_ref)
        ])
        print(f"same tokens under the pre-step weights: mean |dlogp| {old_errs.mean():.4f}")
        self.assertGreater(old_errs.mean(), 5 * new_errs.mean())
        self.sync.push(self.trainer.params, self.trainer.version)

    def test_3_host_optimizer_matches_device_optimizer(self):
        path = snapshot_download(self.MODEL_ID)
        rng = np.random.default_rng(3)
        samples = []
        for L in (300, 700, 500, 900):
            toks = rng.integers(10, 50000, L).tolist()
            samples.append(Sample(0, toks, 50, [0] * 50 + [1] * (L - 50), list(rng.normal(-2, 0.5, L)), [0] * L))
        adv = [1.0, -1.0, 0.5, -0.5]
        states = []
        for host in (False, True):
            model, cfg = load_causal_lm(path, sub_mesh(jax.devices()[:2], 1, 2, 1), jnp.bfloat16)
            trainer = Trainer(model, cfg, dataclasses.replace(self.trainer.cfg, host_optimizer=host))
            del model
            trainer.warmup()
            m = trainer.step(samples, adv)
            print(f"host_optimizer={host}: step {m['train/seconds']:.2f} s, grad norm {m['train/grad_norm']:.6f}")
            states.append(jax.device_get((trainer.params, trainer.opt_state)))
            del trainer
        # One step from identical params sees identical gradients, so params and Adam moments may differ
        # only by CPU vs GPU float32 rounding. A second step would compare gradients of last-bit-different
        # params, which the bfloat16 forward amplifies into ~1 % gradient noise.
        lr = self.trainer.cfg.lr
        diff = max(float(np.max(np.abs(a - b))) for a, b in zip(jax.tree.leaves(states[0][0]), jax.tree.leaves(states[1][0])))
        print(f"device vs host optimizer after one step: max |dp| {diff:.2e} ({diff / lr:.1e} lr)")
        for a, b in zip(jax.tree.leaves(states[0][0]), jax.tree.leaves(states[1][0])):
            np.testing.assert_allclose(a, b, rtol=5e-7, atol=1e-3 * lr)
        # The GPU flushes float32 subnormals (second moments of ~1e-20 gradients) to zero; the CPU keeps them.
        for a, b in zip(jax.tree.leaves(states[0][1]), jax.tree.leaves(states[1][1])):
            np.testing.assert_allclose(a, b, rtol=5e-7, atol=1e-37)


if __name__ == "__main__":
    absltest.main()
