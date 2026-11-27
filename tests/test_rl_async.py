"""Async invariants of the control loop with in-flight weight updates (GPU, real weights).

Every policy token carries the version that sampled it; versions never decrease within a
trajectory; trained tokens respect the staleness bound; a trajectory sampled under one version has
behaviour logprobs equal to that version's trainer recompute; in-flight updates actually occur.
"""

import asyncio
import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import numpy as np
from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from omegalax.engine import Engine, EngineConfig
from omegalax.rl import loop
from omegalax.rl.rollout import RolloutConfig
from omegalax.rl.trainer import TrainConfig, Trainer
from omegalax.rl.weights import WeightSync, engine_replica, load_causal_lm, sub_mesh
from tests.real_weights import requires_real_weights

LOGP_MEAN, LOGP_P99 = 0.03, 0.30
MAX_STALENESS = 2


class ParityEnv:
    """Reward 1 iff the last generated token id is even: random, so every group has variance."""

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def prompt(self, task):
        text = self.tok.apply_chat_template(task["prompt"], tokenize=False, add_generation_prompt=True)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    async def step(self, task, turn, turn_index):
        return None

    async def reward(self, task, sample):
        return float(sample.tokens[-1] % 2 == 0)


class RecordingTrainer(Trainer):
    """Keeps every trained batch and a host copy of the params of every version."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches, self.snapshots = [], {0: jax.device_get(self.params)}

    def step(self, samples, advantages):
        self.batches.append((self.version, list(samples)))
        metrics = super().step(samples, advantages)
        self.snapshots[self.version] = jax.device_get(self.params)
        return metrics


@requires_real_weights
class AsyncInvariantsTest(absltest.TestCase):
    def test_versions_staleness_and_behaviour_logprobs(self):
        devices = jax.devices()
        path = snapshot_download("Qwen/Qwen3.5-0.8B")
        tok = AutoTokenizer.from_pretrained(path)
        eos, pad = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")
        model, cfg = load_causal_lm(path, sub_mesh(devices[:1], 1, 1, 1))
        # A large lr makes consecutive versions' logprobs clearly distinguishable.
        trainer = RecordingTrainer(model, cfg, TrainConfig(
            lr=3e-5, weight_decay=0.0, adam_b1=0.9, adam_b2=0.98, adam_eps=1e-8, max_grad_norm=1.0,
            eps_low=0.2, eps_high=0.28, tokens_per_microbatch=16384, length_buckets=(512, 1024),
            logit_tile_tokens=4096, pad_id=pad, host_optimizer=False,
        ))
        sync = WeightSync(trainer.graphdef, trainer.params)
        replica, shardings = engine_replica(sync.gather(trainer.params), cfg, sub_mesh(devices[1:2], 1, 1, 1))
        engine = Engine(replica, EngineConfig(
            num_pages=512, max_context=1024, decode_steps=8, decode_buckets=(8, 16, 32),
            extend_buckets=(512, 1024), eos_token_ids=(eos, pad), pad_id=pad, max_group=4, max_sessions=32,
        ))
        del replica, model
        engine.start()
        sync.add(engine, shardings)
        prompts = ["Count from one to fifty in words.", "Write a long poem about rivers.",
                   "Explain recursion with three examples.", "List twenty animals and a fact each."]
        tasks = ({"prompt": [{"role": "user", "content": p}]} for _ in iter(int, 1) for p in prompts)
        cfg_loop = loop.LoopConfig(num_steps=4, groups_per_step=2, rollout_ahead=2,
                                   max_staleness=MAX_STALENESS, rollout=RolloutConfig(4, 400, 1.0), seed=0,
                                   eval_every=2, eval_concurrency=4, drop_zero_variance=False,
                                   dump_every=1000)
        metrics = []
        try:
            asyncio.run(loop.run(trainer, [engine], sync, ParityEnv(tok), tasks, [{"prompt": [{"role": "user", "content": p}]} for p in prompts], cfg_loop, metrics.append, None))
        finally:
            engine.stop()

        spanning, single = 0, []
        for train_version, samples in trainer.batches:
            for s in samples:
                v = np.asarray(s.versions)[np.asarray(s.mask) == 1]
                self.assertTrue(np.all(v >= 0) and np.all(v <= train_version))
                self.assertTrue(np.all(np.diff(v) >= 0), "versions decrease within a trajectory")
                self.assertLessEqual(train_version - v.min(), MAX_STALENESS)
                spanning += int(v.min() != v.max())
                if v.min() == v.max():
                    single.append((int(v[0]), s))
        print(f"{len(single)} single-version and {spanning} version-spanning trajectories;",
              "staleness max per step:", [m["staleness/max"] for m in metrics if "step" in m])
        self.assertGreater(spanning, 0, "no trajectory saw an in-flight weight update")

        own, other = [], []
        live = trainer.params
        for version in sorted({v for v, _ in single}):
            group = [s for v, s in single if v == version]
            for scored, errs in ((version, own), ((version + 1) % len(trainer.snapshots), other)):
                trainer.params = jax.device_put(trainer.snapshots[scored], jax.tree.map(lambda x: x.sharding, live))
                ref = trainer.logprobs([s.tokens for s in group])
                errs += [np.abs(np.asarray(s.logprobs) - r)[np.asarray(s.mask) == 1] for s, r in zip(group, ref)]
        trainer.params = live
        own, other = np.concatenate(own), np.concatenate(other)
        print(f"behaviour vs own-version recompute: mean {own.mean():.4f} p99 {np.quantile(own, 0.99):.4f}; "
              f"vs another version: mean {other.mean():.4f}")
        self.assertLess(own.mean(), LOGP_MEAN)
        self.assertLess(np.quantile(own, 0.99), LOGP_P99)
        self.assertGreater(other.mean(), 3 * own.mean())


if __name__ == "__main__":
    absltest.main()
