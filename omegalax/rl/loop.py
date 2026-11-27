"""Asynchronous GRPO control loop: bounded rollout-ahead, one optimizer step per batch."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import numpy as np

from omegalax.rl.rollout import Env, RolloutConfig, group_advantages, rollout_group, summarize
from omegalax.rl.trainer import Trainer
from omegalax.rl.weights import WeightSync


@dataclasses.dataclass(frozen=True)
class LoopConfig:
    """At most ``(rollout_ahead + 1) * groups_per_step`` groups are outstanding; groups with a
    token sampled more than ``max_staleness`` versions before the training step are dropped."""

    num_steps: int
    groups_per_step: int
    rollout_ahead: int
    max_staleness: int
    rollout: RolloutConfig
    seed: int
    eval_every: int
    eval_concurrency: int


def jsonl_tasks(path: str, seed: int) -> Iterator[dict]:
    """Rows of ``path`` in a fresh seeded permutation per epoch, forever."""
    rows = [json.loads(line) for line in open(path)]
    rng = np.random.default_rng(seed)
    while True:
        for i in rng.permutation(len(rows)):
            yield rows[i]


async def evaluate(engines, env: Env, tasks: list[dict], cfg: RolloutConfig, concurrency: int, rng) -> dict:
    """One sample per task, at most ``concurrency`` in flight, concurrent with training."""
    one_sample = dataclasses.replace(cfg, group_size=1)
    slots = asyncio.Semaphore(concurrency)

    async def one(i, task):
        async with slots:
            return (await rollout_group(engines[i % len(engines)], env, task, -1, one_sample, rng))[0]

    t0 = time.perf_counter()
    samples = await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
    ok = [s for s in samples if s.status != "error"]
    versions = [v for s in ok for v, m in zip(s.versions, s.mask) if m]
    return {
        "eval/reward": float(np.mean([s.reward for s in ok])),
        "eval/samples": len(ok),
        "eval/errors": len(samples) - len(ok),
        "eval/response_len": float(np.mean([len(s.tokens) - s.prompt_len for s in ok])),
        "eval/truncated_frac": float(np.mean([s.status == "length" for s in ok])),
        "eval/version_min": min(versions),
        "eval/version_max": max(versions),
        "eval/seconds": time.perf_counter() - t0,
    }


def _engine_busy_s(engines):
    return sum(e.stats["decode_s"] + e.stats["extend_s"] for e in engines)


async def run(trainer: Trainer, engines, sync: WeightSync, env: Env, tasks: Iterator[dict],
              eval_tasks: list[dict], cfg: LoopConfig, log):
    """Trains ``cfg.num_steps`` steps; ``log`` receives each step's metrics and each evaluation
    (started before step 1 and after every ``eval_every`` steps)."""
    G = cfg.groups_per_step
    capacity = asyncio.Semaphore((cfg.rollout_ahead + 1) * G)
    ready: asyncio.Queue = asyncio.Queue()
    load = [0] * len(engines)
    rng = np.random.default_rng(cfg.seed)
    running = set()
    train_thread = ThreadPoolExecutor(1, thread_name_prefix="trainer")
    loop = asyncio.get_running_loop()

    async def one_group(gid, task, e):
        # Env and reward failures are recorded per sample; anything raised here is fatal and is
        # re-raised by the consumer.
        try:
            result = await rollout_group(engines[e], env, task, gid, cfg.rollout, rng)
        except Exception as exc:
            result = exc
        load[e] -= 1
        await ready.put(result)

    async def produce():
        for gid, task in enumerate(tasks):
            await capacity.acquire()
            e = int(np.argmin(load))
            load[e] += 1
            t = asyncio.create_task(one_group(gid, task, e))
            running.add(t)
            t.add_done_callback(running.discard)

    def train(batch):
        samples = [s for g in batch for s in g]
        advantages = [a for g in batch for a in group_advantages([s.reward for s in g])]
        metrics = trainer.step(samples, advantages)
        metrics.update(sync.push(trainer.params, trainer.version))
        return metrics

    async def eval_at(step):
        metrics = await evaluate(engines, env, eval_tasks, cfg.rollout, cfg.eval_concurrency, rng)
        log({"eval_step": step, **metrics})

    evals = [asyncio.create_task(eval_at(0))]
    # The trainer compiles on its own thread while the first groups are generated.
    warmup = loop.run_in_executor(train_thread, trainer.warmup)
    producer = asyncio.create_task(produce())
    t_start = time.perf_counter()
    dropped = {"error": 0, "stale": 0}
    errors = []
    busy0, gen0, t_prev = _engine_busy_s(engines), 0, t_start
    for step in range(cfg.num_steps):
        t_wait = time.perf_counter()
        batch = []
        while len(batch) < G:
            group = await ready.get()
            if isinstance(group, Exception):
                raise group
            if any(s.status == "error" for s in group):
                dropped["error"] += 1
                errors += [s.error for s in group if s.error][:1]
                capacity.release()
                continue
            oldest = min(v for s in group for v, m in zip(s.versions, s.mask) if m)
            if trainer.version - oldest > cfg.max_staleness:
                dropped["stale"] += 1
                capacity.release()
                continue
            batch.append(group)
        wait_s = time.perf_counter() - t_wait
        samples = [s for g in batch for s in g]
        lags = np.asarray([trainer.version - v for s in samples for v, m in zip(s.versions, s.mask) if m])
        rewards = np.asarray([[s.reward for s in g] for g in batch])
        metrics = await loop.run_in_executor(train_thread, train, batch)
        if step == 0:
            metrics["train/warmup_s"] = await warmup
        for _ in batch:
            capacity.release()
        now = time.perf_counter()
        busy = _engine_busy_s(engines)
        gen = sum(e.stats["decode_tokens"] for e in engines)
        metrics.update(summarize(samples))
        metrics.update({
            "step": step + 1,
            "version": trainer.version,
            "wall_s": now - t_start,
            "step_s": now - t_prev,
            "trainer_wait_s": wait_s,
            "staleness/mean": float(lags.mean()),
            "staleness/max": int(lags.max()),
            "groups/zero_variance": int(np.sum(rewards.std(axis=1) == 0)),
            "groups/dropped_error": dropped["error"],
            "groups/dropped_stale": dropped["stale"],
            "engine/busy_frac": (busy - busy0) / (len(engines) * (now - t_prev)),
            "engine/decode_tokens_per_s": (gen - gen0) / (now - t_prev),
            "groups/outstanding": len(running) + ready.qsize(),
        })
        busy0, gen0, t_prev = busy, gen, now
        log(metrics)
        if (step + 1) % cfg.eval_every == 0:
            evals.append(asyncio.create_task(eval_at(step + 1)))
        if errors:
            print(f"step {step + 1} sample errors (first): {errors[:3]}", flush=True)
            errors.clear()
    producer.cancel()
    for t in list(running):
        t.cancel()
    await asyncio.gather(*evals)
    train_thread.shutdown()
