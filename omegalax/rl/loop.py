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
    eval_samples: int
    eval_concurrency: int
    dump_every: int


def jsonl_tasks(path: str, seed: int) -> Iterator[dict]:
    """Rows of ``path`` in a fresh seeded permutation per epoch, forever."""
    rows = [json.loads(line) for line in open(path)]
    rng = np.random.default_rng(seed)
    while True:
        for i in rng.permutation(len(rows)):
            yield rows[i]


async def evaluate(engines, env: Env, tasks: list[dict], cfg: RolloutConfig, samples_per_task: int,
                   slots: list[asyncio.Semaphore], rng) -> dict:
    """``samples_per_task`` samples per task (one group), concurrent with training: task ``i`` runs on
    engine ``i % len(engines)`` under a permit of that engine's ``slots[i % len(engines)]``. Reports
    the mean reward and its standard error over tasks."""
    group = dataclasses.replace(cfg, group_size=samples_per_task)

    async def one(i, task):
        async with slots[i % len(engines)]:
            return await rollout_group(engines[i % len(engines)], env, task, -1, group, rng)

    t0 = time.perf_counter()
    groups = await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
    samples = [s for g in groups for s in g]
    ok = [s for s in samples if s.status != "error"]
    per_task = [np.mean([s.reward for s in g if s.status != "error"]) for g in groups if any(s.status != "error" for s in g)]
    versions = [v for s in ok for v, m in zip(s.versions, s.mask) if m]
    return {
        "eval/reward": float(np.mean([s.reward for s in ok])),
        "eval/reward_se": float(np.std(per_task, ddof=1) / np.sqrt(len(per_task))),
        "eval/samples": len(ok),
        "eval/errors": len(samples) - len(ok),
        "eval/response_len": float(np.mean([len(s.tokens) - s.prompt_len for s in ok])),
        "eval/truncated_frac": float(np.mean([s.status == "length" for s in ok])),
        "eval/version_min": min(versions),
        "eval/version_max": max(versions),
        "eval/seconds": time.perf_counter() - t0,
    }


def _engine_totals(engines) -> np.ndarray:
    keys = ("decode_s", "extend_s", "decode_tokens", "decode_row_steps", "extend_tokens",
            "evict_idle", "evict_waiting", "evicted_tokens")
    return np.asarray([sum(e.stats[k] for e in engines) for k in keys], np.float64)


async def run(trainer: Trainer, engines, sync: WeightSync, env: Env, tasks: Iterator[dict],
              eval_tasks: list[dict], cfg: LoopConfig, log, dump):
    """Trains ``cfg.num_steps`` steps; ``log`` receives each step's metrics and each evaluation
    (started before step 1 and after every ``eval_every`` steps), ``dump`` the trained samples of
    every ``dump_every``-th step."""
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

    # Evaluations can outlast eval_every steps; one budget per engine for all of them keeps every
    # engine's session count within the bound scripts/train_rl.py checks against max_sessions.
    eval_slots = [asyncio.Semaphore(cfg.eval_concurrency // cfg.eval_samples // len(engines)) for _ in engines]

    async def eval_at(step):
        start = time.perf_counter() - t_start
        metrics = await evaluate(engines, env, eval_tasks, cfg.rollout, cfg.eval_samples, eval_slots, rng)
        log({"eval_step": step, "eval/start_wall_s": start, **metrics})

    t_start = time.perf_counter()
    evals = [asyncio.create_task(eval_at(0))]
    # The trainer compiles on its own thread while the first groups are generated.
    warmup = loop.run_in_executor(train_thread, trainer.warmup)
    producer = asyncio.create_task(produce())
    dropped = {"error": 0, "stale": 0}
    errors = []
    totals0, t_prev = _engine_totals(engines), t_start
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
        if (step + 1) % cfg.dump_every == 0:
            dump(step + 1, samples)
        metrics = await loop.run_in_executor(train_thread, train, batch)
        if step == 0:
            metrics["train/warmup_s"] = await warmup
        for _ in batch:
            capacity.release()
        now = time.perf_counter()
        totals = _engine_totals(engines)
        decode_s, extend_s, decode_tokens, row_steps, extend_tokens, *_ = totals - totals0
        evictions = totals[5:]
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
            "engine/busy_frac": (decode_s + extend_s) / (len(engines) * (now - t_prev)),
            "engine/extend_frac": extend_s / max(decode_s + extend_s, 1e-9),
            "engine/decode_tokens_per_s": decode_tokens / (now - t_prev),
            "engine/decode_occupancy": decode_tokens / max(row_steps, 1.0),
            "engine/extend_tokens_per_s": extend_tokens / (now - t_prev),
            "engine/evict_idle": int(evictions[0]),
            "engine/evict_waiting": int(evictions[1]),
            "engine/evicted_tokens": int(evictions[2]),
            "groups/outstanding": len(running) + ready.qsize(),
        })
        totals0, t_prev = totals, now
        log(metrics)
        if (step + 1) % cfg.eval_every == 0:
            evals.append(asyncio.create_task(eval_at(step + 1)))
        if errors:
            print(f"step {step + 1} sample errors (first): {errors[:3]}", flush=True)
            errors.clear()
    producer.cancel()
    await asyncio.gather(*evals)
    for t in list(running):
        t.cancel()
    train_thread.shutdown()
