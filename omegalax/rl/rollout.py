"""Multi-turn trajectories on the rollout engine, built by appending token ids."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any, Protocol

import numpy as np

from omegalax.engine import Engine, Request


class Env(Protocol):
    def prompt(self, task: dict) -> list[int]:
        """Prompt tokens, ending where the policy's first turn starts."""

    async def step(self, task: dict, turn: list[int], turn_index: int) -> list[int] | None:
        """Tokens to append after a policy turn that ended in EOS, or None to end the episode."""

    async def reward(self, task: dict, sample: "Sample") -> float:
        """Raises when the reward cannot be computed; the sample is then missing, not zero."""


@dataclasses.dataclass
class Sample:
    """One trajectory. Per-token lists cover ``tokens``: ``mask`` is 1 on policy-sampled tokens,
    ``logprobs`` their behaviour logprob, ``versions`` the policy version that sampled them
    (``-1`` on prompt and observation tokens).

    ``status``: ``stop`` (the policy or env ended the episode), ``length`` (response budget
    exhausted) or ``error`` (env or reward failure; ``reward`` is None).
    """

    group: int
    tokens: list[int]
    prompt_len: int
    mask: list[int]
    logprobs: list[float]
    versions: list[int]
    turns: int = 0
    status: str = ""
    reward: float | None = None
    error: str = ""

    def append(self, tokens: list[int], logprobs: list[float] | None, versions: list[int] | None):
        self.tokens += tokens
        self.mask += [0 if logprobs is None else 1] * len(tokens)
        self.logprobs += [0.0] * len(tokens) if logprobs is None else logprobs
        self.versions += [-1] * len(tokens) if versions is None else versions


@dataclasses.dataclass(frozen=True)
class RolloutConfig:
    group_size: int
    max_response: int
    temperature: float


async def rollout_group(
    engine: Engine, env: Env, task: dict, group: int, cfg: RolloutConfig, rng: np.random.Generator
) -> list[Sample]:
    """Samples ``cfg.group_size`` trajectories of ``task`` sharing the prompt's KV pages.

    ``cfg.max_response`` bounds every token after the prompt, generated or observed.
    """
    prompt = env.prompt(task)
    n = cfg.group_size
    keys = rng.integers(0, 2**32, size=(n, 2), dtype=np.uint32)
    futures = engine.generate(Request(prompt, keys, [cfg.max_response] * n, [cfg.temperature] * n))
    samples = [Sample(group, list(prompt), len(prompt), [0] * len(prompt), [0.0] * len(prompt), [-1] * len(prompt)) for _ in range(n)]
    await asyncio.gather(
        *(_trajectory(engine, env, task, s, f, k, cfg) for s, f, k in zip(samples, futures, keys))
    )
    return samples


async def _trajectory(engine, env, task, sample: Sample, future, key, cfg: RolloutConfig) -> None:
    c = await asyncio.wrap_future(future)
    try:
        while True:
            sample.append(c.tokens, c.logprobs, c.weight_versions)
            sample.turns += 1
            if c.finish_reason == "length":
                sample.status = "length"
                break
            obs = await env.step(task, c.tokens, sample.turns - 1)
            if obs is None:
                sample.status = "stop"
                break
            budget = cfg.max_response - (len(sample.tokens) - sample.prompt_len) - len(obs)
            if budget < 1:
                sample.status = "length"
                break
            sample.append(obs, None, None)
            req = Request(obs, key[None], [budget], [cfg.temperature], session=c.session)
            c = await asyncio.wrap_future(engine.generate(req)[0])
        sample.reward = float(await env.reward(task, sample))
    except Exception as exc:  # noqa: BLE001 - env and reward failures make the sample missing
        sample.status, sample.reward, sample.error = "error", None, f"{type(exc).__name__}: {exc}"
    # Not in a finally: a cancelled trajectory may still be generating, and only shutdown cancels.
    engine.release(c.session)


def group_advantages(rewards: list[float]) -> list[float]:
    """GRPO: reward minus the group mean, over the unbiased group std (+1e-6) when it is > 0."""
    r = np.asarray(rewards, np.float64)
    a = r - r.mean()
    std = r.std(ddof=1) if len(r) > 1 else 0.0
    return list(a / (std + 1e-6) if std > 0 else a)


def summarize(samples: list[Sample]) -> dict[str, Any]:
    ok = [s for s in samples if s.status != "error"]
    resp = [len(s.tokens) - s.prompt_len for s in ok]
    gen = [sum(s.mask) for s in ok]
    return {
        "rollout/samples": len(samples),
        "rollout/errors": len(samples) - len(ok),
        "rollout/reward": float(np.mean([s.reward for s in ok])) if ok else float("nan"),
        "rollout/truncated_frac": float(np.mean([s.status == "length" for s in ok])) if ok else 0.0,
        "rollout/response_len": float(np.mean(resp)) if ok else 0.0,
        "rollout/generated_tokens": int(sum(gen)),
        "rollout/turns": float(np.mean([s.turns for s in ok])) if ok else 0.0,
    }
