"""Environments under a scripted policy: fixed turns give expected observations, rewards, statuses."""

import asyncio
import itertools
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from omegalax.engine import Completion
from omegalax.rl.envs import CODE_TOOL, MathEnv, PythonToolEnv
from omegalax.rl.rollout import RolloutConfig, rollout_group

CALL = "<tool_call>\n<function=code_interpreter>\n<parameter=code>\n{}\n</parameter>\n</function>\n</tool_call>"


class ScriptedEngine:
    """Engine stand-in: session ``i`` of a group plays ``scripts[i]``, one text per turn."""

    def __init__(self, tokenizer, scripts, version=3):
        self.tok, self.scripts, self.version = tokenizer, scripts, version
        self.ids = itertools.count()
        self.turn, self.seen, self.released = {}, {}, []

    def _complete(self, sid):
        text, reason = self.scripts[self.seen[sid]][self.turn[sid]]
        self.turn[sid] += 1
        toks = self.tok(text, add_special_tokens=False)["input_ids"]
        f = Future()
        f.set_result(Completion(sid, toks, [-0.5] * len(toks), [self.version] * len(toks), reason))
        return f

    def generate(self, req):
        if req.session is None:
            out = []
            for i in range(len(req.keys)):
                sid = next(self.ids)
                self.turn[sid], self.seen[sid] = 0, i
                out.append(self._complete(sid))
            return out
        return [self._complete(req.session)]

    def release(self, sid):
        self.released.append(sid)


class EnvTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen3.5-2B", allow_patterns=["*.json", "*.jinja", "*.txt"]))
        cls.pool = ThreadPoolExecutor(2)
        cls.task = {"prompt": [{"role": "user", "content": "What is 6 times 7?"}], "label": "42"}

    def _run(self, env, scripts, max_response=4096):
        engine = ScriptedEngine(self.tok, scripts)
        cfg = RolloutConfig(len(scripts), max_response, 1.0)
        samples = asyncio.run(rollout_group(engine, env, self.task, 0, cfg, np.random.default_rng(0)))
        self.assertCountEqual(engine.released, range(len(scripts)))
        return samples

    def test_math_rewards_and_statuses(self):
        env = MathEnv(self.tok, self.pool, 10.0, thinking=True)
        s = self._run(env, [
            [("thinking</think>\n\nSo the answer is \\boxed{42}.<|im_end|>", "stop")],
            [("thinking</think>\n\n\\boxed{41}<|im_end|>", "stop")],
            [("no closing think tag \\boxed{42}<|im_end|>", "stop")],
            [("still thinking about \\boxed{42}", "length")],
        ])
        self.assertEqual([x.reward for x in s], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual([x.status for x in s], ["stop", "stop", "stop", "length"])
        prompt = self.tok.decode(s[0].tokens[: s[0].prompt_len])
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n"))
        for x in s:
            gen = len(x.tokens) - x.prompt_len
            self.assertEqual(x.mask, [0] * x.prompt_len + [1] * gen)
            self.assertEqual(x.versions, [-1] * x.prompt_len + [3] * gen)

    def test_math_non_thinking_grades_the_last_box(self):
        env = MathEnv(self.tok, self.pool, 10.0, thinking=False)
        s = self._run(env, [
            [("6 * 7 = 42, so \\boxed{42}.<|im_end|>", "stop")],
            [("first \\boxed{42} then \\boxed{41}<|im_end|>", "stop")],
        ])
        self.assertEqual([x.reward for x in s], [1.0, 0.0])
        prompt = self.tok.decode(s[0].tokens[: s[0].prompt_len])
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_python_tool_turns_observations_and_reward(self):
        env = PythonToolEnv(self.tok, self.pool, 10.0, thinking=True, max_turns=4, tool_timeout_s=5.0,
                            tool_memory_bytes=2 << 30, tool_output_chars=2000, max_tool_procs=4)
        s = self._run(env, [
            [("compute it</think>\n\n" + CALL.format("print(6 * 7)") + "<|im_end|>", "stop"),
             ("got 42</think>\n\nThe answer is \\boxed{42}<|im_end|>", "stop")],
            [("try</think>\n\n" + CALL.format("while True:\n    pass") + "<|im_end|>", "stop"),
             ("gave up</think>\n\n\\boxed{0}<|im_end|>", "stop")],
            [("peek</think>\n\n" + CALL.format("import os, sys\nprint(os.listdir(os.path.dirname(sys.prefix)), os.path.exists(os.path.expanduser('~/.ssh')))") + "<|im_end|>", "stop"),
             ("</think>\n\n\\boxed{42}<|im_end|>", "stop")],
            [("net</think>\n\n" + CALL.format("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=2)") + "<|im_end|>", "stop"),
             ("</think>\n\n\\boxed{42}<|im_end|>", "stop")],
        ])
        self.assertEqual([x.status for x in s], ["stop"] * 4)
        self.assertEqual([x.turns for x in s], [2] * 4)
        self.assertEqual([x.reward for x in s], [1.0, 0.0, 1.0, 1.0])
        text = [self.tok.decode(x.tokens[x.prompt_len :]) for x in s]
        self.assertIn("<|im_end|>\n<|im_start|>user\n<tool_response>\n42\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n", text[0])
        self.assertIn("timed out after 5 s", text[1])
        self.assertIn(f"['{os.path.basename(sys.prefix)}'] False", text[2])
        self.assertIn("Network is unreachable", text[3])
        obs = np.asarray(s[0].mask) == 0
        self.assertEqual(int(obs[s[0].prompt_len :].sum()), len(env.observation(["42\n"])))
        rendered = self.tok.apply_chat_template(
            self.task["prompt"], tools=[CODE_TOOL],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        self.assertEqual(self.tok.decode(s[0].tokens[: s[0].prompt_len]), rendered)

    def test_reward_failure_makes_the_sample_missing(self):
        class Broken(MathEnv):
            async def reward(self, task, sample):
                raise TimeoutError("grader timed out")

        s = self._run(Broken(self.tok, self.pool, 10.0, thinking=True), [[("x</think>\n\n\\boxed{42}<|im_end|>", "stop")]])
        self.assertEqual((s[0].status, s[0].reward), ("error", None))
        self.assertIn("grader timed out", s[0].error)


if __name__ == "__main__":
    absltest.main()
