"""Environments: chat-templated prompts, observations as appended tokens, graded rewards."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from concurrent.futures import Executor

from omegalax.rl.math_grade import deepscaler_reward, grade_answer_verl
from omegalax.rl.rollout import Sample


class MathEnv:
    """Single-turn math (rows: ``prompt`` messages, ``label``). Thinking mode grades the last
    ``\\boxed`` after ``</think>`` (Miles ``deepscaler`` RM), non-thinking the last ``\\boxed`` of
    the response (Miles ``math`` RM); the template's thinking switch is always passed explicitly
    (Qwen3.5-0.8B/2B default to off, 4B+ to on). Grading runs in ``executor``: sympy holds the
    GIL and may hang.
    """

    def __init__(self, tokenizer, executor: Executor, grade_timeout_s: float, *, thinking: bool):
        self.tokenizer = tokenizer
        self.executor = executor
        self.grade_timeout_s = grade_timeout_s
        self.thinking = thinking
        self._grade = deepscaler_reward if thinking else grade_answer_verl

    def prompt(self, task: dict) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            task["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking
        )
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    async def step(self, task: dict, turn: list[int], turn_index: int) -> list[int] | None:
        return None

    async def reward(self, task: dict, sample: Sample) -> float:
        response = self.tokenizer.decode(sample.tokens[sample.prompt_len :], skip_special_tokens=True)
        future = self.executor.submit(self._grade, response, task["label"])
        return await asyncio.wait_for(asyncio.wrap_future(future), self.grade_timeout_s)


_TOOL_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        "description": "A tool for executing Python code in a safe sandbox environment.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "The Python code to execute"}},
            "required": ["code"],
        },
    },
}


class PythonToolEnv(MathEnv):
    """Multi-turn math with a Python interpreter (the Miles ``retool_v2`` task) in the Qwen3.5
    tool-call format. A turn with tool calls after its ``</think>`` gets their outputs appended as
    the chat template renders tool messages; a turn without calls ends the episode. Code runs in
    a bubblewrap sandbox: read-only root, /fast (homes, projects) hidden, no network, address-space
    limit; it is checked once at construction so a broken sandbox never becomes tool output.
    """

    def __init__(self, tokenizer, executor: Executor, grade_timeout_s: float, *, thinking: bool, max_turns: int,
                 tool_timeout_s: float, tool_memory_bytes: int, tool_output_chars: int, max_tool_procs: int):
        super().__init__(tokenizer, executor, grade_timeout_s, thinking=thinking)
        self.max_turns = max_turns
        self.tool_timeout_s = tool_timeout_s
        self.tool_output_chars = tool_output_chars
        self._procs = asyncio.Semaphore(max_tool_procs)
        binds = sorted({os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix)})
        self._cmd = [
            "prlimit", f"--as={tool_memory_bytes}", "--",
            "bwrap", "--ro-bind", "/", "/", "--tmpfs", "/fast", "--tmpfs", "/tmp",
            *[a for p in binds for a in ("--ro-bind", p, p)],
            "--proc", "/proc", "--dev", "/dev", "--unshare-all", "--die-with-parent", "--new-session",
            "--chdir", "/tmp", sys.executable, "-I", "-c",
        ]
        check = subprocess.run([*self._cmd, "print(6 * 7)"], capture_output=True, text=True, timeout=60)
        if check.stdout != "42\n":
            raise RuntimeError(f"Python sandbox does not work: {check.stdout!r} {check.stderr!r}")
        self._base = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "x"}]
        self._without = tokenizer.apply_chat_template(self._base, tokenize=False, enable_thinking=thinking)
        if not self._without.endswith("<|im_end|>\n"):
            raise ValueError("Expected the chat template to end an assistant turn with '<|im_end|>\\n'.")

    def prompt(self, task: dict) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            task["prompt"], tools=[CODE_TOOL], tokenize=False, add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    async def step(self, task: dict, turn: list[int], turn_index: int) -> list[int] | None:
        if turn_index + 1 >= self.max_turns:
            return None
        text = self.tokenizer.decode(turn, skip_special_tokens=True).split("</think>")[-1]
        calls = _TOOL_CALL.findall(text)
        if not calls:
            return None
        outputs = [await self._call(name.strip(), dict(_PARAM.findall(body))) for name, body in calls]
        return self.observation(outputs)

    def observation(self, outputs: list[str]) -> list[int]:
        """Tokens from the policy's ``<|im_end|>`` to the start of the next policy turn."""
        tools = [{"role": "tool", "content": o} for o in outputs]
        full = self.tokenizer.apply_chat_template(
            self._base + tools, tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking
        )
        if not full.startswith(self._without):
            raise ValueError("Tool messages changed the rendering of earlier turns.")
        return self.tokenizer(full[len(self._without) - 1 :], add_special_tokens=False)["input_ids"]

    async def _call(self, name: str, params: dict[str, str]) -> str:
        if name != "code_interpreter" or "code" not in params:
            return f"Error: unknown tool '{name}' or missing 'code' parameter."
        return await self.run_python(params["code"])

    async def run_python(self, code: str) -> str:
        async with self._procs:
            proc = await asyncio.create_subprocess_exec(
                *self._cmd, code, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), self.tool_timeout_s)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return f"Error: execution timed out after {self.tool_timeout_s:g} s."
        text = out.decode(errors="replace")
        if len(text) > self.tool_output_chars:
            text = text[: self.tool_output_chars] + "\n... (output truncated)"
        return text if text.strip() else "(no output)"
