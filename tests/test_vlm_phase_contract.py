"""CPU tests for VLM phase binding and cleanup ownership."""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
from absl.testing import absltest

from omegalax.trainers import vlm


class _Resource:
    def __init__(self, *, wait_error=None, close_error=None):
        self.wait_error = wait_error
        self.close_error = close_error
        self.waits = 0
        self.closes = 0

    def wait_until_finished(self):
        self.waits += 1
        if self.wait_error is not None:
            raise self.wait_error

    def close(self):
        self.closes += 1
        if self.close_error is not None:
            raise self.close_error


class _Context:
    def __init__(self, error=None):
        self.error = error
        self.exit_args = None

    def __exit__(self, *args):
        self.exit_args = args
        if self.error is not None:
            raise self.error


class VLMPhaseContractTest(absltest.TestCase):
    def test_phase_fields_are_strict_and_parent_extension_is_exact(self):
        cfg = vlm.TrainConfig(schedule_horizon=20)
        vlm._validate_training_phase(cfg, 10)
        for horizon, end in ((True, 1), (0, 1), (20, True), (20, 0), (20, 21)):
            with self.subTest(horizon=horizon, end=end), self.assertRaises(ValueError):
                vlm._validate_training_phase(
                    vlm.TrainConfig(schedule_horizon=horizon),
                    end,
                )

        expected = {
            "version": 2,
            "optimizer": [],
            "phase": {"schedule_horizon": 20, "invocation_end_step": 20},
            "input_iter": {},
        }
        parent = dict(expected)
        parent["phase"] = {"schedule_horizon": 20, "invocation_end_step": 10}
        vlm._validate_checkpoint_phase(parent, expected, 10, 20)
        vlm._validate_checkpoint_phase(parent, parent, 7, 10)
        for step, end in ((9, 20), (10, 9), (7, 20)):
            with self.subTest(step=step, end=end), self.assertRaises(ValueError):
                vlm._validate_checkpoint_phase(parent, expected, step, end)
        corrupt = dict(parent)
        corrupt["phase"] = {"schedule_horizon": True, "invocation_end_step": 10}
        with self.assertRaises(ValueError):
            vlm._validate_checkpoint_phase(corrupt, expected, 10, 20)

    def test_cleanup_preserves_typed_fatal_and_drains_every_resource(self):
        manager = _Resource(
            wait_error=OSError("wait failed"),
            close_error=RuntimeError("close failed"),
        )
        context = _Context(ValueError("exit failed"))
        cleanup = vlm._TrainingCleanup(manager, context)
        fatal = FloatingPointError("invalid_candidate_state")

        cleanup.close(fatal, (FloatingPointError, fatal, None))

        self.assertEqual(manager.waits, 1)
        self.assertEqual(manager.closes, 1)
        self.assertIs(context.exit_args[1], fatal)
        self.assertLen(fatal.__notes__, 3)

    def test_cleanup_failure_without_active_error_is_typed(self):
        cleanup = vlm._TrainingCleanup(_Resource(close_error=OSError("close failed")), None)
        with self.assertRaisesRegex(OSError, "close failed"):
            cleanup.close(None, (None, None, None))

    def test_cleanup_restores_prior_signal_handlers_in_reverse_order(self):
        cleanup = vlm._TrainingCleanup()
        usr_handler = object()
        term_handler = object()
        with mock.patch.object(
            vlm.signal,
            "signal",
            side_effect=[usr_handler, term_handler, None, None],
        ) as set_handler:
            cleanup.install_signal_handler(vlm.signal.SIGUSR1, object())
            cleanup.install_signal_handler(vlm.signal.SIGTERM, object())
            cleanup.close(None, (None, None, None))

        self.assertEqual(set_handler.call_args_list[-2].args, (vlm.signal.SIGTERM, term_handler))
        self.assertEqual(set_handler.call_args_list[-1].args, (vlm.signal.SIGUSR1, usr_handler))

    def test_checkpoint_rng_is_absent_and_fused_path_has_no_host_sync(self):
        checkpoint_source = "\n".join(
            inspect.getsource(function)
            for function in (
                vlm._train_state,
                vlm._abstract_train_state,
                vlm._sft_checkpoint_schema,
                vlm._restore_sft_checkpoint,
                vlm._commit_sft_checkpoint,
            )
        )
        self.assertNotIn('"rng"', checkpoint_source)
        update_source = inspect.getsource(vlm.apply_normalized_gradient_sum)
        self.assertNotIn("device_get", update_source)
        self.assertNotIn("block_until_ready", update_source)

    def test_every_supported_training_forward_is_stochastic_free(self):
        root = Path(vlm.__file__).parents[2]
        paths = [
            root / "omegalax/vlm/api.py",
            root / "omegalax/trainers/lora.py",
            *sorted((root / "omegalax/models/qwen3_vl").glob("*.py")),
            *sorted((root / "omegalax/models/qwen3_5").glob("*.py")),
        ]
        violations = []
        for path in paths:
            source = path.read_text()
            tree = ast.parse(source, filename=str(path))
            lines = source.splitlines()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
                    "forward",
                    "__call__",
                }:
                    body = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                    if any(token in body for token in ("jax.random", "nnx.Dropout", "rngs")):
                        violations.append(f"{path.relative_to(root)}:{node.lineno}")
        self.assertEmpty(violations)

    def test_phase_preflight_is_identical_under_python_optimization(self):
        code = """
from omegalax.trainers import vlm
cases = [(True, 1), (0, 1), (20, True), (20, 0), (20, 21)]
for horizon, end in cases:
    try:
        vlm._validate_training_phase(vlm.TrainConfig(schedule_horizon=horizon), end)
    except ValueError:
        continue
    raise SystemExit(f'accepted invalid phase: {horizon!r}, {end!r}')
"""
        env = dict(os.environ)
        env["JAX_PLATFORMS"] = "cpu"
        for optimized in (False, True):
            command = [sys.executable]
            if optimized:
                command.append("-O")
            command.extend(["-c", code])
            with self.subTest(optimized=optimized):
                subprocess.run(command, env=env, check=True, timeout=120)

    def test_fatal_phase_end_never_attempts_save(self):
        class _NoSave:
            def save(self, *args, **kwargs):
                raise AssertionError("fatal phase attempted final save")

        with self.assertRaisesRegex(FloatingPointError, "invalid_candidate_state"):
            vlm._commit_phase_end(
                _NoSave(),
                object(),
                object(),
                20,
                13,
                10,
                jnp.asarray(
                    vlm.OptimizerFatalStatus.INVALID_CANDIDATE_STATE,
                    dtype=jnp.uint8,
                ),
            )


if __name__ == "__main__":
    absltest.main()
