"""CPU tests for VLM phase binding and cleanup ownership."""

from __future__ import annotations

import ast
import contextlib
import inspect
import os
import subprocess
import sys
import tempfile
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


class _CountingIterator:
    def __init__(self):
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        return {
            "token_ids_BT": jnp.ones((1, 4), dtype=jnp.int32),
            "attention_mask_BT": jnp.ones((1, 4), dtype=jnp.int32),
            "loss_mask_BT": jnp.ones((1, 4), dtype=jnp.int32),
        }


class _LoopManager(_Resource):
    def __init__(self):
        super().__init__(
            wait_error=KeyboardInterrupt("wait failed"),
            close_error=SystemExit("close failed"),
        )
        self.saves = []
        self.frontier = 0

    def save(self, step, *, args, force):
        self.saves.append((step, args, force))
        self.frontier = step
        return True

    def latest_step(self):
        return self.frontier


class VLMPhaseContractTest(absltest.TestCase):
    def test_phase_fields_are_strict_and_extension_requires_registrar_authority(self):
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
        with self.assertRaisesRegex(PermissionError, "registrar-authorized"):
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
            wait_error=KeyboardInterrupt("wait failed"),
            close_error=SystemExit("close failed"),
        )
        cleanup = vlm._TrainingCleanup(manager)
        fatal = FloatingPointError("invalid_candidate_state")

        cleanup.close(fatal)

        self.assertEqual(manager.waits, 1)
        self.assertEqual(manager.closes, 1)
        self.assertLen(fatal.__notes__, 2)

    def test_cleanup_failure_without_active_error_is_typed(self):
        cleanup = vlm._TrainingCleanup(_Resource(close_error=OSError("close failed")))
        with self.assertRaisesRegex(OSError, "close failed"):
            cleanup.close(None)

    def test_public_cleanup_owner_preserves_primary_fatal(self):
        manager = _Resource()
        fatal = FloatingPointError("invalid_candidate_state")

        def fail(*args, _cleanup, **kwargs):
            del args, kwargs
            _cleanup.checkpoint_manager = manager
            raise fatal

        with (
            mock.patch.object(
                vlm,
                "_require_registrar_compiled_executable_capability",
            ),
            mock.patch.object(vlm, "_run_sft", side_effect=fail),
            self.assertRaises(FloatingPointError) as raised,
        ):
            vlm.run_sft(
                object(),
                vlm.TrainConfig(schedule_horizon=1),
                object(),
                invocation_end_step=1,
            )

        self.assertIs(raised.exception, fatal)
        self.assertEqual(manager.waits, 1)
        self.assertEqual(manager.closes, 1)

    def test_run_loop_fatal_boundaries_never_save_and_cleanup_once(self):
        mesh = vlm.ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        model = object()
        optimizer = mock.Mock(model=model)
        gradient = object()
        metrics = {
            "ce_loss_sum": jnp.asarray(1.0, dtype=jnp.float32),
            "aux_loss": jnp.asarray(0.0, dtype=jnp.float32),
            "supervised_tokens": jnp.asarray(1.0, dtype=jnp.float32),
            "total_tokens": jnp.asarray(4.0, dtype=jnp.float32),
        }
        fatal_status = jnp.asarray(
            vlm.OptimizerFatalStatus.INVALID_GRADIENT,
            dtype=jnp.uint8,
        )
        cases = (
            ("log", 2, 1, None),
            ("validation", 1, 1, 1),
            ("checkpoint", 1, 0, None),
        )

        for boundary, invocation_end, log_every, val_every in cases:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as tmpdir:
                manager = _LoopManager()
                train_iterator = _CountingIterator()
                val_iterator = _CountingIterator() if val_every is not None else None
                save_every = 2 if boundary == "log" else 1
                patchers = (
                    mock.patch.object(vlm, "_make_checkpoint_manager", return_value=manager),
                    mock.patch.object(
                        vlm,
                        "_require_registrar_compiled_executable_capability",
                    ),
                    mock.patch.object(vlm, "_write_checkpoint_config"),
                    mock.patch.object(vlm, "_write_lora_metadata"),
                    mock.patch.object(vlm.vlm_api, "resolve_config", return_value=object()),
                    mock.patch.object(vlm, "require_zero_router_aux_loss"),
                    mock.patch.object(vlm, "ensure_mesh", return_value=mesh),
                    mock.patch.object(
                        vlm.vlm_api, "align_config_to_mesh", side_effect=lambda cfg, _: cfg
                    ),
                    mock.patch.object(vlm.vlm_api, "batch_partition_spec", return_value=vlm.P()),
                    mock.patch.object(vlm, "required_batch_multiple", return_value=1),
                    mock.patch.object(vlm.vlm_api, "init_model", return_value=(model, object())),
                    mock.patch(
                        "omegalax.models.sharding_runtime.set_attn_backend",
                    ),
                    mock.patch.object(vlm, "record_deltanet_kernel", return_value=None),
                    mock.patch.object(vlm, "build_optimizer", return_value=optimizer),
                    mock.patch.object(
                        vlm,
                        "make_sft_gradient_step",
                        return_value=lambda _model, _batch: (gradient, metrics),
                    ),
                    mock.patch.object(
                        vlm,
                        "make_sft_eval_step",
                        return_value=lambda _model, _batch: (
                            jnp.asarray(1.0),
                            jnp.asarray(1.0),
                        ),
                    ),
                    mock.patch.object(
                        vlm.vlm_api, "shard_batch_dict", side_effect=lambda batch, *_: batch
                    ),
                    mock.patch.object(
                        vlm, "per_device_step_flops", return_value=vlm.StepFlops(0.0, 0.0)
                    ),
                    mock.patch.object(vlm, "initialize_gradient_sum", return_value=gradient),
                    mock.patch.object(
                        vlm,
                        "apply_normalized_gradient_sum",
                        return_value=(fatal_status, jnp.asarray(1.0, dtype=jnp.float32)),
                    ),
                    mock.patch.object(vlm, "startup_log"),
                )
                with contextlib.ExitStack() as stack:
                    for patcher in patchers:
                        stack.enter_context(patcher)
                    with self.assertRaisesRegex(
                        FloatingPointError, f"{boundary}: invalid_gradient"
                    ) as raised:
                        vlm.run_sft(
                            object(),
                            vlm.TrainConfig(
                                batch_size=1,
                                seq_len=4,
                                schedule_horizon=invocation_end,
                            ),
                            train_iterator,
                            invocation_end_step=invocation_end,
                            save_dir=Path(tmpdir) / "run",
                            save_every=save_every,
                            log_every=log_every,
                            val_data_iter=val_iterator,
                            val_every=val_every,
                            val_steps=1,
                            tp_size=1,
                            fsdp_size=1,
                            dp_size=1,
                        )

                self.assertLen(raised.exception.__notes__, 2)
                self.assertEqual(manager.waits, 1)
                self.assertEqual(manager.closes, 1)
                self.assertEmpty(manager.saves)
                self.assertEqual(manager.frontier, 0)
                self.assertEqual(train_iterator.count, 1)
                if val_iterator is not None:
                    self.assertEqual(val_iterator.count, 0)

    def test_single_process_preflight_precedes_iterators_and_trainer(self):
        with (
            mock.patch.object(vlm.jax, "process_count", return_value=2),
            mock.patch.object(vlm, "_run_sft") as private_run,
            self.assertRaisesRegex(RuntimeError, "exactly one JAX process"),
        ):
            vlm.run_sft(
                object(),
                vlm.TrainConfig(schedule_horizon=1),
                object(),
                invocation_end_step=1,
            )
        private_run.assert_not_called()

        with (
            mock.patch.object(vlm.jax, "process_count", return_value=1),
            mock.patch.object(vlm, "_run_sft") as private_run,
            self.assertRaisesRegex(RuntimeError, "registrar-authorized"),
        ):
            vlm.run_sft(
                object(),
                vlm.TrainConfig(schedule_horizon=1),
                object(),
                invocation_end_step=1,
            )
        private_run.assert_not_called()

        code = """
from unittest import mock
from scripts import train_vlm_sft as script
with (
    mock.patch.object(script, "FLAGS") as flag_values,
    mock.patch.object(script, "_validate_flags"),
    mock.patch.object(script, "open_local_vlm_snapshot") as open_snapshot,
    mock.patch.object(script.vlm_api, "resolve_config"),
    mock.patch.object(script, "_load_snapshot_assets", return_value=(mock.Mock(), mock.Mock())),
    mock.patch.object(script.jax.config, "update"),
    mock.patch.object(script.jax.distributed, "initialize"),
    mock.patch.object(
        script.vlm_trainer,
        "_require_single_jax_process",
        side_effect=RuntimeError("exactly one JAX process"),
    ),
    mock.patch.object(
        script,
        "_grain_iter",
        side_effect=AssertionError("iterator built before topology gate"),
    ),
    mock.patch.object(
        script.vlm_trainer,
        "run_sft",
        side_effect=AssertionError("trainer entered before topology gate"),
    ),
):
    flag_values.jax_cache_dir = "/tmp/unused"
    flag_values.model_snapshot = "/sealed/model"
    flag_values.max_length = 1
    open_snapshot.return_value.__enter__.return_value = mock.Mock()
    script._load_snapshot_assets.return_value[0].model_max_length = 2
    try:
        script.main(None)
    except RuntimeError as error:
        if "exactly one JAX process" not in str(error):
            raise
    else:
        raise AssertionError("multi-process CLI topology was accepted")

with (
    mock.patch.object(script, "FLAGS") as flag_values,
    mock.patch.object(script, "_validate_flags"),
    mock.patch.object(script, "open_local_vlm_snapshot") as open_snapshot,
    mock.patch.object(script.vlm_api, "resolve_config"),
    mock.patch.object(script, "_load_snapshot_assets", return_value=(mock.Mock(), mock.Mock())),
    mock.patch.object(script.jax.config, "update"),
    mock.patch.object(script.jax.distributed, "initialize"),
    mock.patch.object(script.vlm_trainer, "_require_single_jax_process"),
    mock.patch.object(
        script,
        "_grain_iter",
        side_effect=AssertionError("iterator built before capability gate"),
    ),
    mock.patch.object(
        script.vlm_trainer,
        "run_sft",
        side_effect=AssertionError("trainer entered before capability gate"),
    ),
):
    flag_values.jax_cache_dir = "/tmp/unused"
    flag_values.model_snapshot = "/sealed/model"
    flag_values.max_length = 1
    open_snapshot.return_value.__enter__.return_value = mock.Mock()
    script._load_snapshot_assets.return_value[0].model_max_length = 2
    try:
        script.main(None)
    except RuntimeError as error:
        if "registrar-authorized" not in str(error):
            raise
    else:
        raise AssertionError("missing registrar capability was accepted")
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

    def test_snapshot_validation_precedes_jax_entrypoint(self):
        from scripts import train_vlm_sft as script

        events = []
        snapshot = mock.Mock()
        snapshot_context = mock.MagicMock()
        snapshot_context.__enter__.side_effect = lambda: events.append("open") or snapshot
        snapshot_context.__exit__.side_effect = lambda *_: events.append("close")
        with (
            mock.patch.object(script, "FLAGS") as flag_values,
            mock.patch.object(
                script, "_validate_flags", side_effect=lambda: events.append("flags")
            ),
            mock.patch.object(
                script,
                "open_local_vlm_snapshot",
                return_value=snapshot_context,
            ),
            mock.patch.object(
                script.vlm_api,
                "resolve_config",
                side_effect=lambda *_: events.append("config"),
            ),
            mock.patch.object(
                script,
                "_load_snapshot_assets",
                side_effect=lambda *_: (
                    events.append("assets") or (mock.Mock(model_max_length=2), mock.Mock())
                ),
            ),
            mock.patch.object(script, "_run", side_effect=lambda *_: events.append("jax")),
        ):
            flag_values.model_snapshot = "/sealed/model"
            flag_values.max_length = 1
            script.main(None)

        self.assertEqual(events, ["flags", "open", "config", "assets", "jax", "close"])

    def test_snapshot_assets_use_one_pinned_local_source(self):
        from scripts import train_vlm_sft as script

        snapshot = mock.Mock()
        snapshot.consume.return_value = mock.MagicMock()
        snapshot.consume.return_value.__enter__.return_value = "/proc/pinned"
        with (
            mock.patch.object(script.AutoTokenizer, "from_pretrained") as tokenizer_load,
            mock.patch.object(script.AutoImageProcessor, "from_pretrained") as processor_load,
        ):
            tokenizer, processor = script._load_snapshot_assets(snapshot)

        tokenizer_load.assert_called_once_with("/proc/pinned", local_files_only=True)
        processor_load.assert_called_once_with(
            "/proc/pinned",
            local_files_only=True,
            use_fast=False,
        )
        self.assertIs(tokenizer, tokenizer_load.return_value)
        self.assertIs(processor, processor_load.return_value)

    def test_fresh_string_source_is_rejected_before_resolution(self):
        with (
            mock.patch.object(
                vlm.vlm_api,
                "resolve_config",
                side_effect=AssertionError("raw source was resolved"),
            ),
            self.assertRaisesRegex(TypeError, "LocalVLMSnapshot"),
        ):
            vlm._run_sft(
                "Qwen/Qwen3-VL-8B-Instruct",
                vlm.TrainConfig(schedule_horizon=1),
                object(),
                invocation_end_step=1,
                _cleanup=vlm._TrainingCleanup(),
            )

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
