"""CPU tests for exact VLM checkpoint commit results."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import grain
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from absl.testing import absltest
from flax import nnx

from omegalax.trainers import vlm
from omegalax.trainers.optim import MixedPrecisionOptimizer

_MODEL_SNAPSHOT_ID = "ab" * 32


class _TinyModel(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.array([0.75, -0.25], dtype=jnp.float32))


def _optimizer() -> MixedPrecisionOptimizer:
    return MixedPrecisionOptimizer(_TinyModel(), vlm.generation_adamw(weight_decay=0.0))


def _iterator():
    return iter(grain.MapDataset.source([10, 20, 30, 40]).to_iter_dataset())


def _snapshot(optimizer):
    return jax.tree.map(lambda value: np.asarray(value).copy(), nnx.pure(nnx.state(optimizer)))


def _set_generation(optimizer, generation):
    optimizer.step[...] = jnp.asarray(generation, dtype=jnp.uint32)
    optimizer.opt_state[0].count[...] = jnp.asarray(generation, dtype=jnp.int32)


def _assert_tree_equal(testcase, actual, expected):
    testcase.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


class _FakeManager:
    def __init__(self, *, save_result=True, latest=None, update_latest=True, error=None):
        self.save_result = save_result
        self.latest = latest
        self.update_latest = update_latest
        self.error = error
        self.saves = []
        self.waits = 0

    def save(self, step, *, args, force):
        self.saves.append((step, args, force))
        if self.error is not None:
            raise self.error
        if self.save_result and self.update_latest:
            self.latest = step
        return self.save_result

    def wait_until_finished(self):
        self.waits += 1

    def latest_step(self):
        return self.latest


class _FakeRestoreManager:
    def __init__(
        self,
        train_state,
        schema,
        *,
        items=("input_iter", "schema", "train_state"),
    ):
        self.train_state = train_state
        self.schema = schema
        self.items = items
        self.restores = []

    def item_metadata(self, step):
        return {item: None for item in self.items}

    def restore(self, step, *, args):
        self.restores.append((step, args))
        if set(args.keys()) == {"schema"}:
            return {"schema": self.schema}
        return {"train_state": self.train_state}


class CheckpointCommitTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.optimizer = _optimizer()
        self.iterator = _iterator()
        self.status = jnp.asarray(vlm.OptimizerFatalStatus.HEALTHY, dtype=jnp.uint8)

    def _commit(self, manager, step, mode, prior_commit=None):
        return vlm._commit_sft_checkpoint(
            manager,
            self.optimizer,
            step,
            self.iterator,
            20,
            20,
            _MODEL_SNAPSHOT_ID,
            self.status,
            vlm.OptimizerStatusBoundary.CHECKPOINT,
            mode,
            prior_commit,
        )

    def test_periodic_and_forced_save_flags(self):
        periodic = _FakeManager()
        _set_generation(self.optimizer, 10)
        commit = self._commit(periodic, 10, vlm._CheckpointCommitMode.PERIODIC)
        self.assertEqual(commit.step, 10)
        self.assertLen(periodic.saves, 1)
        self.assertFalse(periodic.saves[0][2])
        self.assertEqual(periodic.waits, 1)

        forced = _FakeManager(latest=10)
        _set_generation(self.optimizer, 13)
        commit = self._commit(forced, 13, vlm._CheckpointCommitMode.FORCED)
        self.assertEqual(commit.step, 13)
        self.assertLen(forced.saves, 1)
        self.assertTrue(forced.saves[0][2])
        self.assertEqual(forced.waits, 1)

    def test_false_exception_partial_and_latest_mismatch_fail(self):
        _set_generation(self.optimizer, 10)
        with self.assertRaisesRegex(RuntimeError, "did not save"):
            self._commit(
                _FakeManager(save_result=False),
                10,
                vlm._CheckpointCommitMode.PERIODIC,
            )

        _set_generation(self.optimizer, 3)
        with self.assertRaisesRegex(OSError, "write failed"):
            self._commit(
                _FakeManager(error=OSError("write failed")),
                3,
                vlm._CheckpointCommitMode.FORCED,
            )

        for latest in (None, 9, 11):
            _set_generation(self.optimizer, 10)
            with (
                self.subTest(latest=latest),
                self.assertRaisesRegex(RuntimeError, "commit mismatch"),
            ):
                self._commit(
                    _FakeManager(latest=latest, update_latest=False),
                    10,
                    vlm._CheckpointCommitMode.PERIODIC,
                )

    def test_reuse_requires_the_identical_saved_boundary(self):
        manager = _FakeManager()
        _set_generation(self.optimizer, 10)
        commit = self._commit(manager, 10, vlm._CheckpointCommitMode.PERIODIC)
        reused = self._commit(manager, 10, vlm._CheckpointCommitMode.REUSE, commit)
        self.assertIs(reused, commit)
        self.assertLen(manager.saves, 1)
        self.assertEqual(manager.waits, 2)

        with self.assertRaisesRegex(ValueError, "identical step"):
            self._commit(manager, 10, vlm._CheckpointCommitMode.REUSE)
        with self.assertRaisesRegex(ValueError, "identical step"):
            vlm._commit_sft_checkpoint(
                manager,
                self.optimizer,
                10,
                _iterator(),
                20,
                20,
                _MODEL_SNAPSHOT_ID,
                self.status,
                vlm.OptimizerStatusBoundary.CHECKPOINT,
                vlm._CheckpointCommitMode.REUSE,
                commit,
            )
        with self.assertRaisesRegex(ValueError, "identical step"):
            vlm._commit_sft_checkpoint(
                _FakeManager(latest=10),
                self.optimizer,
                10,
                self.iterator,
                20,
                20,
                _MODEL_SNAPSHOT_ID,
                self.status,
                vlm.OptimizerStatusBoundary.CHECKPOINT,
                vlm._CheckpointCommitMode.REUSE,
                commit,
            )
        with self.assertRaisesRegex(ValueError, "identical step"):
            vlm._commit_sft_checkpoint(
                manager,
                self.optimizer,
                10,
                self.iterator,
                21,
                20,
                _MODEL_SNAPSHOT_ID,
                self.status,
                vlm.OptimizerStatusBoundary.CHECKPOINT,
                vlm._CheckpointCommitMode.REUSE,
                commit,
            )

        manager.latest = 9
        with self.assertRaisesRegex(RuntimeError, "commit mismatch"):
            self._commit(manager, 10, vlm._CheckpointCommitMode.REUSE, commit)

    def test_save_rejects_counter_mismatch_before_serialization(self):
        for name, nnx_step, adam_count in (("nnx", 0, 1), ("adam", 1, 0)):
            with self.subTest(name=name):
                optimizer = _optimizer()
                optimizer.step[...] = jnp.asarray(nnx_step, dtype=jnp.uint32)
                optimizer.opt_state[0].count[...] = jnp.asarray(adam_count, dtype=jnp.int32)
                iterator = _iterator()
                manager = _FakeManager()
                before_optimizer = _snapshot(optimizer)
                before_iterator = iterator.get_state()
                with (
                    mock.patch.object(
                        vlm,
                        "_sft_checkpoint_schema",
                        side_effect=AssertionError("invalid generation reached serialization"),
                    ) as checkpoint_schema,
                    self.assertRaisesRegex(ValueError, "must equal NNX step and Adam count"),
                ):
                    vlm._commit_sft_checkpoint(
                        manager,
                        optimizer,
                        1,
                        iterator,
                        20,
                        20,
                        _MODEL_SNAPSHOT_ID,
                        self.status,
                        vlm.OptimizerStatusBoundary.CHECKPOINT,
                        vlm._CheckpointCommitMode.PERIODIC,
                        None,
                    )
                checkpoint_schema.assert_not_called()
                self.assertEmpty(manager.saves)
                self.assertEqual(manager.waits, 0)
                _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
                self.assertEqual(iterator.get_state(), before_iterator)

    def test_real_orbax_forces_off_interval_and_reuses_exact_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = vlm._make_checkpoint_manager(Path(tmpdir), save_interval=10)
            _set_generation(self.optimizer, 10)
            periodic = self._commit(manager, 10, vlm._CheckpointCommitMode.PERIODIC)
            reused = self._commit(manager, 10, vlm._CheckpointCommitMode.REUSE, periodic)
            self.assertIs(reused, periodic)

            _set_generation(self.optimizer, 13)
            forced = self._commit(manager, 13, vlm._CheckpointCommitMode.FORCED)
            self.assertEqual(forced.step, 13)
            self.assertEqual(manager.all_steps(), [10, 13])
            vlm._require_checkpoint_frontier(manager, 13, Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "does not match.*frontier 13"):
                vlm._require_checkpoint_frontier(manager, 10, Path(tmpdir))

            with self.assertRaises(ocp.checkpoint_manager.StepAlreadyExistsError):
                self._commit(manager, 13, vlm._CheckpointCommitMode.FORCED)
            manager.close()

    def test_real_orbax_restore_preserves_exact_boundary(self):
        gradients = nnx.State({"weight": nnx.Param(jnp.array([0.2, -0.4], dtype=jnp.float32))})
        self.optimizer.update(gradients, learning_rate=jnp.asarray(0.03))
        self.assertEqual(int(self.optimizer.step[...]), 1)
        self.assertEqual(next(self.iterator), 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = vlm._make_checkpoint_manager(Path(tmpdir), save_interval=1)
            self._commit(manager, 1, vlm._CheckpointCommitMode.PERIODIC)
            expected_state = _snapshot(self.optimizer)
            expected_next = next(self.iterator)

            restored_optimizer = _optimizer()
            restored_iterator = _iterator()
            restored_optimizer, step, restored_iterator = vlm._restore_sft_checkpoint(
                manager,
                restored_optimizer,
                1,
                restored_iterator,
                20,
                20,
                _MODEL_SNAPSHOT_ID,
            )
            self.assertEqual(step, 1)
            _assert_tree_equal(self, _snapshot(restored_optimizer), expected_state)
            self.assertEqual(next(restored_iterator), expected_next)
            manager.close()

    def test_same_phase_restore_continues_exact_optimizer_and_iterator_boundary(self):
        gradients = nnx.State({"weight": nnx.Param(jnp.array([0.2, -0.4], dtype=jnp.float32))})
        self.optimizer.update(gradients, learning_rate=jnp.asarray(0.03))
        self.assertEqual(next(self.iterator), 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = vlm._make_checkpoint_manager(Path(tmpdir), save_interval=10)
            vlm._commit_sft_checkpoint(
                manager,
                self.optimizer,
                1,
                self.iterator,
                3,
                3,
                _MODEL_SNAPSHOT_ID,
                self.status,
                vlm.OptimizerStatusBoundary.FINAL,
                vlm._CheckpointCommitMode.FORCED,
                None,
            )
            restored, step, restored_iterator = vlm._restore_sft_checkpoint(
                manager,
                _optimizer(),
                1,
                _iterator(),
                3,
                3,
                _MODEL_SNAPSHOT_ID,
            )
            self.assertEqual(step, 1)
            self.assertEqual(next(restored_iterator), next(self.iterator))

            self.optimizer.update(gradients, learning_rate=jnp.asarray(0.01))
            restored.update(gradients, learning_rate=jnp.asarray(0.01))
            _assert_tree_equal(self, _snapshot(restored), _snapshot(self.optimizer))
            manager.close()

    def test_restore_rejects_generation_mismatch_before_iterator_mutation(self):
        for name, nnx_step in (("both", 0), ("adam", 1)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmpdir:
                source_optimizer = _optimizer()
                source_optimizer.step[...] = nnx_step
                source_iterator = _iterator()
                manager = vlm._make_checkpoint_manager(Path(tmpdir), save_interval=1)
                saved = manager.save(
                    1,
                    args=ocp.args.Composite(
                        train_state=ocp.args.PyTreeSave(vlm._train_state(source_optimizer)),
                        input_iter=grain.checkpoint.CheckpointSave(source_iterator),
                        schema=ocp.args.JsonSave(
                            vlm._sft_checkpoint_schema(
                                source_optimizer,
                                source_iterator,
                                20,
                                20,
                                _MODEL_SNAPSHOT_ID,
                            )
                        ),
                    ),
                )
                self.assertTrue(saved)
                manager.wait_until_finished()

                optimizer = _optimizer()
                iterator = _iterator()
                before_optimizer = _snapshot(optimizer)
                before_iterator = iterator.get_state()
                with (
                    mock.patch.object(
                        vlm,
                        "_restore_iterator_checkpoint",
                        wraps=vlm._restore_iterator_checkpoint,
                    ) as restore_iterator,
                    self.assertRaisesRegex(ValueError, "must equal NNX step and Adam count"),
                ):
                    vlm._restore_sft_checkpoint(
                        manager,
                        optimizer,
                        1,
                        iterator,
                        20,
                        20,
                        _MODEL_SNAPSHOT_ID,
                    )
                restore_iterator.assert_not_called()
                _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
                self.assertEqual(iterator.get_state(), before_iterator)
                manager.close()

    def test_restore_rejects_historical_schema_before_state_restore(self):
        optimizer = _optimizer()
        iterator = _iterator()
        before_optimizer = _snapshot(optimizer)
        before_iterator = iterator.get_state()
        for version in (1, 2, 3.0, True, None):
            with self.subTest(version=version):
                schema = vlm._sft_checkpoint_schema(
                    optimizer,
                    iterator,
                    20,
                    20,
                    _MODEL_SNAPSHOT_ID,
                )
                schema["version"] = version
                manager = _FakeRestoreManager({"optimizer": nnx.state(_optimizer())}, schema)
                with self.assertRaisesRegex(ValueError, "incompatible with fresh-run schema 3"):
                    vlm._restore_sft_checkpoint(
                        manager,
                        optimizer,
                        1,
                        iterator,
                        20,
                        20,
                        _MODEL_SNAPSHOT_ID,
                    )
                _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
                self.assertEqual(iterator.get_state(), before_iterator)
                self.assertLen(manager.restores, 1)

    def test_restore_rejects_different_model_snapshot_before_state_restore(self):
        optimizer = _optimizer()
        iterator = _iterator()
        before_optimizer = _snapshot(optimizer)
        before_iterator = iterator.get_state()
        schema = vlm._sft_checkpoint_schema(
            optimizer,
            iterator,
            20,
            20,
            "cd" * 32,
        )
        manager = _FakeRestoreManager({"optimizer": nnx.state(_optimizer())}, schema)

        with self.assertRaisesRegex(ValueError, "schema does not match"):
            vlm._restore_sft_checkpoint(
                manager,
                optimizer,
                1,
                iterator,
                20,
                20,
                _MODEL_SNAPSHOT_ID,
            )

        _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
        self.assertEqual(iterator.get_state(), before_iterator)
        self.assertLen(manager.restores, 1)

    def test_restore_rejects_optimizer_schema_without_mutation(self):
        expected = nnx.state(_optimizer())
        cases = {
            "paths": nnx.State(
                {
                    "model": {},
                    "opt_state": expected["opt_state"],
                    "step": expected["step"],
                }
            ),
            "variable": nnx.State(
                {
                    "model": {"weight": nnx.BatchStat(jnp.array([0.75, -0.25], dtype=jnp.float32))},
                    "opt_state": expected["opt_state"],
                    "step": expected["step"],
                }
            ),
            "shape": nnx.State(
                {
                    "model": {"weight": nnx.Param(jnp.ones(3, dtype=jnp.float32))},
                    "opt_state": expected["opt_state"],
                    "step": expected["step"],
                }
            ),
            "dtype": nnx.State(
                {
                    "model": {"weight": nnx.Param(jnp.ones(2, dtype=jnp.bfloat16))},
                    "opt_state": expected["opt_state"],
                    "step": expected["step"],
                }
            ),
        }
        for name, restored_state in cases.items():
            with self.subTest(name=name):
                optimizer = _optimizer()
                iterator = _iterator()
                before_optimizer = _snapshot(optimizer)
                before_iterator = iterator.get_state()
                manager = _FakeRestoreManager(
                    {"optimizer": restored_state},
                    vlm._sft_checkpoint_schema(
                        optimizer,
                        iterator,
                        20,
                        20,
                        _MODEL_SNAPSHOT_ID,
                    ),
                )
                with self.assertRaisesRegex(ValueError, f"optimizer .*{name}"):
                    vlm._restore_sft_checkpoint(
                        manager,
                        optimizer,
                        1,
                        iterator,
                        20,
                        20,
                        _MODEL_SNAPSHOT_ID,
                    )
                _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
                self.assertEqual(iterator.get_state(), before_iterator)
                self.assertLen(manager.restores, 2)

    def test_restore_rejects_saved_variable_subtype_without_mutation(self):
        optimizer = _optimizer()
        iterator = _iterator()
        before_optimizer = _snapshot(optimizer)
        before_iterator = iterator.get_state()
        schema = vlm._sft_checkpoint_schema(
            optimizer,
            iterator,
            20,
            20,
            _MODEL_SNAPSHOT_ID,
        )
        schema["optimizer"][0]["variable_type"] = "flax.nnx.variablelib.BatchStat"
        manager = _FakeRestoreManager(
            {"optimizer": nnx.state(_optimizer())},
            schema,
        )
        with self.assertRaisesRegex(ValueError, "schema does not match"):
            vlm._restore_sft_checkpoint(
                manager,
                optimizer,
                1,
                iterator,
                20,
                20,
                _MODEL_SNAPSHOT_ID,
            )
        _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
        self.assertEqual(iterator.get_state(), before_iterator)
        self.assertLen(manager.restores, 1)

    def test_restore_rejects_outer_schema_without_mutation(self):
        optimizer = _optimizer()
        iterator = _iterator()
        before_optimizer = _snapshot(optimizer)
        before_iterator = iterator.get_state()
        cases = (
            _FakeRestoreManager(
                {"optimizer": nnx.state(_optimizer())},
                {},
                items=("train_state",),
            ),
            _FakeRestoreManager(
                {"optimizer": nnx.state(_optimizer()), "extra": 1},
                vlm._sft_checkpoint_schema(
                    optimizer,
                    iterator,
                    20,
                    20,
                    _MODEL_SNAPSHOT_ID,
                ),
            ),
        )
        for manager in cases:
            with (
                self.subTest(items=manager.items),
                self.assertRaisesRegex(ValueError, "items|train_state"),
            ):
                vlm._restore_sft_checkpoint(
                    manager,
                    optimizer,
                    1,
                    iterator,
                    20,
                    20,
                    _MODEL_SNAPSHOT_ID,
                )
            _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
            self.assertEqual(iterator.get_state(), before_iterator)

    def test_restore_rejects_corrupt_iterator_before_mutation(self):
        gradients = nnx.State({"weight": nnx.Param(jnp.array([0.2, -0.4], dtype=jnp.float32))})
        self.optimizer.update(gradients, learning_rate=jnp.asarray(0.03))
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = vlm._make_checkpoint_manager(Path(tmpdir), save_interval=1)
            self._commit(manager, 1, vlm._CheckpointCommitMode.PERIODIC)
            iterator_path = Path(tmpdir) / "000001/input_iter/process_0-of-1.json"
            cases = {
                "type": '{"next_index": "1"}',
                "keys": '{"wrong": 1}',
                "duplicate": '{"next_index": 1, "next_index": 2}',
            }
            for name, raw in cases.items():
                with self.subTest(name=name):
                    iterator_path.write_text(raw)
                    optimizer = _optimizer()
                    iterator = _iterator()
                    before_optimizer = _snapshot(optimizer)
                    before_iterator = iterator.get_state()
                    with self.assertRaisesRegex(ValueError, "input_iter|Duplicate"):
                        vlm._restore_sft_checkpoint(
                            manager,
                            optimizer,
                            1,
                            iterator,
                            20,
                            20,
                            _MODEL_SNAPSHOT_ID,
                        )
                    _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
                    self.assertEqual(iterator.get_state(), before_iterator)
            manager.close()

    def test_restore_rejects_symlinked_iterator_checkpoint(self):
        gradients = nnx.State({"weight": nnx.Param(jnp.array([0.2, -0.4], dtype=jnp.float32))})
        self.optimizer.update(gradients, learning_rate=jnp.asarray(0.03))
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = vlm._make_checkpoint_manager(Path(tmpdir), save_interval=1)
            self._commit(manager, 1, vlm._CheckpointCommitMode.PERIODIC)
            iterator_path = Path(tmpdir) / "000001/input_iter/process_0-of-1.json"
            target_path = Path(tmpdir) / "iterator-target.json"
            target_path.write_bytes(iterator_path.read_bytes())
            iterator_path.unlink()
            iterator_path.symlink_to(target_path)

            optimizer = _optimizer()
            iterator = _iterator()
            before_optimizer = _snapshot(optimizer)
            before_iterator = iterator.get_state()
            with self.assertRaisesRegex(ValueError, "no-follow regular file"):
                vlm._restore_sft_checkpoint(
                    manager,
                    optimizer,
                    1,
                    iterator,
                    20,
                    20,
                    _MODEL_SNAPSHOT_ID,
                )
            _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
            self.assertEqual(iterator.get_state(), before_iterator)
            manager.close()

    def test_restore_rejects_oversized_iterator_checkpoint(self):
        gradients = nnx.State({"weight": nnx.Param(jnp.array([0.2, -0.4], dtype=jnp.float32))})
        self.optimizer.update(gradients, learning_rate=jnp.asarray(0.03))
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = vlm._make_checkpoint_manager(Path(tmpdir), save_interval=1)
            self._commit(manager, 1, vlm._CheckpointCommitMode.PERIODIC)
            iterator_path = Path(tmpdir) / "000001/input_iter/process_0-of-1.json"
            os.truncate(iterator_path, vlm._MAX_GRAIN_CHECKPOINT_BYTES + 1)

            optimizer = _optimizer()
            iterator = _iterator()
            before_optimizer = _snapshot(optimizer)
            before_iterator = iterator.get_state()
            with self.assertRaisesRegex(ValueError, "exceeds"):
                vlm._restore_sft_checkpoint(
                    manager,
                    optimizer,
                    1,
                    iterator,
                    20,
                    20,
                    _MODEL_SNAPSHOT_ID,
                )
            _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
            self.assertEqual(iterator.get_state(), before_iterator)
            manager.close()

    def test_restore_uses_validated_bytes_after_path_replacement(self):
        gradients = nnx.State({"weight": nnx.Param(jnp.array([0.2, -0.4], dtype=jnp.float32))})
        self.optimizer.update(gradients, learning_rate=jnp.asarray(0.03))
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = vlm._make_checkpoint_manager(Path(tmpdir), save_interval=1)
            self._commit(manager, 1, vlm._CheckpointCommitMode.PERIODIC)
            iterator_path = Path(tmpdir) / "000001/input_iter/process_0-of-1.json"
            replacement_path = Path(tmpdir) / "iterator-replacement.json"
            original_validate = vlm._validate_json_schema
            replaced = False

            def replace_path(expected, restored, path="input_iter"):
                nonlocal replaced
                if not replaced:
                    replacement_path.write_text('{"next_index": 3}')
                    os.replace(replacement_path, iterator_path)
                    replaced = True
                return original_validate(expected, restored, path)

            optimizer = _optimizer()
            iterator = _iterator()
            with mock.patch.object(vlm, "_validate_json_schema", side_effect=replace_path):
                _, step, restored_iterator = vlm._restore_sft_checkpoint(
                    manager,
                    optimizer,
                    1,
                    iterator,
                    20,
                    20,
                    _MODEL_SNAPSHOT_ID,
                )
            self.assertEqual(step, 1)
            self.assertTrue(replaced)
            self.assertEqual(next(restored_iterator), 10)
            self.assertEqual(iterator_path.read_text(), '{"next_index": 3}')
            manager.close()

    def test_vlm_resume_request_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            missing = root / "missing"
            self.assertFalse(
                vlm._validate_resume_request(
                    vlm.checkpoint_utils.ResumeMode.NEVER,
                    None,
                    missing,
                    10,
                )
            )
            with self.assertRaisesRegex(ValueError, "new checkpoint root"):
                vlm._validate_resume_request(
                    vlm.checkpoint_utils.ResumeMode.NEVER,
                    None,
                    root,
                    10,
                )
            with self.assertRaisesRegex(ValueError, "does not support if_present"):
                vlm._validate_resume_request(
                    vlm.checkpoint_utils.ResumeMode.IF_PRESENT,
                    None,
                    root,
                    10,
                )
            with self.assertRaisesRegex(ValueError, "positive integer"):
                vlm._validate_resume_request(
                    vlm.checkpoint_utils.ResumeMode.REQUIRED,
                    None,
                    root,
                    10,
                )
            self.assertTrue(
                vlm._validate_resume_request(
                    vlm.checkpoint_utils.ResumeMode.REQUIRED,
                    3,
                    root,
                    10,
                )
            )


if __name__ == "__main__":
    absltest.main()
