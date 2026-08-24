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
import optax
import orbax.checkpoint as ocp
from absl.testing import absltest
from flax import nnx

from omegalax.trainers import vlm
from omegalax.trainers.optim import MixedPrecisionOptimizer


class _TinyModel(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.array([0.75, -0.25], dtype=jnp.float32))


def _optimizer() -> MixedPrecisionOptimizer:
    return MixedPrecisionOptimizer(_TinyModel(), optax.adamw(0.03))


def _iterator():
    return iter(grain.MapDataset.source([10, 20, 30, 40]).to_iter_dataset())


def _snapshot(optimizer):
    return jax.tree.map(lambda value: np.asarray(value).copy(), nnx.pure(nnx.state(optimizer)))


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
        self.rng = jax.random.key(7)
        self.iterator = _iterator()

    def _commit(self, manager, step, mode, prior_commit=None):
        return vlm._commit_sft_checkpoint(
            manager,
            self.optimizer,
            self.rng,
            step,
            self.iterator,
            mode,
            prior_commit,
        )

    def test_periodic_and_forced_save_flags(self):
        periodic = _FakeManager()
        commit = self._commit(periodic, 10, vlm._CheckpointCommitMode.PERIODIC)
        self.assertEqual(commit.step, 10)
        self.assertLen(periodic.saves, 1)
        self.assertFalse(periodic.saves[0][2])
        self.assertEqual(periodic.waits, 1)

        forced = _FakeManager(latest=10)
        commit = self._commit(forced, 13, vlm._CheckpointCommitMode.FORCED)
        self.assertEqual(commit.step, 13)
        self.assertLen(forced.saves, 1)
        self.assertTrue(forced.saves[0][2])
        self.assertEqual(forced.waits, 1)

    def test_false_exception_partial_and_latest_mismatch_fail(self):
        with self.assertRaisesRegex(RuntimeError, "did not save"):
            self._commit(
                _FakeManager(save_result=False),
                10,
                vlm._CheckpointCommitMode.PERIODIC,
            )

        with self.assertRaisesRegex(OSError, "write failed"):
            self._commit(
                _FakeManager(error=OSError("write failed")),
                3,
                vlm._CheckpointCommitMode.FORCED,
            )

        for latest in (None, 9, 11):
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
                self.rng,
                10,
                _iterator(),
                vlm._CheckpointCommitMode.REUSE,
                commit,
            )
        with self.assertRaisesRegex(ValueError, "identical step"):
            vlm._commit_sft_checkpoint(
                _FakeManager(latest=10),
                self.optimizer,
                self.rng,
                10,
                self.iterator,
                vlm._CheckpointCommitMode.REUSE,
                commit,
            )
        with self.assertRaisesRegex(ValueError, "identical step"):
            vlm._commit_sft_checkpoint(
                manager,
                self.optimizer,
                jax.random.key(7),
                10,
                self.iterator,
                vlm._CheckpointCommitMode.REUSE,
                commit,
            )

        manager.latest = 9
        with self.assertRaisesRegex(RuntimeError, "commit mismatch"):
            self._commit(manager, 10, vlm._CheckpointCommitMode.REUSE, commit)

    def test_real_orbax_forces_off_interval_and_reuses_exact_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = vlm._make_checkpoint_manager(Path(tmpdir), save_interval=10)
            periodic = self._commit(manager, 10, vlm._CheckpointCommitMode.PERIODIC)
            reused = self._commit(manager, 10, vlm._CheckpointCommitMode.REUSE, periodic)
            self.assertIs(reused, periodic)

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
        self.optimizer.update(gradients)
        self.assertEqual(int(self.optimizer.step[...]), 1)
        self.assertEqual(next(self.iterator), 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = vlm._make_checkpoint_manager(Path(tmpdir), save_interval=1)
            self._commit(manager, 1, vlm._CheckpointCommitMode.PERIODIC)
            expected_state = _snapshot(self.optimizer)
            expected_rng = np.asarray(jax.random.key_data(self.rng))
            expected_next = next(self.iterator)

            restored_optimizer = _optimizer()
            restored_iterator = _iterator()
            restored_optimizer, step, restored_rng, restored_iterator = vlm._restore_sft_checkpoint(
                manager,
                restored_optimizer,
                jax.random.key(0),
                1,
                restored_iterator,
            )
            self.assertEqual(step, 1)
            _assert_tree_equal(self, _snapshot(restored_optimizer), expected_state)
            np.testing.assert_array_equal(jax.random.key_data(restored_rng), expected_rng)
            self.assertEqual(next(restored_iterator), expected_next)
            manager.close()

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
                    {"optimizer": restored_state, "rng": jax.random.key(7)},
                    vlm._sft_checkpoint_schema(optimizer, jax.random.key(0), iterator),
                )
                with self.assertRaisesRegex(ValueError, f"optimizer .*{name}"):
                    vlm._restore_sft_checkpoint(
                        manager,
                        optimizer,
                        jax.random.key(0),
                        1,
                        iterator,
                    )
                _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
                self.assertEqual(iterator.get_state(), before_iterator)
                self.assertLen(manager.restores, 2)

    def test_restore_rejects_rng_schema_without_mutation(self):
        restored_optimizer = nnx.state(_optimizer())
        cases = {
            "type": jnp.array([0, 7], dtype=jnp.uint32),
            "shape": jax.random.split(jax.random.key(7), 2),
            "dtype": jax.random.key(7, impl="rbg"),
        }
        for name, restored_rng in cases.items():
            with self.subTest(name=name):
                optimizer = _optimizer()
                iterator = _iterator()
                before_optimizer = _snapshot(optimizer)
                before_iterator = iterator.get_state()
                manager = _FakeRestoreManager(
                    {"optimizer": restored_optimizer, "rng": restored_rng},
                    vlm._sft_checkpoint_schema(optimizer, jax.random.key(0), iterator),
                )
                with self.assertRaisesRegex(ValueError, f"RNG {name}"):
                    vlm._restore_sft_checkpoint(
                        manager,
                        optimizer,
                        jax.random.key(0),
                        1,
                        iterator,
                    )
                _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
                self.assertEqual(iterator.get_state(), before_iterator)
                self.assertLen(manager.restores, 2)

    def test_restore_rejects_saved_variable_subtype_without_mutation(self):
        optimizer = _optimizer()
        iterator = _iterator()
        before_optimizer = _snapshot(optimizer)
        before_iterator = iterator.get_state()
        schema = vlm._sft_checkpoint_schema(optimizer, jax.random.key(0), iterator)
        schema["optimizer"][0]["variable_type"] = "flax.nnx.variablelib.BatchStat"
        manager = _FakeRestoreManager(
            {"optimizer": nnx.state(_optimizer()), "rng": jax.random.key(7)},
            schema,
        )
        with self.assertRaisesRegex(ValueError, "schema does not match"):
            vlm._restore_sft_checkpoint(
                manager,
                optimizer,
                jax.random.key(0),
                1,
                iterator,
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
                {"optimizer": nnx.state(_optimizer()), "rng": jax.random.key(7)},
                {},
                items=("train_state",),
            ),
            _FakeRestoreManager(
                {"optimizer": nnx.state(_optimizer())},
                vlm._sft_checkpoint_schema(optimizer, jax.random.key(0), iterator),
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
                    jax.random.key(0),
                    1,
                    iterator,
                )
            _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
            self.assertEqual(iterator.get_state(), before_iterator)

    def test_restore_rejects_corrupt_iterator_before_mutation(self):
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
                            jax.random.key(0),
                            1,
                            iterator,
                        )
                    _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
                    self.assertEqual(iterator.get_state(), before_iterator)
            manager.close()

    def test_restore_rejects_symlinked_iterator_checkpoint(self):
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
                    jax.random.key(0),
                    1,
                    iterator,
                )
            _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
            self.assertEqual(iterator.get_state(), before_iterator)
            manager.close()

    def test_restore_rejects_oversized_iterator_checkpoint(self):
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
                    jax.random.key(0),
                    1,
                    iterator,
                )
            _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
            self.assertEqual(iterator.get_state(), before_iterator)
            manager.close()

    def test_restore_uses_validated_bytes_after_path_replacement(self):
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
                _, step, _, restored_iterator = vlm._restore_sft_checkpoint(
                    manager,
                    optimizer,
                    jax.random.key(0),
                    1,
                    iterator,
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
