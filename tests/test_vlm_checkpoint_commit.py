"""CPU tests for exact VLM checkpoint commit results."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

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
                restored_iterator,
            )
            self.assertEqual(step, 1)
            _assert_tree_equal(self, _snapshot(restored_optimizer), expected_state)
            np.testing.assert_array_equal(jax.random.key_data(restored_rng), expected_rng)
            self.assertEqual(next(restored_iterator), expected_next)
            manager.close()


if __name__ == "__main__":
    absltest.main()
