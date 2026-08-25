from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import grain
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx

from omegalax.trainers import checkpoint_utils, vlm
from omegalax.trainers.optim import MixedPrecisionOptimizer


class _TinyModel(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.array([0.75, -0.25], dtype=jnp.float32))


class _EightShardIterator(grain.DataLoaderIterator):
    def __init__(self, position: int = 0):
        self.position = position
        self.values = (10, 20, 30, 40)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.values[self.position]
        self.position += 1
        return value

    def get_state(self) -> bytes:
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "logical_shards": 8,
                    "states": [{"position": self.position} for _ in range(8)],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()

    def set_state(self, state: bytes) -> None:
        payload = json.loads(state)
        positions = [shard["position"] for shard in payload["states"]]
        if len(set(positions)) != 1:
            raise ValueError("Logical shard positions differ.")
        self.position = positions[0]

    def start_prefetch(self) -> None:
        return None


def _optimizer() -> MixedPrecisionOptimizer:
    return MixedPrecisionOptimizer(_TinyModel(), vlm.generation_adamw(weight_decay=0.0))


def _rng() -> jax.Array:
    return jax.random.key(7)


def _status(value=vlm.OptimizerFatalStatus.HEALTHY) -> jax.Array:
    return jnp.asarray(value, dtype=jnp.uint8)


def _identities() -> checkpoint_utils.CheckpointIdentities:
    return checkpoint_utils.CheckpointIdentities(
        model_sha256="1" * 64,
        dataset_sha256="2" * 64,
        source_sha256="3" * 64,
        runtime_sha256="4" * 64,
    )


def _receipt(step: int) -> checkpoint_utils.ValidationReceipt:
    return checkpoint_utils.ValidationReceipt(
        step=step,
        batches=2,
        loss_sum_hex=(3.5).hex(),
        supervised_tokens=17,
        dataset_sha256="5" * 64,
    )


def _snapshot(optimizer):
    return jax.tree.map(lambda value: np.asarray(value).copy(), nnx.pure(nnx.state(optimizer)))


def _set_generation(optimizer, generation):
    optimizer.step[...] = jnp.asarray(generation, dtype=jnp.uint32)
    optimizer.opt_state[0].count[...] = jnp.asarray(generation, dtype=jnp.int32)


def _assert_tree_equal(testcase, actual, expected):
    testcase.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def _receipt_dict(value):
    return {
        "step": value.step,
        "batches": value.batches,
        "loss_sum_hex": value.loss_sum_hex,
        "supervised_tokens": value.supervised_tokens,
        "dataset_sha256": value.dataset_sha256,
    }


class _FakeStore:
    def __init__(self, *, latest=None, result_step=None, error=None):
        self.latest = latest
        self.result_step = result_step
        self.error = error
        self.saves = []
        self.waits = 0

    def save(self, step, **kwargs):
        self.saves.append((step, kwargs))
        if self.error is not None:
            raise self.error
        published_step = step if self.result_step is None else self.result_step
        self.latest = step
        return checkpoint_utils.VerifiedCheckpoint(
            path=Path(f"/{published_step:06d}"),
            manifest={"validation": _receipt_dict(kwargs["validation"])},
            sha256="6" * 64,
            step=published_step,
            identities=kwargs["identities"],
        )

    def wait_until_finished(self):
        self.waits += 1

    def latest_step(self):
        return self.latest


class _FakeRestoreStore:
    def __init__(self, train_state, schema, *, items=("input_iter", "schema", "train_state")):
        self.train_state = train_state
        self.schema = schema
        self.items = items
        self.restores = []
        self.directory = Path("/unread")

    @contextlib.contextmanager
    def open(self, step):
        verified = checkpoint_utils.VerifiedCheckpoint(
            path=self.directory / f"{step:06d}",
            manifest={},
            sha256="6" * 64,
            step=step,
            identities=_identities(),
        )
        yield mock.Mock(checkpoint=verified)

    def item_metadata(self, checkpoint):
        del checkpoint
        return {item: None for item in self.items}

    def restore(self, checkpoint, *, args):
        self.restores.append((checkpoint, args))
        if set(args.keys()) == {"schema"}:
            return {"schema": self.schema}
        return {"train_state": self.train_state}


class CheckpointCommitTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.optimizer = _optimizer()
        self.iterator = _EightShardIterator()
        self.rng = _rng()
        self.status = _status()
        self.identities = _identities()

    def _commit(self, manager, step, mode, prior_commit=None):
        return vlm._commit_sft_checkpoint(
            manager,
            self.optimizer,
            self.rng,
            step,
            self.iterator,
            20,
            20,
            self.status,
            self.identities,
            _receipt(step),
            vlm.OptimizerStatusBoundary.CHECKPOINT,
            mode,
            prior_commit,
        )

    def _schema(self, optimizer=None, iterator=None, rng=None, status=None):
        return vlm._sft_checkpoint_schema(
            optimizer or self.optimizer,
            rng if rng is not None else self.rng,
            status if status is not None else self.status,
            iterator or self.iterator,
            20,
            20,
        )

    def test_periodic_forced_and_reuse_bind_one_boundary(self):
        store = _FakeStore()
        _set_generation(self.optimizer, 10)
        commit = self._commit(store, 10, vlm._CheckpointCommitMode.PERIODIC)
        self.assertFalse(store.saves[0][1]["force"])
        self.assertEqual(commit.verified.step, 10)

        reused = self._commit(store, 10, vlm._CheckpointCommitMode.REUSE, commit)
        self.assertIs(reused, commit)
        self.assertLen(store.saves, 1)
        self.assertEqual(store.waits, 2)

        with self.assertRaisesRegex(ValueError, "identical step"):
            self._commit(store, 10, vlm._CheckpointCommitMode.REUSE)
        with self.assertRaisesRegex(ValueError, "identical step"):
            vlm._commit_sft_checkpoint(
                store,
                self.optimizer,
                self.rng,
                10,
                _EightShardIterator(),
                20,
                20,
                self.status,
                self.identities,
                _receipt(10),
                vlm.OptimizerStatusBoundary.CHECKPOINT,
                vlm._CheckpointCommitMode.REUSE,
                commit,
            )

        forced = _FakeStore()
        _set_generation(self.optimizer, 13)
        self._commit(forced, 13, vlm._CheckpointCommitMode.FORCED)
        self.assertTrue(forced.saves[0][1]["force"])

    def test_invalid_state_never_reaches_serialization(self):
        cases = (
            (0, 1, self.status, "must equal NNX step and Adam count"),
            (1, 0, self.status, "must equal NNX step and Adam count"),
            (1, 1, _status(vlm.OptimizerFatalStatus.INVALID_GRADIENT), "invalid_gradient"),
        )
        for nnx_step, adam_count, status, message in cases:
            with self.subTest(nnx_step=nnx_step, adam_count=adam_count, status=int(status)):
                optimizer = _optimizer()
                optimizer.step[...] = nnx_step
                optimizer.opt_state[0].count[...] = adam_count
                store = _FakeStore()
                with (
                    mock.patch.object(
                        vlm,
                        "_sft_checkpoint_schema",
                        side_effect=AssertionError("invalid state reached serialization"),
                    ),
                    self.assertRaisesRegex((ValueError, FloatingPointError), message),
                ):
                    vlm._commit_sft_checkpoint(
                        store,
                        optimizer,
                        self.rng,
                        1,
                        self.iterator,
                        20,
                        20,
                        status,
                        self.identities,
                        _receipt(1),
                        vlm.OptimizerStatusBoundary.CHECKPOINT,
                        vlm._CheckpointCommitMode.PERIODIC,
                        None,
                    )
                self.assertEmpty(store.saves)

    def test_single_iterator_and_stale_validation_fail_before_save(self):
        _set_generation(self.optimizer, 1)
        with self.assertRaisesRegex(ValueError, "eight canonical logical shards"):
            vlm._commit_sft_checkpoint(
                _FakeStore(),
                self.optimizer,
                self.rng,
                1,
                iter(grain.MapDataset.source([1, 2]).to_iter_dataset()),
                20,
                20,
                self.status,
                self.identities,
                _receipt(1),
                vlm.OptimizerStatusBoundary.CHECKPOINT,
                vlm._CheckpointCommitMode.PERIODIC,
                None,
            )
        with self.assertRaisesRegex(ValueError, "receipt step"):
            vlm._commit_sft_checkpoint(
                _FakeStore(),
                self.optimizer,
                self.rng,
                1,
                self.iterator,
                20,
                20,
                self.status,
                self.identities,
                _receipt(2),
                vlm.OptimizerStatusBoundary.CHECKPOINT,
                vlm._CheckpointCommitMode.PERIODIC,
                None,
            )

    def test_real_orbax_commit_and_restore_preserve_exact_boundary(self):
        gradients = nnx.State({"weight": nnx.Param(jnp.array([0.2, -0.4], dtype=jnp.float32))})
        self.optimizer.update(gradients, learning_rate=jnp.asarray(0.03))
        self.assertEqual(next(self.iterator), 10)
        with tempfile.TemporaryDirectory() as tmpdir:
            store = vlm._SFTCheckpointStore(Path(tmpdir), save_interval=1)
            commit = self._commit(store, 1, vlm._CheckpointCommitMode.PERIODIC)
            expected_optimizer = _snapshot(self.optimizer)
            expected_rng = np.asarray(jax.random.key_data(self.rng))
            expected_next = next(self.iterator)

            restored = vlm._restore_sft_checkpoint(
                store,
                _optimizer(),
                _rng(),
                _status(),
                1,
                _EightShardIterator(),
                20,
                20,
            )
            restored_optimizer, step, restored_rng, restored_status, restored_iterator = restored
            self.assertEqual(step, 1)
            _assert_tree_equal(self, _snapshot(restored_optimizer), expected_optimizer)
            np.testing.assert_array_equal(jax.random.key_data(restored_rng), expected_rng)
            self.assertEqual(int(restored_status), int(vlm.OptimizerFatalStatus.HEALTHY))
            self.assertEqual(next(restored_iterator), expected_next)
            self.assertEqual(
                checkpoint_utils.verify_checkpoint(commit.verified.path), commit.verified
            )
            self.assertEqual(store.all_steps(), [1])
            store.close()

    def test_restore_is_pinned_when_numeric_path_is_swapped(self):
        _set_generation(self.optimizer, 1)
        expected_optimizer = _snapshot(self.optimizer)
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as other_tmpdir:
            root = Path(tmpdir)
            other_root = Path(other_tmpdir)
            store = vlm._SFTCheckpointStore(root, save_interval=1)
            original = self._commit(store, 1, vlm._CheckpointCommitMode.PERIODIC).verified

            alternate_optimizer = _optimizer()
            alternate_optimizer.model.weight[...] = jnp.asarray([9.0, -9.0], dtype=jnp.float32)
            _set_generation(alternate_optimizer, 1)
            alternate_store = vlm._SFTCheckpointStore(other_root, save_interval=1)
            vlm._commit_sft_checkpoint(
                alternate_store,
                alternate_optimizer,
                jax.random.key(99),
                1,
                _EightShardIterator(position=1),
                20,
                20,
                self.status,
                self.identities,
                _receipt(1),
                vlm.OptimizerStatusBoundary.CHECKPOINT,
                vlm._CheckpointCommitMode.PERIODIC,
                None,
            )

            real_restore = store.restore
            swapped = False

            def swap_then_restore(checkpoint, *, args):
                nonlocal swapped
                if not swapped:
                    os.rename(root / "000001", root / "original-step")
                    os.rename(other_root / "000001", root / "000001")
                    swapped = True
                return real_restore(checkpoint, args=args)

            restored_optimizer = _optimizer()
            with mock.patch.object(store, "restore", side_effect=swap_then_restore):
                restored = vlm._restore_sft_checkpoint(
                    store,
                    restored_optimizer,
                    _rng(),
                    _status(),
                    1,
                    _EightShardIterator(),
                    20,
                    20,
                    original,
                )

            self.assertTrue(swapped)
            _assert_tree_equal(self, _snapshot(restored[0]), expected_optimizer)
            self.assertNotEqual(checkpoint_utils.verify_checkpoint(root / "000001"), original)
            store.close()
            alternate_store.close()

    def test_restore_rejects_schema_before_live_mutation(self):
        optimizer = _optimizer()
        iterator = _EightShardIterator()
        before_optimizer = _snapshot(optimizer)
        before_iterator = iterator.get_state()
        schema = self._schema(optimizer=optimizer, iterator=iterator)
        schema["version"] = 2
        manager = _FakeRestoreStore(
            vlm._train_state(optimizer, self.rng, self.status),
            schema,
        )
        with self.assertRaisesRegex(ValueError, "fresh-run schema 3"):
            vlm._restore_sft_checkpoint(
                manager,
                optimizer,
                self.rng,
                self.status,
                1,
                iterator,
                20,
                20,
            )
        _assert_tree_equal(self, _snapshot(optimizer), before_optimizer)
        self.assertEqual(iterator.get_state(), before_iterator)
        self.assertLen(manager.restores, 1)

    def test_frontier_rejects_older_and_corrupt_latest_without_fallback(self):
        _set_generation(self.optimizer, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = vlm._SFTCheckpointStore(root, save_interval=1)
            self._commit(store, 1, vlm._CheckpointCommitMode.PERIODIC)
            vlm._require_checkpoint_frontier(store, 1, root, self.identities)
            (root / "000002").mkdir()
            with self.assertRaisesRegex(ValueError, "frontier 2"):
                vlm._require_checkpoint_frontier(store, 1, root, self.identities)
            with self.assertRaisesRegex(ValueError, "regular file"):
                vlm._require_checkpoint_frontier(store, 2, root, self.identities)
            store.close()

    def test_hidden_interrupted_write_is_ignored(self):
        _set_generation(self.optimizer, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = vlm._SFTCheckpointStore(root, save_interval=1)
            self._commit(store, 1, vlm._CheckpointCommitMode.PERIODIC)
            interrupted = root / ".pending-000002-crash" / "000002"
            interrupted.mkdir(parents=True)
            (interrupted / "partial").write_bytes(b"partial")

            self.assertEqual(store.all_steps(), [1])
            self.assertEqual(store.latest_step(), 1)
            vlm._require_checkpoint_frontier(store, 1, root, self.identities)
            store.close()

    def test_retention_preserves_periodic_and_latest_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = vlm._SFTCheckpointStore(
                root,
                save_interval=1,
                keep_period=2,
                keep_latest=1,
            )
            for step in (1, 2, 3):
                _set_generation(self.optimizer, step)
                self._commit(store, step, vlm._CheckpointCommitMode.PERIODIC)

            self.assertEqual(store.all_steps(), [2, 3])
            self.assertEqual(store.latest_step(), 3)
            store.close()

    def test_vlm_resume_request_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            missing = root / "missing"
            self.assertFalse(
                vlm._validate_resume_request(
                    checkpoint_utils.ResumeMode.NEVER,
                    None,
                    missing,
                    10,
                )
            )
            with self.assertRaisesRegex(ValueError, "new checkpoint root"):
                vlm._validate_resume_request(
                    checkpoint_utils.ResumeMode.NEVER,
                    None,
                    root,
                    10,
                )
            with self.assertRaisesRegex(ValueError, "does not support if_present"):
                vlm._validate_resume_request(
                    checkpoint_utils.ResumeMode.IF_PRESENT,
                    None,
                    root,
                    10,
                )
            self.assertTrue(
                vlm._validate_resume_request(
                    checkpoint_utils.ResumeMode.REQUIRED,
                    3,
                    root,
                    10,
                )
            )


if __name__ == "__main__":
    absltest.main()
