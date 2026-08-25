"""VLM update, checkpoint, and cleanup contracts."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import grain
import jax
import jax.numpy as jnp
import optax
from absl.testing import absltest
from flax import nnx

from omegalax.trainers import checkpoint_utils, vlm
from omegalax.trainers.loss import chunked_cross_entropy_loss_sum
from omegalax.trainers.optim import (
    MixedPrecisionOptimizer,
    accumulate_gradient_sum,
    apply_normalized_gradient_sum,
    initialize_gradient_sum,
)

_MODEL_IDENTITY = jnp.arange(32, dtype=jnp.uint8)


class _ScalarModel(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.asarray([1.0], dtype=jnp.bfloat16))


def _make_optimizer() -> MixedPrecisionOptimizer:
    schedule = optax.linear_schedule(0.1, 0.01, 100)
    return MixedPrecisionOptimizer(_ScalarModel(), optax.adamw(schedule), wrt=nnx.Param)


def _make_iterator():
    return iter(grain.MapDataset.source([{"value": value} for value in range(4)]).batch(1))


class _Closeable:
    def __init__(self, error: BaseException | None = None):
        self.error = error
        self.closed = 0

    def close(self):
        self.closed += 1
        if self.error is not None:
            raise self.error


class _PanelIterator:
    def __init__(self, batches):
        self.batches = batches
        self.index = 0

    def __next__(self):
        batch = self.batches[self.index]
        self.index += 1
        return dict(batch)

    def get_state(self):
        return self.index

    def set_state(self, state):
        self.index = state


class VLMTrainingContractTest(absltest.TestCase):
    def test_non_eight_topology_is_supported_normal_and_optimized(self):
        code = """
import jax
import numpy as np
from omegalax.distributed.mesh import ensure_mesh
from omegalax.models.qwen3_vl.config import make_vl_config
from omegalax.vlm import api as vlm_api
mesh = ensure_mesh(tp_size=1, fsdp_size=2, dp_size=1)
if tuple(mesh.axis_names) != ('tp', 'fsdp', 'dp') or mesh.size != 2:
    raise RuntimeError(mesh)
cfg = vlm_api.align_config_to_mesh(make_vl_config('qwen3-vl-smoke'), mesh)
batch = {'token_ids_BT': np.arange(16, dtype=np.int32).reshape(4, 4)}
sharded = vlm_api.shard_batch_dict(batch, cfg, mesh)['token_ids_BT']
if sharded.shape != (4, 4) or sorted(s.data.shape for s in sharded.addressable_shards) != [(2, 4)] * 2:
    raise RuntimeError(sharded)
"""
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
        for optimized in (False, True):
            command = [sys.executable]
            if optimized:
                command.append("-O")
            command.extend(["-c", code])
            result = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_resume_requires_exact_generation_and_schedule_horizon(self):
        config = vlm.TrainConfig(num_steps=10, schedule_horizon=100)
        save_path = Path("/tmp/checkpoints")
        self.assertTrue(
            vlm._validate_training_request(
                config,
                checkpoint_utils.ResumeMode.REQUIRED,
                4,
                save_path,
            )
        )
        with self.assertRaisesRegex(ValueError, "requires save_dir and resume_step"):
            vlm._validate_training_request(
                config,
                checkpoint_utils.ResumeMode.REQUIRED,
                None,
                save_path,
            )
        with self.assertRaisesRegex(ValueError, "Unsupported resume mode"):
            vlm._validate_training_request(
                config,
                checkpoint_utils.ResumeMode.IF_PRESENT,
                None,
                save_path,
            )
        with self.assertRaisesRegex(ValueError, "exceeds schedule_horizon"):
            vlm._validate_training_request(
                vlm.TrainConfig(num_steps=101, schedule_horizon=100),
                checkpoint_utils.ResumeMode.NEVER,
                None,
                save_path,
            )

    def test_gradient_sum_accumulates_in_fp32_and_updates_once(self):
        optimizer = _make_optimizer()
        gradients = jax.tree.map(
            lambda value: jnp.full_like(value, 1.0),
            nnx.state(optimizer.model),
        )
        gradient_sum = initialize_gradient_sum(gradients)
        gradient_sum = accumulate_gradient_sum(gradient_sum, gradients)
        self.assertEqual(jax.tree.leaves(gradient_sum)[0].dtype, jnp.float32)

        grad_norm, healthy = apply_normalized_gradient_sum(
            optimizer,
            gradient_sum,
            jnp.asarray(2.0),
            jnp.asarray(1.0),
        )

        self.assertTrue(bool(healthy))
        self.assertAlmostEqual(float(grad_norm), 1.0)
        self.assertEqual(int(optimizer.step[...]), 1)

    def test_cross_entropy_is_weighted_by_supervised_tokens(self):
        hidden = jnp.asarray(
            [
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0], [1.0, -1.0], [0.5, 0.5]],
            ]
        )
        kernel = jnp.asarray([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
        targets = jnp.asarray([[0, 1, 2, 0], [1, 0, 2, 1]])
        masks = jnp.asarray([[0, 1, 0, 0], [0, 1, 1, 1]])

        combined_sum, combined_count = chunked_cross_entropy_loss_sum(
            hidden,
            kernel,
            targets,
            masks,
            num_tiles=1,
        )
        first_sum, first_count = chunked_cross_entropy_loss_sum(
            hidden[:1],
            kernel,
            targets[:1],
            masks[:1],
            num_tiles=1,
        )
        second_sum, second_count = chunked_cross_entropy_loss_sum(
            hidden[1:],
            kernel,
            targets[1:],
            masks[1:],
            num_tiles=1,
        )

        self.assertAlmostEqual(float(combined_sum), float(first_sum + second_sum), places=6)
        self.assertEqual(float(combined_count), 4.0)
        self.assertAlmostEqual(float(combined_count), float(first_count + second_count))
        weighted = (first_sum + second_sum) / (first_count + second_count)
        unweighted = (first_sum / first_count + second_sum / second_count) / 2
        self.assertAlmostEqual(float(combined_sum / combined_count), float(weighted), places=6)
        self.assertNotAlmostEqual(float(weighted), float(unweighted), places=4)

    def test_validation_is_token_weighted_and_resets_fixed_panel(self):
        iterator = _PanelIterator(
            [
                {"loss_sum": jnp.asarray(2.0), "tokens": jnp.asarray(1.0)},
                {"loss_sum": jnp.asarray(30.0), "tokens": jnp.asarray(3.0)},
            ]
        )

        def eval_step(_model, batch):
            return batch["loss_sum"], batch["tokens"]

        with mock.patch.object(
            vlm.vlm_api, "shard_batch_dict", side_effect=lambda batch, *_: batch
        ):
            loss, tokens, healthy = vlm._evaluate_validation_panel(
                eval_step,
                None,
                iterator,
                2,
                None,
                None,
            )

        self.assertEqual(float(loss), 8.0)
        self.assertEqual(float(tokens), 4.0)
        self.assertTrue(bool(healthy))
        self.assertEqual(iterator.index, 0)

    def test_compiled_update_has_no_host_read(self):
        optimizer = _make_optimizer()
        gradients = initialize_gradient_sum(nnx.state(optimizer.model))
        with mock.patch.object(jax, "device_get", side_effect=AssertionError("host read")):
            grad_norm, healthy = apply_normalized_gradient_sum(
                optimizer,
                gradients,
                jnp.asarray(1.0),
                jnp.asarray(1.0),
            )
            jax.block_until_ready((grad_norm, healthy))

    def test_real_orbax_resume_restores_optimizer_and_iterator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = vlm._make_checkpoint_manager(Path(tmpdir), save_interval=None)
            iterator = _make_iterator()
            next(iterator)
            optimizer = _make_optimizer()
            gradients = initialize_gradient_sum(nnx.state(optimizer.model))
            apply_normalized_gradient_sum(
                optimizer,
                gradients,
                jnp.asarray(1.0),
                jnp.asarray(1.0),
            )
            vlm._save_sft_checkpoint(
                manager,
                optimizer,
                jax.random.key(0),
                1,
                iterator,
                100,
                _MODEL_IDENTITY,
                jnp.asarray(True),
            )
            expected_next = next(iterator)["value"].copy()
            apply_normalized_gradient_sum(
                optimizer,
                gradients,
                jnp.asarray(1.0),
                jnp.asarray(1.0),
            )
            expected_weight = optimizer.model.weight[...]

            restored_optimizer, step, _, restored_iterator = vlm._restore_sft_checkpoint(
                manager,
                _make_optimizer(),
                jax.random.key(1),
                _make_iterator(),
                1,
                100,
                _MODEL_IDENTITY,
            )
            self.assertEqual(step, 1)
            self.assertEqual(int(restored_optimizer.step[...]), 1)
            apply_normalized_gradient_sum(
                restored_optimizer,
                gradients,
                jnp.asarray(1.0),
                jnp.asarray(1.0),
            )
            self.assertTrue(
                bool(jnp.array_equal(restored_optimizer.model.weight[...], expected_weight))
            )
            self.assertSequenceEqual(
                restored_iterator.__next__()["value"].tolist(), expected_next.tolist()
            )
            manager.close()

    def test_schedule_mismatch_rejects_before_optimizer_mutation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = vlm._make_checkpoint_manager(Path(tmpdir), save_interval=None)
            optimizer = _make_optimizer()
            gradients = initialize_gradient_sum(nnx.state(optimizer.model))
            apply_normalized_gradient_sum(
                optimizer,
                gradients,
                jnp.asarray(1.0),
                jnp.asarray(1.0),
            )
            vlm._save_sft_checkpoint(
                manager,
                optimizer,
                jax.random.key(0),
                1,
                _make_iterator(),
                100,
                _MODEL_IDENTITY,
                jnp.asarray(True),
            )
            candidate = _make_optimizer()

            with self.assertRaisesRegex(ValueError, "schedule horizon"):
                vlm._restore_sft_checkpoint(
                    manager,
                    candidate,
                    jax.random.key(1),
                    _make_iterator(),
                    1,
                    101,
                    _MODEL_IDENTITY,
                )

            self.assertEqual(int(candidate.step[...]), 0)
            manager.close()

    def test_model_identity_mismatch_rejects_before_optimizer_mutation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = vlm._make_checkpoint_manager(Path(tmpdir), save_interval=None)
            optimizer = _make_optimizer()
            gradients = initialize_gradient_sum(nnx.state(optimizer.model))
            apply_normalized_gradient_sum(
                optimizer,
                gradients,
                jnp.asarray(1.0),
                jnp.asarray(1.0),
            )
            vlm._save_sft_checkpoint(
                manager,
                optimizer,
                jax.random.key(0),
                1,
                _make_iterator(),
                100,
                _MODEL_IDENTITY,
                jnp.asarray(True),
            )
            candidate = _make_optimizer()

            with self.assertRaisesRegex(ValueError, "model snapshot"):
                vlm._restore_sft_checkpoint(
                    manager,
                    candidate,
                    jax.random.key(1),
                    _make_iterator(),
                    1,
                    100,
                    _MODEL_IDENTITY.at[0].set(255),
                )

            self.assertEqual(int(candidate.step[...]), 0)
            manager.close()

    def test_numerical_failure_rejects_before_save(self):
        manager = mock.Mock()
        with self.assertRaisesRegex(FloatingPointError, "step 1"):
            vlm._save_sft_checkpoint(
                manager,
                _make_optimizer(),
                jax.random.key(0),
                1,
                _make_iterator(),
                100,
                _MODEL_IDENTITY,
                jnp.asarray(False),
            )
        manager.save.assert_not_called()

    def test_cleanup_preserves_primary_and_closes_every_owned_iterator(self):
        first = _Closeable(RuntimeError("first cleanup"))
        second = _Closeable(RuntimeError("second cleanup"))
        cleanup = vlm._TrainingCleanup()
        cleanup.own_iterator(first)
        cleanup.own_iterator(second)
        primary = KeyboardInterrupt("primary")

        cleanup.close(primary)

        self.assertEqual(first.closed, 1)
        self.assertEqual(second.closed, 1)
        self.assertLen(primary.__notes__, 2)

    def test_cleanup_attempts_manager_close_after_wait_failure(self):
        manager = mock.Mock()
        manager.wait_until_finished.side_effect = RuntimeError("wait failed")
        cleanup = vlm._TrainingCleanup(checkpoint_manager=manager)
        primary = SystemExit(3)

        cleanup.close(primary)

        manager.close.assert_called_once_with()
        self.assertIn("wait failed", primary.__notes__[0])


if __name__ == "__main__":
    absltest.main()
