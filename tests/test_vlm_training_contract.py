"""Focused tests for the VLM update and resume boundaries."""

import inspect
import os
import tempfile
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax
from absl.testing import absltest
from flax import nnx

from omegalax.distributed.mesh import ensure_mesh
from omegalax.trainers import checkpoint_utils, vlm
from omegalax.trainers.optim import (
    MixedPrecisionOptimizer,
    accumulate_gradient_sum,
    apply_normalized_gradient_sum,
    initialize_gradient_sum,
)


class _ScalarModel(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.asarray([1.0], dtype=jnp.bfloat16))


class _Closable:
    def __init__(self, events: list[str], name: str):
        self.events = events
        self.name = name

    def close(self):
        self.events.append(self.name)


class VLMTrainingContractTest(absltest.TestCase):
    def test_single_device_mesh_is_supported(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        self.assertEqual(mesh.size, 1)

    def test_resume_requires_exact_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            self.assertTrue(
                vlm._validate_resume_request(
                    checkpoint_utils.ResumeMode.REQUIRED,
                    4,
                    save_path,
                    10,
                )
            )
            with self.assertRaisesRegex(ValueError, "requires save_dir and resume_step"):
                vlm._validate_resume_request(
                    checkpoint_utils.ResumeMode.REQUIRED,
                    None,
                    save_path,
                    10,
                )
            with self.assertRaisesRegex(ValueError, "only valid"):
                vlm._validate_resume_request(
                    checkpoint_utils.ResumeMode.NEVER,
                    4,
                    save_path,
                    10,
                )

    def test_gradient_sum_accumulates_in_fp32(self):
        total = initialize_gradient_sum({"weight": jnp.asarray([1.0], dtype=jnp.bfloat16)})
        total = accumulate_gradient_sum(
            total,
            {"weight": jnp.asarray([2.0], dtype=jnp.bfloat16)},
        )
        self.assertEqual(total["weight"].dtype, jnp.float32)
        self.assertEqual(float(total["weight"][0]), 3.0)

    def test_optimizer_update_has_no_host_read(self):
        source = inspect.getsource(apply_normalized_gradient_sum)
        self.assertNotIn("device_get", source)
        self.assertNotIn("np.asarray", source)

    def test_gradient_sum_is_normalized_once(self):
        model = _ScalarModel()
        optimizer = MixedPrecisionOptimizer(model, optax.sgd(0.1))
        gradients = jax.tree.map(
            lambda value: jnp.full_like(value, 2.0),
            nnx.state(model),
        )
        gradient_sum = initialize_gradient_sum(gradients)

        grad_norm, loss, healthy = apply_normalized_gradient_sum(
            optimizer,
            gradient_sum,
            jnp.asarray(2.0),
            jnp.asarray(1.0),
        )

        self.assertTrue(bool(healthy))
        self.assertAlmostEqual(float(grad_norm), 1.0)
        self.assertAlmostEqual(float(loss), 0.5)
        self.assertAlmostEqual(float(model.weight[0]), 0.9, places=2)

    def test_numerical_failure_stops_at_boundary(self):
        with self.assertRaisesRegex(FloatingPointError, "step 7"):
            vlm._require_healthy_at_boundary(jnp.asarray(False), 7)

    def test_owned_iterators_close_after_failure(self):
        events: list[str] = []
        data_iter = _Closable(events, "train")
        val_data_iter = _Closable(events, "validation")
        with (
            mock.patch.object(vlm, "_run_sft", side_effect=RuntimeError("training failed")),
            self.assertRaisesRegex(RuntimeError, "training failed"),
        ):
            vlm.run_sft(
                "unused",
                vlm.TrainConfig(),
                data_iter,
                val_data_iter=val_data_iter,
            )

        self.assertEqual(events, ["train", "validation"])


if __name__ == "__main__":
    absltest.main()
