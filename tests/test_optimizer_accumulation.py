import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import optax
from absl.testing import absltest, parameterized
from flax import nnx

from omegalax.trainers import text, vlm
from omegalax.trainers.optim import MixedPrecisionOptimizer


class _TinyModel(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 16)


def _build(trainer, model, grad_accum_steps):
    cfg = trainer.TrainConfig(
        num_steps=4,
        learning_rate=1e-2,
        weight_decay=0.01,
        max_grad_norm=1.0,
        grad_accum_steps=grad_accum_steps,
    )
    if trainer is vlm:
        return trainer.build_optimizer(model, 1e-2, cfg)
    return trainer.build_optimizer(model, cfg)


class OptimizerAccumulationTest(parameterized.TestCase):
    @parameterized.parameters(text, vlm)
    def test_invalid_accumulation_count_fails(self, trainer):
        with self.assertRaisesRegex(ValueError, "grad_accum_steps must be at least 1"):
            _build(trainer, _TinyModel(), grad_accum_steps=0)

    @parameterized.parameters(text, vlm)
    def test_single_step_has_no_accumulator(self, trainer):
        optimizer = _build(trainer, _TinyModel(), grad_accum_steps=1)
        state = nnx.to_pure_dict(nnx.state(optimizer.opt_state))
        self.assertNotIn("acc_grads", state)

    @parameterized.parameters(text, vlm)
    def test_multiple_steps_has_accumulator(self, trainer):
        optimizer = _build(trainer, _TinyModel(), grad_accum_steps=2)
        state = nnx.to_pure_dict(nnx.state(optimizer.opt_state))
        self.assertIn("acc_grads", state)

    def test_single_step_update_matches_multisteps_one(self):
        optimizer = _build(vlm, _TinyModel(), grad_accum_steps=1)

        reference_model = _TinyModel()
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(1e-2, weight_decay=0.01),
        )
        reference = MixedPrecisionOptimizer(
            reference_model,
            optax.MultiSteps(tx, every_k_schedule=1),
        )

        def update(opt):
            grads = nnx.grad(lambda model: jnp.sum(model.weight[...] ** 2))(opt.model)
            opt.update(grads)

        for _ in range(2):
            update(optimizer)
            update(reference)

        np.testing.assert_array_equal(
            np.asarray(optimizer.model.weight[...]),
            np.asarray(reference.model.weight[...]),
        )


if __name__ == "__main__":
    absltest.main()
