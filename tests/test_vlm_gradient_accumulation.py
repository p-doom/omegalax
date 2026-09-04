from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl.testing import absltest
from flax import nnx

from omegalax.trainers.loss import chunked_cross_entropy_loss, chunked_cross_entropy_loss_sum
from omegalax.trainers.optim import (
    MixedPrecisionOptimizer,
    accumulate_gradient_sum,
    apply_normalized_gradient_sum,
    initialize_gradient_sum,
)


class _TinyModel(nnx.Module):
    def __init__(self, dtype=jnp.float32):
        self.weight = nnx.Param(jnp.array([[0.75, -0.25, 0.5], [-0.4, 0.6, 0.2]], dtype=dtype))


def _make_optimizer(*, clip_norm: float = 0.0, dtype=jnp.float32) -> MixedPrecisionOptimizer:
    transforms = []
    if clip_norm:
        transforms.append(optax.clip_by_global_norm(clip_norm))
    transforms.append(optax.adamw(0.03, b1=0.8, b2=0.9, weight_decay=0.02))
    return MixedPrecisionOptimizer(_TinyModel(dtype=dtype), optax.chain(*transforms))


def _micro_gradient(optimizer, hidden, targets, mask):
    def loss_fn(model):
        return chunked_cross_entropy_loss_sum(
            hidden,
            model.weight[...],
            targets,
            mask,
            num_tiles=1,
        )

    return nnx.value_and_grad(loss_fn, has_aux=True)(optimizer.model)


def _snapshot(optimizer):
    return jax.tree.map(lambda value: np.asarray(value).copy(), nnx.pure(nnx.state(optimizer)))


def _assert_tree_allclose(testcase, actual, expected, *, atol=1e-7):
    testcase.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=0.0, atol=atol)


class VLMGradientAccumulationTest(absltest.TestCase):
    def test_tiled_ce_sum_and_count_define_mean(self):
        hidden = jnp.arange(2 * 7 * 2, dtype=jnp.float32).reshape(2, 7, 2) / 10.0
        kernel = jnp.array([[0.2, -0.3, 0.5], [0.7, 0.1, -0.4]], dtype=jnp.float32)
        targets = jnp.array([[0, 1, 2, 1, 0, 2, 1], [2, 0, 1, 2, 1, 0, 2]], dtype=jnp.int32)
        mask = jnp.array([[0, 1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1]], dtype=jnp.float32)

        loss_sum, supervised_tokens = chunked_cross_entropy_loss_sum(
            hidden, kernel, targets, mask, num_tiles=3
        )
        mean_loss = chunked_cross_entropy_loss(hidden, kernel, targets, mask, num_tiles=3)

        self.assertEqual(float(supervised_tokens), 7.0)
        np.testing.assert_allclose(mean_loss, loss_sum / supervised_tokens, rtol=0.0, atol=1e-7)

    def test_unequal_masks_equal_combined_clipped_update(self):
        candidate = _make_optimizer(clip_norm=0.15)
        reference = _make_optimizer(clip_norm=0.15)
        micros = (
            (
                jnp.array(
                    [[[1.0, 0.5], [-0.5, 1.0], [0.25, -1.0], [1.5, 0.5]]],
                    dtype=jnp.float32,
                ),
                jnp.array([[0, 1, 2, 1]], dtype=jnp.int32),
                jnp.array([[0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32),
            ),
            (
                jnp.array(
                    [[[2.0, -1.0], [0.25, 1.5], [-1.0, -0.5], [0.5, 0.75]]],
                    dtype=jnp.float32,
                ),
                jnp.array([[2, 0, 1, 2]], dtype=jnp.int32),
                jnp.array([[0.0, 1.0, 1.0, 1.0]], dtype=jnp.float32),
            ),
        )

        gradient_sum = None
        total_supervised = jnp.array(0.0, dtype=jnp.float32)
        total_loss = jnp.array(0.0, dtype=jnp.float32)
        for micro in micros:
            (loss_sum, supervised_tokens), gradients = _micro_gradient(candidate, *micro)
            gradient_sum = (
                initialize_gradient_sum(gradients)
                if gradient_sum is None
                else accumulate_gradient_sum(gradient_sum, gradients)
            )
            total_loss += loss_sum
            total_supervised += supervised_tokens
        candidate_grad_norm, _, _ = apply_normalized_gradient_sum(
            candidate,
            gradient_sum,
            total_supervised,
            total_loss,
        )

        def combined_objective(model):
            hidden = jnp.concatenate([micro[0] for micro in micros], axis=0)
            targets = jnp.concatenate([micro[1] for micro in micros], axis=0)
            mask = jnp.concatenate([micro[2] for micro in micros], axis=0)
            loss_sum, supervised_tokens = chunked_cross_entropy_loss_sum(
                hidden,
                model.weight[...],
                targets,
                mask,
                num_tiles=1,
            )
            return loss_sum / supervised_tokens

        reference_gradient = nnx.grad(combined_objective)(reference.model)
        reference_grad_norm = optax.tree.norm(reference_gradient)
        reference.update(reference_gradient)

        np.testing.assert_allclose(candidate_grad_norm, reference_grad_norm, rtol=0.0, atol=1e-7)
        _assert_tree_allclose(self, _snapshot(candidate), _snapshot(reference))
        self.assertEqual(int(candidate.step[...]), 1)

    def test_grad_accum_one_equals_direct_update(self):
        candidate = _make_optimizer()
        reference = _make_optimizer()
        micro = (
            jnp.array([[[1.0, -2.0], [0.5, 1.5], [-0.25, 0.75]]], dtype=jnp.float32),
            jnp.array([[0, 2, 1]], dtype=jnp.int32),
            jnp.array([[0.0, 1.0, 1.0]], dtype=jnp.float32),
        )

        (loss_sum, supervised_tokens), gradients = _micro_gradient(candidate, *micro)
        apply_normalized_gradient_sum(
            candidate,
            initialize_gradient_sum(gradients),
            supervised_tokens,
            loss_sum,
        )

        def direct_objective(model):
            hidden, targets, mask = micro
            direct_loss_sum, direct_supervised_tokens = chunked_cross_entropy_loss_sum(
                hidden,
                model.weight[...],
                targets,
                mask,
                num_tiles=1,
            )
            return direct_loss_sum / direct_supervised_tokens

        reference.update(nnx.grad(direct_objective)(reference.model))
        _assert_tree_allclose(self, _snapshot(candidate), _snapshot(reference))

    def test_bf16_gradients_upcast_before_cancellation(self):
        candidate = _make_optimizer(clip_norm=0.15, dtype=jnp.bfloat16)
        reference = _make_optimizer(clip_norm=0.15, dtype=jnp.bfloat16)
        micros = (
            (
                jnp.array(
                    [[[1.0, 0.5], [-0.5, 1.0], [0.25, -1.0], [1.5, 0.5]]],
                    dtype=jnp.float32,
                ),
                jnp.array([[0, 1, 2, 1]], dtype=jnp.int32),
                jnp.array([[0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32),
            ),
            (
                jnp.array(
                    [[[2.0, -1.0], [0.25, 1.5], [-1.0, -0.5], [0.5, 0.75]]],
                    dtype=jnp.float32,
                ),
                jnp.array([[2, 0, 1, 2]], dtype=jnp.int32),
                jnp.array([[0.0, 1.0, 1.0, 1.0]], dtype=jnp.float32),
            ),
        )

        raw_gradients = []
        total_loss = jnp.array(0.0, dtype=jnp.float32)
        total_supervised = jnp.array(0.0, dtype=jnp.float32)
        for micro in micros:
            (loss_sum, supervised_tokens), gradients = _micro_gradient(candidate, *micro)
            raw_gradients.append(gradients)
            total_loss += loss_sum
            total_supervised += supervised_tokens

        gradient_sum = accumulate_gradient_sum(
            initialize_gradient_sum(raw_gradients[0]), raw_gradients[1]
        )
        reference_sum = jax.tree.map(
            lambda first, second: first.astype(jnp.float32) + second.astype(jnp.float32),
            raw_gradients[0],
            raw_gradients[1],
        )
        bf16_sum = jax.tree.map(jnp.add, raw_gradients[0], raw_gradients[1])

        self.assertTrue(all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(gradient_sum)))
        for accumulated, first in zip(
            jax.tree.leaves(gradient_sum), jax.tree.leaves(raw_gradients[0])
        ):
            self.assertEqual(accumulated.sharding, first.sharding)
        self.assertTrue(
            any(
                not np.array_equal(np.asarray(bf16, dtype=np.float32), np.asarray(fp32))
                for bf16, fp32 in zip(jax.tree.leaves(bf16_sum), jax.tree.leaves(reference_sum))
            )
        )

        apply_normalized_gradient_sum(candidate, gradient_sum, total_supervised, total_loss)
        apply_normalized_gradient_sum(reference, reference_sum, total_supervised, total_loss)
        _assert_tree_allclose(self, _snapshot(candidate), _snapshot(reference), atol=0.0)


if __name__ == "__main__":
    absltest.main()
