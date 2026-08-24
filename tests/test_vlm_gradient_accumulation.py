"""CPU properties for the VLM token-normalized accumulation contract."""

from __future__ import annotations

import dataclasses
import os
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl.testing import absltest
from flax import nnx

from omegalax.models.qwen3_vl import make_vl_config
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.trainers import vlm as vlm_trainer
from omegalax.trainers.loss import chunked_cross_entropy_loss, chunked_cross_entropy_loss_sum
from omegalax.trainers.optim import (
    MixedPrecisionOptimizer,
    accumulate_gradient_sum,
    apply_normalized_gradient_sum,
    initialize_gradient_sum,
)
from omegalax.trainers.vlm import require_zero_router_aux_loss


class _TinyModel(nnx.Module):
    def __init__(self, dtype=jnp.float32):
        self.weight = nnx.Param(jnp.array([[0.75, -0.25, 0.5], [-0.4, 0.6, 0.2]], dtype=dtype))


def _make_optimizer(*, clip_norm: float = 0.0, dtype=jnp.float32) -> MixedPrecisionOptimizer:
    chain = []
    if clip_norm:
        chain.append(optax.clip_by_global_norm(clip_norm))
    chain.append(optax.adamw(0.03, b1=0.8, b2=0.9, weight_decay=0.02))
    return MixedPrecisionOptimizer(_TinyModel(dtype=dtype), optax.chain(*chain))


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


def _assert_tree_equal(testcase, actual, expected):
    testcase.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


class VLMGradientAccumulationTest(absltest.TestCase):
    def test_bf16_microgradients_accumulate_in_fp32_before_cancellation(self):
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
        total_ce_loss = jnp.array(0.0, dtype=jnp.float32)
        total_supervised = jnp.array(0.0, dtype=jnp.float32)
        for micro in micros:
            (ce_loss_sum, supervised_tokens), gradients = _micro_gradient(candidate, *micro)
            raw_gradients.append(gradients)
            total_ce_loss += ce_loss_sum
            total_supervised += supervised_tokens
        self.assertTrue(
            all(leaf.dtype == jnp.bfloat16 for leaf in jax.tree.leaves(raw_gradients[0]))
        )

        candidate_sum = initialize_gradient_sum(raw_gradients[0])
        candidate_sum = accumulate_gradient_sum(candidate_sum, raw_gradients[1])
        concatenated_fp32_sum = jax.tree.map(
            lambda first, second: jnp.sum(
                jnp.stack([first.astype(jnp.float32), second.astype(jnp.float32)]), axis=0
            ),
            raw_gradients[0],
            raw_gradients[1],
        )
        naive_bf16_sum = jax.tree.map(jnp.add, raw_gradients[0], raw_gradients[1])

        self.assertTrue(all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(candidate_sum)))
        for accumulated, first in zip(
            jax.tree.leaves(candidate_sum), jax.tree.leaves(raw_gradients[0])
        ):
            self.assertEqual(accumulated.sharding, first.sharding)
        self.assertTrue(
            any(
                not np.array_equal(
                    np.asarray(naive, dtype=np.float32), np.asarray(fp32_accumulated)
                )
                for naive, fp32_accumulated in zip(
                    jax.tree.leaves(naive_bf16_sum), jax.tree.leaves(concatenated_fp32_sum)
                )
            )
        )

        zero_aux = jnp.array(0.0, dtype=jnp.float32)
        apply_normalized_gradient_sum(
            candidate, candidate_sum, total_ce_loss, total_supervised, zero_aux
        )
        apply_normalized_gradient_sum(
            reference, concatenated_fp32_sum, total_ce_loss, total_supervised, zero_aux
        )
        _assert_tree_allclose(self, _snapshot(candidate), _snapshot(reference), atol=0.0)

    def test_tiled_ce_sum_and_count_define_the_existing_mean(self):
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

    def test_real_dense_vlm_unequal_masks_equal_concatenated_update(self):
        cfg = make_vl_config("qwen3-vl-smoke")
        cfg = dataclasses.replace(
            cfg,
            dtype=jnp.float32,
            vision=dataclasses.replace(cfg.vision, dtype=jnp.float32),
        )
        token_ids = np.array(
            [[11, 12, 13, 14], [21, 22, 23, 24]],
            dtype=np.int32,
        )
        attention_mask = np.ones_like(token_ids)
        loss_mask = np.array(
            [[0, 0, 0, 1], [0, 1, 1, 1]],
            dtype=np.int32,
        )
        micros = [
            {
                "token_ids_BT": token_ids[index : index + 1],
                "attention_mask_BT": attention_mask[index : index + 1],
                "loss_mask_BT": loss_mask[index : index + 1],
            }
            for index in range(2)
        ]
        combined = {
            "token_ids_BT": token_ids,
            "attention_mask_BT": attention_mask,
            "loss_mask_BT": loss_mask,
        }
        common = {
            "seed": 7,
            "seq_len": 4,
            "num_steps": 1,
            "learning_rate": 3e-3,
            "weight_decay": 0.01,
            "max_grad_norm": 0.2,
            "print_every": 0,
        }

        accumulated, accumulated_metrics = vlm_trainer.run_sft(
            cfg,
            vlm_trainer.TrainConfig(batch_size=1, grad_accum_steps=2, **common),
            iter(micros),
            log_every=0,
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
            text_attn_backend="xla",
        )
        reference, reference_metrics = vlm_trainer.run_sft(
            cfg,
            vlm_trainer.TrainConfig(batch_size=2, grad_accum_steps=1, **common),
            iter([combined]),
            log_every=0,
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
            text_attn_backend="xla",
        )

        _assert_tree_allclose(
            self,
            _snapshot(accumulated),
            _snapshot(reference),
            atol=2e-6,
        )
        np.testing.assert_allclose(
            accumulated_metrics["loss"], reference_metrics["loss"], rtol=0.0, atol=2e-6
        )
        self.assertEqual(int(accumulated.step[...]), 1)
        self.assertEqual(float(accumulated_metrics["aux_loss"]), 0.0)

    def test_unequal_masks_equal_one_combined_clipped_update(self):
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
        total_ce_loss = jnp.array(0.0, dtype=jnp.float32)
        total_supervised = jnp.array(0.0, dtype=jnp.float32)
        for micro in micros:
            (ce_loss_sum, supervised_tokens), gradients = _micro_gradient(candidate, *micro)
            gradient_sum = (
                initialize_gradient_sum(gradients)
                if gradient_sum is None
                else accumulate_gradient_sum(gradient_sum, gradients)
            )
            total_ce_loss += ce_loss_sum
            total_supervised += supervised_tokens
        candidate_grad_norm = apply_normalized_gradient_sum(
            candidate,
            gradient_sum,
            total_ce_loss,
            total_supervised,
            jnp.array(0.0, dtype=jnp.float32),
        )

        def combined_objective(model):
            hidden = jnp.concatenate([micro[0] for micro in micros], axis=0)
            targets = jnp.concatenate([micro[1] for micro in micros], axis=0)
            mask = jnp.concatenate([micro[2] for micro in micros], axis=0)
            ce_loss_sum, supervised_tokens = chunked_cross_entropy_loss_sum(
                hidden,
                model.weight[...],
                targets,
                mask,
                num_tiles=1,
            )
            return ce_loss_sum / supervised_tokens

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

        (ce_loss_sum, supervised_tokens), gradient_sum = _micro_gradient(candidate, *micro)
        apply_normalized_gradient_sum(
            candidate,
            initialize_gradient_sum(gradient_sum),
            ce_loss_sum,
            supervised_tokens,
            jnp.array(0.0, dtype=jnp.float32),
        )

        def direct_objective(model):
            hidden, targets, mask = micro
            ce_loss_sum, supervised_tokens = chunked_cross_entropy_loss_sum(
                hidden,
                model.weight[...],
                targets,
                mask,
                num_tiles=1,
            )
            return ce_loss_sum / supervised_tokens

        reference.update(nnx.grad(direct_objective)(reference.model))
        _assert_tree_allclose(self, _snapshot(candidate), _snapshot(reference))

    def test_invalid_loss_or_count_rejected_before_optimizer_mutation(self):
        optimizer = _make_optimizer(clip_norm=0.15)
        before = _snapshot(optimizer)
        zero_gradients = jax.tree.map(jnp.zeros_like, nnx.state(optimizer.model, nnx.Param))

        cases = (
            ("negative loss", -1.0, 1.0, "CE loss sum"),
            ("NaN loss", jnp.nan, 1.0, "CE loss sum"),
            ("infinite loss", jnp.inf, 1.0, "CE loss sum"),
            ("zero count", 1.0, 0.0, "supervised-token count"),
            ("negative count", 1.0, -1.0, "supervised-token count"),
            ("NaN count", 1.0, jnp.nan, "supervised-token count"),
            ("infinite count", 1.0, jnp.inf, "supervised-token count"),
        )
        for name, loss, count, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                apply_normalized_gradient_sum(
                    optimizer,
                    initialize_gradient_sum(zero_gradients),
                    jnp.array(loss, dtype=jnp.float32),
                    jnp.array(count, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                )
            _assert_tree_equal(self, _snapshot(optimizer), before)

    def test_nonfinite_gradient_or_norm_rejected_before_optimizer_mutation(self):
        optimizer = _make_optimizer(clip_norm=0.15)
        before = _snapshot(optimizer)
        gradient_template = nnx.state(optimizer.model, nnx.Param)

        cases = (
            ("NaN gradient", jnp.nan),
            ("infinite gradient", jnp.inf),
            ("overflowing norm", jnp.finfo(jnp.float32).max / 2),
        )
        for name, value in cases:
            gradients = jax.tree.map(
                lambda gradient, fill=value: jnp.full_like(gradient, fill), gradient_template
            )
            if name == "overflowing norm":
                self.assertTrue(
                    all(np.all(np.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))
                )
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "global norm"):
                apply_normalized_gradient_sum(
                    optimizer,
                    gradients,
                    jnp.array(1.0, dtype=jnp.float32),
                    jnp.array(1.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                )
            _assert_tree_equal(self, _snapshot(optimizer), before)

    def test_unexpected_runtime_aux_is_rejected_before_optimizer_mutation(self):
        cfg = make_vl_config("qwen3-vl-smoke")
        require_zero_router_aux_loss(cfg)
        model, cfg = vlm_trainer.vlm_api.init_model(
            cfg,
            jax.random.key(3),
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )
        set_attn_backend(model, text_backend="xla")
        mesh = vlm_trainer.ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        with vlm_trainer.mesh_rules(mesh):
            optimizer = vlm_trainer.build_optimizer(
                model,
                1e-3,
                vlm_trainer.TrainConfig(batch_size=1, seq_len=4, num_steps=1),
            )
        gradient_step = vlm_trainer.make_sft_gradient_step(cfg, num_loss_tiles=1)
        batch = {
            "token_ids_BT": jnp.array([[11, 12, 13, 14]], dtype=jnp.int32),
            "attention_mask_BT": jnp.ones((1, 4), dtype=jnp.int32),
            "loss_mask_BT": jnp.array([[0, 0, 1, 1]], dtype=jnp.int32),
        }
        real_forward = vlm_trainer.vlm_api.forward

        def forward_with_unexpected_aux(*args, **kwargs):
            hidden, _ = real_forward(*args, **kwargs)
            return hidden, jnp.array(0.125, dtype=jnp.float32)

        before = _snapshot(optimizer)
        with mock.patch.object(
            vlm_trainer.vlm_api, "forward", side_effect=forward_with_unexpected_aux
        ):
            gradients, metrics = gradient_step(optimizer.model, batch)
        for invalid_aux in (
            jnp.abs(metrics["aux_loss"]),
            jnp.array(jnp.nan, dtype=jnp.float32),
            jnp.array(jnp.inf, dtype=jnp.float32),
        ):
            with self.assertRaisesRegex(ValueError, "router auxiliary loss must be finite"):
                apply_normalized_gradient_sum(
                    optimizer,
                    initialize_gradient_sum(gradients),
                    metrics["ce_loss_sum"],
                    metrics["supervised_tokens"],
                    invalid_aux,
                )

        after = _snapshot(optimizer)
        _assert_tree_equal(self, after, before)

    def test_real_dense_target_is_supported_but_moe_aux_is_rejected(self):
        require_zero_router_aux_loss(make_vl_config("qwen3-vl-smoke"))

        with (
            mock.patch.object(vlm_trainer.vlm_api, "init_model") as init_model,
            self.assertRaisesRegex(ValueError, "router auxiliary loss"),
        ):
            vlm_trainer.run_sft(
                make_vl_config("qwen3-vl-smoke-moe"),
                vlm_trainer.TrainConfig(num_steps=1),
                iter(()),
            )
        init_model.assert_not_called()


if __name__ == "__main__":
    absltest.main()
