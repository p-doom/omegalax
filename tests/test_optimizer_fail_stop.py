"""CPU properties for the named fused optimizer transaction."""

from __future__ import annotations

import inspect
import os
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl.testing import absltest
from flax import nnx

from omegalax.trainers import optim as optim_lib


class _TinyModel(nnx.Module):
    def __init__(self, *, dtype=jnp.float32, size=2):
        self.weight = nnx.Param(jnp.linspace(-0.25, 0.75, size, dtype=dtype))


def _optimizer(*, dtype=jnp.float32, weight_decay=0.02, size=2):
    return optim_lib.MixedPrecisionOptimizer(
        _TinyModel(dtype=dtype, size=size),
        optim_lib.generation_adamw(weight_decay=weight_decay),
    )


def _gradients(values=(0.2, -0.4)):
    return nnx.State({"weight": nnx.Param(jnp.asarray(values, dtype=jnp.float32))})


def _healthy_status():
    return jnp.asarray(optim_lib.OptimizerFatalStatus.HEALTHY, dtype=jnp.uint8)


def _apply(
    optimizer,
    gradients=None,
    *,
    loss=1.0,
    supervised=2.0,
    auxiliary=0.0,
    status=None,
    clip_norm=0.15,
    learning_rate=0.03,
    generation=None,
):
    if gradients is None:
        gradients = _gradients()
    if status is None:
        status = _healthy_status()
    if generation is None:
        generation = int(optimizer.step[...]) + 1
    return optim_lib.apply_normalized_gradient_sum(
        optimizer,
        gradients,
        jnp.asarray(loss, dtype=jnp.float32),
        jnp.asarray(supervised, dtype=jnp.float32),
        jnp.asarray(auxiliary, dtype=jnp.float32),
        status,
        clip_norm,
        jnp.asarray(learning_rate, dtype=jnp.float32),
        jnp.asarray(generation, dtype=jnp.int32),
    )


def _snapshot(optimizer):
    return jax.tree.map(lambda value: np.asarray(value).copy(), nnx.pure(nnx.state(optimizer)))


def _assert_tree_bit_equal(testcase, actual, expected):
    testcase.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        testcase.assertEqual(actual_leaf.dtype, expected_leaf.dtype)
        testcase.assertEqual(actual_leaf.shape, expected_leaf.shape)
        testcase.assertEqual(actual_leaf.tobytes(), expected_leaf.tobytes())


def _adam_state(optimizer):
    leaves = jax.tree.leaves(nnx.pure(optimizer.opt_state))
    return tuple(np.asarray(value) for value in leaves if np.issubdtype(value.dtype, np.inexact))


class OptimizerFailStopTest(absltest.TestCase):
    def test_generation_adamw_matches_canonical_one_and_two_steps(self):
        schedules = {
            "constant": (0.03, 0.03),
            "scheduled": (0.03, 0.007),
        }
        gradients = (_gradients((0.8, -0.6)), _gradients((-0.1, 0.5)))
        for schedule_name, rates in schedules.items():
            for weight_decay in (0.0, 0.02):
                with self.subTest(schedule=schedule_name, weight_decay=weight_decay):
                    candidate = _optimizer(dtype=jnp.bfloat16, weight_decay=weight_decay)
                    reference = optim_lib.MixedPrecisionOptimizer(
                        _TinyModel(dtype=jnp.bfloat16),
                        optax.adamw(
                            lambda count, schedule_rates=rates: jnp.asarray(schedule_rates)[count],
                            weight_decay=weight_decay,
                        ),
                    )
                    status = _healthy_status()
                    for generation, (gradient, rate) in enumerate(
                        zip(gradients, rates, strict=True), start=1
                    ):
                        status, _ = _apply(
                            candidate,
                            gradient,
                            status=status,
                            learning_rate=rate,
                            generation=generation,
                        )
                        normalized = nnx.State(
                            {
                                "weight": nnx.Param(
                                    gradient["weight"].get_value()
                                    / jnp.asarray(2.0, dtype=jnp.float32)
                                )
                            }
                        )
                        clipped, _ = optax.clip_by_global_norm(0.15).update(normalized, ())
                        reference.update(clipped)

                        self.assertEqual(int(status), optim_lib.OptimizerFatalStatus.HEALTHY)
                        np.testing.assert_array_equal(
                            candidate.model.weight[...], reference.model.weight[...]
                        )
                        for actual, expected in zip(
                            _adam_state(candidate), _adam_state(reference), strict=True
                        ):
                            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
                        self.assertEqual(int(candidate.step[...]), generation)

    def test_dispatch_has_no_explicit_host_read_or_second_clip(self):
        optimizer = _optimizer()
        with (
            mock.patch.object(
                optim_lib.jax,
                "device_get",
                side_effect=AssertionError("device_get in update dispatch"),
            ),
            mock.patch.object(
                optim_lib.jax,
                "block_until_ready",
                side_effect=AssertionError("block_until_ready in update dispatch"),
            ),
        ):
            status, grad_norm = _apply(optimizer)

        self.assertEqual(int(status), optim_lib.OptimizerFatalStatus.HEALTHY)
        self.assertTrue(np.isfinite(np.asarray(grad_norm)).item())
        source = inspect.getsource(optim_lib)
        self.assertNotIn("clip_by_global_norm", source)
        self.assertNotIn("_normalized_gradient_health", source)

    def test_named_generation_requires_exact_scalar_int32(self):
        for generation in (
            jnp.asarray(True),
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray([1], dtype=jnp.int32),
        ):
            optimizer = _optimizer()
            before = _snapshot(optimizer)
            with (
                self.subTest(generation=generation),
                self.assertRaisesRegex(TypeError, "scalar int32"),
            ):
                optim_lib.apply_normalized_gradient_sum(
                    optimizer,
                    _gradients(),
                    jnp.asarray(1.0, dtype=jnp.float32),
                    jnp.asarray(2.0, dtype=jnp.float32),
                    jnp.asarray(0.0, dtype=jnp.float32),
                    _healthy_status(),
                    0.15,
                    jnp.asarray(0.03, dtype=jnp.float32),
                    generation,
                )
            _assert_tree_bit_equal(self, _snapshot(optimizer), before)

    def test_invalid_health_and_generation_preserve_every_optimizer_bit(self):
        max_float = jnp.finfo(jnp.float32).max
        cases = (
            ("loss", {"loss": jnp.nan}, optim_lib.OptimizerFatalStatus.INVALID_LOSS),
            (
                "supervision",
                {"supervised": 0.0},
                optim_lib.OptimizerFatalStatus.INVALID_SUPERVISION,
            ),
            (
                "auxiliary",
                {"auxiliary": 0.25},
                optim_lib.OptimizerFatalStatus.INVALID_AUXILIARY_LOSS,
            ),
            (
                "gradient",
                {"gradients": _gradients((jnp.nan, 0.0))},
                optim_lib.OptimizerFatalStatus.INVALID_GRADIENT,
            ),
            (
                "gradient_norm",
                {"gradients": _gradients((max_float, max_float))},
                optim_lib.OptimizerFatalStatus.INVALID_GRADIENT_NORM,
            ),
            (
                "clip_norm",
                {"clip_norm": jnp.nan},
                optim_lib.OptimizerFatalStatus.INVALID_CLIP_NORM,
            ),
            (
                "learning_rate",
                {"learning_rate": jnp.nan},
                optim_lib.OptimizerFatalStatus.INVALID_LEARNING_RATE,
            ),
            (
                "generation_zero",
                {"generation": 0},
                optim_lib.OptimizerFatalStatus.INVALID_GENERATION,
            ),
            (
                "generation_skip",
                {"generation": 2},
                optim_lib.OptimizerFatalStatus.INVALID_GENERATION,
            ),
        )

        for name, kwargs, expected_status in cases:
            optimizer = _optimizer()
            before = _snapshot(optimizer)
            with self.subTest(name=name):
                status, _ = _apply(optimizer, **kwargs)
                self.assertEqual(int(status), expected_status)
                _assert_tree_bit_equal(self, _snapshot(optimizer), before)

        poisoned = _optimizer()
        poisoned.model.weight[...] = jnp.array([jnp.nan, -0.25], dtype=jnp.float32)
        before = _snapshot(poisoned)
        status, _ = _apply(poisoned)
        self.assertEqual(int(status), optim_lib.OptimizerFatalStatus.INVALID_CURRENT_STATE)
        _assert_tree_bit_equal(self, _snapshot(poisoned), before)

        poisoned_momentum = _optimizer()
        poisoned_momentum.opt_state[0].mu["weight"][...] = jnp.array(
            [jnp.inf, 0.0], dtype=jnp.float32
        )
        before = _snapshot(poisoned_momentum)
        status, _ = _apply(poisoned_momentum)
        self.assertEqual(int(status), optim_lib.OptimizerFatalStatus.INVALID_CURRENT_STATE)
        _assert_tree_bit_equal(self, _snapshot(poisoned_momentum), before)

    def test_counter_schema_and_candidate_overflow_fail_before_commit(self):
        canonical = _optimizer()
        exact_leaves = [
            value
            for value in jax.tree.leaves(nnx.pure(canonical.opt_state))
            if not np.issubdtype(value.dtype, np.inexact)
        ]
        self.assertLen(exact_leaves, 1)
        self.assertEqual(exact_leaves[0].dtype, jnp.int32)
        self.assertEqual(canonical.step.dtype, jnp.uint32)

        optimizer = _optimizer()
        optimizer.opt_state[0].count[...] = jnp.asarray(1, dtype=jnp.int32)
        before = _snapshot(optimizer)
        status, _ = _apply(optimizer, generation=1)
        self.assertEqual(int(status), optim_lib.OptimizerFatalStatus.INVALID_GENERATION)
        _assert_tree_bit_equal(self, _snapshot(optimizer), before)

        historical = optim_lib.MixedPrecisionOptimizer(
            _TinyModel(),
            optax.adamw(lambda count: jnp.asarray(0.03), weight_decay=0.02),
        )
        before = _snapshot(historical)
        with self.assertRaisesRegex(TypeError, "exactly one Adam count"):
            _apply(historical)
        _assert_tree_bit_equal(self, _snapshot(historical), before)

        overflow = _optimizer(weight_decay=1.0)
        overflow.model.weight[...] = jnp.full(
            (2,), jnp.finfo(jnp.float32).max / 2, dtype=jnp.float32
        )
        before = _snapshot(overflow)
        status, _ = _apply(
            overflow,
            _gradients((0.0, 0.0)),
            learning_rate=4.0,
        )
        self.assertEqual(int(status), optim_lib.OptimizerFatalStatus.INVALID_CANDIDATE_STATE)
        _assert_tree_bit_equal(self, _snapshot(overflow), before)

    def test_fatal_status_is_sticky_and_typed_boundary_preserves_code(self):
        optimizer = _optimizer()
        before = _snapshot(optimizer)
        status, _ = _apply(optimizer, _gradients((jnp.nan, 0.0)))
        status, _ = _apply(optimizer, status=status)
        self.assertEqual(int(status), optim_lib.OptimizerFatalStatus.INVALID_GRADIENT)
        _assert_tree_bit_equal(self, _snapshot(optimizer), before)
        with self.assertRaisesRegex(
            FloatingPointError,
            "final: invalid_gradient",
        ):
            optim_lib.require_healthy_optimizer_status(
                status, optim_lib.OptimizerStatusBoundary.FINAL
            )

    def test_compiled_transaction_donates_state_and_gradient(self):
        optimizer = _optimizer(size=1024)
        gradients = _gradients(jnp.linspace(-0.4, 0.2, 1024, dtype=jnp.float32))
        compiled = optim_lib.apply_normalized_gradient_sum.lower(
            optimizer,
            gradients,
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(2.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            _healthy_status(),
            0.15,
            jnp.asarray(0.03, dtype=jnp.float32),
            jnp.asarray(1, dtype=jnp.int32),
        ).compile()
        optimizer_bytes = sum(value.nbytes for value in jax.tree.leaves(_snapshot(optimizer)))
        gradient_bytes = sum(value.nbytes for value in jax.tree.leaves(nnx.pure(gradients)))
        memory = compiled.memory_analysis()

        self.assertEqual(memory.alias_size_in_bytes, optimizer_bytes + gradient_bytes)
        self.assertLess(memory.temp_size_in_bytes, optimizer_bytes)


if __name__ == "__main__":
    absltest.main()
