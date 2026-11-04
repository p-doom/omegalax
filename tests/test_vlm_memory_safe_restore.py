"""Tests for array-free VLM checkpoint reconstruction."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from omegalax.trainers import vlm


class _TinyModel(nnx.Module):
    def __init__(self):
        self.linear = nnx.Linear(4, 3, rngs=nnx.Rngs(7))

    def __call__(self, x):
        return self.linear(x)


class _TinyBF16Model(nnx.Module):
    def __init__(self):
        self.linear = nnx.Linear(
            4,
            3,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=nnx.Rngs(9),
        )

    def __call__(self, x):
        return self.linear(x)


def _host_state(state):
    return jax.tree.map(lambda value: np.asarray(jax.device_get(value)).copy(), state)


class MemorySafeRestoreTest(absltest.TestCase):
    def test_trained_promotion_paths_are_exact(self):
        self.assertEqual(
            vlm._trained_promotion_group(("opt_state", "acc_grads", "layer", "kernel")),
            "acc_grads",
        )
        self.assertEqual(
            vlm._trained_promotion_group(
                ("opt_state", "inner_opt_state", 1, 0, "mu", "layer", "kernel")
            ),
            "mu",
        )
        self.assertIsNone(
            vlm._trained_promotion_group(
                ("opt_state", "inner_opt_state", 1, "trace", "mu", "kernel")
            )
        )
        self.assertIsNone(
            vlm._trained_promotion_group(
                ("opt_state", "inner_opt_state", 1, 0, "mu", "layer", "nu")
            )
        )

    def test_trained_fp32_accumulators_and_moments_are_preserved_without_cast(self):
        cfg = vlm.TrainConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            max_grad_norm=1.0,
            grad_accum_steps=2,
        )

        @nnx.jit
        def finite_step(opt, x):
            def loss_fn(module):
                return jnp.mean(module(x).astype(jnp.float32) ** 2)

            loss, grads = nnx.value_and_grad(loss_fn)(opt.model)
            opt.update(grads)
            return loss

        trained = vlm.build_optimizer(_TinyBF16Model(), 1e-3, cfg)
        batch = jnp.ones((2, 4), dtype=jnp.bfloat16)
        finite_step(trained, batch).block_until_ready()
        finite_step(trained, batch).block_until_ready()
        _, trained_state = nnx.split(trained)

        fresh = vlm.build_optimizer(_TinyBF16Model(), 1e-3, cfg)
        fresh_graphdef, fresh_state = nnx.split(fresh)
        expected = vlm._abstract_train_state_from_optimizer_state(
            fresh_state, jax.random.key(0)
        )["optimizer"]
        trained_flat = nnx.to_flat_state(trained_state)
        promoted_before = {
            path: variable.get_value()
            for path, variable in trained_flat
            if variable.dtype == jnp.float32
            and (
                path[:2] == ("opt_state", "acc_grads")
                or path[:2] == ("opt_state", "inner_opt_state")
                and any(name in path for name in ("mu", "nu"))
            )
        }
        contract = vlm._assert_restored_optimizer_contract(expected, trained_state)
        self.assertEqual(contract["promoted_leaf_count"], 6)
        self.assertEqual(contract["converted_leaf_count"], 0)
        self.assertEqual(set(contract["groups"]), {"acc_grads", "mu", "nu"})
        for path, array in promoted_before.items():
            self.assertIs(dict(nnx.to_flat_state(trained_state))[path].get_value(), array)

        restored = nnx.merge(fresh_graphdef, trained_state)
        merged_contract = vlm._assert_restored_optimizer_contract(expected, nnx.state(restored))
        self.assertEqual(merged_contract["promoted_leaf_records"], contract["promoted_leaf_records"])
        for path, array in promoted_before.items():
            self.assertIs(dict(nnx.to_flat_state(nnx.state(restored)))[path].get_value(), array)

        next_loss = finite_step(restored, batch)
        next_loss.block_until_ready()
        self.assertTrue(np.isfinite(np.asarray(next_loss)).item())
        post_update = dict(nnx.to_flat_state(nnx.state(restored)))
        for path in promoted_before:
            self.assertEqual(post_update[path].dtype, jnp.float32)

        wrong_promotion = jax.tree.map(lambda value: value, trained_state)
        wrong_accumulator = next(
            variable
            for path, variable in nnx.to_flat_state(wrong_promotion)
            if vlm._trained_promotion_group(path) == "acc_grads"
        )
        wrong_accumulator.set_value(
            wrong_accumulator.get_value().astype(jnp.float16)
        )
        with self.assertRaisesRegex(RuntimeError, "unpermitted"):
            vlm._assert_restored_optimizer_contract(expected, wrong_promotion)

        bad_state = jax.tree.map(lambda value: value, trained_state)
        bad_variable = dict(nnx.to_flat_state(bad_state))[("model", "linear", "kernel")]
        bad_variable.set_value(bad_variable.get_value().astype(jnp.float32))
        with self.assertRaisesRegex(RuntimeError, "unpermitted"):
            vlm._assert_restored_optimizer_contract(expected, bad_state)

    def test_alias_blocks_release_then_graphdef_reconstructs_exact_state(self):
        model = _TinyModel()
        cfg = vlm.TrainConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            grad_accum_steps=2,
        )
        optimizer = vlm.build_optimizer(model, 1e-3, cfg)
        rng = jax.random.key(123)

        expected_optimizer = _host_state(nnx.state(optimizer))
        expected_rng = np.asarray(jax.device_get(jax.random.key_data(rng))).copy()
        blueprint = vlm._prepare_memory_safe_restore(optimizer, rng)

        # Reproduce the production bug: the standalone model alias keeps all
        # parameter arrays alive even after the optimizer owner is deleted.
        leaked_model_alias = model
        del optimizer, model
        gc.collect()
        with self.assertRaisesRegex(RuntimeError, "live aliases"):
            vlm._verify_initialized_state_released(blueprint)

        del leaked_model_alias
        gc.collect()
        report = vlm._verify_initialized_state_released(blueprint)
        self.assertTrue(report["initialized_optimizer_collected"])
        self.assertTrue(report["initialized_model_collected"])
        self.assertEqual(report["live_initialized_array_count_after_gc"], 0)

        # Simulate Orbax's contract-directed restore: values are placed using
        # the expected dtypes/shardings, then NNX reconstructs from GraphDef.
        restored_state = jax.tree.map(
            lambda wanted, value: jax.device_put(
                np.asarray(value, dtype=wanted.dtype), wanted.sharding
            ),
            blueprint.abstract_train_state["optimizer"],
            expected_optimizer,
            is_leaf=lambda value: isinstance(value, jax.ShapeDtypeStruct),
        )
        optimizer_contract = vlm._assert_restored_optimizer_contract(
            blueprint.abstract_train_state["optimizer"], restored_state
        )
        restored_optimizer = nnx.merge(blueprint.optimizer_graphdef, restored_state)

        actual_optimizer = _host_state(nnx.state(restored_optimizer))
        expected_leaves = jax.tree.leaves(expected_optimizer)
        actual_leaves = jax.tree.leaves(actual_optimizer)
        self.assertLen(actual_leaves, len(expected_leaves))
        for expected, actual in zip(expected_leaves, actual_leaves):
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, expected.dtype)
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(jax.random.key_data(rng))), expected_rng
        )

        # The optimizer step and MultiSteps accumulation counters are included
        # in the exact all-leaf comparison above; also assert both are present.
        paths = {
            jax.tree_util.keystr(path)
            for path, _ in jax.tree_util.tree_leaves_with_path(nnx.state(restored_optimizer))
        }
        self.assertTrue(any("['step']" in path for path in paths))
        self.assertTrue(any("['mini_step']" in path for path in paths))

        iterator_state = {
            "next_index_in_cycle": 0,
            "iterators_in_use_states": [{"next_index": 8}, {"next_index": 8}],
        }

        class _RestoredIterator:
            def get_state(self):
                return iterator_state

        counters = vlm._restored_optimizer_counters(nnx.state(restored_optimizer))
        rng_data = [int(value) for value in jax.random.key_data(rng)]
        iterator_sha = hashlib.sha256(
            json.dumps(iterator_state, indent=4).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {
                "OMEGALAX_REQUIRE_EXACT_RESTORE_ATTESTATION": "1",
                "OMEGALAX_EXPECT_RESUME_STEP": "600",
                "OMEGALAX_EXPECT_OPTIMIZER_COUNTERS_JSON": json.dumps(counters),
                "OMEGALAX_EXPECT_RNG_KEY_DATA_JSON": json.dumps(rng_data),
                "OMEGALAX_EXPECT_ITERATOR_STATE_JSON": json.dumps(iterator_state),
                "OMEGALAX_EXPECT_ITERATOR_SHA256": iterator_sha,
                "OMEGALAX_EXPECT_PROMOTED_OPTIMIZER_STATE_JSON": json.dumps(
                    {
                        "promoted_leaf_count": 0,
                        "promoted_source_bytes": 0,
                        "fresh_zero_state_bytes": 0,
                    }
                ),
            },
            clear=False,
        ):
            vlm._write_exact_restore_attestation(
                Path(tmpdir),
                600,
                restored_optimizer,
                rng,
                _RestoredIterator(),
                len(actual_leaves),
                {"tp": 1, "fsdp": 1, "dp": 1},
                optimizer_contract,
            )
            attestation = json.loads(
                (Path(tmpdir) / "restore_exact_state.json").read_text()
            )
        self.assertEqual(attestation["status"], "restore_pass")
        self.assertEqual(attestation["optimizer_counters"], counters)
        self.assertEqual(attestation["rng_key_data"], rng_data)
        self.assertEqual(attestation["restored_iterator_state"], iterator_state)
        self.assertEqual(attestation["input_iterator_sha256"], iterator_sha)
        self.assertEqual(attestation["target_topology"], {"tp": 1, "fsdp": 1, "dp": 1})

        @nnx.jit
        def finite_step(opt, x):
            def loss_fn(module):
                return jnp.mean(module(x) ** 2)

            loss, grads = nnx.value_and_grad(loss_fn)(opt.model)
            opt.update(grads)
            return loss

        loss = finite_step(restored_optimizer, jnp.ones((2, 4), dtype=jnp.float32))
        self.assertTrue(np.isfinite(np.asarray(loss)).item())


if __name__ == "__main__":
    absltest.main()
