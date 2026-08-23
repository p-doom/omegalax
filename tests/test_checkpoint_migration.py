import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from absl.testing import absltest
from flax import nnx

from omegalax.trainers.checkpoint_migration import (
    migrate_multisteps_k1_checkpoint,
    migrate_multisteps_k1_train_state,
)
from omegalax.trainers.optim import MixedPrecisionOptimizer


class _TinyModel(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 16)


def _base_tx():
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lambda count: jnp.asarray(1e-2), weight_decay=0.01),
    )


def _optimizer(*, legacy: bool):
    tx = _base_tx()
    if legacy:
        tx = optax.MultiSteps(tx, every_k_schedule=1)
    return MixedPrecisionOptimizer(_TinyModel(), tx)


def _update(optimizer):
    grads = nnx.grad(lambda model: jnp.sum(model.weight[...] ** 2))(optimizer.model)
    optimizer.update(grads)


def _pure_train_state(optimizer, rng):
    return {
        "optimizer": nnx.to_pure_dict(nnx.state(optimizer)),
        "rng": rng,
    }


def _assert_trees_equal(testcase, expected, actual):
    testcase.assertEqual(jax.tree.structure(expected), jax.tree.structure(actual))
    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(expected_leaf), np.asarray(actual_leaf))


class CheckpointMigrationTest(absltest.TestCase):
    def test_writer_migration_and_strict_restore(self):
        legacy = _optimizer(legacy=True)
        for _ in range(3):
            _update(legacy)
        rng = jax.random.key(17)
        legacy_state = _pure_train_state(legacy, rng)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "source"
            destination_root = Path(temp_dir) / "destination"
            source_step = source_root / "000003"
            source_step.mkdir(parents=True)
            (source_root / "config.json").write_text('{"model_type": "test"}\n')
            (source_root / "lora_metadata.json").write_text('{"enable_lora": false}\n')
            input_iter_bytes = b'{"last_seen_indices": {"0": 23}}\n'
            (source_step / "input_iter").mkdir()
            (source_step / "input_iter" / "process_0-of-1.json").write_bytes(input_iter_bytes)
            (source_step / "_CHECKPOINT_METADATA").write_text(
                json.dumps(
                    {
                        "item_handlers": {
                            "input_iter": "grain._src.python.checkpoint.handler.CheckpointHandler",
                            "train_state": (
                                "orbax.checkpoint._src.handlers.pytree_checkpoint_handler."
                                "PyTreeCheckpointHandler"
                            ),
                        },
                        "metrics": {},
                        "performance_metrics": {},
                        "custom_metadata": {},
                    }
                )
            )

            checkpointer = ocp.PyTreeCheckpointer()
            try:
                checkpointer.save(source_step / "train_state", legacy_state)
                current_target = _pure_train_state(_optimizer(legacy=False), jax.random.key(0))
                with self.assertRaises(ValueError):
                    checkpointer.restore(source_step / "train_state", item=current_target)

                migrated_step = migrate_multisteps_k1_checkpoint(
                    source_root, destination_root, checkpoint_step=3
                )
                current_restore_args = ocp.checkpoint_utils.construct_restore_args(current_target)
                restored = checkpointer.restore(
                    migrated_step / "train_state",
                    item=current_target,
                    restore_args=current_restore_args,
                )
            finally:
                checkpointer.close()

            expected_optimizer = legacy_state["optimizer"]
            self.assertNotIn("acc_grads", restored["optimizer"]["opt_state"])
            _assert_trees_equal(self, expected_optimizer["model"], restored["optimizer"]["model"])
            _assert_trees_equal(
                self,
                expected_optimizer["opt_state"]["inner_opt_state"],
                restored["optimizer"]["opt_state"],
            )
            np.testing.assert_array_equal(
                np.asarray(expected_optimizer["step"]), np.asarray(restored["optimizer"]["step"])
            )
            np.testing.assert_array_equal(
                np.asarray(jax.random.key_data(rng)),
                np.asarray(jax.random.key_data(restored["rng"])),
            )
            self.assertEqual(
                (migrated_step / "input_iter" / "process_0-of-1.json").read_bytes(),
                input_iter_bytes,
            )
            self.assertEqual(
                json.loads((destination_root / "checkpoint_migration.json").read_text()),
                {
                    "schema": "omegalax.multisteps_k1_to_direct.v1",
                    "checkpoint_step": 3,
                },
            )
            self.assertEqual(
                (destination_root / "config.json").read_text(), '{"model_type": "test"}\n'
            )
            self.assertEqual(
                (destination_root / "lora_metadata.json").read_text(),
                '{"enable_lora": false}\n',
            )

    def test_rejects_multisteps_k2_checkpoint(self):
        optimizer = MixedPrecisionOptimizer(
            _TinyModel(), optax.MultiSteps(_base_tx(), every_k_schedule=2)
        )
        _update(optimizer)
        _update(optimizer)
        train_state = _pure_train_state(optimizer, jax.random.key(0))

        with self.assertRaisesRegex(ValueError, "not a completed MultiSteps\\(k=1\\) boundary"):
            migrate_multisteps_k1_train_state(train_state, checkpoint_step=1)


if __name__ == "__main__":
    absltest.main()
