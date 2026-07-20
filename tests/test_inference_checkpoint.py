"""Checkpoint resolution and params-only restore for text inference."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=1"
).strip()

from absl.testing import absltest
from flax import nnx
import jax
import numpy as np
import orbax.checkpoint as ocp

from omegalax import export as export_lib
from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.models.params_utils import save_hf_config
from omegalax.text import api as text_api
from omegalax.text.checkpoint import resolve_checkpoint, restore_model_params
from omegalax.trainers import text as text_trainer


def _save_checkpoint(root: Path, step: int, model, cfg) -> None:
    save_hf_config(export_lib.model_config_to_hf_dict(cfg), root)
    with mesh_rules(ensure_mesh()):
        optimizer = text_trainer.build_optimizer(
            model,
            text_trainer.TrainConfig(num_steps=1),
        )
    train_state = text_trainer._train_state(optimizer, jax.random.key(17))
    registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    registry.add("train_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)
    manager = ocp.CheckpointManager(
        root,
        options=ocp.CheckpointManagerOptions(step_format_fixed_length=6),
        handler_registry=registry,
    )
    manager.save(
        step,
        args=ocp.args.Composite(train_state=ocp.args.PyTreeSave(train_state)),
    )
    manager.wait_until_finished()
    manager.close()


class InferenceCheckpointTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source, cls.cfg = text_api.init_model(
            "qwen3.5-smoke-dense",
            jax.random.key(0),
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )

    def test_resolves_explicit_step_and_latest_complete_root_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _save_checkpoint(root, 3, self.source, self.cfg)
            _save_checkpoint(root, 7, self.source, self.cfg)
            incomplete_step = root / "000009"
            incomplete_step.mkdir()

            from_root = resolve_checkpoint(root)
            from_step = resolve_checkpoint(root / "000003")

            self.assertEqual(from_root.root, root.resolve())
            self.assertEqual(from_root.step, 7)
            self.assertEqual(from_root.step_path, root.resolve() / "000007")
            self.assertEqual(from_step.root, root.resolve())
            self.assertEqual(from_step.step, 3)
            with self.assertRaisesRegex(ValueError, "complete checkpoint"):
                resolve_checkpoint(incomplete_step)

    def test_restores_exact_model_subtree_without_optimizer_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _save_checkpoint(root, 11, self.source, self.cfg)
            target, _ = text_api.init_model(
                self.cfg,
                jax.random.key(1),
                tp_size=1,
                fsdp_size=1,
                dp_size=1,
            )

            restore_model_params(target, resolve_checkpoint(root))

            source_leaves = dict(jax.tree_util.tree_flatten_with_path(nnx.state(self.source))[0])
            target_leaves = dict(jax.tree_util.tree_flatten_with_path(nnx.state(target))[0])
            self.assertEqual(target_leaves.keys(), source_leaves.keys())
            for path, source in source_leaves.items():
                restored = target_leaves[path]
                self.assertEqual(restored.shape, source.shape, msg=jax.tree_util.keystr(path))
                self.assertEqual(restored.dtype, source.dtype, msg=jax.tree_util.keystr(path))
                np.testing.assert_array_equal(np.asarray(restored), np.asarray(source))

    def test_rejects_missing_config_and_non_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "config.json"):
                resolve_checkpoint(root)

            (root / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "complete checkpoint"):
                resolve_checkpoint(root)


if __name__ == "__main__":
    absltest.main()
