"""Force-save smoke: off-cadence steps must write when force=True.

Orbax's ``save_interval_steps`` policy silently refuses steps that are not
multiples of the interval; the trainer's signal/end-of-run saves rely on
``force=True`` to bypass it.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import tempfile
from pathlib import Path

import jax.numpy as jnp
import orbax.checkpoint as ocp
from absl.testing import absltest
from orbax.checkpoint.checkpoint_manager import StepAlreadyExistsError


def _make_manager(save_dir: Path) -> ocp.CheckpointManager:
    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add("train_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)
    handler_registry.add(
        "train_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler
    )
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=2500,
        step_format_fixed_length=6,
        cleanup_tmp_directories=True,
    )
    return ocp.CheckpointManager(save_dir, options=options, handler_registry=handler_registry)


class ForceSaveTest(absltest.TestCase):
    def _tree(self):
        return {"w": jnp.arange(4, dtype=jnp.float32)}

    def _save(self, cm, step, force=False):
        return cm.save(
            step,
            args=ocp.args.Composite(train_state=ocp.args.PyTreeSave(self._tree())),
            force=force,
        )

    def test_force_save_writes_off_cadence_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp).resolve()
            cm = _make_manager(save_dir)
            self.assertTrue(self._save(cm, 1, force=True))
            cm.wait_until_finished()
            self.assertIn(1, cm.all_steps())
            self.assertTrue((save_dir / "000001" / "_CHECKPOINT_METADATA").exists())
            cm.close()

    def test_interval_policy_refuses_off_cadence_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp).resolve()
            cm = _make_manager(save_dir)
            self.assertTrue(self._save(cm, 2500))
            cm.wait_until_finished()
            self.assertFalse(self._save(cm, 2501))
            self.assertNotIn(2501, cm.all_steps())
            self.assertTrue(self._save(cm, 2501, force=True))
            cm.wait_until_finished()
            self.assertIn(2501, cm.all_steps())
            self.assertTrue((save_dir / "002501" / "_CHECKPOINT_METADATA").exists())
            cm.close()

    def test_force_save_at_existing_step_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp).resolve()
            cm = _make_manager(save_dir)
            self.assertTrue(self._save(cm, 2500))
            cm.wait_until_finished()
            with self.assertRaises(StepAlreadyExistsError):
                self._save(cm, 2500, force=True)
            cm.close()


if __name__ == "__main__":
    absltest.main()
