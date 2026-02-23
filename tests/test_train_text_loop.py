"""Smoke test for the text training loop and checkpoint resume."""

import dataclasses
import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest  # noqa: E402

from omegalax.trainers import text as text_trainer  # noqa: E402


class TrainLoopTest(absltest.TestCase):
    def test_train_and_resume(self):
        train_cfg = text_trainer.TrainConfig(
            seed=0,
            batch_size=2,
            seq_len=8,
            num_steps=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # First run: create checkpoint and metrics.
            _, metrics = text_trainer.run_training(
                "qwen3-smoke",
                train_cfg,
                save_dir=tmpdir_path,
                save_every=1,
                log_every=1,
                log_jsonl=tmpdir_path / "metrics.jsonl",
            )
            self.assertIn("loss", metrics)
            ckpt_mgr = text_trainer._make_checkpoint_manager(tmpdir_path, save_interval=None)  # type: ignore
            self.assertEqual(ckpt_mgr.latest_step(), 2)
            ckpt_mgr.close()

            # Resume for an extra step.
            resumed_cfg = dataclasses.replace(train_cfg, num_steps=3)
            _, resumed_metrics = text_trainer.run_training(
                "qwen3-smoke",
                resumed_cfg,
                save_dir=tmpdir_path,
                save_every=1,
                log_every=0,
                resume=True,
            )
            self.assertGreaterEqual(int(resumed_metrics["step"]), 3)
            ckpt_mgr = text_trainer._make_checkpoint_manager(tmpdir_path, save_interval=None)  # type: ignore
            self.assertGreaterEqual(ckpt_mgr.latest_step() or 0, 3)
            ckpt_mgr.close()


if __name__ == "__main__":
    absltest.main()
