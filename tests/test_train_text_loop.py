"""Smoke test for the text training loop and checkpoint resume."""

import dataclasses
import os
import tempfile
from pathlib import Path
from unittest import mock

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
                tp_size=1,
                fsdp_size=1,
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
                tp_size=1,
                fsdp_size=1,
            )
            self.assertGreaterEqual(int(resumed_metrics["step"]), 3)
            ckpt_mgr = text_trainer._make_checkpoint_manager(tmpdir_path, save_interval=None)  # type: ignore
            self.assertGreaterEqual(ckpt_mgr.latest_step() or 0, 3)
            ckpt_mgr.close()

    def test_train_qwen3_5_smoke_by_model_id(self):
        train_cfg = text_trainer.TrainConfig(
            seed=0,
            batch_size=1,
            seq_len=4,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )

        _, metrics = text_trainer.run_training(
            "qwen3.5-smoke",
            train_cfg,
            log_every=0,
            tp_size=1,
            fsdp_size=1,
        )
        self.assertIn("loss", metrics)
        self.assertEqual(int(metrics["step"]), 1)

    def test_replicated_batch_not_split_by_process_count(self):
        train_cfg = text_trainer.TrainConfig(
            seed=0,
            batch_size=2,
            seq_len=8,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )
        seen_batch_sizes: list[int] = []
        original_make_batch = text_trainer.make_synthetic_batch

        def _capture_batch_size(rng, batch_size, seq_len, vocab_size, pad_id=0):
            seen_batch_sizes.append(batch_size)
            return original_make_batch(rng, batch_size, seq_len, vocab_size, pad_id)

        with mock.patch("omegalax.trainers.text.jax.process_count", return_value=2):
            with mock.patch(
                "omegalax.trainers.text.make_synthetic_batch",
                side_effect=_capture_batch_size,
            ):
                text_trainer.run_training(
                    "qwen3-smoke",
                    train_cfg,
                    log_every=0,
                    tp_size=1,
                    fsdp_size=1,
                )

        self.assertEqual(seen_batch_sizes, [train_cfg.batch_size])

    def test_replicated_training_is_process_index_invariant(self):
        train_cfg = text_trainer.TrainConfig(
            seed=0,
            batch_size=2,
            seq_len=8,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )

        with mock.patch("omegalax.trainers.text.jax.process_index", return_value=0):
            _, metrics_rank0 = text_trainer.run_training(
                "qwen3-smoke",
                train_cfg,
                log_every=0,
                tp_size=1,
                fsdp_size=1,
            )
        with mock.patch("omegalax.trainers.text.jax.process_index", return_value=1):
            _, metrics_rank1 = text_trainer.run_training(
                "qwen3-smoke",
                train_cfg,
                log_every=0,
                tp_size=1,
                fsdp_size=1,
            )

        self.assertAlmostEqual(metrics_rank0["loss"], metrics_rank1["loss"], places=6)
        self.assertAlmostEqual(metrics_rank0["grad_norm"], metrics_rank1["grad_norm"], places=6)

    def test_non_primary_process_does_not_write_jsonl_logs(self):
        train_cfg = text_trainer.TrainConfig(
            seed=0,
            batch_size=2,
            seq_len=8,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "metrics.jsonl"
            with mock.patch("omegalax.trainers.text.jax.process_index", return_value=1):
                _, metrics = text_trainer.run_training(
                    "qwen3-smoke",
                    train_cfg,
                    log_every=1,
                    log_jsonl=log_path,
                    tp_size=1,
                    fsdp_size=1,
                )
            self.assertIn("loss", metrics)
            self.assertFalse(log_path.exists())


if __name__ == "__main__":
    absltest.main()
