"""Smoke test for the VLM training loop across supported families."""

import os
import tempfile
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest  # noqa: E402
import jax  # noqa: E402

from omegalax.trainers import vlm as vlm_trainer  # noqa: E402


class TrainVLMTest(absltest.TestCase):
    def test_train_qwen3_vl_smoke(self):
        train_cfg = vlm_trainer.TrainConfig(
            seed=0,
            batch_size=1,
            seq_len=4,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )

        _, metrics = vlm_trainer.run_training(
            "qwen3-vl-smoke",
            train_cfg,
            log_every=0,
            tp_size=1,
            fsdp_size=1,
        )
        self.assertIn("loss", metrics)
        self.assertEqual(int(metrics["step"]), 1)

    def test_train_qwen3_5_vlm_smoke(self):
        train_cfg = vlm_trainer.TrainConfig(
            seed=0,
            batch_size=1,
            seq_len=4,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )

        _, metrics = vlm_trainer.run_training(
            "qwen3.5-smoke",
            train_cfg,
            log_every=0,
            tp_size=1,
            fsdp_size=1,
        )
        self.assertIn("loss", metrics)
        self.assertEqual(int(metrics["step"]), 1)

    def test_replicated_batch_not_split_by_process_count(self):
        train_cfg = vlm_trainer.TrainConfig(
            seed=0,
            batch_size=2,
            seq_len=4,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )
        seen_batch_sizes: list[int] = []
        original_make_batch = vlm_trainer.make_synthetic_batch

        def _capture_batch_size(rng, batch_size, seq_len, vocab_size, pad_id=0):
            seen_batch_sizes.append(batch_size)
            return original_make_batch(rng, batch_size, seq_len, vocab_size, pad_id)

        with mock.patch("omegalax.trainers.vlm.jax.process_count", return_value=2):
            with mock.patch(
                "omegalax.trainers.vlm.make_synthetic_batch",
                side_effect=_capture_batch_size,
            ):
                vlm_trainer.run_training(
                    "qwen3-vl-smoke",
                    train_cfg,
                    log_every=0,
                    tp_size=1,
                    fsdp_size=1,
                )

        self.assertEqual(seen_batch_sizes, [train_cfg.batch_size])

    def test_replicated_training_is_process_index_invariant(self):
        train_cfg = vlm_trainer.TrainConfig(
            seed=0,
            batch_size=2,
            seq_len=4,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )

        with mock.patch("omegalax.trainers.vlm.jax.process_index", return_value=0):
            _, metrics_rank0 = vlm_trainer.run_training(
                "qwen3-vl-smoke",
                train_cfg,
                log_every=0,
                tp_size=1,
                fsdp_size=1,
            )
        with mock.patch("omegalax.trainers.vlm.jax.process_index", return_value=1):
            _, metrics_rank1 = vlm_trainer.run_training(
                "qwen3-vl-smoke",
                train_cfg,
                log_every=0,
                tp_size=1,
                fsdp_size=1,
            )

        self.assertAlmostEqual(metrics_rank0["loss"], metrics_rank1["loss"], places=6)
        self.assertAlmostEqual(metrics_rank0["grad_norm"], metrics_rank1["grad_norm"], places=6)

    def test_non_primary_process_does_not_write_jsonl_logs(self):
        train_cfg = vlm_trainer.TrainConfig(
            seed=0,
            batch_size=1,
            seq_len=4,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "metrics.jsonl"
            with mock.patch("omegalax.trainers.vlm.jax.process_index", return_value=1):
                _, metrics = vlm_trainer.run_training(
                    "qwen3-vl-smoke",
                    train_cfg,
                    log_every=1,
                    log_jsonl=log_path,
                    tp_size=1,
                    fsdp_size=1,
                )
            self.assertIn("loss", metrics)
            self.assertFalse(log_path.exists())

    def test_abstract_train_state_preserves_sharding_metadata(self):
        rng = jax.random.key(0)
        model = vlm_trainer.init_model("qwen3-vl-smoke", rng, tp_size=1, fsdp_size=1)
        optimizer = vlm_trainer.build_optimizer(model, vlm_trainer.TrainConfig.smoke())

        abstract_state = vlm_trainer._abstract_train_state(optimizer, rng)  # type: ignore
        self.assertIsNotNone(getattr(abstract_state["rng"], "sharding", None))
        optimizer_leaves = jax.tree_util.tree_leaves(abstract_state["optimizer"])
        leaves_with_sharding = [
            leaf
            for leaf in optimizer_leaves
            if hasattr(leaf, "shape") and getattr(leaf, "sharding", None) is not None
        ]
        self.assertTrue(len(leaves_with_sharding) > 0)


if __name__ == "__main__":
    absltest.main()
