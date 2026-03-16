"""Smoke tests for SFT training loops and shard_batch_dict."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest

import jax
import numpy as np

from omegalax.distributed.mesh import ensure_mesh
from omegalax.models.qwen3_vl.config import make_vl_config
from omegalax.models.qwen3_vl.model import get_rope_index
from omegalax.models.sharding_runtime import shard_batch_dict
from omegalax.models.shard_config import ShardConfig
from omegalax.trainers import text as text_trainer
from omegalax.trainers import vlm as vlm_trainer


def _make_synthetic_sft_batch(batch_size: int, seq_len: int, vocab_size: int) -> dict[str, np.ndarray]:
    """Build a minimal SFT batch dict with random data and a simple loss mask."""
    rng = np.random.RandomState(42)
    token_ids = rng.randint(1, vocab_size, size=(batch_size, seq_len)).astype(np.int32)
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int32)
    # Supervise the second half of the sequence
    loss_mask = np.zeros((batch_size, seq_len), dtype=np.int32)
    loss_mask[:, seq_len // 2:] = 1
    return {
        "token_ids_BT": token_ids,
        "attention_mask_BT": attention_mask,
        "loss_mask_BT": loss_mask,
    }


def _make_multimodal_qwen3_vl_smoke_batch(seq_len: int = 8) -> dict[str, np.ndarray]:
    cfg = make_vl_config("qwen3-vl-smoke")
    llm_grid_t, h, w = 1, 4, 4
    num_vision_tokens = llm_grid_t * (h // cfg.vision.spatial_merge_size) * (w // cfg.vision.spatial_merge_size)

    seq = [11, cfg.vision_start_token_id, *([cfg.image_token_id] * num_vision_tokens), 21, 22]
    token_ids = np.zeros((1, seq_len), dtype=np.int32)
    attention_mask = np.zeros((1, seq_len), dtype=np.int32)
    loss_mask = np.zeros((1, seq_len), dtype=np.int32)
    token_ids[0, : len(seq)] = np.asarray(seq, dtype=np.int32)
    attention_mask[0, : len(seq)] = 1
    loss_mask[0, 1 : len(seq)] = 1

    image_grid_thw = np.asarray([[llm_grid_t, h, w]], dtype=np.int32)
    vision_cu_seqlens = np.concatenate(
        [np.zeros(1, dtype=np.int32), np.cumsum(np.asarray([h * w] * llm_grid_t, dtype=np.int32), dtype=np.int32)]
    )
    in_features = cfg.vision.in_channels * cfg.vision.temporal_patch_size * cfg.vision.patch_size**2
    pixel_values = np.random.default_rng(0).standard_normal((llm_grid_t * h * w, in_features), dtype=np.float32)

    position_ids, _ = get_rope_index(
        token_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        spatial_merge_size=cfg.vision.spatial_merge_size,
        image_token_id=cfg.image_token_id,
        video_token_id=cfg.video_token_id,
        vision_start_token_id=cfg.vision_start_token_id,
    )

    return {
        "token_ids_BT": token_ids,
        "attention_mask_BT": attention_mask,
        "loss_mask_BT": loss_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "vision_cu_seqlens": vision_cu_seqlens,
        "position_ids_ZBT": position_ids.astype(np.int32),
    }


class ShardBatchDictTest(absltest.TestCase):
    def test_rank2_arrays(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1)
        shd_cfg = ShardConfig.no_sharding()
        batch = {
            "token_ids_BT": np.ones((2, 4), dtype=np.int32),
            "loss_mask_BT": np.zeros((2, 4), dtype=np.int32),
        }
        out = shard_batch_dict(batch, shd_cfg, mesh)
        self.assertIn("token_ids_BT", out)
        self.assertIn("loss_mask_BT", out)
        self.assertEqual(out["token_ids_BT"].shape, (2, 4))

    def test_mixed_rank_arrays(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1)
        shd_cfg = ShardConfig.no_sharding()
        batch = {
            "token_ids_BT": np.ones((2, 4), dtype=np.int32),
            "pixel_values": np.ones((2, 3, 8, 8), dtype=np.float32),
        }
        out = shard_batch_dict(batch, shd_cfg, mesh)
        self.assertEqual(out["token_ids_BT"].shape, (2, 4))
        self.assertEqual(out["pixel_values"].shape, (2, 3, 8, 8))


class TextSFTTrainingTest(absltest.TestCase):
    def test_one_step_sft(self):
        train_cfg = text_trainer.TrainConfig(
            seed=0,
            batch_size=2,
            seq_len=8,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )
        batch = _make_synthetic_sft_batch(2, 8, 32000)
        data_iter = iter([batch])

        _, metrics = text_trainer.run_sft(
            "qwen3-smoke",
            train_cfg,
            data_iter,
            log_every=0,
            tp_size=1,
            fsdp_size=1,
        )
        self.assertIn("loss", metrics)
        self.assertIn("supervised_tokens", metrics)
        self.assertGreater(metrics["supervised_tokens"], 0)
        self.assertEqual(int(metrics["step"]), 1)


class VLMSFTTrainingTest(absltest.TestCase):
    def test_one_step_sft_text_only(self):
        train_cfg = vlm_trainer.TrainConfig(
            seed=0,
            batch_size=1,
            seq_len=4,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )
        batch = _make_synthetic_sft_batch(1, 4, 32000)
        data_iter = iter([batch])

        _, metrics = vlm_trainer.run_sft(
            "qwen3-vl-smoke",
            train_cfg,
            data_iter,
            log_every=0,
            tp_size=1,
            fsdp_size=1,
        )
        self.assertIn("loss", metrics)
        self.assertIn("supervised_tokens", metrics)
        self.assertGreater(metrics["supervised_tokens"], 0)
        self.assertEqual(int(metrics["step"]), 1)

    def test_one_step_sft_multimodal_qwen3_vl(self):
        train_cfg = vlm_trainer.TrainConfig(
            seed=0,
            batch_size=1,
            seq_len=8,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )
        batch = _make_multimodal_qwen3_vl_smoke_batch(seq_len=8)
        data_iter = iter([batch])

        _, metrics = vlm_trainer.run_sft(
            "qwen3-vl-smoke",
            train_cfg,
            data_iter,
            log_every=0,
            tp_size=1,
            fsdp_size=1,
        )
        self.assertIn("loss", metrics)
        self.assertIn("supervised_tokens", metrics)
        self.assertGreater(metrics["supervised_tokens"], 0)
        self.assertEqual(int(metrics["step"]), 1)


if __name__ == "__main__":
    absltest.main()
