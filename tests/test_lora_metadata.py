"""Tests for checkpoint LoRA metadata."""

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest

from omegalax.export import read_lora_metadata
from omegalax.trainers import vlm as vlm_trainer


class LoraMetadataTest(absltest.TestCase):
    def test_writer_reader_contract(self):
        cases = (
            (
                "full_ft",
                vlm_trainer.TrainConfig(enable_lora=False, lora_rank=None, lora_alpha=None),
                {
                    "enable_lora": False,
                    "lora_rank": None,
                    "lora_alpha": None,
                    "lora_qwen3_5_deltanet": False,
                },
            ),
            (
                "lora",
                vlm_trainer.TrainConfig(enable_lora=True, lora_rank=16, lora_alpha=32.0),
                {
                    "enable_lora": True,
                    "lora_rank": 16,
                    "lora_alpha": 32.0,
                    "lora_qwen3_5_deltanet": False,
                },
            ),
            (
                "qwen3_5_deltanet_lora",
                vlm_trainer.TrainConfig(
                    enable_lora=True,
                    lora_rank=16,
                    lora_alpha=32.0,
                    lora_qwen3_5_deltanet=True,
                ),
                {
                    "enable_lora": True,
                    "lora_rank": 16,
                    "lora_alpha": 32.0,
                    "lora_qwen3_5_deltanet": True,
                },
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            for name, train_cfg, expected in cases:
                with self.subTest(name=name):
                    save_dir = Path(tmpdir) / name
                    save_dir.mkdir()
                    vlm_trainer._write_lora_metadata(save_dir, train_cfg)
                    raw = json.loads((save_dir / "lora_metadata.json").read_text())
                    self.assertEqual(raw, expected)
                    self.assertEqual(read_lora_metadata(save_dir), expected)

                    if train_cfg.enable_lora:
                        self.assertIsInstance(raw["lora_rank"], int)
                        self.assertIsInstance(raw["lora_alpha"], float)

    def test_reader_requires_deltanet_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)
            (save_dir / "lora_metadata.json").write_text(
                json.dumps({"enable_lora": True, "lora_rank": 16, "lora_alpha": 32.0})
            )
            with self.assertRaisesRegex(ValueError, "lora_qwen3_5_deltanet"):
                read_lora_metadata(save_dir)

    def test_reader_rejects_non_boolean_modes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)
            (save_dir / "lora_metadata.json").write_text(
                json.dumps(
                    {
                        "enable_lora": True,
                        "lora_rank": 16,
                        "lora_alpha": 32.0,
                        "lora_qwen3_5_deltanet": "false",
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                read_lora_metadata(save_dir)


if __name__ == "__main__":
    absltest.main()
