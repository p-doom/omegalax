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
                {"enable_lora": False, "lora_rank": None, "lora_alpha": None},
            ),
            (
                "lora",
                vlm_trainer.TrainConfig(enable_lora=True, lora_rank=16, lora_alpha=32.0),
                {"enable_lora": True, "lora_rank": 16, "lora_alpha": 32.0},
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


if __name__ == "__main__":
    absltest.main()
