import json
import tempfile
from pathlib import Path

from absl.testing import absltest
from transformers import AutoConfig

from omegalax.models.params_utils import load_hf_config


class HfConfigNormalizationTest(absltest.TestCase):
    def _load(self, hf_cfg):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "config.json").write_text(json.dumps(hf_cfg))
            return load_hf_config(path)

    def test_qwen3_moe_uses_internal_expert_name(self):
        hf_cfg = AutoConfig.for_model("qwen3_moe", num_experts=4).to_dict()
        self.assertIn("num_local_experts", hf_cfg)

        normalized = self._load(hf_cfg)

        self.assertEqual(normalized["num_experts"], 4)
        self.assertNotIn("num_local_experts", normalized)

    def test_qwen3_vl_moe_uses_internal_expert_name(self):
        hf_cfg = AutoConfig.for_model("qwen3_vl_moe", text_config={"num_experts": 4}).to_dict()
        self.assertIn("num_local_experts", hf_cfg["text_config"])

        normalized = self._load(hf_cfg)

        self.assertEqual(normalized["text_config"]["num_experts"], 4)
        self.assertNotIn("num_local_experts", normalized["text_config"])

    def test_qwen3_5_moe_keeps_production_expert_name(self):
        hf_cfg = AutoConfig.for_model("qwen3_5_moe", text_config={"num_experts": 4}).to_dict()
        self.assertIn("num_experts", hf_cfg["text_config"])

        normalized = self._load(hf_cfg)

        self.assertEqual(normalized["text_config"]["num_experts"], 4)
        self.assertNotIn("num_local_experts", normalized["text_config"])

    def test_ambiguous_expert_count_raises(self):
        hf_cfg = AutoConfig.for_model("qwen3_moe", num_experts=4).to_dict()
        hf_cfg["num_experts"] = 5

        with self.assertRaisesRegex(ValueError, "exactly one expert-count field"):
            self._load(hf_cfg)


if __name__ == "__main__":
    absltest.main()
