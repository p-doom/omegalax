from absl.testing import absltest

from omegalax.models.qwen3.config import make_config, make_config_from_hf
from omegalax.models.qwen3.loader import _assert_config


def _dense_hf_cfg() -> dict:
    return {
        "model_type": "qwen3",
        "dtype": "bfloat16",
        "vocab_size": 151_936,
        "hidden_size": 4_096,
        "intermediate_size": 12_288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1_000_000, "rope_type": "default"},
        "tie_word_embeddings": False,
    }


def _moe_hf_cfg() -> dict:
    return {
        "model_type": "qwen3_moe",
        "dtype": "bfloat16",
        "vocab_size": 151_936,
        "hidden_size": 2_048,
        "intermediate_size": 6_144,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10_000_000, "rope_type": "default"},
        "tie_word_embeddings": False,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "mlp_only_layers": [],
        "decoder_sparse_step": 1,
        "norm_topk_prob": True,
    }


class Qwen3RegistryTest(absltest.TestCase):
    def test_dense_hf_config_maps_cleanly(self):
        cfg = make_config_from_hf(_dense_hf_cfg())
        self.assertEqual(cfg.vocab_size, 151_936)
        self.assertEqual(cfg.emb_dim, 4_096)
        self.assertEqual(cfg.mlp_dim, 12_288)
        self.assertEqual(cfg.num_layers, 36)
        self.assertEqual(cfg.num_heads, 32)
        self.assertEqual(cfg.head_dim, 128)
        self.assertEqual(cfg.num_kv_heads, 8)
        self.assertFalse(cfg.tie_word_embeddings)

    def test_moe_hf_config_maps_cleanly(self):
        cfg = make_config_from_hf(_moe_hf_cfg())
        self.assertEqual(cfg.num_experts, 128)
        self.assertEqual(cfg.num_experts_per_tok, 8)
        self.assertEqual(cfg.moe_intermediate_size, 768)
        self.assertTrue(cfg.is_moe)

    def test_moe_hf_config_accepts_num_local_experts_alias(self):
        hf_cfg = _moe_hf_cfg()
        hf_cfg["num_local_experts"] = hf_cfg.pop("num_experts")
        cfg = make_config_from_hf(hf_cfg)
        self.assertEqual(cfg.num_experts, 128)
        self.assertTrue(cfg.is_moe)

    def test_unknown_alias_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Qwen3 model_id"):
            make_config("qwen3-0.6b-base")

    def test_tie_word_embeddings_mismatch_raises(self):
        cfg = make_config_from_hf(_dense_hf_cfg())
        hf_cfg = dict(_dense_hf_cfg())
        hf_cfg["tie_word_embeddings"] = not cfg.tie_word_embeddings

        with self.assertRaisesRegex(ValueError, "tie_word_embeddings"):
            _assert_config(cfg, hf_cfg)


if __name__ == "__main__":
    absltest.main()
