from absl.testing import absltest

from omegalax.models.qwen3.dense.params_dense import assert_dense_config
from omegalax.text import api


EXPECTED_QWEN3_SPECS = {
    "Qwen/Qwen3-0.6B": {
        "vocab_size": 151_936,
        "emb_dim": 1_024,
        "mlp_dim": 3_072,
        "num_layers": 28,
        "num_heads": 16,
        "head_dim": 128,
        "num_kv_heads": 8,
        "tie_word_embeddings": True,
    },
    "Qwen/Qwen3-1.7B": {
        "vocab_size": 151_936,
        "emb_dim": 2_048,
        "mlp_dim": 6_144,
        "num_layers": 28,
        "num_heads": 16,
        "head_dim": 128,
        "num_kv_heads": 8,
        "tie_word_embeddings": True,
    },
    "Qwen/Qwen3-4B": {
        "vocab_size": 151_936,
        "emb_dim": 2_560,
        "mlp_dim": 9_728,
        "num_layers": 36,
        "num_heads": 32,
        "head_dim": 128,
        "num_kv_heads": 8,
        "tie_word_embeddings": True,
    },
    "Qwen/Qwen3-8B": {
        "vocab_size": 151_936,
        "emb_dim": 4_096,
        "mlp_dim": 12_288,
        "num_layers": 36,
        "num_heads": 32,
        "head_dim": 128,
        "num_kv_heads": 8,
        "tie_word_embeddings": False,
    },
    "Qwen/Qwen3-14B": {
        "vocab_size": 151_936,
        "emb_dim": 5_120,
        "mlp_dim": 17_408,
        "num_layers": 40,
        "num_heads": 40,
        "head_dim": 128,
        "num_kv_heads": 8,
        "tie_word_embeddings": False,
    },
    "Qwen/Qwen3-32B": {
        "vocab_size": 151_936,
        "emb_dim": 5_120,
        "mlp_dim": 25_600,
        "num_layers": 64,
        "num_heads": 64,
        "head_dim": 128,
        "num_kv_heads": 8,
        "tie_word_embeddings": False,
    },
}


class Qwen3RegistryTest(absltest.TestCase):
    def test_registry_values_match(self):
        for model_id, expected in EXPECTED_QWEN3_SPECS.items():
            cfg = api.registry.build_config(model_id)
            for field, value in expected.items():
                self.assertEqual(getattr(cfg, field), value, msg=f"{model_id}: {field}")

    def test_unknown_alias_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Qwen3 dense model_id"):
            api.registry.build_config("qwen3-0.6b-base")

    def test_tie_word_embeddings_mismatch_raises(self):
        cfg = api.registry.build_config("Qwen/Qwen3-8B")
        hf_cfg = {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.emb_dim,
            "intermediate_size": cfg.mlp_dim,
            "num_hidden_layers": cfg.num_layers,
            "num_attention_heads": cfg.num_heads,
            "num_key_value_heads": cfg.num_kv_heads,
            "head_dim": cfg.head_dim,
            "rope_theta": cfg.rope_theta,
            "tie_word_embeddings": not cfg.tie_word_embeddings,
        }

        with self.assertRaisesRegex(ValueError, "tie_word_embeddings"):
            assert_dense_config(cfg, hf_cfg)


if __name__ == "__main__":
    absltest.main()
