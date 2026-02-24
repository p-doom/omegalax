"""Smoke test for Qwen3-VL MoE HF parity on a tiny config."""

import json
import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from transformers import AutoConfig, Qwen3VLMoeForConditionalGeneration

from omegalax.models.qwen3_vl import create_qwen3_vl_from_safetensors

jax.config.update("jax_default_matmul_precision", "highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _random_tokens(batch: int, seq: int, vocab: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(1, vocab, size=(batch, seq), dtype=np.int64)


class Qwen3VLMoeSmokeTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmpdir = tempfile.mkdtemp()

        # Tiny MoE config
        vision_cfg = {
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "spatial_merge_size": 2,
            "out_hidden_size": 128,
            "depth": 2,
            "hidden_act": "gelu_pytorch_tanh",
            "num_position_embeddings": 256,
            "deepstack_visual_indexes": [0],
            "model_type": "qwen3_vl_moe",
        }
        text_cfg = {
            "vocab_size": 1024,
            "hidden_size": 128,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": False,
            "rope_parameters": {"rope_theta": 1_000_000, "mrope_section": [8, 4, 4]},
            "moe_intermediate_size": 128,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "mlp_only_layers": [],
            "decoder_sparse_step": 1,
            "norm_topk_prob": True,
            "attention_bias": False,
            "model_type": "qwen3_vl_moe_text",
        }
        cls.config_dict = {
            "architectures": ["Qwen3VLMoeForConditionalGeneration"],
            "model_type": "qwen3_vl_moe",
            "tie_word_embeddings": False,
            "image_token_id": 2,
            "video_token_id": 3,
            "vision_start_token_id": 4,
            "vision_end_token_id": 5,
            "vision_config": vision_cfg,
            "text_config": text_cfg,
        }

        cfg_path = os.path.join(cls.tmpdir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cls.config_dict, f)

        hf_cfg = AutoConfig.from_pretrained(cls.tmpdir)
        cls.hf_model = Qwen3VLMoeForConditionalGeneration(hf_cfg).eval().to(torch.float32)
        cls.hf_model.save_pretrained(cls.tmpdir, safe_serialization=True)

        cls.jax_loaded, _ = create_qwen3_vl_from_safetensors(cls.tmpdir, "qwen3-vl-smoke-moe")

    def test_forward_logits_match_hf(self):
        tokens = _random_tokens(batch=1, seq=8, vocab=self.config_dict["text_config"]["vocab_size"])
        attention_mask = np.ones_like(tokens, dtype=np.int64)

        with torch.no_grad():
            hf_logits = self.hf_model(
                input_ids=torch.tensor(tokens, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask, dtype=torch.long),
                pixel_values=None,
                image_grid_thw=None,
            ).logits.cpu().float().numpy()

        logits_jax, _ = self.jax_loaded(
            jnp.asarray(tokens, dtype=jnp.int32),
            jnp.asarray(attention_mask, dtype=jnp.int32),
        )
        logits_jax = np.asarray(logits_jax, dtype=np.float32)

        np.testing.assert_allclose(logits_jax, hf_logits, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    absltest.main()
