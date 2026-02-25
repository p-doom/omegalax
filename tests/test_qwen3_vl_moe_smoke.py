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

_JNP_TO_TORCH = {jnp.float32: torch.float32, jnp.bfloat16: torch.bfloat16, jnp.float16: torch.float16}


def _tolerances(jnp_dtype):
    if jnp_dtype == jnp.float32:
        return 1e-5, 1e-5
    return 1e-2, 1e-2


def _random_tokens(batch: int, seq: int, vocab: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(1, vocab, size=(batch, seq), dtype=np.int64)


class Qwen3VLMoeSmokeTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmpdir = tempfile.mkdtemp()

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
        hf_model = Qwen3VLMoeForConditionalGeneration(hf_cfg).eval()
        hf_model.save_pretrained(cls.tmpdir, safe_serialization=True)

        with open(cfg_path, "w") as f:
            json.dump(hf_cfg.to_dict(), f)

        cls.jax_model, cls.jax_cfg = create_qwen3_vl_from_safetensors(cls.tmpdir)

        torch_dtype = _JNP_TO_TORCH[cls.jax_cfg.dtype]
        cls.hf_model = hf_model.to(torch_dtype)
        cls.RTOL, cls.ATOL = _tolerances(cls.jax_cfg.dtype)

    def test_forward_logits_match_hf(self):
        token_ids_BT = _random_tokens(batch=1, seq=8, vocab=self.config_dict["text_config"]["vocab_size"])
        attention_mask_BT = np.ones_like(token_ids_BT, dtype=np.int64)

        with torch.no_grad():
            hf_logits_BTV = self.hf_model(
                input_ids=torch.tensor(token_ids_BT, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask_BT, dtype=torch.long),
                pixel_values=None,
                image_grid_thw=None,
            ).logits.cpu().float().numpy()

        jax_logits_BTV, _ = self.jax_model(
            jnp.asarray(token_ids_BT, dtype=jnp.int32),
            jnp.asarray(attention_mask_BT, dtype=jnp.int32),
        )
        jax_logits_BTV = np.asarray(jax_logits_BTV, dtype=np.float32)

        max_abs_diff = np.max(np.abs(jax_logits_BTV - hf_logits_BTV))
        print(f"\n  max_abs_diff = {max_abs_diff:.6e}")
        np.testing.assert_allclose(jax_logits_BTV, hf_logits_BTV, rtol=self.RTOL, atol=self.ATOL)


if __name__ == "__main__":
    absltest.main()
