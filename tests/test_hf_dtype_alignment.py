import copy
import unittest

import jax.numpy as jnp

from omegalax.models.qwen3_5.config import make_config_from_hf
from omegalax.models.qwen3_vl.config import make_vl_config_from_hf


def _qwen3_vl_hf_cfg() -> dict:
    return {
        "model_type": "qwen3_vl",
        "tie_word_embeddings": False,
        "image_token_id": 2,
        "video_token_id": 3,
        "vision_start_token_id": 4,
        "text_config": {
            "dtype": "bfloat16",
            "num_hidden_layers": 2,
            "vocab_size": 1024,
            "hidden_size": 128,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "head_dim": 32,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000,
            "rope_scaling": {
                "mrope_section": [8, 4, 4],
            },
        },
        "vision_config": {
            "hidden_size": 64,
            "intermediate_size": 256,
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
        },
    }


def _qwen3_5_hf_cfg() -> dict:
    return {
        "tie_word_embeddings": False,
        "image_token_id": 11,
        "video_token_id": 12,
        "vision_start_token_id": 13,
        "vision_end_token_id": 14,
        "text_config": {
            "dtype": "bfloat16",
            "vocab_size": 1024,
            "hidden_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "rms_norm_eps": 1e-6,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "rope_parameters": {
                "rope_theta": 10_000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
            "attention_bias": False,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 32,
            "intermediate_size": 256,
        },
        "vision_config": {
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "in_channels": 3,
            "out_hidden_size": 128,
            "num_position_embeddings": 100,
        },
    }


class HFDtypeAlignmentTest(unittest.TestCase):
    def test_qwen3_vl_defaults_vision_dtype_to_text_dtype(self):
        cfg = make_vl_config_from_hf(_qwen3_vl_hf_cfg())
        self.assertEqual(cfg.dtype, jnp.bfloat16)
        self.assertEqual(cfg.vision.dtype, cfg.dtype)

    def test_qwen3_vl_honors_explicit_vision_dtype(self):
        hf_cfg = copy.deepcopy(_qwen3_vl_hf_cfg())
        hf_cfg["vision_config"]["dtype"] = "float32"
        cfg = make_vl_config_from_hf(hf_cfg)
        self.assertEqual(cfg.dtype, jnp.bfloat16)
        self.assertEqual(cfg.vision.dtype, jnp.float32)

    def test_qwen3_5_defaults_vision_dtype_to_text_dtype(self):
        cfg = make_config_from_hf(_qwen3_5_hf_cfg())
        self.assertEqual(cfg.text_config.dtype, jnp.bfloat16)
        self.assertEqual(cfg.vision_config.dtype, cfg.text_config.dtype)

    def test_qwen3_5_honors_explicit_vision_dtype(self):
        hf_cfg = copy.deepcopy(_qwen3_5_hf_cfg())
        hf_cfg["vision_config"]["dtype"] = "float32"
        cfg = make_config_from_hf(hf_cfg)
        self.assertEqual(cfg.text_config.dtype, jnp.bfloat16)
        self.assertEqual(cfg.vision_config.dtype, jnp.float32)


if __name__ == "__main__":
    unittest.main()
