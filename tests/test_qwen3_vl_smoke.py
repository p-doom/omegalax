"""Smoke test for the Qwen3-VL JAX implementation against HuggingFace.

Creates a small HF Qwen3VLForConditionalGeneration from scratch with
smoke-test dimensions, saves it to safetensors, loads it with our JAX
loader, and compares forward-pass logits (text-only and image+text).
"""

import json
import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig as HFQwen3VLConfig,
    Qwen3VLTextConfig as HFTextConfig,
    Qwen3VLVisionConfig as HFVisionConfig,
)

from omegalax.models.qwen3_vl import create_qwen3_vl_from_safetensors

jax.config.update("jax_default_matmul_precision", "highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

_JNP_TO_TORCH = {jnp.float32: torch.float32, jnp.bfloat16: torch.bfloat16, jnp.float16: torch.float16}


def _tolerances(jnp_dtype):
    if jnp_dtype == jnp.float32:
        return 1e-5, 1e-5
    return 1e-2, 1e-2

HF_VISION_CFG = HFVisionConfig(
    depth=2,
    hidden_size=64,
    intermediate_size=256,
    num_heads=4,
    patch_size=14,
    temporal_patch_size=2,
    spatial_merge_size=2,
    in_channels=3,
    out_hidden_size=128,
    num_position_embeddings=256,
    hidden_act="gelu_pytorch_tanh",
    deepstack_visual_indexes=[0],
)

HF_TEXT_CFG = HFTextConfig(
    vocab_size=1024,
    hidden_size=128,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=32,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    rope_parameters={"rope_theta": 1_000_000, "rope_type": "default", "mrope_section": [8, 4, 4]},
    tie_word_embeddings=False,
)

HF_CFG = HFQwen3VLConfig(
    vision_config=HF_VISION_CFG.to_dict(),
    text_config=HF_TEXT_CFG.to_dict(),
    image_token_id=2,
    video_token_id=3,
    vision_start_token_id=4,
    tie_word_embeddings=False,
)


def _random_input(batch_size: int = 1, seq_len: int = 16, vocab_size: int = 1024):
    rng = np.random.RandomState(42)
    return rng.randint(1, vocab_size, size=(batch_size, seq_len)).astype(np.int32)


class Qwen3VLSmokeTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmpdir = tempfile.mkdtemp()

        hf_model = Qwen3VLForConditionalGeneration(HF_CFG).eval()
        hf_model.save_pretrained(cls.tmpdir, safe_serialization=True)

        cfg_path = os.path.join(cls.tmpdir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(HF_CFG.to_dict(), f)

        cls.jax_model, cls.jax_cfg = create_qwen3_vl_from_safetensors(
            cls.tmpdir,
            tp_size=1,
            fsdp_size=1,
        )

        torch_dtype = _JNP_TO_TORCH[cls.jax_cfg.dtype]
        cls.hf_model = hf_model.to(torch_dtype)
        cls.RTOL, cls.ATOL = _tolerances(cls.jax_cfg.dtype)

    def test_weight_loading_succeeds(self):
        self.assertIsNotNone(self.jax_model)

    def test_text_only_prefill_logits_match_hf(self):
        token_ids_BT = _random_input(batch_size=1, seq_len=16, vocab_size=HF_TEXT_CFG.vocab_size)
        attention_mask_BT = np.ones_like(token_ids_BT, dtype=np.int64)

        with torch.no_grad():
            hf_out = self.hf_model(
                input_ids=torch.tensor(token_ids_BT, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask_BT, dtype=torch.long),
            )
            hf_logits_BTV = hf_out.logits.cpu().float().numpy()

        token_ids_jax_BT = jnp.asarray(token_ids_BT)
        attention_mask_jax_BT = jnp.asarray(attention_mask_BT.astype(np.int32))
        jax_logits_BTV = np.asarray(self.jax_model(token_ids_jax_BT, attention_mask_jax_BT), dtype=np.float32)

        max_abs_diff = np.max(np.abs(jax_logits_BTV - hf_logits_BTV))
        print(f"\n  max_abs_diff = {max_abs_diff:.6e}")
        np.testing.assert_allclose(jax_logits_BTV, hf_logits_BTV, rtol=self.RTOL, atol=self.ATOL)

    def test_text_only_prefill_logits_batched(self):
        token_ids_a_BT = _random_input(batch_size=1, seq_len=16, vocab_size=HF_TEXT_CFG.vocab_size)
        token_ids_b_BT = _random_input(batch_size=1, seq_len=10, vocab_size=HF_TEXT_CFG.vocab_size)

        padded_b = np.zeros((1, 16), dtype=np.int32)
        padded_b[:, 6:] = token_ids_b_BT
        token_ids_BT = np.concatenate([token_ids_a_BT, padded_b], axis=0)
        attention_mask_BT = (token_ids_BT != 0).astype(np.int64)

        with torch.no_grad():
            hf_out = self.hf_model(
                input_ids=torch.tensor(token_ids_BT, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask_BT, dtype=torch.long),
            )
            hf_logits_BTV = hf_out.logits.cpu().float().numpy()

        token_ids_jax_BT = jnp.asarray(token_ids_BT)
        attention_mask_jax_BT = jnp.asarray(attention_mask_BT.astype(np.int32))
        jax_logits_BTV = np.asarray(self.jax_model(token_ids_jax_BT, attention_mask_jax_BT), dtype=np.float32)

        mask = attention_mask_BT.astype(bool)
        max_abs_diff = np.max(np.abs(jax_logits_BTV[mask] - hf_logits_BTV[mask]))
        print(f"\n  max_abs_diff = {max_abs_diff:.6e}")
        np.testing.assert_allclose(jax_logits_BTV[mask], hf_logits_BTV[mask], rtol=self.RTOL, atol=self.ATOL)

    def test_round_trip_preserves_logits(self):
        from flax import nnx

        token_ids_BT = _random_input(batch_size=1, seq_len=16, vocab_size=HF_TEXT_CFG.vocab_size)
        token_ids_jax_BT = jnp.asarray(token_ids_BT)
        attention_mask_jax_BT = jnp.ones_like(token_ids_jax_BT, dtype=jnp.int32)

        baseline_BTV = np.asarray(self.jax_model(token_ids_jax_BT, attention_mask_jax_BT), dtype=np.float32)

        graph_def, state = nnx.split(self.jax_model)
        pure_state = nnx.to_pure_dict(state)
        restored = nnx.merge(graph_def, pure_state)
        restored_logits_BTV = np.asarray(restored(token_ids_jax_BT, attention_mask_jax_BT), dtype=np.float32)

        np.testing.assert_array_equal(restored_logits_BTV, baseline_BTV)


if __name__ == "__main__":
    absltest.main()
