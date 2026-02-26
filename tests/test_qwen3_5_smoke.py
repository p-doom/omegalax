"""Correctness test for the Qwen3.5 MoE JAX implementation against HuggingFace.

Creates a small HF Qwen3_5MoeForConditionalGeneration model from scratch with
smoke-test dimensions, saves it to safetensors, loads it with our JAX weight
converter, and compares forward-pass logits (text-only, no vision input).
"""

import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig as HFConfig,
    Qwen3_5MoeTextConfig as HFTextConfig,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5VisionConfig as HFVisionConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration as HFModel,
)

from omegalax.models.qwen3_5.params import create_qwen3_5_from_safetensors

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
    intermediate_size=128,
    num_heads=4,
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    in_channels=3,
    out_hidden_size=128,
    num_position_embeddings=100,
)

HF_TEXT_CFG = HFTextConfig(
    vocab_size=1024,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    rms_norm_eps=1e-6,
    rope_parameters={
        "rope_theta": 10_000,
        "partial_rotary_factor": 0.25,
        "rope_type": "default",
        "mrope_section": [2, 1, 1],
        "mrope_interleaved": True,
    },
    layer_types=[
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    linear_conv_kernel_dim=4,
    linear_key_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_value_head_dim=32,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
    num_experts=4,
    num_experts_per_tok=2,
    tie_word_embeddings=False,
)

HF_CFG = HFConfig(
    vision_config=HF_VISION_CFG.to_dict(),
    text_config=HF_TEXT_CFG.to_dict(),
    tie_word_embeddings=False,
)


def _random_input(
    batch_size: int = 1,
    seq_len: int = 16,
    vocab_size: int = 1024,
):
    rng = np.random.RandomState(42)
    return rng.randint(1, vocab_size, size=(batch_size, seq_len)).astype(np.int32)


class Qwen3_5WeightsTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmpdir = tempfile.mkdtemp()

        hf_model = HFModel(HF_CFG).eval()
        hf_model.save_pretrained(cls.tmpdir, safe_serialization=True)

        cls.jax_model, cls.jax_cfg = create_qwen3_5_from_safetensors(
            cls.tmpdir,
            tp_size=1,
            fsdp_size=1,
        )

        torch_dtype = _JNP_TO_TORCH[cls.jax_cfg.text_config.dtype]
        cls.hf_model = hf_model.to(torch_dtype)
        cls.RTOL, cls.ATOL = _tolerances(cls.jax_cfg.text_config.dtype)
        cls.pad_id = 0

    def _jax_prefill_logits(self, tokens_np: np.ndarray) -> np.ndarray:
        token_ids_BT = jnp.asarray(tokens_np)
        segment_ids_BT = (token_ids_BT != self.pad_id).astype(jnp.int32)
        logits_BTV, _ = self.jax_model(
            token_ids_BT, segment_ids_BT, None, jnp.array(0, dtype=jnp.int32)
        )
        return np.asarray(logits_BTV, dtype=np.float32)

    def test_weight_loading_succeeds(self):
        self.assertIsNotNone(self.jax_model)

    def test_prefill_logits_match_hf(self):
        """Single-sequence text-only forward pass should match HuggingFace."""
        token_ids_BT = _random_input(batch_size=1, seq_len=16, vocab_size=HF_TEXT_CFG.vocab_size)
        attention_mask_BT = np.ones_like(token_ids_BT, dtype=np.int64)

        with torch.no_grad():
            hf_out = self.hf_model(
                input_ids=torch.tensor(token_ids_BT, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask_BT, dtype=torch.long),
                use_cache=False,
            )
            hf_logits_BTV = hf_out.logits.cpu().float().numpy()

        jax_logits_BTV = self._jax_prefill_logits(token_ids_BT)

        max_abs_diff = np.max(np.abs(jax_logits_BTV - hf_logits_BTV))
        print(f"\n  max_abs_diff = {max_abs_diff:.6e}")
        np.testing.assert_allclose(jax_logits_BTV, hf_logits_BTV, rtol=self.RTOL, atol=self.ATOL)

    def test_prefill_logits_match_hf_batched(self):
        """Batched forward pass with left-padding should match HuggingFace.

        HF's create_causal_mask handles padding differently for B=1 vs B>1,
        so we test padding only in the batched case (B=2) where HF is reliable.
        """
        token_ids_a_BT = _random_input(batch_size=1, seq_len=16, vocab_size=HF_TEXT_CFG.vocab_size)
        token_ids_b_BT = _random_input(batch_size=1, seq_len=10, vocab_size=HF_TEXT_CFG.vocab_size)

        padded_b = np.zeros((1, 16), dtype=np.int32)
        padded_b[:, 6:] = token_ids_b_BT
        token_ids_BT = np.concatenate([token_ids_a_BT, padded_b], axis=0)
        attention_mask_BT = (token_ids_BT != self.pad_id).astype(np.int64)

        with torch.no_grad():
            hf_out = self.hf_model(
                input_ids=torch.tensor(token_ids_BT, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask_BT, dtype=torch.long),
                use_cache=False,
            )
            hf_logits_BTV = hf_out.logits.cpu().float().numpy()

        jax_logits_BTV = self._jax_prefill_logits(token_ids_BT)

        mask = attention_mask_BT.astype(bool)
        max_abs_diff = np.max(np.abs(jax_logits_BTV[mask] - hf_logits_BTV[mask]))
        print(f"\n  max_abs_diff = {max_abs_diff:.6e}")
        np.testing.assert_allclose(jax_logits_BTV[mask], hf_logits_BTV[mask], rtol=self.RTOL, atol=self.ATOL)

    def test_round_trip_preserves_logits(self):
        """Split → merge round-trip should produce identical logits."""
        from flax import nnx

        token_ids_BT = _random_input(batch_size=1, seq_len=16, vocab_size=HF_TEXT_CFG.vocab_size)
        baseline_BTV = self._jax_prefill_logits(token_ids_BT)

        graph_def, state = nnx.split(self.jax_model)
        pure_state = nnx.to_pure_dict(state)
        restored = nnx.merge(graph_def, pure_state)

        jax_token_ids_BT = jnp.asarray(token_ids_BT)
        segment_ids_BT = (jax_token_ids_BT != self.pad_id).astype(jnp.int32)
        restored_logits_BTV, _ = restored(
            jax_token_ids_BT, segment_ids_BT, None, jnp.array(0, dtype=jnp.int32)
        )
        restored_logits_BTV = np.asarray(restored_logits_BTV)

        np.testing.assert_array_equal(restored_logits_BTV, baseline_BTV)


if __name__ == "__main__":
    absltest.main()
