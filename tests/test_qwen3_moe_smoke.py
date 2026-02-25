"""Correctness test for the Qwen3 MoE JAX implementation against HuggingFace.

Creates a small HF Qwen3MoeForCausalLM from scratch with smoke-test dimensions,
saves it to safetensors, loads it with our JAX loader, and compares
forward-pass logits.
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
from transformers import Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig as HFQwen3MoeConfig

from omegalax.text import api
from omegalax.models.qwen3.moe.config import make_moe_config
from omegalax.models.qwen3.moe.params_moe import create_qwen3_moe_from_safetensors

jax.config.update("jax_default_matmul_precision", "highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

_JNP_TO_TORCH = {jnp.float32: torch.float32, jnp.bfloat16: torch.bfloat16, jnp.float16: torch.float16}


def _tolerances(jnp_dtype):
    if jnp_dtype == jnp.float32:
        return 1e-5, 1e-5
    return 1e-2, 1e-2

SMOKE_MOE_ID = "qwen3-smoke-moe"

HF_SMOKE_CFG = HFQwen3MoeConfig(
    hidden_size=128,
    intermediate_size=256,
    moe_intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=32,
    num_experts=4,
    num_experts_per_tok=2,
    vocab_size=512,
    rope_theta=1_000_000,
    decoder_sparse_step=1,
    mlp_only_layers=[],
    norm_topk_prob=True,
    tie_word_embeddings=False,
    hidden_act="silu",
    rms_norm_eps=1e-6,
)


def _random_input(batch_size: int = 1, seq_len: int = 16, vocab_size: int = 512, pad_id: int = 0):
    rng = np.random.RandomState(42)
    token_ids_BT = rng.randint(1, vocab_size, size=(batch_size, seq_len)).astype(np.int32)
    token_ids_BT[:, 0] = pad_id
    return token_ids_BT


class Qwen3MoeWeightsTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmpdir = tempfile.mkdtemp()

        hf_model = Qwen3MoeForCausalLM(HF_SMOKE_CFG).eval()
        hf_model.save_pretrained(cls.tmpdir, safe_serialization=True)

        cfg_path = os.path.join(cls.tmpdir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(HF_SMOKE_CFG.to_dict(), f)

        cls.jax_cfg = make_moe_config(SMOKE_MOE_ID)
        cls.jax_model = create_qwen3_moe_from_safetensors(cls.tmpdir, SMOKE_MOE_ID)

        torch_dtype = _JNP_TO_TORCH[cls.jax_cfg.dtype]
        cls.hf_model = hf_model.to(torch_dtype)
        cls.RTOL, cls.ATOL = _tolerances(cls.jax_cfg.dtype)
        cls.pad_id = 0

    def test_weight_loading_succeeds(self):
        """The MoE weight loader should successfully load all parameters."""
        self.assertIsNotNone(self.jax_model)

    def test_prefill_logits_match_hf(self):
        """JAX MoE forward pass logits must match HF within tolerance."""
        token_ids_BT = _random_input(batch_size=1, seq_len=16, vocab_size=HF_SMOKE_CFG.vocab_size)
        attention_mask_BT = (token_ids_BT != self.pad_id).astype(np.int64)

        with torch.no_grad():
            hf_out = self.hf_model(
                input_ids=torch.tensor(token_ids_BT, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask_BT, dtype=torch.long),
            )
            hf_logits_BTV = hf_out.logits.cpu().float().numpy()

        jax_token_ids_BT = jnp.asarray(token_ids_BT)
        jax_logits_BTV, _ = api.forward(self.jax_model, jax_token_ids_BT, self.pad_id, self.jax_cfg)
        jax_logits_BTV = np.asarray(jax_logits_BTV, dtype=np.float32)

        mask = attention_mask_BT.astype(bool)
        max_abs_diff = np.max(np.abs(jax_logits_BTV[mask] - hf_logits_BTV[mask]))
        print(f"\n  max_abs_diff = {max_abs_diff:.6e}")
        np.testing.assert_allclose(jax_logits_BTV[mask], hf_logits_BTV[mask], rtol=self.RTOL, atol=self.ATOL)

    def test_prefill_logits_match_hf_batched(self):
        """Batched forward pass with padding should match HF."""
        token_ids_a_BT = _random_input(batch_size=1, seq_len=16, vocab_size=HF_SMOKE_CFG.vocab_size)
        token_ids_b_BT = _random_input(batch_size=1, seq_len=10, vocab_size=HF_SMOKE_CFG.vocab_size)

        # Pad shorter sequence on the left to match longer
        padded_b = np.zeros((1, 16), dtype=np.int32)
        padded_b[:, 6:] = token_ids_b_BT
        token_ids_BT = np.concatenate([token_ids_a_BT, padded_b], axis=0)
        attention_mask_BT = (token_ids_BT != self.pad_id).astype(np.int64)

        with torch.no_grad():
            hf_out = self.hf_model(
                input_ids=torch.tensor(token_ids_BT, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask_BT, dtype=torch.long),
            )
            hf_logits_BTV = hf_out.logits.cpu().float().numpy()

        jax_token_ids_BT = jnp.asarray(token_ids_BT)
        jax_logits_BTV, _ = api.forward(self.jax_model, jax_token_ids_BT, self.pad_id, self.jax_cfg)
        jax_logits_BTV = np.asarray(jax_logits_BTV, dtype=np.float32)

        mask = attention_mask_BT.astype(bool)
        max_abs_diff = np.max(np.abs(jax_logits_BTV[mask] - hf_logits_BTV[mask]))
        print(f"\n  max_abs_diff = {max_abs_diff:.6e}")
        np.testing.assert_allclose(jax_logits_BTV[mask], hf_logits_BTV[mask], rtol=self.RTOL, atol=self.ATOL)

    def test_round_trip_preserves_logits(self):
        """Split → merge round-trip should preserve forward pass output."""
        from flax import nnx

        token_ids_BT = _random_input(batch_size=1, seq_len=16, vocab_size=HF_SMOKE_CFG.vocab_size)
        jax_token_ids_BT = jnp.asarray(token_ids_BT)

        baseline_BTV, _ = api.forward(self.jax_model, jax_token_ids_BT, self.pad_id, self.jax_cfg)
        baseline_BTV = np.asarray(baseline_BTV)

        graph_def, state = nnx.split(self.jax_model)
        pure_state = nnx.to_pure_dict(state)
        restored = nnx.merge(graph_def, pure_state)
        restored_logits_BTV, _ = api.forward(restored, jax_token_ids_BT, self.pad_id, self.jax_cfg)
        restored_logits_BTV = np.asarray(restored_logits_BTV)

        np.testing.assert_array_equal(restored_logits_BTV, baseline_BTV)


if __name__ == "__main__":
    absltest.main()
