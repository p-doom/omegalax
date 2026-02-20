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
from omegalax.models.qwen3.moe.params_moe import create_qwen3_moe_from_safe_tensors

jax.config.update("jax_default_matmul_precision", "highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

RTOL = 1e-6
ATOL = 1e-6

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
    tokens = rng.randint(1, vocab_size, size=(batch_size, seq_len)).astype(np.int32)
    tokens[:, 0] = pad_id
    return tokens


class Qwen3MoeWeightsTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.hf_model = Qwen3MoeForCausalLM(HF_SMOKE_CFG).eval()
        cls.tmpdir = tempfile.mkdtemp()
        cls.hf_model.save_pretrained(cls.tmpdir, safe_serialization=True)

        # Write config.json so our loader can validate it
        cfg_path = os.path.join(cls.tmpdir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(HF_SMOKE_CFG.to_dict(), f)

        cls.jax_cfg = make_moe_config(SMOKE_MOE_ID)
        cls.jax_model = create_qwen3_moe_from_safe_tensors(cls.tmpdir, SMOKE_MOE_ID)
        cls.pad_id = 0

    def test_weight_loading_succeeds(self):
        """The MoE weight loader should successfully load all parameters."""
        self.assertIsNotNone(self.jax_model)

    def test_prefill_logits_match_hf(self):
        """JAX MoE forward pass logits must match HF within tolerance."""
        tokens = _random_input(batch_size=1, seq_len=16, vocab_size=HF_SMOKE_CFG.vocab_size)
        attention_mask = (tokens != self.pad_id).astype(np.int64)

        with torch.no_grad():
            hf_out = self.hf_model(
                input_ids=torch.tensor(tokens, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            )
            hf_logits = hf_out.logits.cpu().numpy()

        jax_tokens = jnp.asarray(tokens)
        jax_logits, _ = api.forward(self.jax_model, jax_tokens, self.pad_id, self.jax_cfg)
        jax_logits = np.asarray(jax_logits, dtype=np.float32)

        mask = attention_mask.astype(bool)
        max_abs_diff = np.max(np.abs(jax_logits[mask] - hf_logits[mask]))
        print(f"\n  max_abs_diff = {max_abs_diff:.6e}")
        np.testing.assert_allclose(jax_logits[mask], hf_logits[mask], rtol=RTOL, atol=ATOL)

    def test_prefill_logits_match_hf_batched(self):
        """Batched forward pass with padding should match HF."""
        tokens_a = _random_input(batch_size=1, seq_len=16, vocab_size=HF_SMOKE_CFG.vocab_size)
        tokens_b = _random_input(batch_size=1, seq_len=10, vocab_size=HF_SMOKE_CFG.vocab_size)

        # Pad shorter sequence on the left to match longer
        padded_b = np.zeros((1, 16), dtype=np.int32)
        padded_b[:, 6:] = tokens_b
        tokens = np.concatenate([tokens_a, padded_b], axis=0)
        attention_mask = (tokens != self.pad_id).astype(np.int64)

        with torch.no_grad():
            hf_out = self.hf_model(
                input_ids=torch.tensor(tokens, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            )
            hf_logits = hf_out.logits.cpu().numpy()

        jax_tokens = jnp.asarray(tokens)
        jax_logits, _ = api.forward(self.jax_model, jax_tokens, self.pad_id, self.jax_cfg)
        jax_logits = np.asarray(jax_logits, dtype=np.float32)

        mask = attention_mask.astype(bool)
        max_abs_diff = np.max(np.abs(jax_logits[mask] - hf_logits[mask]))
        print(f"\n  max_abs_diff = {max_abs_diff:.6e}")
        np.testing.assert_allclose(jax_logits[mask], hf_logits[mask], rtol=RTOL, atol=ATOL)

    def test_round_trip_preserves_logits(self):
        """Split → merge round-trip should preserve forward pass output."""
        from flax import nnx

        tokens = _random_input(batch_size=1, seq_len=16, vocab_size=HF_SMOKE_CFG.vocab_size)
        jax_tokens = jnp.asarray(tokens)

        baseline, _ = api.forward(self.jax_model, jax_tokens, self.pad_id, self.jax_cfg)
        baseline = np.asarray(baseline)

        graph_def, state = nnx.split(self.jax_model)
        pure_state = nnx.to_pure_dict(state)
        restored = nnx.merge(graph_def, pure_state)
        restored_logits, _ = api.forward(restored, jax_tokens, self.pad_id, self.jax_cfg)
        restored_logits = np.asarray(restored_logits)

        np.testing.assert_array_equal(restored_logits, baseline)


if __name__ == "__main__":
    absltest.main()
