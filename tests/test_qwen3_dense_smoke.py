"""Smoke test for the Qwen3 dense JAX implementation against HuggingFace.

Creates a small HF Qwen3ForCausalLM from scratch with smoke-test dimensions,
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
from transformers import Qwen3Config as HFQwen3Config
from transformers import Qwen3ForCausalLM

from omegalax.text import api
from omegalax.models.qwen3.dense.config import make_dense_config
from omegalax.models.qwen3.dense.params_dense import create_qwen3_dense_from_safe_tensors

jax.config.update("jax_default_matmul_precision", "highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

RTOL = 1e-6
ATOL = 1e-6

SMOKE_ID = "qwen3-smoke"

HF_SMOKE_CFG = HFQwen3Config(
    vocab_size=1024,
    hidden_size=128,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=32,
    rope_theta=1_000_000,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    hidden_act="silu",
)


def _random_input(batch_size: int = 1, seq_len: int = 16, vocab_size: int = 1024, pad_id: int = 0):
    rng = np.random.RandomState(42)
    tokens = rng.randint(1, vocab_size, size=(batch_size, seq_len)).astype(np.int32)
    tokens[:, 0] = pad_id
    return tokens


class Qwen3DenseSmokeTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.hf_model = Qwen3ForCausalLM(HF_SMOKE_CFG).eval()
        cls.tmpdir = tempfile.mkdtemp()
        cls.hf_model.save_pretrained(cls.tmpdir, safe_serialization=True)

        cfg_path = os.path.join(cls.tmpdir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(HF_SMOKE_CFG.to_dict(), f)

        cls.jax_cfg = make_dense_config(SMOKE_ID)
        cls.jax_model = create_qwen3_dense_from_safe_tensors(cls.tmpdir, SMOKE_ID)
        cls.pad_id = 0

    def _jax_prefill_logits(self, tokens_np: np.ndarray) -> np.ndarray:
        tokens = jnp.asarray(tokens_np)
        logits, _ = api.forward(self.jax_model, tokens, self.pad_id, self.jax_cfg)
        return np.asarray(logits, dtype=np.float32)

    def test_weight_loading_succeeds(self):
        self.assertIsNotNone(self.jax_model)

    def test_prefill_logits_match_hf(self):
        tokens = _random_input(batch_size=1, seq_len=16, vocab_size=HF_SMOKE_CFG.vocab_size)
        attention_mask = (tokens != self.pad_id).astype(np.int64)

        with torch.no_grad():
            hf_out = self.hf_model(
                input_ids=torch.tensor(tokens, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            )
            hf_logits = hf_out.logits.cpu().numpy()

        jax_logits = self._jax_prefill_logits(tokens)

        mask = attention_mask.astype(bool)
        max_abs_diff = np.max(np.abs(jax_logits[mask] - hf_logits[mask]))
        print(f"\n  max_abs_diff = {max_abs_diff:.6e}")
        np.testing.assert_allclose(jax_logits[mask], hf_logits[mask], rtol=RTOL, atol=ATOL)

    def test_prefill_logits_match_hf_batched(self):
        tokens_a = _random_input(batch_size=1, seq_len=16, vocab_size=HF_SMOKE_CFG.vocab_size)
        tokens_b = _random_input(batch_size=1, seq_len=10, vocab_size=HF_SMOKE_CFG.vocab_size)

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

        jax_logits = self._jax_prefill_logits(tokens)

        mask = attention_mask.astype(bool)
        max_abs_diff = np.max(np.abs(jax_logits[mask] - hf_logits[mask]))
        print(f"\n  max_abs_diff = {max_abs_diff:.6e}")
        np.testing.assert_allclose(jax_logits[mask], hf_logits[mask], rtol=RTOL, atol=ATOL)

    def test_round_trip_preserves_logits(self):
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
