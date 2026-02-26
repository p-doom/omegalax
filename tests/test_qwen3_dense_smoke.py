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
from omegalax.models.qwen3.dense.params_dense import create_qwen3_dense_from_safetensors

jax.config.update("jax_default_matmul_precision", "highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

_JNP_TO_TORCH = {jnp.float32: torch.float32, jnp.bfloat16: torch.bfloat16, jnp.float16: torch.float16}


def _tolerances(jnp_dtype):
    if jnp_dtype == jnp.float32:
        return 1e-5, 1e-5
    return 1e-2, 1e-2

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
    token_ids_BT = rng.randint(1, vocab_size, size=(batch_size, seq_len)).astype(np.int32)
    token_ids_BT[:, 0] = pad_id
    return token_ids_BT


class Qwen3DenseSmokeTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmpdir = tempfile.mkdtemp()
        torch.manual_seed(0)
        np.random.seed(0)

        hf_model = Qwen3ForCausalLM(HF_SMOKE_CFG).eval()
        hf_model.save_pretrained(cls.tmpdir, safe_serialization=True)

        cfg_path = os.path.join(cls.tmpdir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(HF_SMOKE_CFG.to_dict(), f)

        cls.jax_cfg = make_dense_config(SMOKE_ID)
        cls.jax_model = create_qwen3_dense_from_safetensors(
            cls.tmpdir,
            SMOKE_ID,
            tp_size=1,
            fsdp_size=1,
        )

        torch_dtype = _JNP_TO_TORCH[cls.jax_cfg.dtype]
        cls.hf_model = hf_model.to(torch_dtype)
        cls.RTOL, cls.ATOL = _tolerances(cls.jax_cfg.dtype)
        cls.pad_id = 0

    def _jax_prefill_logits(self, tokens_np: np.ndarray) -> np.ndarray:
        token_ids_BT = jnp.asarray(tokens_np)
        logits_BTV, _ = api.forward(self.jax_model, token_ids_BT, self.pad_id, self.jax_cfg)
        return np.asarray(logits_BTV, dtype=np.float32)

    def test_weight_loading_succeeds(self):
        self.assertIsNotNone(self.jax_model)

    def test_prefill_logits_match_hf(self):
        token_ids_BT = _random_input(batch_size=1, seq_len=16, vocab_size=HF_SMOKE_CFG.vocab_size)
        attention_mask_BT = (token_ids_BT != self.pad_id).astype(np.int64)

        with torch.no_grad():
            hf_out = self.hf_model(
                input_ids=torch.tensor(token_ids_BT, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask_BT, dtype=torch.long),
            )
            hf_logits_BTV = hf_out.logits.cpu().float().numpy()

        jax_logits_BTV = self._jax_prefill_logits(token_ids_BT)

        mask = attention_mask_BT.astype(bool)
        max_abs_diff = np.max(np.abs(jax_logits_BTV[mask] - hf_logits_BTV[mask]))
        print(f"\n  max_abs_diff = {max_abs_diff:.6e}")
        np.testing.assert_allclose(jax_logits_BTV[mask], hf_logits_BTV[mask], rtol=self.RTOL, atol=self.ATOL)

    def test_prefill_logits_match_hf_batched(self):
        token_ids_a_BT = _random_input(batch_size=1, seq_len=16, vocab_size=HF_SMOKE_CFG.vocab_size)
        token_ids_b_BT = _random_input(batch_size=1, seq_len=10, vocab_size=HF_SMOKE_CFG.vocab_size)

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

        jax_logits_BTV = self._jax_prefill_logits(token_ids_BT)

        mask = attention_mask_BT.astype(bool)
        max_abs_diff = np.max(np.abs(jax_logits_BTV[mask] - hf_logits_BTV[mask]))
        print(f"\n  max_abs_diff = {max_abs_diff:.6e}")
        np.testing.assert_allclose(jax_logits_BTV[mask], hf_logits_BTV[mask], rtol=self.RTOL, atol=self.ATOL)

    def test_round_trip_preserves_logits(self):
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
