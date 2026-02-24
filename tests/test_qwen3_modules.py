"""Module-level numerical equivalence tests.

Each test feeds the **identical** input (produced by HF) into both the JAX
and HuggingFace implementations of a single module so that per-module error
is measured in isolation, without accumulation from prior layers.

Empirically measured per-module error on Qwen3-0.6B (CPU, float32):

  Module            max_diff    ULPs
  ──────────────────────────────────
  RMSNorm           ~1e-6       ~1  
  RoPE cos/sin      ~6e-8       <1  
  apply_rope        ~5e-7       ~2  
  linear (Q/K/V/O)  ~7e-6       6-13
  q_norm / k_norm   ~8e-6       ~2  
  full attention     ~1e-6       ~4  
  full MLP           ~1e-6       ~44
  full decoder layer ~3e-6       ~5  

Linear projections dominate at 6-13 ULPs because XLA CPU and PyTorch CPU
use different BLAS implementations with different dot-product accumulation
orders.  For a 1024-element inner dimension the theoretical bound is
~sqrt(1024) ≈ 32 ULPs, so 6-13 ULPs is well within expectations.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from omegalax.text import api
from omegalax.models.qwen3.dense import params_dense
from omegalax.models.qwen3.rope import apply_rope, generate_pos_embeddings

jax.config.update("jax_default_matmul_precision", "highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "Why is the sky blue instead of another color like purple?"

# Per-module absolute tolerances, derived from empirical measurements.
# Each tolerance is set to ~2x the observed max_diff for that module to
# guard against regressions while staying as tight as empirically feasible.
NORM_ATOL = 2e-6          # RMSNorm: measured ~1e-6 (1 ULP)
ROPE_ATOL = 1e-6          # RoPE: measured ~5e-7 (<1-2 ULPs)
LINEAR_ATOL = 2e-5        # matmuls: measured ~7e-6 (6-13 ULPs)
ATTENTION_ATOL = 5e-6     # full attention: measured ~1.2e-6 (3-4 ULPs)
MLP_ATOL = 5e-6           # full MLP: measured ~1e-6
DECODER_LAYER_ATOL = 1e-5 # full layer: measured ~3e-6 (5 ULPs)


def _to_jax(t: torch.Tensor) -> jax.Array:
    return jnp.asarray(t.detach().cpu().float().numpy())


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


class Qwen3ModuleTest(absltest.TestCase):
    """Feed identical HF-produced inputs into each module pair and compare."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cfg = api.registry.build_config(MODEL_ID)
        cls.model_path = snapshot_download(MODEL_ID)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_path)
        hf_cfg = AutoConfig.from_pretrained(cls.model_path)
        hf_cfg.tie_word_embeddings = cls.cfg.tie_word_embeddings
        cls.hf_model = Qwen3ForCausalLM.from_pretrained(
            cls.model_path, config=hf_cfg, torch_dtype=torch.float32
        ).eval()
        cls.jax_model = params_dense.create_qwen3_dense_from_safetensors(cls.model_path, MODEL_ID)

        chat_text = cls.tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        cls.inputs = cls.tokenizer(chat_text, return_tensors="pt", padding=True)
        cls.pad_id = cls.tokenizer.pad_token_id or 0

        with torch.no_grad():
            cls.hf_emb = cls.hf_model.model.embed_tokens(cls.inputs["input_ids"])

        tokens_jax = jnp.asarray(np.array(cls.inputs["input_ids"].cpu(), dtype=np.int32))
        cls.tokens_jax = tokens_jax
        cls.segment_ids = jnp.array(1 * (tokens_jax != cls.pad_id), dtype=jnp.int32)

    # Helpers
    def _assert_close(self, jax_val, hf_val, *, atol, rtol=0.0, msg=""):
        j = _to_np(jax_val)
        h = _to_np(hf_val)
        np.testing.assert_allclose(j, h, atol=atol, rtol=rtol, err_msg=msg)

    # 1. RMSNorm
    def test_rms_norm(self):
        """Isolated RMSNorm with identical input."""
        jax_inp = _to_jax(self.hf_emb)
        with torch.no_grad():
            hf_out = self.hf_model.model.layers[0].input_layernorm(self.hf_emb)
        jax_out = self.jax_model.layers[0].input_layernorm(jax_inp)
        self._assert_close(jax_out, hf_out, atol=NORM_ATOL, msg="RMSNorm")

    # 2. RoPE
    def test_rope(self):
        """RoPE sin/cos generation and application."""
        seq_len = self.hf_emb.shape[1]
        head_dim = self.cfg.head_dim

        position_ids_torch = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            hf_cos, hf_sin = self.hf_model.model.rotary_emb(self.hf_emb, position_ids_torch)

        position_ids_jax = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        jax_sin, jax_cos = generate_pos_embeddings(position_ids_jax, head_dim)

        hf_cos_half = _to_np(hf_cos)[..., :head_dim // 2]
        hf_sin_half = _to_np(hf_sin)[..., :head_dim // 2]
        self._assert_close(jax_cos, hf_cos_half, atol=ROPE_ATOL, msg="RoPE cos")
        self._assert_close(jax_sin, hf_sin_half, atol=ROPE_ATOL, msg="RoPE sin")

        rng = np.random.RandomState(42)
        q_np = rng.randn(1, seq_len, self.cfg.num_heads, head_dim).astype(np.float32)
        q_torch = torch.from_numpy(q_np).transpose(1, 2)
        q_jax = jnp.asarray(q_np)

        with torch.no_grad():
            hf_q_rope, _ = apply_rotary_pos_emb(q_torch, q_torch, hf_cos, hf_sin)
        hf_q_rope = hf_q_rope.transpose(1, 2)

        jax_q_rope = apply_rope(q_jax, jax_sin, jax_cos)
        self._assert_close(jax_q_rope, hf_q_rope, atol=ROPE_ATOL, msg="apply_rope")

    # 3. Linear (Q projection)
    def test_linear_q_proj(self):
        """Single linear projection."""
        with torch.no_grad():
            hf_normed = self.hf_model.model.layers[0].input_layernorm(self.hf_emb)
            hf_q = self.hf_model.model.layers[0].self_attn.q_proj(hf_normed)

        jax_normed = _to_jax(hf_normed)
        jax_q = self.jax_model.layers[0].attn.q_proj(jax_normed)
        self._assert_close(jax_q, hf_q, atol=LINEAR_ATOL, msg="q_proj (same input)")

    # 4. Full Attention block
    def test_attention(self):
        """Full attention block."""
        hf_layer = self.hf_model.model.layers[0]
        with torch.no_grad():
            hf_normed = hf_layer.input_layernorm(self.hf_emb)

        seq_len = self.hf_emb.shape[1]
        position_ids_torch = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            pos_emb = self.hf_model.model.rotary_emb(self.hf_emb, position_ids_torch)
            from transformers.masking_utils import create_causal_mask
            causal_mask = create_causal_mask(
                config=self.hf_model.config,
                inputs_embeds=self.hf_emb,
                attention_mask=self.inputs["attention_mask"],
                cache_position=torch.arange(seq_len),
                past_key_values=None,
                position_ids=position_ids_torch,
            )
            hf_attn_out, _ = hf_layer.self_attn(
                hidden_states=hf_normed,
                position_embeddings=pos_emb,
                attention_mask=causal_mask,
            )

        jax_normed = _to_jax(hf_normed)
        jax_attn_out = self.jax_model.layers[0].attn(jax_normed, None, self.segment_ids)
        self._assert_close(jax_attn_out, hf_attn_out, atol=ATTENTION_ATOL, msg="attention")

    # 5. MLP
    def test_mlp(self):
        """Full MLP block."""
        with torch.no_grad():
            hf_normed = self.hf_model.model.layers[0].input_layernorm(self.hf_emb)
            seq_len = self.hf_emb.shape[1]
            pos_emb = self.hf_model.model.rotary_emb(
                self.hf_emb, torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
            )
            hf_attn_out, _ = self.hf_model.model.layers[0].self_attn(
                hidden_states=hf_normed, position_embeddings=pos_emb, attention_mask=None
            )
            hf_after_attn = self.hf_emb + hf_attn_out
            hf_post_ln = self.hf_model.model.layers[0].post_attention_layernorm(hf_after_attn)
            hf_mlp_out = self.hf_model.model.layers[0].mlp(hf_post_ln)

        jax_mlp_out = self.jax_model.layers[0].mlp(_to_jax(hf_post_ln))
        self._assert_close(jax_mlp_out, hf_mlp_out, atol=MLP_ATOL, msg="MLP (real activations)")

    # 6. Single DecoderLayer
    def test_decoder_layer(self):
        """Full decoder layer."""
        hf_layer = self.hf_model.model.layers[0]
        seq_len = self.hf_emb.shape[1]
        position_ids_torch = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            pos_emb = self.hf_model.model.rotary_emb(self.hf_emb, position_ids_torch)
            from transformers.masking_utils import create_causal_mask
            causal_mask = create_causal_mask(
                config=self.hf_model.config,
                inputs_embeds=self.hf_emb,
                attention_mask=self.inputs["attention_mask"],
                cache_position=torch.arange(seq_len),
                past_key_values=None,
                position_ids=position_ids_torch,
            )
            hf_out = hf_layer(
                self.hf_emb,
                attention_mask=causal_mask,
                position_embeddings=pos_emb,
                position_ids=position_ids_torch,
            )

        jax_out = self.jax_model.layers[0](_to_jax(self.hf_emb), None, self.segment_ids)
        self._assert_close(jax_out, hf_out, atol=DECODER_LAYER_ATOL, msg="decoder layer 0")


if __name__ == "__main__":
    absltest.main()
