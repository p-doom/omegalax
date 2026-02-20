"""Full attention for Qwen3.5 text model.

Key differences from standard Qwen3 attention:
  - q_proj outputs 2x (query + output gate)
  - Partial RoPE (only partial_rotary_factor * head_dim dimensions)
  - MRoPE (multi-dimensional RoPE for 3-D position IDs)
  - RMSNorm uses (1 + weight) * norm(x)
"""

import jax
import jax.numpy as jnp
from flax import nnx

from .config import Qwen3_5TextConfig
from .norms import RMSNorm
from .rope import apply_text_rope

_K_MASK: float = float(jnp.finfo(jnp.float32).min)


class Attention(nnx.Module):
    def __init__(self, cfg: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        hd = cfg.head_dim
        nh = cfg.num_attention_heads
        nkv = cfg.num_key_value_heads

        self.q_proj = nnx.Linear(cfg.hidden_size, nh * hd * 2, use_bias=cfg.attention_bias, rngs=rngs)
        self.k_proj = nnx.Linear(cfg.hidden_size, nkv * hd, use_bias=cfg.attention_bias, rngs=rngs)
        self.v_proj = nnx.Linear(cfg.hidden_size, nkv * hd, use_bias=cfg.attention_bias, rngs=rngs)
        self.o_proj = nnx.Linear(nh * hd, cfg.hidden_size, use_bias=cfg.attention_bias, rngs=rngs)

        self.q_norm = RMSNorm(hd, cfg.rms_norm_eps, rngs=rngs)
        self.k_norm = RMSNorm(hd, cfg.rms_norm_eps, rngs=rngs)

        self.num_heads = nh
        self.num_kv_heads = nkv
        self.head_dim = hd
        self.n_rep = nh // nkv
        self.scale = hd ** -0.5

    @jax.named_scope("attention")
    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        segment_ids: jax.Array,
        position_ids: jax.Array,
    ) -> jax.Array:
        """Forward pass (prefill only, no cache).

        Args:
            x: (B, T, D)
            cos, sin: (B, T, rotary_dim)
            segment_ids: (B, T) — non-zero for real tokens
            position_ids: (B, T) — for causal mask
        """
        B, T, _ = x.shape

        # Q projection → query + output gate
        q_out = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim * 2)
        query, gate = jnp.split(q_out, 2, axis=-1)
        gate = gate.reshape(B, T, -1)

        # K, V projections
        query = self.q_norm(query).transpose(0, 2, 1, 3)  # (B, nh, T, hd)
        key = self.k_norm(
            self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)  # (B, nkv, T, hd)
        value = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply partial RoPE
        query, key = apply_text_rope(query, key, cos, sin)

        # GQA: expand K/V to match Q head count
        if self.n_rep > 1:
            key = jnp.repeat(key, self.n_rep, axis=1)
            value = jnp.repeat(value, self.n_rep, axis=1)

        # Attention
        attn_logits = jnp.matmul(query, key.transpose(0, 1, 3, 2)) * self.scale

        # Causal + segment mask
        q_pos = position_ids[:, :, None]  # (B, T, 1)
        k_pos = position_ids[:, None, :]  # (B, 1, T)
        causal_mask = k_pos <= q_pos
        seg_mask = segment_ids[:, :, None] == segment_ids[:, None, :]
        combined_mask = (causal_mask & seg_mask)[:, None, :, :]  # (B, 1, T, T)
        attn_logits = jnp.where(combined_mask, attn_logits, _K_MASK)

        attn_weights = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=-1).astype(attn_logits.dtype)
        attn_out = jnp.matmul(attn_weights, value)  # (B, nh, T, hd)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, -1)

        # Output gating
        attn_out = attn_out * jax.nn.sigmoid(gate)
        return self.o_proj(attn_out)
