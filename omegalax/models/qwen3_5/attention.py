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

def _mask_value(dtype: jnp.dtype) -> float:
    return float(jnp.finfo(dtype).min)


class Attention(nnx.Module):
    def __init__(self, cfg: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        hd = cfg.head_dim
        nh = cfg.num_attention_heads
        nkv = cfg.num_key_value_heads

        self.q_proj = nnx.Linear(cfg.hidden_size, nh * hd * 2, use_bias=cfg.attention_bias, rngs=rngs, dtype=cfg.dtype)
        self.k_proj = nnx.Linear(cfg.hidden_size, nkv * hd, use_bias=cfg.attention_bias, rngs=rngs, dtype=cfg.dtype)
        self.v_proj = nnx.Linear(cfg.hidden_size, nkv * hd, use_bias=cfg.attention_bias, rngs=rngs, dtype=cfg.dtype)
        self.o_proj = nnx.Linear(nh * hd, cfg.hidden_size, use_bias=cfg.attention_bias, rngs=rngs, dtype=cfg.dtype)

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
        hidden_BTD: jax.Array,
        cos_BTK: jax.Array,
        sin_BTK: jax.Array,
        segment_ids_BT: jax.Array,
        position_ids_BT: jax.Array,
    ) -> jax.Array:
        B, T, _ = hidden_BTD.shape

        q_out_BTHK2 = self.q_proj(hidden_BTD).reshape(B, T, self.num_heads, self.head_dim * 2)
        q_BTHK, gate_BTHK = jnp.split(q_out_BTHK2, 2, axis=-1)
        gate_BTD = gate_BTHK.reshape(B, T, -1)

        q_BHTK = self.q_norm(q_BTHK).transpose(0, 2, 1, 3)
        k_BGTK = self.k_norm(
            self.k_proj(hidden_BTD).reshape(B, T, self.num_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        v_BGTK = self.v_proj(hidden_BTD).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q_BHTK, k_BGTK = apply_text_rope(q_BHTK, k_BGTK, cos_BTK, sin_BTK)

        if self.n_rep > 1:
            k_BHTK = jnp.repeat(k_BGTK, self.n_rep, axis=1)
            v_BHTK = jnp.repeat(v_BGTK, self.n_rep, axis=1)
        else:
            k_BHTK = k_BGTK
            v_BHTK = v_BGTK

        logits_BHTS = jnp.matmul(q_BHTK, k_BHTK.transpose(0, 1, 3, 2)) * self.scale

        q_pos_BT1 = position_ids_BT[:, :, None]
        k_pos_B1T = position_ids_BT[:, None, :]
        causal_mask_BTS = k_pos_B1T <= q_pos_BT1
        seg_mask_BTS = segment_ids_BT[:, :, None] == segment_ids_BT[:, None, :]
        combined_mask_BTS = (causal_mask_BTS & seg_mask_BTS)[:, None, :, :]
        logits_BHTS = jnp.where(combined_mask_BTS, logits_BHTS, _mask_value(logits_BHTS.dtype))

        weights_BHTS = jax.nn.softmax(logits_BHTS.astype(jnp.float32), axis=-1).astype(logits_BHTS.dtype)
        attn_out_BHTK = jnp.matmul(weights_BHTS, v_BHTK)
        attn_out_BTD = attn_out_BHTK.transpose(0, 2, 1, 3).reshape(B, T, -1)

        attn_out_BTD = attn_out_BTD * jax.nn.sigmoid(gate_BTD)
        return self.o_proj(attn_out_BTD)
