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
from jax.sharding import PartitionSpec, reshard

from .config import Qwen3_5TextConfig
from .norms import RMSNorm
from .rope import apply_text_rope

P = PartitionSpec


def _mask_value(dtype: jnp.dtype) -> float:
    return float(jnp.finfo(dtype).min)


class Attention(nnx.Module):
    def __init__(self, cfg: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        hd = cfg.head_dim
        nh = cfg.num_attention_heads
        nkv = cfg.num_key_value_heads

        self.shd_cfg = cfg.shd_cfg
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
        self.hidden_shd = cfg.shd_cfg.act_btd
        self.logits_shd = P(cfg.shd_cfg.act_btd[0], cfg.shd_cfg.act_btnh[2], None, None)

    @jax.named_scope("attention")
    def __call__(
        self,
        hidden_BTD: jax.Array,
        cos_BTK: jax.Array,
        sin_BTK: jax.Array,
        segment_ids_BT: jax.Array,
        position_ids_BT: jax.Array,
    ) -> jax.Array:
        hidden_BTD = reshard(hidden_BTD, self.hidden_shd)
        B, T, _ = hidden_BTD.shape

        q_out_BTHK2 = self.q_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf).reshape(
            B, T, self.num_heads, self.head_dim * 2
        )
        q_BTHK, gate_BTHK = jnp.split(q_out_BTHK2, 2, axis=-1)
        gate_BTD = gate_BTHK.reshape(B, T, -1)

        q_BHTK = reshard(self.q_norm(q_BTHK), self.shd_cfg.act_btnh).transpose(0, 2, 1, 3)
        k_BGTK = reshard(
            self.k_norm(
                self.k_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf).reshape(B, T, self.num_kv_heads, self.head_dim)
            ),
            self.shd_cfg.act_btnh,
        ).transpose(0, 2, 1, 3)
        v_BGTK = reshard(
            self.v_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf).reshape(B, T, self.num_kv_heads, self.head_dim),
            self.shd_cfg.act_btnh,
        ).transpose(0, 2, 1, 3)

        q_BHTK, k_BGTK = apply_text_rope(q_BHTK, k_BGTK, cos_BTK, sin_BTK)

        if self.n_rep > 1:
            k_BHTK = jnp.broadcast_to(
                k_BGTK[:, :, None, :, :],
                (B, self.num_kv_heads, self.n_rep, T, self.head_dim),
            ).reshape(B, self.num_heads, T, self.head_dim)
            v_BHTK = jnp.broadcast_to(
                v_BGTK[:, :, None, :, :],
                (B, self.num_kv_heads, self.n_rep, T, self.head_dim),
            ).reshape(B, self.num_heads, T, self.head_dim)
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
        logits_BHTS = reshard(logits_BHTS, self.logits_shd)

        weights_BHTS = jax.nn.softmax(logits_BHTS.astype(jnp.float32), axis=-1).astype(logits_BHTS.dtype)
        attn_out_BHTK = jnp.matmul(weights_BHTS, v_BHTK)
        attn_out_BTD = attn_out_BHTK.transpose(0, 2, 1, 3).reshape(B, T, -1)

        attn_out_BTD = attn_out_BTD * jax.nn.sigmoid(gate_BTD)
        out_BTD = self.o_proj(attn_out_BTD, out_sharding=self.shd_cfg.act_btd)
        return reshard(out_BTD, self.shd_cfg.act_btd)
