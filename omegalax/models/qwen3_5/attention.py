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
from tokamax import dot_product_attention

from .config import Qwen3_5TextConfig
from .norms import RMSNorm
from .rope import apply_text_rope

P = PartitionSpec
wp = nnx.with_partitioning


class Attention(nnx.Module):
    def __init__(self, cfg: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        hd = cfg.head_dim
        nh = cfg.num_attention_heads
        nkv = cfg.num_key_value_heads

        self.shd_cfg = cfg.shd_cfg
        init_fn = nnx.initializers.lecun_normal()
        qkv_init = wp(init_fn, ("embed", "heads"))
        o_init = wp(init_fn, ("heads", "embed"))
        self.q_proj = nnx.Linear(
            cfg.hidden_size,
            nh * hd * 2,
            use_bias=cfg.attention_bias,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=qkv_init,
        )
        self.k_proj = nnx.Linear(
            cfg.hidden_size,
            nkv * hd,
            use_bias=cfg.attention_bias,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=qkv_init,
        )
        self.v_proj = nnx.Linear(
            cfg.hidden_size,
            nkv * hd,
            use_bias=cfg.attention_bias,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=qkv_init,
        )
        self.o_proj = nnx.Linear(
            nh * hd,
            cfg.hidden_size,
            use_bias=cfg.attention_bias,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=o_init,
        )

        self.q_norm = RMSNorm(hd, cfg.rms_norm_eps, rngs=rngs, sharding=(None,))
        self.k_norm = RMSNorm(hd, cfg.rms_norm_eps, rngs=rngs, sharding=(None,))

        self.num_heads = nh
        self.num_kv_heads = nkv
        self.head_dim = hd
        self.n_rep = nh // nkv
        self.scale = hd ** -0.5
        self.hidden_shd = cfg.shd_cfg.act_btd

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

        q_BTHK = reshard(self.q_norm(q_BTHK), self.shd_cfg.act_btnh)
        k_BTGK = reshard(
            self.k_norm(
                self.k_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf).reshape(B, T, self.num_kv_heads, self.head_dim)
            ),
            self.shd_cfg.act_btnh,
        )
        v_BTGK = reshard(
            self.v_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf).reshape(B, T, self.num_kv_heads, self.head_dim),
            self.shd_cfg.act_btnh,
        )

        q_BTHK, k_BTGK = apply_text_rope(q_BTHK, k_BTGK, cos_BTK, sin_BTK)

        attn_BTHK = dot_product_attention(
            q_BTHK, k_BTGK, v_BTGK,
            is_causal=True, scale=self.scale, implementation="mosaic",
        )
        attn_out_BTD = attn_BTHK.reshape(B, T, -1)

        attn_out_BTD = attn_out_BTD * jax.nn.sigmoid(gate_BTD)
        out_BTD = self.o_proj(attn_out_BTD, out_sharding=self.shd_cfg.act_btd)
        return reshard(out_BTD, self.shd_cfg.act_btd)
