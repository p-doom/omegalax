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
from jax.sharding import PartitionSpec
from tokamax import dot_product_attention

from .config import Qwen3_5TextConfig
from .norms import RMSNorm
from .rope import apply_text_rope

P = PartitionSpec
wp = nnx.with_partitioning


def _mask_value(dtype: jnp.dtype) -> float:
    return float(jnp.finfo(dtype).min)


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
        self.scale = hd**-0.5
        self.hidden_shd = cfg.shd_cfg.act_btd
        object.__setattr__(self, "_q_sharding", None)
        object.__setattr__(self, "_q_sharding_spec", P(*cfg.shd_cfg.act_btnh))
        object.__setattr__(self, "_attn_backend", "mosaic_gpu")
        object.__setattr__(self, "_attn_kind", "text")

    @jax.named_scope("attention")
    def __call__(
        self,
        hidden_BTD: jax.Array,
        cos_BTK: jax.Array,
        sin_BTK: jax.Array,
        segment_ids_BT: jax.Array,
        position_ids_BT: jax.Array,
        *,
        cache=None,
    ) -> jax.Array:
        B, T, _ = hidden_BTD.shape

        heads_shd = self.shd_cfg.act_btnh
        q_out_BTHK2 = jax.lax.reshape(
            self.q_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf),
            (B, T, self.num_heads, self.head_dim * 2),
            out_sharding=P(heads_shd[0], heads_shd[1], heads_shd[2], None),
        )
        q_BTHK, gate_BTHK = jnp.split(q_out_BTHK2, 2, axis=-1)
        gate_BTD = jax.lax.reshape(
            gate_BTHK, (B, T, self.num_heads * self.head_dim), out_sharding=self.shd_cfg.act_btf
        )

        q_BTHK = self.q_norm(q_BTHK)
        k_BTGK = self.k_norm(
            jax.lax.reshape(
                self.k_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf),
                (B, T, self.num_kv_heads, self.head_dim),
                out_sharding=heads_shd,
            )
        )
        v_BTGK = jax.lax.reshape(
            self.v_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf),
            (B, T, self.num_kv_heads, self.head_dim),
            out_sharding=heads_shd,
        )

        q_BTHK, k_BTGK = apply_text_rope(q_BTHK, k_BTGK, cos_BTK, sin_BTK)

        if cache is None:
            attn_BTHK = dot_product_attention(
                q_BTHK,
                k_BTGK,
                v_BTGK,
                is_causal=True,
                scale=self.scale,
                implementation=self._attn_backend,
                q_sharding=self._q_sharding,
            )
        else:
            slice_indices = (0, cache.cur_ind[...], 0, 0)
            cache.k_cache[...] = jax.lax.dynamic_update_slice(
                cache.k_cache[...], k_BTGK, slice_indices
            )
            cache.v_cache[...] = jax.lax.dynamic_update_slice(
                cache.v_cache[...], v_BTGK, slice_indices
            )

            q_BTGRK = jax.lax.reshape(
                q_BTHK,
                (B, T, self.num_kv_heads, self.n_rep, self.head_dim),
            )
            logits_BTSGR = jnp.einsum("BTGRK,BSGK->BTSGR", q_BTGRK, cache.k_cache[...]) * self.scale
            query_positions_T = cache.cur_ind[...] + jnp.arange(T, dtype=jnp.int32)
            key_positions_S = jnp.arange(cache.size, dtype=jnp.int32)
            valid_S = key_positions_S < cache.cur_ind[...] + T
            causal_TS = key_positions_S[None, :] <= query_positions_T[:, None]
            mask_TS = valid_S[None, :] & causal_TS
            logits_BTSGR = jnp.where(
                mask_TS[None, :, :, None, None],
                logits_BTSGR,
                _mask_value(logits_BTSGR.dtype),
            )
            weights_BTSGR = jax.nn.softmax(logits_BTSGR.astype(jnp.float32), axis=2).astype(
                logits_BTSGR.dtype
            )
            attn_BTGRK = jnp.einsum("BTSGR,BSGK->BTGRK", weights_BTSGR, cache.v_cache[...])
            attn_BTHK = jax.lax.reshape(
                attn_BTGRK,
                (B, T, self.num_heads, self.head_dim),
                out_sharding=self.shd_cfg.act_btnh,
            )
            cache.cur_ind[...] = cache.cur_ind[...] + T
        attn_out_BTD = jax.lax.reshape(
            attn_BTHK, (B, T, self.num_heads * self.head_dim), out_sharding=self.shd_cfg.act_btf
        )

        attn_out_BTD = attn_out_BTD * jax.nn.sigmoid(gate_BTD)
        out_BTD = self.o_proj(attn_out_BTD, out_sharding=self.shd_cfg.act_btd)
        return out_BTD
