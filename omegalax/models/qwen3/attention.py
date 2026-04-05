import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P, reshard
from tokamax import dot_product_attention

from .norms import RMSNorm
from .rope import apply_rope, generate_pos_embeddings
from .utils import compute_positions_from_segment_ids, count_left_pads

wp = nnx.with_partitioning

def _mask_value(dtype: jnp.dtype) -> float:
    return float(jnp.finfo(dtype).min)


class Attention(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
        self.dtype = cfg.dtype
        init_fn = nnx.initializers.lecun_normal()
        qkv_init = wp(init_fn, ("embed", "heads"))
        o_init = wp(init_fn, ("heads", "embed"))
        self.q_proj = nnx.Linear(
            cfg.emb_dim,
            cfg.num_heads * cfg.head_dim,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=qkv_init,
        )
        self.k_proj = nnx.Linear(
            cfg.emb_dim,
            cfg.num_kv_heads * cfg.head_dim,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=qkv_init,
        )
        self.v_proj = nnx.Linear(
            cfg.emb_dim,
            cfg.num_kv_heads * cfg.head_dim,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=qkv_init,
        )
        self.o_proj = nnx.Linear(
            cfg.num_heads * cfg.head_dim,
            cfg.emb_dim,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=o_init,
        )

        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps, rngs=rngs, sharding=(None,))
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps, rngs=rngs, sharding=(None,))
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.scale = cfg.head_dim**-0.5
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        object.__setattr__(self, "_q_sharding", None)
        object.__setattr__(self, "_q_sharding_spec", P(*cfg.shd_cfg.act_btnh))

    @jax.named_scope("attention")
    def __call__(self, hidden_BTD: jax.Array, cache, segment_ids_BT: jax.Array) -> jax.Array:
        q_proj_BTF = self.q_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf)
        k_proj_BTF = self.k_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf)
        v_proj_BTF = self.v_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf)
        B, T = hidden_BTD.shape[:2]
        q_BTHK = self.q_norm(jax.lax.reshape(q_proj_BTF, (B, T, self.num_heads, self.head_dim), out_sharding=self.shd_cfg.act_btnh))
        k_BTGK = self.k_norm(jax.lax.reshape(k_proj_BTF, (B, T, self.num_kv_heads, self.head_dim), out_sharding=self.shd_cfg.act_btnh))
        v_BTGK = jax.lax.reshape(v_proj_BTF, (B, T, self.num_kv_heads, self.head_dim), out_sharding=self.shd_cfg.act_btnh)

        if cache is None:
            positions_BT = compute_positions_from_segment_ids(segment_ids_BT)
            sin_BTK, cos_BTK = generate_pos_embeddings(positions_BT, self.head_dim)
            sin_BTK = sin_BTK.astype(self.dtype)
            cos_BTK = cos_BTK.astype(self.dtype)

            q_BTHK = apply_rope(q_BTHK, sin_BTK, cos_BTK)
            k_BTGK = apply_rope(k_BTGK, sin_BTK, cos_BTK)

            B, T, H, K = q_BTHK.shape
            attn_BTHK = dot_product_attention(
                q_BTHK, k_BTGK, v_BTGK,
                is_causal=True, scale=self.scale, implementation="mosaic",
                q_sharding=self._q_sharding,
            )
            out_BTD = self.o_proj(jax.lax.reshape(attn_BTHK, (B, T, self.num_heads * K), out_sharding=self.shd_cfg.act_btf), out_sharding=self.shd_cfg.act_btd)
            return out_BTD

        left_pads_B = count_left_pads(segment_ids_BT)
        left_pads_B = reshard(left_pads_B, P(self.shd_cfg.act_btnh[0]))
        cache.start_ind.set_value(jnp.where(cache.start_ind[...] < 0, left_pads_B, cache.start_ind[...]))
        positions_BT = compute_positions_from_segment_ids(segment_ids_BT) + cache.cur_ind[...]
        sin_BTK, cos_BTK = generate_pos_embeddings(positions_BT, self.head_dim)
        sin_BTK = sin_BTK.astype(self.dtype)
        cos_BTK = cos_BTK.astype(self.dtype)
        q_BTHK = apply_rope(q_BTHK, sin_BTK, cos_BTK)
        k_BTGK = apply_rope(k_BTGK, sin_BTK, cos_BTK)

        slice_indices = (0, cache.cur_ind[...], 0, 0)
        cache.v_cache[...] = jax.lax.dynamic_update_slice(cache.v_cache[...], v_BTGK, slice_indices)
        cache.k_cache[...] = jax.lax.dynamic_update_slice(cache.k_cache[...], k_BTGK, slice_indices)

        B, T, H, K = q_BTHK.shape
        q_BTGRK = jax.lax.reshape(q_BTHK, (B, T, self.num_kv_heads, self.n_rep, K), out_sharding=P(self.shd_cfg.act_btnh[0], None, self.shd_cfg.act_btnh[2], None, None))
        logits_BTSGR: jax.Array = jnp.asarray(
            jnp.einsum("BTGRK,BSGK->BTSGR", q_BTGRK, cache.k_cache[...])
            * self.scale
        )

        q_pos_BT = cache.cur_ind[...] + jnp.arange(T, dtype=jnp.int32)[None, :] - cache.start_ind[:, None]
        ts = jnp.arange(cache.size, dtype=jnp.int32)
        kv_valid_BS = (ts[None, :] >= cache.start_ind[:, None]) & (ts[None, :] < cache.cur_ind[...] + T)
        k_pos_BS = ts[None, :] - cache.start_ind[:, None]
        causal_mask_BTS = k_pos_BS[:, None, :] <= q_pos_BT[:, :, None]
        segment_mask_BTS = kv_valid_BS[:, None, :] == segment_ids_BT[:, :, None]
        final_mask_BTS = causal_mask_BTS & segment_mask_BTS
        attn_mask = final_mask_BTS[:, :, :, None, None]
        logits_BTSGR = jnp.where(attn_mask, logits_BTSGR, _mask_value(logits_BTSGR.dtype))

        weights_BTSGR = jax.nn.softmax(logits_BTSGR.astype(jnp.float32), axis=2).astype(logits_BTSGR.dtype)
        attn_BTGRK = jnp.einsum("BTSGR,BSGK->BTGRK", weights_BTSGR, cache.v_cache[...])
        attn_BTHK = jax.lax.reshape(attn_BTGRK, (B, T, self.num_heads, K), out_sharding=self.shd_cfg.act_btnh)

        cache.cur_ind[...] = cache.cur_ind[...] + T
        out_BTD = self.o_proj(jax.lax.reshape(attn_BTHK, (B, T, self.num_heads * K), out_sharding=self.shd_cfg.act_btf), out_sharding=self.shd_cfg.act_btd)
        return out_BTD
