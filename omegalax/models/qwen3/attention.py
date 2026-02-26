import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P, reshard

from .norms import RMSNorm
from .rope import apply_rope, generate_pos_embeddings
from .utils import compute_positions_from_segment_ids, count_left_pads

def _mask_value(dtype: jnp.dtype) -> float:
    return float(jnp.finfo(dtype).min)


class Attention(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
        self.dtype = cfg.dtype
        self.q_proj = nnx.Linear(
            cfg.emb_dim, cfg.num_heads * cfg.head_dim, use_bias=False, rngs=rngs, dtype=cfg.dtype
        )
        self.k_proj = nnx.Linear(
            cfg.emb_dim, cfg.num_kv_heads * cfg.head_dim, use_bias=False, rngs=rngs, dtype=cfg.dtype
        )
        self.v_proj = nnx.Linear(
            cfg.emb_dim, cfg.num_kv_heads * cfg.head_dim, use_bias=False, rngs=rngs, dtype=cfg.dtype
        )
        self.o_proj = nnx.Linear(
            cfg.num_heads * cfg.head_dim, cfg.emb_dim, use_bias=False, rngs=rngs, dtype=cfg.dtype
        )

        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps, self.shd_cfg.rms_norm, rngs=rngs)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps, self.shd_cfg.rms_norm, rngs=rngs)
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.scale = cfg.head_dim**-0.5
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads

    @jax.named_scope("attention")
    def __call__(self, hidden_BTD: jax.Array, cache, segment_ids_BT: jax.Array) -> jax.Array:
        q_proj_BTF = self.q_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf)
        k_proj_BTF = self.k_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf)
        v_proj_BTF = self.v_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf)
        q_BTHK = reshard(self.q_norm(q_proj_BTF.reshape((*hidden_BTD.shape[:2], self.num_heads, self.head_dim))), self.shd_cfg.act_btnh)
        k_BTGK = reshard(self.k_norm(k_proj_BTF.reshape((*hidden_BTD.shape[:2], self.num_kv_heads, self.head_dim))), self.shd_cfg.act_btnh)
        v_BTGK = reshard(v_proj_BTF.reshape((*hidden_BTD.shape[:2], self.num_kv_heads, self.head_dim)), self.shd_cfg.act_btnh)

        if cache is None:
            positions_BT = compute_positions_from_segment_ids(segment_ids_BT)
            sin_BTK, cos_BTK = generate_pos_embeddings(positions_BT, self.head_dim)
            sin_BTK = sin_BTK.astype(self.dtype)
            cos_BTK = cos_BTK.astype(self.dtype)

            q_BTHK = apply_rope(q_BTHK, sin_BTK, cos_BTK)
            k_BTGK = apply_rope(k_BTGK, sin_BTK, cos_BTK)

            B, T, H, K = q_BTHK.shape
            q_BTGRK = q_BTHK.reshape((B, T, self.num_kv_heads, self.n_rep, K))
            logits_BTSGR: jax.Array = jnp.asarray(
                jnp.einsum("BTGRK,BSGK->BTSGR", q_BTGRK, k_BTGK)
                * self.scale
            )

            q_pos_BT = positions_BT
            k_pos_BT = positions_BT
            causal_mask_BTS = k_pos_BT[:, None, :] <= q_pos_BT[:, :, None]
            segment_mask_BTS = segment_ids_BT[:, None, :] == segment_ids_BT[:, :, None]
            final_mask_BTS = causal_mask_BTS & segment_mask_BTS
            attn_mask = final_mask_BTS[:, :, :, None, None]
            logits_BTSGR = jnp.where(
                attn_mask, logits_BTSGR, _mask_value(logits_BTSGR.dtype)
            )

            weights_BTSGR = jax.nn.softmax(logits_BTSGR.astype(jnp.float32), axis=2).astype(
                logits_BTSGR.dtype
            )
            attn_BTGRK = jnp.einsum("BTSGR,BSGK->BTGRK", weights_BTSGR, v_BTGK)
            attn_BTHK = attn_BTGRK.reshape((B, T, self.num_heads, K))
            out_BTD = self.o_proj(attn_BTHK.reshape(B, T, self.num_heads * K), out_sharding=self.shd_cfg.act_btd)
            return reshard(out_BTD, self.shd_cfg.act_btd)

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
        q_BTGRK = q_BTHK.reshape((B, T, self.num_kv_heads, self.n_rep, K))
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
        attn_BTHK = attn_BTGRK.reshape((B, T, self.num_heads, K))

        cache.cur_ind[...] = cache.cur_ind[...] + T
        out_BTD = self.o_proj(attn_BTHK.reshape(B, T, self.num_heads * K), out_sharding=self.shd_cfg.act_btd)
        return reshard(out_BTD, self.shd_cfg.act_btd)
