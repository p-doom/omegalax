import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P

from .norms import RMSNorm
from .rope import apply_rope, generate_pos_embeddings
from .utils import compute_positions_from_segment_ids, count_left_pads, shard

_K_MASK: float = float(jnp.finfo(jnp.float32).min)


class Attention(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
        self.q_proj = shard(
            nnx.Linear(
                cfg.emb_dim, cfg.num_heads * cfg.head_dim, use_bias=False, rngs=rngs, dtype=jnp.float32
            ),
            self.shd_cfg.q_weight_ndh,
        )
        self.k_proj = shard(
            nnx.Linear(
                cfg.emb_dim, cfg.num_kv_heads * cfg.head_dim, use_bias=False, rngs=rngs, dtype=jnp.float32
            ),
            self.shd_cfg.kv_weight_ndh,
        )
        self.v_proj = shard(
            nnx.Linear(
                cfg.emb_dim, cfg.num_kv_heads * cfg.head_dim, use_bias=False, rngs=rngs, dtype=jnp.float32
            ),
            self.shd_cfg.kv_weight_ndh,
        )
        self.o_proj = shard(
            nnx.Linear(
                cfg.num_heads * cfg.head_dim, cfg.emb_dim, use_bias=False, rngs=rngs, dtype=jnp.float32
            ),
            self.shd_cfg.o_weight_nhd,
        )

        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps, self.shd_cfg.rms_norm, rngs=rngs)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps, self.shd_cfg.rms_norm, rngs=rngs)
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.scale = cfg.head_dim**-0.5
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads

    @jax.named_scope("attention")
    def __call__(self, x: jax.Array, cache, segment_ids: jax.Array) -> jax.Array:
        query_proj = shard(self.q_norm(self.q_proj(x).reshape((*x.shape[:2], self.num_heads, self.head_dim))), self.shd_cfg.act_btnh)
        key_proj = shard(self.k_norm(self.k_proj(x).reshape((*x.shape[:2], self.num_kv_heads, self.head_dim))), self.shd_cfg.act_btnh)
        value_proj = shard(self.v_proj(x).reshape((*x.shape[:2], self.num_kv_heads, self.head_dim)), self.shd_cfg.act_btnh)

        if cache is None:
            position_ids = compute_positions_from_segment_ids(segment_ids)
            sin, cos = generate_pos_embeddings(position_ids, self.head_dim)
            query_proj = apply_rope(query_proj, sin, cos)
            key_proj = apply_rope(key_proj, sin, cos)

            b, t, n, h = query_proj.shape
            query_proj_gqa = query_proj.reshape((b, t, self.num_kv_heads, self.n_rep, h))
            prefill_attn_logits: jax.Array = jnp.asarray(
                jnp.einsum(
                    "BTKGH,BSKH->BTSKG", query_proj_gqa, key_proj, precision=jax.lax.Precision.HIGHEST
                )
                * self.scale
            )

            q_pos = position_ids
            k_pos = position_ids
            causal_mask = k_pos[:, None, :] <= q_pos[:, :, None]
            segment_mask = segment_ids[:, None, :] == segment_ids[:, :, None]
            final_mask = causal_mask & segment_mask
            attn_mask = final_mask[:, :, :, None, None]
            prefill_attn_logits = jnp.where(
                attn_mask, prefill_attn_logits, jnp.full_like(prefill_attn_logits, _K_MASK)
            )

            attn_weights = jax.nn.softmax(prefill_attn_logits.astype(jnp.float32), axis=2).astype(
                prefill_attn_logits.dtype
            )
            qkv = jnp.einsum(
                "BTSKG,BSKH->BTKGH", attn_weights, value_proj, precision=jax.lax.Precision.HIGHEST
            )
            qkv = qkv.reshape((b, t, self.num_heads, h))
            return shard(self.o_proj(qkv.reshape(b, t, self.num_heads * h)), self.shd_cfg.act_btd)

        left_pads = count_left_pads(segment_ids)
        left_pads = shard(left_pads, P(self.shd_cfg.act_btnh[0]))
        cache.start_ind.set_value(jnp.where(cache.start_ind[...] < 0, left_pads, cache.start_ind[...]))
        position_ids = compute_positions_from_segment_ids(segment_ids) + cache.cur_ind[...]
        sin, cos = generate_pos_embeddings(position_ids, self.head_dim)
        query_proj = apply_rope(query_proj, sin, cos)
        key_proj = apply_rope(key_proj, sin, cos)

        slice_indices = (0, cache.cur_ind[...], 0, 0)
        cache.v_cache[...] = jax.lax.dynamic_update_slice(cache.v_cache[...], value_proj, slice_indices)
        cache.k_cache[...] = jax.lax.dynamic_update_slice(cache.k_cache[...], key_proj, slice_indices)

        b, t, n, h = query_proj.shape
        query_proj_gqa = query_proj.reshape((b, t, self.num_kv_heads, self.n_rep, h))
        decode_attn_logits: jax.Array = jnp.asarray(
            jnp.einsum(
                "BTKGH,BSKH->BTSKG", query_proj_gqa, cache.k_cache[...], precision=jax.lax.Precision.HIGHEST
            )
            * self.scale
        )

        q_pos = cache.cur_ind[...] + jnp.arange(t, dtype=jnp.int32)[None, :] - cache.start_ind[:, None]
        ts = jnp.arange(cache.size, dtype=jnp.int32)
        kv_segment_ids = (ts[None, :] >= cache.start_ind[:, None]) & (ts[None, :] < cache.cur_ind[...] + t)
        k_pos = ts[None, :] - cache.start_ind[:, None]
        causal_mask = k_pos[:, None, :] <= q_pos[:, :, None]
        segment_mask = kv_segment_ids[:, None, :] == segment_ids[:, :, None]
        final_mask = causal_mask & segment_mask
        attn_mask = final_mask[:, :, :, None, None]
        decode_attn_logits = jnp.where(attn_mask, decode_attn_logits, jnp.full_like(decode_attn_logits, _K_MASK))

        attn_weights = jax.nn.softmax(decode_attn_logits.astype(jnp.float32), axis=2).astype(decode_attn_logits.dtype)
        qkv = jnp.einsum(
            "BTSKG,BSKH->BTKGH", attn_weights, cache.v_cache[...], precision=jax.lax.Precision.HIGHEST
        )
        qkv = qkv.reshape((b, t, self.num_heads, h))

        cache.cur_ind[...] = cache.cur_ind[...] + t
        return shard(self.o_proj(qkv.reshape(b, t, self.num_heads * h)), self.shd_cfg.act_btd)
