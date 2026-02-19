import dataclasses
import math
from functools import partial
from typing import TypeAlias, TypeVar, cast

import jax
from flax import nnx
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec, get_abstract_mesh, reshard
from jaxtyping import Array, ArrayLike

_K_MASK: float = float(np.finfo(np.float32).min)
P = PartitionSpec
ShardingSpec = PartitionSpec
T = TypeVar("T")


@dataclasses.dataclass(slots=True, frozen=True)
class ShardConfig:
    emb_vd: ShardingSpec
    emb_dv: ShardingSpec
    q_weight_ndh: ShardingSpec
    kv_weight_ndh: ShardingSpec
    o_weight_nhd: ShardingSpec
    ffw_weight_df: ShardingSpec
    ffw_weight_fd: ShardingSpec
    rms_norm: ShardingSpec
    act_btd: ShardingSpec
    act_btf: ShardingSpec
    act_btnh: ShardingSpec

    @staticmethod
    def no_sharding():
        """Configuration with no sharding (all None)."""
        return ShardConfig(
            emb_vd=P(None, None),
            emb_dv=P(None, None),
            q_weight_ndh=P(None, None, None),
            kv_weight_ndh=P(None, None, None),
            o_weight_nhd=P(None, None, None),
            ffw_weight_df=P(None, None),
            ffw_weight_fd=P(None, None),
            rms_norm=P(None),
            act_btd=P(None, None, None),
            act_btf=P(None, None, None),
            act_btnh=P(None, None, None, None),
        )

    @staticmethod
    def default():
        return ShardConfig(
            emb_vd=P("tp", "fsdp"),
            emb_dv=P("fsdp", "tp"),
            q_weight_ndh=P("tp", "fsdp", None),
            kv_weight_ndh=P("tp", "fsdp", None),
            o_weight_nhd=P("tp", None, "fsdp"),
            ffw_weight_df=P("fsdp", "tp"),
            ffw_weight_fd=P("tp", "fsdp"),
            rms_norm=P("tp"),
            act_btd=P("fsdp", None, "tp"),
            act_btf=P("fsdp", None, "tp"),
            act_btnh=P("fsdp", None, "tp", None),
        )


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    vocab_size: int
    emb_dim: int
    mlp_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    rope_theta: int
    rope_scaling_factor: float
    local_rope_theta: float
    norm_eps: float
    tie_word_embeddings: bool
    shd_cfg: ShardConfig = dataclasses.field(default_factory=ShardConfig.no_sharding)

    @classmethod
    def _from_param(cls, use_sharding: bool, **kwargs):
        if use_sharding:
            kwargs["shd_cfg"] = ShardConfig.default()
        return cls(**kwargs)

    @classmethod
    def smoke(cls, use_sharding: bool = False):
        return cls._from_param(
            use_sharding,
            num_layers=2,
            vocab_size=1024,
            emb_dim=128,
            mlp_dim=512,
            num_heads=4,
            head_dim=32,
            num_kv_heads=4,
            norm_eps=1e-06,
            rope_theta=1_000_000,
            rope_scaling_factor=8.0,
            local_rope_theta=1e4,
            tie_word_embeddings=False,
        )

    @classmethod
    def qwen3_0_6b(cls, use_sharding: bool = False):
        return cls._from_param(
            use_sharding,
            num_layers=28,
            vocab_size=151936,
            emb_dim=1024,
            mlp_dim=3072,
            num_heads=16,
            head_dim=128,
            num_kv_heads=8,
            norm_eps=1e-06,
            rope_theta=1_000_000,
            rope_scaling_factor=8.0,
            local_rope_theta=1e4,
            tie_word_embeddings=True,
        )


def shard(x: T, s: ShardingSpec) -> T:
    mesh = get_abstract_mesh()
    if not mesh.empty and len(mesh.axis_names) > 0:
        return cast(T, reshard(x, s))
    return x


class LayerCache(nnx.Module):
    def __init__(self, cfg: ModelConfig, batch_size: int, cache_size: int, dtype: jnp.dtype):
        cache_shape = (batch_size, cache_size, cfg.num_kv_heads, cfg.head_dim)
        self.k_cache = shard(nnx.Cache(jnp.zeros(cache_shape, dtype=dtype)), cfg.shd_cfg.act_btnh)
        self.v_cache = shard(nnx.Cache(jnp.zeros(cache_shape, dtype=dtype)), cfg.shd_cfg.act_btnh)
        self.size = self.k_cache.shape[1]
        batch_sharding = P(cfg.shd_cfg.act_btnh[0]) if cfg.shd_cfg.act_btnh else P(None)
        self.start_ind = shard(nnx.Variable(-1 * jnp.ones((batch_size,), dtype=jnp.int32)), batch_sharding)
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))


Cache: TypeAlias = list[LayerCache]


class Einsum(nnx.Module):
    def __init__(self, einsum_str: str, shape: tuple[int, ...], *, shd: ShardingSpec, rngs: nnx.Rngs):
        self.einsum_str = einsum_str
        self.shape = shape
        self.w = shard(nnx.Param(nnx.initializers.normal()(rngs.params(), shape)), shd)

    @jax.named_scope("einsum")
    def __call__(self, x: ArrayLike) -> Array:
        return jnp.einsum(self.einsum_str, x, self.w[...])


def _generate_pos_embeddings(
    positions: jax.Array, head_dim: int, rope_theta: int = 1_000_000
) -> tuple[jax.Array, jax.Array]:
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction
    rotational_frequency = 1.0 / timescale
    sinusoid_inp = jnp.einsum("BT,k->BTk", positions, rotational_frequency, precision=jax.lax.Precision.HIGHEST)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def apply_rope(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.scale = shard(nnx.Param(nnx.initializers.ones_init()(rngs.params(), (dim,))), cfg.shd_cfg.rms_norm)
        self.norm_eps = cfg.norm_eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: Array) -> Array:
        dtype = x.dtype
        rms = jnp.sqrt(jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True) + self.norm_eps)
        return jnp.astype(self.scale[...] * x / rms, dtype)


def count_left_pads(x: jax.Array) -> jax.Array:
    return jnp.sum(jnp.cumsum(x != 0, axis=-1) == 0, -1)


def count_right_pads(x: jax.Array, pad_id: int) -> jax.Array:
    result = jnp.where(
        jnp.all(x == pad_id, axis=1), x.shape[1], jnp.argmin(jnp.flip(x == pad_id, axis=1).astype(jnp.int32), axis=1)
    )
    return jnp.max(result)


def compute_positions_from_segment_ids(seg_ids: jax.Array) -> jax.Array:
    token_positions = jnp.arange(seg_ids.shape[1], dtype=jnp.int32)[None, :]
    row_offsets = jnp.argmax(seg_ids, axis=1, keepdims=True)
    relative_positions = token_positions - row_offsets
    default_positions = jnp.full_like(relative_positions, jnp.int32(2**30))
    return jax.lax.select(seg_ids != 0, relative_positions, default_positions)


class Attention(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
        einsum_fn = partial(Einsum, rngs=rngs)
        self.q_proj = einsum_fn(
            "BTD,DNH->BTNH", (cfg.emb_dim, cfg.num_heads, cfg.head_dim), shd=self.shd_cfg.q_weight_ndh
        )
        self.k_proj = einsum_fn(
            "BSD,DKH->BSKH", (cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim), shd=self.shd_cfg.kv_weight_ndh
        )
        self.v_proj = einsum_fn(
            "BSD,DKH->BSKH", (cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim), shd=self.shd_cfg.kv_weight_ndh
        )
        self.o_proj = einsum_fn(
            "BTNH,NHD->BTD", (cfg.num_heads, cfg.head_dim, cfg.emb_dim), shd=self.shd_cfg.o_weight_nhd
        )

        self.q_norm = RMSNorm(cfg.head_dim, cfg, rngs=rngs)
        self.k_norm = RMSNorm(cfg.head_dim, cfg, rngs=rngs)
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.scale = cfg.head_dim**-0.5

    @jax.named_scope("attention")
    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array) -> Array:
        query_proj = shard(self.q_norm(self.q_proj(x)), self.shd_cfg.act_btnh)
        key_proj = shard(self.k_norm(self.k_proj(x)), self.shd_cfg.act_btnh)
        value_proj = shard(self.v_proj(x), self.shd_cfg.act_btnh)

        if cache is None:
            position_ids = compute_positions_from_segment_ids(segment_ids)
            sin, cos = _generate_pos_embeddings(position_ids, self.head_dim)
            query_proj = apply_rope(query_proj, sin, cos)
            key_proj = apply_rope(key_proj, sin, cos)

            b, t, n, h = query_proj.shape
            query_proj_gqa = query_proj.reshape((b, t, self.num_kv_heads, self.n_rep, h))
            prefill_attn_logits: jax.Array = jnp.asarray(
                jnp.einsum("BTKGH,BSKH->BTSKG", query_proj_gqa, key_proj) * self.scale
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
            qkv = jnp.einsum("BTSKG,BSKH->BTKGH", attn_weights, value_proj)
            qkv = qkv.reshape((b, t, n, h))
            return shard(self.o_proj(qkv), self.shd_cfg.act_btd)

        left_pads = count_left_pads(segment_ids)
        left_pads = shard(left_pads, P(self.shd_cfg.act_btnh[0]))
        cache.start_ind.set_value(jnp.where(cache.start_ind[...] < 0, left_pads, cache.start_ind[...]))
        position_ids = compute_positions_from_segment_ids(segment_ids) + cache.cur_ind[...]
        sin, cos = _generate_pos_embeddings(position_ids, self.head_dim)
        query_proj = apply_rope(query_proj, sin, cos)
        key_proj = apply_rope(key_proj, sin, cos)

        slice_indices = (0, cache.cur_ind[...], 0, 0)
        cache.v_cache[...] = jax.lax.dynamic_update_slice(cache.v_cache[...], value_proj, slice_indices)
        cache.k_cache[...] = jax.lax.dynamic_update_slice(cache.k_cache[...], key_proj, slice_indices)

        b, t, n, h = query_proj.shape
        query_proj_gqa = query_proj.reshape((b, t, self.num_kv_heads, self.n_rep, h))
        decode_attn_logits: jax.Array = jnp.asarray(
            jnp.einsum("BTKGH,BSKH->BTSKG", query_proj_gqa, cache.k_cache[...]) * self.scale
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
        qkv = jnp.einsum("BTSKG,BSKH->BTKGH", attn_weights, cache.v_cache[...])
        qkv = qkv.reshape((b, t, n, h))

        cache.cur_ind[...] = cache.cur_ind[...] + t
        return shard(self.o_proj(qkv), self.shd_cfg.act_btd)

    @property
    def head_dim(self):
        return self.o_proj.shape[1]

    @property
    def num_heads(self):
        return self.q_proj.shape[1]

    @property
    def num_kv_heads(self):
        return self.k_proj.shape[1]


class MLP(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
        linear = partial(nnx.Linear, use_bias=False, rngs=rngs)
        self.gate_proj = shard(linear(cfg.emb_dim, cfg.mlp_dim), self.shd_cfg.ffw_weight_df)
        self.up_proj = shard(linear(cfg.emb_dim, cfg.mlp_dim), self.shd_cfg.ffw_weight_df)
        self.down_proj = shard(linear(cfg.mlp_dim, cfg.emb_dim), self.shd_cfg.ffw_weight_fd)

    @jax.named_scope("feed_forward")
    def __call__(self, x: Array) -> Array:
        activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
        activations = shard(activations, self.shd_cfg.act_btf)
        return self.down_proj(activations)


class DecoderLayer(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.input_layernorm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
        self.attn = Attention(cfg=cfg, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
        self.mlp = MLP(cfg=cfg, rngs=rngs)

    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array) -> Array:
        inputs_normalized = self.input_layernorm(x)
        attn_output = x + self.attn(inputs_normalized, cache, segment_ids)
        return attn_output + self.mlp(self.post_attention_layernorm(attn_output))


class Qwen3(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.embedder = shard(
            nnx.Embed(num_embeddings=cfg.vocab_size, features=cfg.emb_dim, dtype=jnp.bfloat16, rngs=rngs),
            cfg.shd_cfg.emb_vd,
        )
        self.out_emb_shd = None if get_abstract_mesh().empty else cfg.shd_cfg.act_btd
        self.layers = nnx.List([DecoderLayer(cfg=cfg, rngs=rngs) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
        self.lm_head = Einsum(
            einsum_str="BTD,DV->BTV", shape=(cfg.emb_dim, cfg.vocab_size), shd=cfg.shd_cfg.emb_dv, rngs=rngs
        )

    def init_cache(
        self, cfg: ModelConfig, batch_size: int, token_len: int, generate_steps: int, dtype: jnp.dtype = jnp.bfloat16
    ) -> Cache:
        cache_size = 2 ** math.ceil(math.log2(max(token_len + generate_steps, 1)))
        return [LayerCache(cfg, batch_size, cache_size, dtype) for _ in range(cfg.num_layers)]

    def __call__(self, tokens, segment_ids, cache: Cache | None, num_right_pads):
        del num_right_pads
        x = self.embedder.embedding[...].at[(tokens,)].get(out_sharding=self.out_emb_shd)
        for i, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[i]
            x = layer(x, layer_cache, segment_ids)
        return self.lm_head(self.final_norm(x))


@jax.jit
def decode(model: nnx.Module, cache: Cache, tokens: Array, pad_id: int) -> tuple[Array, Cache]:
    segment_ids = 1 * (tokens != pad_id)
    num_right_pads = count_right_pads(tokens, pad_id)
    logits = model(tokens, segment_ids, cache, num_right_pads)
    target_ind = tokens.shape[-1] - num_right_pads - 1
    return logits[:, target_ind], cache


@jax.jit
def forward(model: nnx.Module, tokens: Array, pad_id: int) -> Array:
    segment_ids = 1 * (tokens != pad_id)
    return model(tokens, segment_ids, None, jnp.array(0, dtype=jnp.int32))
