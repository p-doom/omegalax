import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

from ..attention import Attention
from ..config import Qwen3Config
from ..norms import RMSNorm
from ..utils import shard


class MLP(nnx.Module):
    def __init__(self, cfg: Qwen3Config, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
        linear = partial(nnx.Linear, use_bias=False, rngs=rngs, dtype=jnp.float32)
        self.gate_proj = shard(linear(cfg.emb_dim, cfg.mlp_dim), self.shd_cfg.ffw_weight_df)
        self.up_proj = shard(linear(cfg.emb_dim, cfg.mlp_dim), self.shd_cfg.ffw_weight_df)
        self.down_proj = shard(linear(cfg.mlp_dim, cfg.emb_dim), self.shd_cfg.ffw_weight_fd)

    @jax.named_scope("feed_forward")
    def __call__(self, x: jax.Array) -> jax.Array:
        activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
        activations = shard(activations, self.shd_cfg.act_btf)
        return self.down_proj(activations)


class DecoderLayer(nnx.Module):
    def __init__(self, cfg: Qwen3Config, *, rngs: nnx.Rngs):
        self.input_layernorm = RMSNorm(cfg.emb_dim, cfg.norm_eps, cfg.shd_cfg.rms_norm, rngs=rngs)
        self.attn = Attention(cfg=cfg, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim, cfg.norm_eps, cfg.shd_cfg.rms_norm, rngs=rngs)
        self.mlp = MLP(cfg=cfg, rngs=rngs)

    def __call__(self, x: jax.Array, cache, segment_ids: jax.Array) -> jax.Array:
        inputs_normalized = self.input_layernorm(x)
        attn_output = x + self.attn(inputs_normalized, cache, segment_ids)
        return attn_output + self.mlp(self.post_attention_layernorm(attn_output))


class Qwen3Dense(nnx.Module):
    def __init__(self, cfg: Qwen3Config, *, rngs: nnx.Rngs):
        self.embedder = shard(
            nnx.Embed(num_embeddings=cfg.vocab_size, features=cfg.emb_dim, dtype=jnp.float32, rngs=rngs),
            cfg.shd_cfg.emb_vd,
        )
        self.out_emb_shd = None
        self.layers = nnx.List([DecoderLayer(cfg=cfg, rngs=rngs) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.emb_dim, cfg.norm_eps, cfg.shd_cfg.rms_norm, rngs=rngs)
        self.lm_head = shard(
            nnx.Linear(cfg.emb_dim, cfg.vocab_size, use_bias=False, rngs=rngs), cfg.shd_cfg.emb_dv
        )

    def __call__(self, tokens, segment_ids, cache, num_right_pads):
        del num_right_pads
        x = self.embedder.embedding[...].at[(tokens,)].get(out_sharding=self.out_emb_shd)
        for i, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[i]
            x = layer(x, layer_cache, segment_ids)
        logits = self.lm_head(self.final_norm(x))
        return logits
