import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

from ..attention import Attention
from ..dense.model import MLP as DenseMLP
from ..norms import RMSNorm
from ..utils import shard
from .config import Qwen3MoeConfig


class MoEFeedForward(nnx.Module):
    """Sparse MoE block matching the HuggingFace Qwen3MoeSparseMoeBlock architecture.

    Expert weights are stored as batched 3-D parameters [num_experts, ...] and
    the forward computes all experts in parallel, then selects top-k per token.
    Crucially, each expert applies its own down_proj *before* the weighted merge,
    matching the official per-expert output projection semantics.
    """

    def __init__(self, cfg: Qwen3MoeConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.shd_cfg = cfg.shd_cfg
        E, D, F = cfg.num_experts, cfg.emb_dim, cfg.moe_intermediate_size

        init = nnx.initializers.lecun_normal()
        self.gate_proj = nnx.Param(init(rngs.params(), (E, D, F)))
        self.up_proj = nnx.Param(init(rngs.params(), (E, D, F)))
        self.down_proj = nnx.Param(init(rngs.params(), (E, F, D)))
        self.router = nnx.Linear(D, E, use_bias=False, rngs=rngs, dtype=jnp.float32)

    @jax.named_scope("moe_feed_forward")
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        cfg = self.cfg

        router_logits = self.router(x)
        probs = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1)
        topk_weights, topk_idx = jax.lax.top_k(probs, cfg.num_experts_per_tok)
        if cfg.norm_topk_prob:
            topk_weights = topk_weights / jnp.clip(
                jnp.sum(topk_weights, axis=-1, keepdims=True), min=1e-9
            )
        topk_weights = topk_weights.astype(probs.dtype)

        # Per-expert projections (batched over all experts)
        gate_proj = self.gate_proj[...]
        up_proj = self.up_proj[...]
        down_proj = self.down_proj[...]

        gate = jnp.einsum("btd,edf->btef", x, gate_proj)
        up = jnp.einsum("btd,edf->btef", x, up_proj)
        expert_hidden = nnx.silu(gate) * up
        expert_out = jnp.einsum("btef,efd->bted", expert_hidden, down_proj)

        # Select top-k experts per token and weighted-merge
        b, t = x.shape[:2]
        flat_out = expert_out.reshape(b * t, cfg.num_experts, cfg.emb_dim)
        flat_idx = topk_idx.reshape(b * t, cfg.num_experts_per_tok)
        gathered = jnp.take_along_axis(flat_out, flat_idx[..., None], axis=1)
        gathered = gathered.reshape(b, t, cfg.num_experts_per_tok, cfg.emb_dim)
        merged = jnp.sum(gathered * topk_weights[..., None], axis=-2)

        # HF-style load-balancing loss (Switch Transformer eqs. 4-6)
        expert_mask = jax.nn.one_hot(topk_idx, cfg.num_experts, dtype=probs.dtype)
        tokens_per_expert = jnp.mean(expert_mask, axis=(0, 1))  # (top_k, E)
        router_prob_per_expert = jnp.mean(probs, axis=(0, 1))  # (E,)
        aux_loss_raw = jnp.sum(tokens_per_expert * router_prob_per_expert[None, :]) * cfg.num_experts
        aux_loss = cfg.aux_loss_coef * aux_loss_raw
        return merged, aux_loss


class DecoderLayer(nnx.Module):
    def __init__(self, cfg: Qwen3MoeConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.input_layernorm = RMSNorm(cfg.emb_dim, cfg.norm_eps, cfg.shd_cfg.rms_norm, rngs=rngs)
        self.attn = Attention(cfg=cfg, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim, cfg.norm_eps, cfg.shd_cfg.rms_norm, rngs=rngs)

        self.is_moe = cfg.is_moe_layer(layer_idx)
        if self.is_moe:
            self.mlp = MoEFeedForward(cfg=cfg, rngs=rngs)
        else:
            self.mlp = DenseMLP(cfg=cfg, rngs=rngs)

    def __call__(self, x: jax.Array, cache, segment_ids: jax.Array):
        inputs_normalized = self.input_layernorm(x)
        attn_output = x + self.attn(inputs_normalized, cache, segment_ids)
        post_norm = self.post_attention_layernorm(attn_output)
        if self.is_moe:
            ff_out, aux_loss = self.mlp(post_norm)
        else:
            ff_out = self.mlp(post_norm)
            aux_loss = jnp.array(0.0, dtype=x.dtype)
        return attn_output + ff_out, aux_loss


class Qwen3Moe(nnx.Module):
    def __init__(self, cfg: Qwen3MoeConfig, *, rngs: nnx.Rngs):
        self.embedder = shard(
            nnx.Embed(num_embeddings=cfg.vocab_size, features=cfg.emb_dim, dtype=jnp.float32, rngs=rngs),
            cfg.shd_cfg.emb_vd,
        )
        self.out_emb_shd = None
        self.layers = nnx.List([DecoderLayer(cfg=cfg, layer_idx=i, rngs=rngs) for i in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.emb_dim, cfg.norm_eps, cfg.shd_cfg.rms_norm, rngs=rngs)
        self.lm_head = shard(
            nnx.Linear(cfg.emb_dim, cfg.vocab_size, use_bias=False, rngs=rngs), cfg.shd_cfg.emb_dv
        )

    def __call__(self, tokens, segment_ids, cache, num_right_pads):
        del num_right_pads
        aux_losses = []
        x = self.embedder.embedding[...].at[(tokens,)].get(out_sharding=self.out_emb_shd)
        for i, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[i]
            x, aux = layer(x, layer_cache, segment_ids)
            aux_losses.append(aux)
        logits = self.lm_head(self.final_norm(x))
        total_aux = jnp.sum(jnp.stack(aux_losses)) if aux_losses else jnp.array(0.0, dtype=logits.dtype)
        return logits, total_aux
