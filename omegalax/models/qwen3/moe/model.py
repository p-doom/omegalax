import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P, reshard

from ..attention import Attention
from ..dense.model import MLP as DenseMLP
from ..norms import RMSNorm
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
        self.router = nnx.Linear(D, E, use_bias=False, rngs=rngs, dtype=cfg.dtype)

    @jax.named_scope("moe_feed_forward")
    def __call__(self, hidden_BTD: jax.Array) -> tuple[jax.Array, jax.Array]:
        cfg = self.cfg

        router_logits_BTE = self.router(hidden_BTD, out_sharding=P(self.shd_cfg.act_btd[0], None, None))
        probs_BTE = jax.nn.softmax(router_logits_BTE.astype(jnp.float32), axis=-1)
        topk_weights_BTk, topk_idx_BTk = jax.lax.top_k(probs_BTE, cfg.num_experts_per_tok)
        if cfg.norm_topk_prob:
            topk_weights_BTk = topk_weights_BTk / jnp.clip(
                jnp.sum(topk_weights_BTk, axis=-1, keepdims=True), min=1e-9
            )
        topk_weights_BTk = topk_weights_BTk.astype(probs_BTE.dtype)

        gate_proj_EDF = self.gate_proj[...]
        up_proj_EDF = self.up_proj[...]
        down_proj_EFD = self.down_proj[...]
        batch_axis = self.shd_cfg.act_btd[0]
        hidden_axis = self.shd_cfg.act_btd[2]
        ff_axis = self.shd_cfg.act_btf[2]

        dense_hidden_BTD = reshard(hidden_BTD, P(self.shd_cfg.act_btd[0], None, None))
        gate_BTEF = jnp.einsum(
            "BTD,EDF->BTEF",
            dense_hidden_BTD,
            gate_proj_EDF,
            out_sharding=P(batch_axis, None, None, ff_axis),
        )
        up_BTEF = jnp.einsum(
            "BTD,EDF->BTEF",
            dense_hidden_BTD,
            up_proj_EDF,
            out_sharding=P(batch_axis, None, None, ff_axis),
        )
        expert_hidden_BTEF = nnx.silu(gate_BTEF) * up_BTEF
        expert_out_BTED = jnp.einsum(
            "BTEF,EFD->BTED",
            expert_hidden_BTEF,
            down_proj_EFD,
            out_sharding=P(batch_axis, None, None, hidden_axis),
        )

        B, T = hidden_BTD.shape[:2]
        flat_out = expert_out_BTED.reshape(B * T, cfg.num_experts, cfg.emb_dim)
        flat_idx = topk_idx_BTk.reshape(B * T, cfg.num_experts_per_tok)
        gathered = jnp.take_along_axis(flat_out, flat_idx[..., None], axis=1)
        gathered = gathered.reshape(B, T, cfg.num_experts_per_tok, cfg.emb_dim)
        merged_BTD = jnp.sum(gathered * topk_weights_BTk[..., None], axis=-2)
        merged_BTD = reshard(merged_BTD, self.shd_cfg.act_btd)

        # HF-style load-balancing loss (Switch Transformer eqs. 4-6)
        expert_mask_BTkE = jax.nn.one_hot(topk_idx_BTk, cfg.num_experts, dtype=probs_BTE.dtype)
        tokens_per_expert = jnp.mean(expert_mask_BTkE, axis=(0, 1))
        router_prob_per_expert_E = jnp.mean(probs_BTE, axis=(0, 1))
        aux_loss_raw = jnp.sum(tokens_per_expert * router_prob_per_expert_E[None, :]) * cfg.num_experts
        aux_loss = cfg.aux_loss_coef * aux_loss_raw
        return merged_BTD, aux_loss


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

    def __call__(self, hidden_BTD: jax.Array, cache, segment_ids_BT: jax.Array):
        normed_BTD = self.input_layernorm(hidden_BTD)
        attn_out_BTD = hidden_BTD + self.attn(normed_BTD, cache, segment_ids_BT)
        post_norm_BTD = self.post_attention_layernorm(attn_out_BTD)
        if self.is_moe:
            ff_out_BTD, aux_loss = self.mlp(post_norm_BTD)
        else:
            ff_out_BTD = self.mlp(post_norm_BTD)
            aux_loss = jnp.array(0.0, dtype=jnp.float32)
        return attn_out_BTD + ff_out_BTD, aux_loss


class Qwen3Moe(nnx.Module):
    def __init__(self, cfg: Qwen3MoeConfig, *, rngs: nnx.Rngs):
        self.embedder = nnx.Embed(num_embeddings=cfg.vocab_size, features=cfg.emb_dim, dtype=cfg.dtype, rngs=rngs)
        self.out_emb_shd = cfg.shd_cfg.act_btd
        self.logits_shd = P(cfg.shd_cfg.act_btd[0], None, None)
        self.layers = nnx.List([DecoderLayer(cfg=cfg, layer_idx=i, rngs=rngs) for i in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.emb_dim, cfg.norm_eps, cfg.shd_cfg.rms_norm, rngs=rngs)
        self.lm_head = nnx.Linear(cfg.emb_dim, cfg.vocab_size, use_bias=False, rngs=rngs, dtype=cfg.dtype)

    def __call__(self, token_ids_BT, segment_ids_BT, cache, num_right_pads):
        del num_right_pads
        aux_losses = []
        hidden_BTD = self.embedder.embedding[...].at[(token_ids_BT,)].get(out_sharding=self.out_emb_shd)
        for i, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[i]
            hidden_BTD, aux = layer(hidden_BTD, layer_cache, segment_ids_BT)
            aux_losses.append(aux)
        logits_BTV = self.lm_head(self.final_norm(hidden_BTD), out_sharding=self.logits_shd)
        total_aux = jnp.sum(jnp.stack(aux_losses)) if aux_losses else jnp.array(0.0, dtype=jnp.float32)
        return logits_BTV, total_aux
