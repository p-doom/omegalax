"""Unified Qwen3 model (dense + MoE)."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P, reshard
from omegalax.models.moe_grouped import grouped_moe
from omegalax.models.remat_policy import resolve_remat_policy, tag_offload_residual
from .attention import Attention
from .config import Qwen3Config
from .norms import RMSNorm

wp = nnx.with_partitioning


class MLP(nnx.Module):
    def __init__(self, cfg: Qwen3Config, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
        init_fn = nnx.initializers.lecun_normal()
        col_parallel = partial(
            nnx.Linear,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=wp(init_fn, ("embed", "mlp")),
        )
        row_parallel = partial(
            nnx.Linear,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=wp(init_fn, ("mlp", "embed")),
        )
        self.gate_proj = col_parallel(cfg.emb_dim, cfg.mlp_dim)
        self.up_proj = col_parallel(cfg.emb_dim, cfg.mlp_dim)
        self.down_proj = row_parallel(cfg.mlp_dim, cfg.emb_dim)

    @jax.named_scope("feed_forward")
    def __call__(self, hidden_BTD: jax.Array) -> jax.Array:
        gate_BTF = self.gate_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf)
        up_BTF = self.up_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf)
        activated_BTF = nnx.silu(gate_BTF) * up_BTF
        return self.down_proj(activated_BTF, out_sharding=self.shd_cfg.act_btd)


class MoEFeedForward(nnx.Module):
    """Sparse MoE block matching the HuggingFace Qwen3MoeSparseMoeBlock architecture."""

    # Logical sharding of the expert-stacked projections, read by
    # trainers.lora.inject_lora to attach per-expert LoRA (mirrors the Params below).
    _EXPERT_LORA_SHARDING = {
        "gate_proj": (None, "embed", "mlp"),
        "up_proj": (None, "embed", "mlp"),
        "down_proj": (None, "mlp", "embed"),
    }

    def __init__(self, cfg: Qwen3Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.shd_cfg = cfg.shd_cfg
        E, D, F = cfg.num_experts, cfg.emb_dim, cfg.moe_intermediate_size

        init = nnx.initializers.lecun_normal()
        self.gate_proj = nnx.Param(
            init(rngs.params(), (E, D, F)),
            sharding=(None, "embed", "mlp"),
        )
        self.up_proj = nnx.Param(
            init(rngs.params(), (E, D, F)),
            sharding=(None, "embed", "mlp"),
        )
        self.down_proj = nnx.Param(
            init(rngs.params(), (E, F, D)),
            sharding=(None, "mlp", "embed"),
        )
        # Optional per-expert LoRA adapters (populated by inject_lora; None = no-op).
        # nnx.data(None) makes these data slots so a module can be assigned later
        # (a plain None would be a static attribute).
        self.gate_proj_lora = nnx.data(None)
        self.up_proj_lora = nnx.data(None)
        self.down_proj_lora = nnx.data(None)
        self.router = nnx.Linear(
            D,
            E,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=wp(init, ("embed", None)),
        )

    @jax.named_scope("moe_feed_forward")
    def __call__(self, hidden_BTD: jax.Array) -> tuple[jax.Array, jax.Array]:
        cfg = self.cfg
        router_logits_BTE = self.router(
            hidden_BTD, out_sharding=P(self.shd_cfg.act_btd[0], None, None)
        )
        probs_BTE = jax.nn.softmax(router_logits_BTE.astype(jnp.float32), axis=-1)
        topk_weights_BTk, topk_idx_BTk = jax.lax.top_k(probs_BTE, cfg.num_experts_per_tok)
        if cfg.norm_topk_prob:
            topk_weights_BTk = topk_weights_BTk / jnp.clip(
                jnp.sum(topk_weights_BTk, axis=-1, keepdims=True), min=1e-9
            )
        topk_weights_BTk = topk_weights_BTk.astype(probs_BTE.dtype)

        gate_proj_EDF = jnp.astype(self.gate_proj[...], hidden_BTD.dtype)
        up_proj_EDF = jnp.astype(self.up_proj[...], hidden_BTD.dtype)
        down_proj_EFD = jnp.astype(self.down_proj[...], hidden_BTD.dtype)
        B, T = hidden_BTD.shape[:2]

        # Dropless grouped-GEMM MoE (single-device grouped path).
        flat_hidden_ND = hidden_BTD.reshape(B * T, cfg.emb_dim)
        flat_idx_Nk = topk_idx_BTk.reshape(B * T, cfg.num_experts_per_tok)
        flat_w_Nk = topk_weights_BTk.reshape(B * T, cfg.num_experts_per_tok)
        merged_ND = grouped_moe(
            flat_hidden_ND,
            flat_idx_Nk,
            flat_w_Nk,
            gate_proj_EDF,
            up_proj_EDF,
            down_proj_EFD,
            num_experts=cfg.num_experts,
            gate_lora=self.gate_proj_lora,
            up_lora=self.up_proj_lora,
            down_lora=self.down_proj_lora,
        )
        merged_BTD = reshard(merged_ND.reshape(B, T, cfg.emb_dim), self.shd_cfg.act_btd)

        expert_mask_BTkE = jax.nn.one_hot(topk_idx_BTk, cfg.num_experts, dtype=probs_BTE.dtype)
        tokens_per_expert = jnp.mean(expert_mask_BTkE, axis=(0, 1))
        router_prob_per_expert_E = jnp.mean(probs_BTE, axis=(0, 1))
        aux_loss_raw = (
            jnp.sum(tokens_per_expert * router_prob_per_expert_E[None, :]) * cfg.num_experts
        )
        aux_loss = cfg.aux_loss_coef * aux_loss_raw
        return merged_BTD, aux_loss


class DecoderLayer(nnx.Module):
    def __init__(self, cfg: Qwen3Config, layer_idx: int, *, rngs: nnx.Rngs):
        self.input_layernorm = RMSNorm(cfg.emb_dim, cfg.norm_eps, rngs=rngs)
        self.attn = Attention(cfg=cfg, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim, cfg.norm_eps, rngs=rngs)

        self.is_moe = cfg.is_moe_layer(layer_idx)
        if self.is_moe:
            self.mlp = MoEFeedForward(cfg=cfg, rngs=rngs)
        else:
            self.mlp = MLP(cfg=cfg, rngs=rngs)

        self._remat_policy = resolve_remat_policy(cfg.remat_policy)
        # Static string (keeps the graphdef stable) so the named-offload policy can
        # tag the residual for host offload; inert under every other policy.
        self._remat_policy_name = cfg.remat_policy

    def __call__(self, hidden_BTD: jax.Array, cache, segment_ids_BT: jax.Array,
                 position_ids_BT: jax.Array | None = None):
        # nnx.remat on the UNBOUND method (no static_argnums): nnx functionalizes
        # self via split/merge, so it must not be static; building the transform
        # inline keeps graphdefs equal across instances (one trace, not one each).
        return nnx.remat(type(self)._impl, policy=self._remat_policy)(
            self, hidden_BTD, cache, segment_ids_BT, position_ids_BT
        )

    def _impl(self, hidden_BTD: jax.Array, cache, segment_ids_BT: jax.Array,
              position_ids_BT: jax.Array | None = None):
        normed_BTD = self.input_layernorm(hidden_BTD)
        attn_out_BTD = hidden_BTD + self.attn(normed_BTD, cache, segment_ids_BT, position_ids_BT)
        post_norm_BTD = self.post_attention_layernorm(attn_out_BTD)
        if self.is_moe:
            ff_out_BTD, aux_loss = self.mlp(post_norm_BTD)
        else:
            ff_out_BTD = self.mlp(post_norm_BTD)
            aux_loss = jnp.array(0.0, dtype=jnp.float32)
        out_BTD = tag_offload_residual(attn_out_BTD + ff_out_BTD, self._remat_policy_name)
        return out_BTD, aux_loss


class Qwen3(nnx.Module):
    """Unified Qwen3 model (dense and MoE)."""

    def __init__(self, cfg: Qwen3Config, *, rngs: nnx.Rngs):
        embed_init = nnx.initializers.normal(stddev=0.02)
        self.embedder = nnx.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.emb_dim,
            dtype=cfg.dtype,
            rngs=rngs,
            embedding_init=wp(embed_init, ("vocab", "embed")),
        )
        self.out_emb_shd = cfg.shd_cfg.act_btd
        self.logits_shd = P(cfg.shd_cfg.act_btd[0], None, None)
        self.cfg = cfg
        self.layers = nnx.List(
            [DecoderLayer(cfg=cfg, layer_idx=i, rngs=rngs) for i in range(cfg.num_layers)]
        )
        self.final_norm = RMSNorm(cfg.emb_dim, cfg.norm_eps, rngs=rngs)
        lm_head_init = nnx.initializers.lecun_normal()
        self.lm_head = nnx.Linear(
            cfg.emb_dim,
            cfg.vocab_size,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=wp(lm_head_init, ("embed", "vocab")),
        )

    def __call__(self, token_ids_BT, segment_ids_BT, cache, num_right_pads,
                 position_ids_BT=None):
        del num_right_pads
        hidden_BTD = jnp.astype(
            self.embedder.embedding[...].at[(token_ids_BT,)].get(out_sharding=self.out_emb_shd),
            self.embedder.dtype,
        )

        # Scan path only for the forward pass (cache is None) on a homogeneous
        # stack; decode and heterogeneous (mixed dense/MoE) stacks use the loop below.
        if cache is None and self.cfg.is_homogeneous:
            hidden_BTD, total_aux = _scan_layers(
                list(self.layers), hidden_BTD, segment_ids_BT, position_ids_BT
            )
            hidden_BTD = self.final_norm(hidden_BTD)
            return hidden_BTD, total_aux

        aux_losses = []
        # Unrolled fallback: each layer self-remats with cfg.remat_policy (no double remat).
        for i, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[i]
            hidden_BTD, aux = layer(hidden_BTD, layer_cache, segment_ids_BT, position_ids_BT)
            aux_losses.append(aux)
        hidden_BTD = self.final_norm(hidden_BTD)
        total_aux = (
            jnp.sum(jnp.stack(aux_losses)) if aux_losses else jnp.array(0.0, dtype=jnp.float32)
        )
        return hidden_BTD, total_aux


def _scan_layers(
    layers: list[DecoderLayer], hidden_BTD: jax.Array, segment_ids_BT: jax.Array,
    position_ids_BT: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Run homogeneous decoder layers with a single ``nnx.scan`` body: stack each
    layer's state on a new (replicated) layer axis and scan (per-layer sharding
    preserved; layers self-remat, no double remat; aux losses summed).

    The per-step output is cast back to the carry dtype because ``nnx.scan`` needs an
    invariant carry and the MoE block can promote bf16 -> fp32 (no-op for fp32).
    """
    carry_dtype = hidden_BTD.dtype
    graphdef, _ = nnx.split(layers[0])
    states = [nnx.split(layer)[1] for layer in layers]
    stacked_state = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)

    @nnx.scan(in_axes=(0, nnx.Carry, None, None), out_axes=(nnx.Carry, 0))
    def run(layer_state, carry_BTD, seg_BT, pos_BT):
        layer = nnx.merge(graphdef, layer_state)
        out_BTD, aux = layer(carry_BTD, None, seg_BT, pos_BT)
        return out_BTD.astype(carry_dtype), aux

    hidden_BTD, aux_L = run(stacked_state, hidden_BTD, segment_ids_BT, position_ids_BT)
    total_aux = jnp.sum(aux_L)
    return hidden_BTD, total_aux
