"""Unified Qwen3 model (dense + MoE)."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P, reshard
from omegalax.models.moe_grouped import grouped_moe, grouped_moe_ep
from omegalax.models.remat_policy import resolve_remat_policy
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

    # Logical sharding of the expert-stacked projections, consumed by
    # ``omegalax.trainers.lora.inject_lora`` to attach per-expert LoRA
    # adapters (mirrors the ``nnx.Param`` sharding tuples below).
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
        # Optional per-expert LoRA adapters, populated by inject_lora. When
        # None (default) the expert einsums below are numerically unchanged.
        # nnx.data(None) marks these as data slots so a LoRAMoEExperts module
        # can be assigned in later (a plain None would be a static attribute).
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
        batch_axis = self.shd_cfg.act_btd[0]
        hidden_axis = self.shd_cfg.act_btd[2]
        ff_axis = self.shd_cfg.act_btf[2]
        B, T = hidden_BTD.shape[:2]

        if cfg.moe_backend == "dense":
            dense_hidden_BTD = reshard(hidden_BTD, P(self.shd_cfg.act_btd[0], None, None))
            ff_sharding = P(batch_axis, None, None, ff_axis)
            hidden_sharding = P(batch_axis, None, None, hidden_axis)
            gate_BTEF = jnp.einsum(
                "BTD,EDF->BTEF",
                dense_hidden_BTD,
                gate_proj_EDF,
                out_sharding=ff_sharding,
            )
            up_BTEF = jnp.einsum(
                "BTD,EDF->BTEF",
                dense_hidden_BTD,
                up_proj_EDF,
                out_sharding=ff_sharding,
            )
            # Per-expert LoRA on gate/up (added inside the expert map, before the
            # nonlinearity and top-k gather). No-op when adapters are unattached.
            if self.gate_proj_lora is not None:
                gate_BTEF += self.gate_proj_lora.delta_shared(
                    dense_hidden_BTD, out_sharding=ff_sharding
                )
            if self.up_proj_lora is not None:
                up_BTEF += self.up_proj_lora.delta_shared(
                    dense_hidden_BTD, out_sharding=ff_sharding
                )
            expert_hidden_BTEF = nnx.silu(gate_BTEF) * up_BTEF
            expert_out_BTED = jnp.einsum(
                "BTEF,EFD->BTED",
                expert_hidden_BTEF,
                down_proj_EFD,
                out_sharding=hidden_sharding,
            )
            if self.down_proj_lora is not None:
                expert_out_BTED += self.down_proj_lora.delta_per_expert(
                    expert_hidden_BTEF, out_sharding=hidden_sharding
                )

            flat_out = jax.lax.reshape(
                expert_out_BTED,
                (B * T, cfg.num_experts, cfg.emb_dim),
                out_sharding=P(batch_axis, None, hidden_axis),
            )
            flat_idx = topk_idx_BTk.reshape(B * T, cfg.num_experts_per_tok)
            gathered = jnp.take_along_axis(flat_out, flat_idx[..., None], axis=1)
            gathered = jax.lax.reshape(
                gathered,
                (B, T, cfg.num_experts_per_tok, cfg.emb_dim),
                out_sharding=P(batch_axis, None, None, hidden_axis),
            )
            merged_BTD = jnp.sum(gathered * topk_weights_BTk[..., None], axis=-2)
            merged_BTD = reshard(merged_BTD, self.shd_cfg.act_btd)
        else:
            # Dropless grouped-GEMM path (EP=1 or EP via ragged all-to-all).
            flat_hidden_ND = hidden_BTD.reshape(B * T, cfg.emb_dim)
            flat_idx_Nk = topk_idx_BTk.reshape(B * T, cfg.num_experts_per_tok)
            flat_w_Nk = topk_weights_BTk.reshape(B * T, cfg.num_experts_per_tok)
            moe_fn = grouped_moe_ep if cfg.moe_backend == "grouped_ep" else grouped_moe
            merged_ND = moe_fn(
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

    def __call__(self, hidden_BTD: jax.Array, cache, segment_ids_BT: jax.Array,
                 position_ids_BT: jax.Array | None = None):
        # Inline nnx.remat on the UNBOUND method (no static_argnums): nnx
        # functionalizes ``self`` via split/merge, so it must not be static.
        # Building the transform inline keeps graphdefs equal across fresh
        # instances (stable hash -> one trace, not one per instance).
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
        return attn_out_BTD + ff_out_BTD, aux_loss


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

        # Scan path: only for the training/eval forward pass (cache is None) and
        # homogeneous layer stacks. Decode (cache is not None) and heterogeneous
        # (mixed dense/MoE) stacks stay on the unrolled loop below. ``position_ids``
        # (used only for zig-zag CP) is broadcast to every layer.
        if cache is None and self.cfg.scan_layers and self.cfg.is_homogeneous:
            hidden_BTD, total_aux = _scan_layers(
                list(self.layers), hidden_BTD, segment_ids_BT, position_ids_BT
            )
            hidden_BTD = self.final_norm(hidden_BTD)
            return hidden_BTD, total_aux

        aux_losses = []
        # Unrolled fallback. Each layer self-remats inside DecoderLayer.__call__
        # with cfg.remat_policy, so we call it directly (no extra nnx.remat here):
        # a single, policy-honoring remat level shared with the scan path below.
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
    """Run homogeneous decoder layers with a single ``nnx.scan`` layer body.

    Stacks each layer's state along a new leading layer axis (replicated, i.e.
    not in the mesh) and scans over it. The per-layer parameter sharding is
    preserved because we stack the already-annotated per-layer states; only the
    leading layer axis is added and it carries no partition spec. Activation
    checkpointing is NOT applied here: the merged layer self-remats inside
    ``DecoderLayer.__call__`` with ``cfg.remat_policy``, giving a single,
    policy-honoring remat level shared with the unrolled path (no double remat).
    Per-layer aux losses are collected as a scanned output (ys) and summed after
    the scan, reproducing the unrolled sum.

    ``nnx.scan`` requires the carry dtype to be invariant across the loop. The
    qwen3/qwen3.5 MoE block returns its output in fp32 (its top-k combine weights
    ride ``probs`` in fp32), so under a bf16 config a layer can promote the hidden
    stream bf16 -> fp32; the unrolled Python loop tolerates that drift but a scan
    carry cannot. We therefore cast the per-step output hidden back to the input
    carry dtype so the carry type is stable. This is a no-op for fp32 configs (so
    the scan-vs-unrolled equivalence tests, which use fp32, are unaffected) and a
    single bf16 round of the residual stream otherwise.
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
