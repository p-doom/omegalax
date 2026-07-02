"""Qwen3.5 model: text decoder and VLM composite."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec, reshard

from omegalax.models.moe_grouped import grouped_moe, grouped_moe_ep
from omegalax.models.remat_policy import resolve_remat_policy
from omegalax.models.shard_config import ShardConfig
from .attention import Attention
from .config import Qwen3_5Config, Qwen3_5TextConfig
from .deltanet import GatedDeltaNet
from .norms import RMSNorm
from .rope import generate_text_rope
from .vision import VisionModel

P = PartitionSpec
wp = nnx.with_partitioning


# Feed-forward blocks
class MLP(nnx.Module):
    """Standard gated MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        shd_cfg: ShardConfig,
        *,
        dtype=None,
        rngs: nnx.Rngs,
    ):
        self.shd_cfg = shd_cfg
        init_fn = nnx.initializers.lecun_normal()
        col_parallel = partial(
            nnx.Linear,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            kernel_init=wp(init_fn, ("embed", "mlp")),
        )
        row_parallel = partial(
            nnx.Linear,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            kernel_init=wp(init_fn, ("mlp", "embed")),
        )
        self.gate_proj = col_parallel(hidden_size, intermediate_size)
        self.up_proj = col_parallel(hidden_size, intermediate_size)
        self.down_proj = row_parallel(intermediate_size, hidden_size)

    @jax.named_scope("mlp")
    def __call__(self, hidden_BTD: jax.Array) -> jax.Array:
        gate_BTF = self.gate_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf)
        up_BTF = self.up_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf)
        activated_BTF = nnx.silu(gate_BTF) * up_BTF
        return self.down_proj(activated_BTF, out_sharding=self.shd_cfg.act_btd)


class MoEFeedForward(nnx.Module):
    """Sparse Mixture-of-Experts block with a shared expert and shared expert gate."""

    def __init__(self, cfg: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.shd_cfg = cfg.shd_cfg
        E = cfg.num_experts
        D = cfg.hidden_size
        F_moe = cfg.moe_intermediate_size

        init = nnx.initializers.lecun_normal()
        self.gate_proj = nnx.Param(
            init(rngs.params(), (E, D, F_moe)),
            sharding=(None, "embed", "mlp"),
        )
        self.up_proj = nnx.Param(
            init(rngs.params(), (E, D, F_moe)),
            sharding=(None, "embed", "mlp"),
        )
        self.down_proj = nnx.Param(
            init(rngs.params(), (E, F_moe, D)),
            sharding=(None, "mlp", "embed"),
        )
        self.router = nnx.Linear(
            D,
            E,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=wp(init, ("embed", None)),
        )

        self.shared_expert = MLP(
            D,
            cfg.shared_expert_intermediate_size,
            shd_cfg=cfg.shd_cfg,
            dtype=cfg.dtype,
            rngs=rngs,
        )
        self.shared_expert_gate = nnx.Linear(
            D,
            1,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=wp(init, ("embed", None)),
        )

    @jax.named_scope("moe_ffn")
    def __call__(self, hidden_BTD: jax.Array) -> tuple[jax.Array, jax.Array]:
        cfg = self.cfg
        B, T = hidden_BTD.shape[:2]
        batch_axis = self.shd_cfg.act_btd[0]
        ff_axis = self.shd_cfg.act_btf[2]
        hidden_axis = self.shd_cfg.act_btd[2]

        router_logits_BTE = self.router(hidden_BTD, out_sharding=P(batch_axis, None, None))
        probs_BTE = jax.nn.softmax(router_logits_BTE.astype(jnp.float32), axis=-1)
        topk_weights_BTk, topk_idx_BTk = jax.lax.top_k(probs_BTE, cfg.num_experts_per_tok)
        topk_weights_BTk = topk_weights_BTk / jnp.clip(
            jnp.sum(topk_weights_BTk, axis=-1, keepdims=True), min=1e-9
        )
        topk_weights_BTk = topk_weights_BTk.astype(probs_BTE.dtype)

        compute_dtype = hidden_BTD.dtype
        gate_proj = jnp.astype(self.gate_proj[...], compute_dtype)
        up_proj = jnp.astype(self.up_proj[...], compute_dtype)
        down_proj = jnp.astype(self.down_proj[...], compute_dtype)

        if cfg.moe_backend == "dense":
            dense_hidden_BTD = reshard(hidden_BTD, P(batch_axis, None, None))
            gate_BTEF = jnp.einsum(
                "BTD,EDF->BTEF",
                dense_hidden_BTD,
                gate_proj,
                out_sharding=P(batch_axis, None, None, ff_axis),
            )
            up_BTEF = jnp.einsum(
                "BTD,EDF->BTEF",
                dense_hidden_BTD,
                up_proj,
                out_sharding=P(batch_axis, None, None, ff_axis),
            )
            expert_hidden_BTEF = nnx.silu(gate_BTEF) * up_BTEF
            expert_out_BTED = jnp.einsum(
                "BTEF,EFD->BTED",
                expert_hidden_BTEF,
                down_proj,
                out_sharding=P(batch_axis, None, None, hidden_axis),
            )

            flat_out = jax.lax.reshape(
                expert_out_BTED,
                (B * T, cfg.num_experts, cfg.hidden_size),
                out_sharding=P(batch_axis, None, None),
            )
            flat_idx = topk_idx_BTk.reshape(B * T, cfg.num_experts_per_tok)
            gathered = jnp.take_along_axis(flat_out, flat_idx[..., None], axis=1)
            gathered = jax.lax.reshape(
                gathered,
                (B, T, cfg.num_experts_per_tok, cfg.hidden_size),
                out_sharding=P(batch_axis, None, None, None),
            )
            moe_out_BTD = reshard(
                jnp.sum(gathered * topk_weights_BTk[..., None], axis=-2), self.shd_cfg.act_btd
            )
        else:
            # Dropless grouped-GEMM path for the routed experts. The shared expert /
            # shared-expert gate below are computed separately and unchanged.
            flat_hidden_ND = hidden_BTD.reshape(B * T, cfg.hidden_size)
            flat_idx_Nk = topk_idx_BTk.reshape(B * T, cfg.num_experts_per_tok)
            flat_w_Nk = topk_weights_BTk.reshape(B * T, cfg.num_experts_per_tok)
            moe_fn = grouped_moe_ep if cfg.moe_backend == "grouped_ep" else grouped_moe
            moe_out_ND = moe_fn(
                flat_hidden_ND,
                flat_idx_Nk,
                flat_w_Nk,
                gate_proj,
                up_proj,
                down_proj,
                num_experts=cfg.num_experts,
            )
            moe_out_BTD = reshard(
                moe_out_ND.reshape(B, T, cfg.hidden_size), self.shd_cfg.act_btd
            )

        shared_out_BTD = self.shared_expert(hidden_BTD)
        shared_gate = jax.nn.sigmoid(
            self.shared_expert_gate(hidden_BTD, out_sharding=P(batch_axis, None, None))
        )
        shared_out_BTD = shared_gate * shared_out_BTD
        output_BTD = moe_out_BTD + shared_out_BTD

        load_E = jnp.mean(probs_BTE, axis=(0, 1))
        uniform_E = jnp.full_like(load_E, 1.0 / cfg.num_experts)
        aux_loss = cfg.router_aux_loss_coef * jnp.sum((load_E - uniform_E) ** 2)

        return output_BTD, aux_loss


# Decoder Layer
class DecoderLayer(nnx.Module):
    """Hybrid decoder layer: full_attention or linear_attention + dense MLP or MoE."""

    def __init__(self, cfg: Qwen3_5TextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.layer_type = cfg.layer_types[layer_idx]
        self.is_moe = cfg.is_moe

        if self.layer_type == "full_attention":
            self.attn = Attention(cfg, rngs=rngs)
        else:
            self.linear_attn = GatedDeltaNet(cfg, rngs=rngs)

        if cfg.is_moe:
            self.mlp = MoEFeedForward(cfg, rngs=rngs)
        else:
            self.mlp = MLP(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.shd_cfg,
                dtype=cfg.dtype,
                rngs=rngs,
            )
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, rngs=rngs)

        self._remat_policy = resolve_remat_policy(cfg.remat_policy)

    def __call__(
        self,
        hidden_BTD: jax.Array,
        cos_BTK: jax.Array,
        sin_BTK: jax.Array,
        segment_ids_BT: jax.Array,
        position_ids_BT: jax.Array,
        attention_mask_BT: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        # Inline nnx.remat on the UNBOUND method (no static_argnums): nnx
        # functionalizes ``self`` via split/merge, so it must not be static.
        # Building the transform inline keeps graphdefs equal across fresh
        # instances (stable hash -> one trace, not one per instance).
        return nnx.remat(type(self)._impl, policy=self._remat_policy)(
            self,
            hidden_BTD,
            cos_BTK,
            sin_BTK,
            segment_ids_BT,
            position_ids_BT,
            attention_mask_BT,
        )

    def _impl(
        self,
        hidden_BTD: jax.Array,
        cos_BTK: jax.Array,
        sin_BTK: jax.Array,
        segment_ids_BT: jax.Array,
        position_ids_BT: jax.Array,
        attention_mask_BT: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        residual_BTD = hidden_BTD
        normed_BTD = self.input_layernorm(hidden_BTD)

        if self.layer_type == "full_attention":
            attn_out_BTD = self.attn(normed_BTD, cos_BTK, sin_BTK, segment_ids_BT, position_ids_BT)
        else:
            attn_out_BTD = self.linear_attn(normed_BTD, attention_mask_BT)

        hidden_BTD = residual_BTD + attn_out_BTD

        residual_BTD = hidden_BTD
        normed_BTD = self.post_attention_layernorm(hidden_BTD)
        if self.is_moe:
            ff_out_BTD, aux_loss = self.mlp(normed_BTD)
        else:
            ff_out_BTD = self.mlp(normed_BTD)
            aux_loss = jnp.array(0.0, dtype=jnp.float32)
        hidden_BTD = residual_BTD + ff_out_BTD

        return hidden_BTD, aux_loss


# Text Model
class TextModel(nnx.Module):
    """Qwen3.5 text decoder."""

    def __init__(self, cfg: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        embed_init = nnx.initializers.normal(stddev=0.02)
        self.embedder = nnx.Embed(
            cfg.vocab_size,
            cfg.hidden_size,
            rngs=rngs,
            dtype=cfg.dtype,
            embedding_init=wp(embed_init, ("vocab", "embed")),
        )
        self.out_emb_shd = cfg.shd_cfg.act_btd
        self.layers = nnx.List(
            [DecoderLayer(cfg, i, rngs=rngs) for i in range(cfg.num_hidden_layers)]
        )
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, rngs=rngs)

    @jax.named_scope("text_model")
    def __call__(
        self,
        token_ids_BT: jax.Array | None = None,
        inputs_embeds_BTD: jax.Array | None = None,
        segment_ids_BT: jax.Array | None = None,
        position_ids_ZBT: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        cfg = self.cfg

        if inputs_embeds_BTD is None:
            hidden_BTD = jnp.astype(
                self.embedder.embedding[...].at[(token_ids_BT,)].get(out_sharding=self.out_emb_shd),
                self.embedder.dtype,
            )
        else:
            hidden_BTD = inputs_embeds_BTD

        B, T, _ = hidden_BTD.shape

        if segment_ids_BT is None:
            segment_ids_BT = jnp.ones((B, T), dtype=jnp.int32)

        if position_ids_ZBT is None:
            seq_pos = jnp.arange(T, dtype=jnp.int32)[None, :]
            position_ids_BT = jnp.broadcast_to(seq_pos, (B, T))
            position_ids_ZBT = jnp.stack([position_ids_BT] * 3, axis=0)
        elif position_ids_ZBT.ndim == 2:
            position_ids_ZBT = jnp.stack([position_ids_ZBT] * 3, axis=0)

        cos_BTK, sin_BTK = generate_text_rope(
            position_ids_ZBT,
            cfg.head_dim,
            cfg.partial_rotary_factor,
            cfg.rope_theta,
            cfg.mrope_section,
        )
        cos_BTK = cos_BTK.astype(cfg.dtype)
        sin_BTK = sin_BTK.astype(cfg.dtype)

        attention_mask_BT = (segment_ids_BT != 0).astype(jnp.float32)
        text_position_ids_BT = position_ids_ZBT[0]

        layer_args = (cos_BTK, sin_BTK, segment_ids_BT, text_position_ids_BT, attention_mask_BT)

        # Block-scan path (training/eval forward): Qwen3.5 interleaves
        # linear_attention (Gated DeltaNet) and full_attention layers with
        # DIFFERENT param pytrees, so they cannot be stacked into ONE scan.
        # Instead we scan the repeating BLOCK (period p, e.g. [lin,lin,lin,full])
        # over num_blocks = num_layers // p, following MaxText's Gemma3 scannable
        # block pattern: each of the p positions is homogeneous across blocks and
        # its params are stacked along the block axis; the block body applies the
        # p positions in original order. Falls back to the unrolled loop for
        # irregular patterns (scan_block_period is None) or when disabled.
        period = cfg.scan_block_period
        # Only scan when there are >= 2 blocks (period < num_layers); otherwise the
        # single "block" is the whole irregular stack and a scan buys nothing, so
        # we drop to the unrolled loop below.
        if cfg.scan_layers and period is not None and period < len(self.layers):
            hidden_BTD, total_aux = _scan_hybrid_blocks(
                list(self.layers), period, hidden_BTD, layer_args
            )
            hidden_BTD = self.final_norm(hidden_BTD)
            return hidden_BTD, total_aux

        aux_losses = []
        # Unrolled fallback. Each layer self-remats inside DecoderLayer.__call__
        # with cfg.remat_policy, so we call it directly (no extra nnx.remat): one
        # policy-honoring remat level shared with the block-scan path above.
        for layer in self.layers:
            hidden_BTD, aux = layer(hidden_BTD, *layer_args)
            aux_losses.append(aux)

        hidden_BTD = self.final_norm(hidden_BTD)
        total_aux = jnp.sum(jnp.stack(aux_losses)) if aux_losses else jnp.array(0.0)
        return hidden_BTD, total_aux


def _scan_hybrid_blocks(
    layers: list[DecoderLayer],
    period: int,
    hidden_BTD: jax.Array,
    layer_args: tuple,
) -> tuple[jax.Array, jax.Array]:
    """Scan the repeating hybrid block over ``num_blocks = len(layers) // period``.

    Each of the ``period`` positions is homogeneous across blocks (all
    ``linear_attention`` or all ``full_attention`` with the same MLP/MoE), so we
    stack that position's per-layer state along a new leading block axis
    (replicated, i.e. NOT in the mesh) — giving ``period`` separate stacked-state
    arrays plus their graphdefs. The scan carries ``hidden`` over the blocks; the
    body applies the ``period`` positions in original order (a small Python unroll
    inside one scan step). Activation checkpointing is NOT applied here: each
    merged layer self-remats inside ``DecoderLayer.__call__`` with
    ``cfg.remat_policy``, giving a single policy-honoring remat level shared with
    the unrolled path (no double remat). Per-layer aux losses are summed within the
    block and the per-block totals are a scanned output, summed after the scan,
    reproducing the unrolled sum. Per-layer parameter sharding is preserved
    because we stack already-annotated per-layer states; only the leading block
    axis is added and it carries no partition spec.

    ``nnx.scan`` requires the carry dtype to be invariant across blocks. The MoE
    block returns fp32 (its top-k combine weights ride ``probs`` in fp32), so a
    bf16 config can promote the hidden stream bf16 -> fp32 within a block; we cast
    the block-output hidden back to the input carry dtype so the carry type is
    stable. No-op for fp32 configs (equivalence tests use fp32).
    """
    carry_dtype = hidden_BTD.dtype
    n = len(layers)
    num_blocks = n // period

    # One (graphdef, stacked_state) per position within the block.
    pos_graphdefs = []
    pos_stacked = []
    for pos in range(period):
        pos_layers = [layers[b * period + pos] for b in range(num_blocks)]
        graphdef, _ = nnx.split(pos_layers[0])
        states = [nnx.split(layer)[1] for layer in pos_layers]
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)
        pos_graphdefs.append(graphdef)
        pos_stacked.append(stacked)

    # in_axes: one stacked state per position (block axis 0), Carry for hidden,
    # None (broadcast) for the shared per-step layer args.
    in_axes = (*([0] * period), nnx.Carry, *([None] * len(layer_args)))

    @nnx.scan(in_axes=in_axes, out_axes=(nnx.Carry, 0))
    def run(*args):
        block_states = args[:period]
        carry_BTD = args[period]
        step_args = args[period + 1 :]

        block_aux = jnp.array(0.0, dtype=jnp.float32)
        for pos in range(period):
            # Merge this position's stacked state with its graphdef and call the
            # layer directly; DecoderLayer.__call__ self-remats with the config
            # policy (single remat level, no double remat).
            layer = nnx.merge(pos_graphdefs[pos], block_states[pos])
            carry_BTD, aux = layer(carry_BTD, *step_args)
            block_aux = block_aux + aux
        return carry_BTD.astype(carry_dtype), block_aux

    hidden_BTD, block_aux_L = run(*pos_stacked, hidden_BTD, *layer_args)
    return hidden_BTD, jnp.sum(block_aux_L)


# Causal LM
class Qwen3_5ForCausalLM(nnx.Module):
    """Text-only causal language model."""

    def __init__(self, cfg: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        self.text = TextModel(cfg, rngs=rngs)
        self.logits_shd = P(cfg.shd_cfg.act_btd[0], None, None)
        lm_head_init = nnx.initializers.lecun_normal()
        self.lm_head = nnx.Linear(
            cfg.hidden_size,
            cfg.vocab_size,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=wp(lm_head_init, ("embed", "vocab")),
        )

    def __call__(self, token_ids_BT, segment_ids_BT, cache, num_right_pads):
        del cache, num_right_pads
        return self.text(token_ids_BT=token_ids_BT, segment_ids_BT=segment_ids_BT)


# VLM
class Qwen3_5ForConditionalGeneration(nnx.Module):
    """Vision-Language Model."""

    def __init__(self, cfg: Qwen3_5Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.vision = VisionModel(cfg.vision_config, shd_cfg=cfg.text_config.shd_cfg, rngs=rngs)
        self.text = TextModel(cfg.text_config, rngs=rngs)
        self.logits_shd = P(cfg.text_config.shd_cfg.act_btd[0], None, None)
        lm_head_init = nnx.initializers.lecun_normal()
        self.lm_head = nnx.Linear(
            cfg.text_config.hidden_size,
            cfg.text_config.vocab_size,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.text_config.dtype,
            kernel_init=wp(lm_head_init, ("embed", "vocab")),
        )

    def __call__(
        self,
        token_ids_BT: jax.Array,
        segment_ids_BT: jax.Array,
        cache,
        num_right_pads,
        pixel_values: jax.Array | None = None,
        image_grid_thw: jax.Array | None = None,
        vision_cu_seqlens: jax.Array | None = None,
        position_ids_ZBT: jax.Array | None = None,
    ):
        del cache, num_right_pads
        inputs_embeds_BTD = jnp.astype(
            self.text.embedder.embedding[...]
            .at[(token_ids_BT,)]
            .get(out_sharding=self.text.out_emb_shd),
            self.text.embedder.dtype,
        )

        if pixel_values is not None and image_grid_thw is not None:
            image_embeds_ND = self.vision(pixel_values, image_grid_thw, vision_cu_seqlens)
            image_mask_BT = token_ids_BT == self.cfg.image_token_id
            image_mask_BTD = jnp.broadcast_to(image_mask_BT[:, :, None], inputs_embeds_BTD.shape)
            inputs_embeds_BTD = jnp.where(image_mask_BTD, 0.0, inputs_embeds_BTD)
            n_embeds = image_embeds_ND.shape[0]  # static after padding
            seq_len = token_ids_BT.shape[1]
            batch_indices, seq_indices = jnp.where(
                image_mask_BT,
                size=n_embeds,
                fill_value=(0, seq_len - 1),
            )
            num_real = jnp.sum(image_mask_BT)
            valid = jnp.arange(n_embeds) < num_real
            safe_embeds = jnp.where(
                valid[:, None],
                image_embeds_ND,
                0.0,
            ).astype(inputs_embeds_BTD.dtype)
            inputs_embeds_BTD = inputs_embeds_BTD.at[batch_indices, seq_indices].set(
                safe_embeds,
                out_sharding=self.text.out_emb_shd,
            )

        return self.text(
            inputs_embeds_BTD=inputs_embeds_BTD,
            segment_ids_BT=segment_ids_BT,
            position_ids_ZBT=position_ids_ZBT,
        )
