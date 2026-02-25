"""Qwen3.5 model: text decoder and VLM composite."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from .attention import Attention
from .config import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from .deltanet import GatedDeltaNet
from .norms import RMSNorm
from .rope import generate_text_rope
from .vision import VisionModel


# Feed-forward blocks
class MLP(nnx.Module):
    """Standard gated MLP (shared expert or dense fallback)."""

    def __init__(self, hidden_size: int, intermediate_size: int, *, dtype=None, rngs: nnx.Rngs):
        linear = partial(nnx.Linear, use_bias=False, rngs=rngs, dtype=dtype)
        self.gate_proj = linear(hidden_size, intermediate_size)
        self.up_proj = linear(hidden_size, intermediate_size)
        self.down_proj = linear(intermediate_size, hidden_size)

    @jax.named_scope("mlp")
    def __call__(self, hidden_BTD: jax.Array) -> jax.Array:
        return self.down_proj(nnx.silu(self.gate_proj(hidden_BTD)) * self.up_proj(hidden_BTD))


class MoEFeedForward(nnx.Module):
    """Sparse Mixture-of-Experts block with a shared expert and shared expert gate."""

    def __init__(self, cfg: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        E = cfg.num_experts
        D = cfg.hidden_size
        F_moe = cfg.moe_intermediate_size

        init = nnx.initializers.lecun_normal()
        self.gate_up_proj = nnx.Param(init(rngs.params(), (E, 2 * F_moe, D)))
        self.down_proj = nnx.Param(init(rngs.params(), (E, D, F_moe)))
        self.router = nnx.Linear(D, E, use_bias=False, rngs=rngs, dtype=cfg.dtype)

        self.shared_expert = MLP(D, cfg.shared_expert_intermediate_size, dtype=cfg.dtype, rngs=rngs)
        self.shared_expert_gate = nnx.Linear(D, 1, use_bias=False, rngs=rngs, dtype=cfg.dtype)

    @jax.named_scope("moe_ffn")
    def __call__(self, hidden_BTD: jax.Array) -> tuple[jax.Array, jax.Array]:
        cfg = self.cfg
        B, T = hidden_BTD.shape[:2]

        router_logits_BTE = self.router(hidden_BTD)
        probs_BTE = jax.nn.softmax(router_logits_BTE.astype(jnp.float32), axis=-1)
        topk_weights_BTk, topk_idx_BTk = jax.lax.top_k(probs_BTE, cfg.num_experts_per_tok)
        topk_weights_BTk = topk_weights_BTk / jnp.clip(
            jnp.sum(topk_weights_BTk, axis=-1, keepdims=True), min=1e-9
        )
        topk_weights_BTk = topk_weights_BTk.astype(probs_BTE.dtype)

        gate_up_BTEF = jnp.einsum("BTD,EFD->BTEF", hidden_BTD, self.gate_up_proj[...])
        gate_BTEF, up_BTEF = jnp.split(gate_up_BTEF, 2, axis=-1)
        expert_hidden_BTEF = nnx.silu(gate_BTEF) * up_BTEF
        expert_out_BTED = jnp.einsum("BTEF,EDF->BTED", expert_hidden_BTEF, self.down_proj[...])

        flat_out = expert_out_BTED.reshape(B * T, cfg.num_experts, cfg.hidden_size)
        flat_idx = topk_idx_BTk.reshape(B * T, cfg.num_experts_per_tok)
        gathered = jnp.take_along_axis(flat_out, flat_idx[..., None], axis=1)
        gathered = gathered.reshape(B, T, cfg.num_experts_per_tok, cfg.hidden_size)
        moe_out_BTD = jnp.sum(gathered * topk_weights_BTk[..., None], axis=-2)

        shared_out_BTD = self.shared_expert(hidden_BTD)
        shared_gate = jax.nn.sigmoid(self.shared_expert_gate(hidden_BTD))
        shared_out_BTD = shared_gate * shared_out_BTD
        output_BTD = moe_out_BTD + shared_out_BTD

        load_E = jnp.mean(probs_BTE, axis=(0, 1))
        uniform_E = jnp.full_like(load_E, 1.0 / cfg.num_experts)
        aux_loss = cfg.router_aux_loss_coef * jnp.sum((load_E - uniform_E) ** 2)

        return output_BTD, aux_loss


# Decoder Layer
class DecoderLayer(nnx.Module):
    """Hybrid decoder layer: full_attention or linear_attention + MoE MLP."""

    def __init__(self, cfg: Qwen3_5TextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.layer_type = cfg.layer_types[layer_idx]

        if self.layer_type == "full_attention":
            self.attn = Attention(cfg, rngs=rngs)
        else:
            self.linear_attn = GatedDeltaNet(cfg, rngs=rngs)

        self.mlp = MoEFeedForward(cfg, rngs=rngs)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, rngs=rngs)

    def __call__(
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
            linear_mask = attention_mask_BT
            if attention_mask_BT is not None and jnp.all(attention_mask_BT == 1):
                linear_mask = None
            attn_out_BTD = self.linear_attn(normed_BTD, linear_mask)

        hidden_BTD = residual_BTD + attn_out_BTD

        residual_BTD = hidden_BTD
        normed_BTD = self.post_attention_layernorm(hidden_BTD)
        ff_out_BTD, aux_loss = self.mlp(normed_BTD)
        hidden_BTD = residual_BTD + ff_out_BTD

        return hidden_BTD, aux_loss


# Text Model
class TextModel(nnx.Module):
    """Qwen3.5 text decoder."""

    def __init__(self, cfg: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.embedder = nnx.Embed(cfg.vocab_size, cfg.hidden_size, rngs=rngs, dtype=cfg.dtype)
        self.layers = nnx.List([
            DecoderLayer(cfg, i, rngs=rngs) for i in range(cfg.num_hidden_layers)
        ])
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
            hidden_BTD = self.embedder(token_ids_BT)
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

        aux_losses = []
        for layer in self.layers:
            hidden_BTD, aux = layer(hidden_BTD, cos_BTK, sin_BTK, segment_ids_BT, text_position_ids_BT, attention_mask_BT)
            aux_losses.append(aux)

        hidden_BTD = self.final_norm(hidden_BTD)
        total_aux = jnp.sum(jnp.stack(aux_losses)) if aux_losses else jnp.array(0.0)
        return hidden_BTD, total_aux


# Causal LM
class Qwen3_5ForCausalLM(nnx.Module):
    """Text-only causal language model."""

    def __init__(self, cfg: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        self.text = TextModel(cfg, rngs=rngs)
        self.lm_head = nnx.Linear(cfg.hidden_size, cfg.vocab_size, use_bias=False, rngs=rngs, dtype=cfg.dtype)

    def __call__(self, token_ids_BT, segment_ids_BT, cache, num_right_pads):
        del cache, num_right_pads
        hidden_BTD, aux = self.text(token_ids_BT=token_ids_BT, segment_ids_BT=segment_ids_BT)
        return self.lm_head(hidden_BTD), aux


# VLM
class Qwen3_5ForConditionalGeneration(nnx.Module):
    """Vision-Language Model."""

    def __init__(self, cfg: Qwen3_5Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.vision = VisionModel(cfg.vision_config, rngs=rngs)
        self.text = TextModel(cfg.text_config, rngs=rngs)
        self.lm_head = nnx.Linear(
            cfg.text_config.hidden_size, cfg.text_config.vocab_size, use_bias=False, rngs=rngs, dtype=cfg.text_config.dtype,
        )

    def __call__(
        self,
        token_ids_BT: jax.Array,
        segment_ids_BT: jax.Array,
        cache,
        num_right_pads,
        pixel_values: jax.Array | None = None,
        image_grid_thw: jax.Array | None = None,
        position_ids_ZBT: jax.Array | None = None,
    ):
        del cache, num_right_pads

        inputs_embeds_BTD = self.text.embedder(token_ids_BT)

        if pixel_values is not None and image_grid_thw is not None:
            image_embeds_ND = self.vision(pixel_values, image_grid_thw)
            image_mask_BT = (token_ids_BT == self.cfg.image_token_id)
            image_mask_BTD = jnp.broadcast_to(
                image_mask_BT[:, :, None], inputs_embeds_BTD.shape
            )
            inputs_embeds_BTD = jnp.where(image_mask_BTD, 0.0, inputs_embeds_BTD)
            batch_indices, seq_indices = jnp.where(image_mask_BT)
            inputs_embeds_BTD = inputs_embeds_BTD.at[batch_indices, seq_indices].set(image_embeds_ND)

        hidden_BTD, aux = self.text(
            inputs_embeds_BTD=inputs_embeds_BTD,
            segment_ids_BT=segment_ids_BT,
            position_ids_ZBT=position_ids_ZBT,
        )
        return self.lm_head(hidden_BTD), aux
