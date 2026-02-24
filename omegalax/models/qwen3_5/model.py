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

    def __init__(self, hidden_size: int, intermediate_size: int, *, rngs: nnx.Rngs):
        linear = partial(nnx.Linear, use_bias=False, rngs=rngs)
        self.gate_proj = linear(hidden_size, intermediate_size)
        self.up_proj = linear(hidden_size, intermediate_size)
        self.down_proj = linear(intermediate_size, hidden_size)

    @jax.named_scope("mlp")
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(nnx.silu(self.gate_proj(x)) * self.up_proj(x))


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
        self.router = nnx.Linear(D, E, use_bias=False, rngs=rngs)

        self.shared_expert = MLP(D, cfg.shared_expert_intermediate_size, rngs=rngs)
        self.shared_expert_gate = nnx.Linear(D, 1, use_bias=False, rngs=rngs)

    @jax.named_scope("moe_ffn")
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        cfg = self.cfg
        B, T = x.shape[:2]

        # Router
        router_logits = self.router(x)
        probs = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1)
        topk_weights, topk_idx = jax.lax.top_k(probs, cfg.num_experts_per_tok)
        topk_weights = topk_weights / jnp.clip(
            jnp.sum(topk_weights, axis=-1, keepdims=True), min=1e-9
        )
        topk_weights = topk_weights.astype(probs.dtype)

        # All-expert computation with fused gate_up_proj: (E, 2*F, D)
        gate_up = jnp.einsum("btd,efd->btef", x, self.gate_up_proj[...])
        gate_out, up_out = jnp.split(gate_up, 2, axis=-1)
        expert_hidden = nnx.silu(gate_out) * up_out
        expert_out = jnp.einsum("btef,edf->bted", expert_hidden, self.down_proj[...])

        # Top-k selection and merge
        flat_out = expert_out.reshape(B * T, cfg.num_experts, cfg.hidden_size)
        flat_idx = topk_idx.reshape(B * T, cfg.num_experts_per_tok)
        gathered = jnp.take_along_axis(flat_out, flat_idx[..., None], axis=1)
        gathered = gathered.reshape(B, T, cfg.num_experts_per_tok, cfg.hidden_size)
        moe_out = jnp.sum(gathered * topk_weights[..., None], axis=-2)

        # Shared expert with sigmoid gate
        shared_out = self.shared_expert(x)
        shared_gate = jax.nn.sigmoid(self.shared_expert_gate(x))
        shared_out = shared_gate * shared_out
        output = moe_out + shared_out

        # Auxiliary load-balancing loss
        load = jnp.mean(probs, axis=(0, 1))
        uniform = jnp.full_like(load, 1.0 / cfg.num_experts)
        aux_loss = cfg.router_aux_loss_coef * jnp.sum((load - uniform) ** 2)

        return output, aux_loss


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
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        segment_ids: jax.Array,
        position_ids: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        residual = x
        normed = self.input_layernorm(x)

        if self.layer_type == "full_attention":
            attn_out = self.attn(normed, cos, sin, segment_ids, position_ids)
        else:
            linear_mask = attention_mask
            if attention_mask is not None and jnp.all(attention_mask == 1):
                linear_mask = None
            attn_out = self.linear_attn(normed, linear_mask)

        x = residual + attn_out

        residual = x
        normed = self.post_attention_layernorm(x)
        ff_out, aux_loss = self.mlp(normed)
        x = residual + ff_out

        return x, aux_loss


# Text Model
class TextModel(nnx.Module):
    """Qwen3.5 text decoder."""

    def __init__(self, cfg: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.embedder = nnx.Embed(cfg.vocab_size, cfg.hidden_size, rngs=rngs)
        self.layers = nnx.List([
            DecoderLayer(cfg, i, rngs=rngs) for i in range(cfg.num_hidden_layers)
        ])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, rngs=rngs)

    @jax.named_scope("text_model")
    def __call__(
        self,
        tokens: jax.Array | None = None,
        inputs_embeds: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
        position_ids: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass (prefill).

        Args:
            tokens: (B, T) token IDs — or None if inputs_embeds given
            inputs_embeds: (B, T, D) — optional pre-computed embeddings
            segment_ids: (B, T) — 1 for real tokens, 0 for padding
            position_ids: (3, B, T) — MRoPE position IDs
        Returns:
            hidden_states: (B, T, D)
            total_aux_loss: scalar
        """
        cfg = self.cfg

        if inputs_embeds is None:
            x = self.embedder(tokens)
        else:
            x = inputs_embeds

        B, T, _ = x.shape

        if segment_ids is None:
            segment_ids = jnp.ones((B, T), dtype=jnp.int32)

        # Build position IDs for text-only (all 3 dims identical)
        if position_ids is None:
            seq_pos = jnp.arange(T, dtype=jnp.int32)[None, :]
            position_ids = jnp.broadcast_to(seq_pos, (B, T))
            position_ids_3d = jnp.stack([position_ids] * 3, axis=0)
        elif position_ids.ndim == 2:
            position_ids_3d = jnp.stack([position_ids] * 3, axis=0)
        else:
            position_ids_3d = position_ids

        # Generate MRoPE cos/sin
        cos, sin = generate_text_rope(
            position_ids_3d,
            cfg.head_dim,
            cfg.partial_rotary_factor,
            cfg.rope_theta,
            cfg.mrope_section,
        )

        # Attention mask for linear attention layers
        attention_mask = (segment_ids != 0).astype(jnp.float32)

        # Use first row of position_ids for causal mask positions
        text_position_ids = position_ids_3d[0]

        aux_losses = []
        for layer in self.layers:
            x, aux = layer(x, cos, sin, segment_ids, text_position_ids, attention_mask)
            aux_losses.append(aux)

        x = self.final_norm(x)
        total_aux = jnp.sum(jnp.stack(aux_losses)) if aux_losses else jnp.array(0.0)
        return x, total_aux


# Causal LM
class Qwen3_5ForCausalLM(nnx.Module):
    """Text-only causal language model."""

    def __init__(self, cfg: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        self.text = TextModel(cfg, rngs=rngs)
        self.lm_head = nnx.Linear(cfg.hidden_size, cfg.vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, tokens, segment_ids, cache, num_right_pads):
        del cache, num_right_pads
        hidden, aux = self.text(tokens=tokens, segment_ids=segment_ids)
        return self.lm_head(hidden), aux


# VLM
class Qwen3_5ForConditionalGeneration(nnx.Module):
    """Vision-Language Model."""

    def __init__(self, cfg: Qwen3_5Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.vision = VisionModel(cfg.vision_config, rngs=rngs)
        self.text = TextModel(cfg.text_config, rngs=rngs)
        self.lm_head = nnx.Linear(
            cfg.text_config.hidden_size, cfg.text_config.vocab_size, use_bias=False, rngs=rngs,
        )

    def __call__(
        self,
        tokens: jax.Array,
        segment_ids: jax.Array,
        cache,
        num_right_pads,
        pixel_values: jax.Array | None = None,
        image_grid_thw: jax.Array | None = None,
        position_ids: jax.Array | None = None,
    ):
        """Forward pass.

        For text-only input (no pixel_values), behaves like Qwen3_5ForCausalLM.
        """
        del cache, num_right_pads

        inputs_embeds = self.text.embedder(tokens)

        if pixel_values is not None and image_grid_thw is not None:
            image_embeds = self.vision(pixel_values, image_grid_thw)
            image_mask = (tokens == self.cfg.image_token_id)
            image_mask_3d = jnp.broadcast_to(
                image_mask[:, :, None], inputs_embeds.shape
            )
            inputs_embeds = jnp.where(image_mask_3d, 0.0, inputs_embeds)
            # Scatter image embeddings into placeholder positions
            batch_indices, seq_indices = jnp.where(image_mask)
            inputs_embeds = inputs_embeds.at[batch_indices, seq_indices].set(image_embeds)

        hidden, aux = self.text(
            inputs_embeds=inputs_embeds,
            segment_ids=segment_ids,
            position_ids=position_ids,
        )
        return self.lm_head(hidden), aux
