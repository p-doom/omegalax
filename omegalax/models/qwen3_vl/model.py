"""Qwen3-VL composite model: vision encoder + text decoder with M-RoPE and DeepStack."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .config import Qwen3VLConfig
from .vision import VisionModel

def _mask_value(dtype: jnp.dtype) -> float:
    return float(jnp.finfo(dtype).min)


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float, *, rngs: nnx.Rngs):
        self.scale = nnx.Param(jnp.ones(dim))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype
        variance = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return (self.scale[...] * normed).astype(dtype)


def apply_rope(x_BTHK: jax.Array, sin_BTK: jax.Array, cos_BTK: jax.Array) -> jax.Array:
    half = x_BTHK.shape[-1] // 2
    x1, x2 = x_BTHK[..., :half], x_BTHK[..., half:]
    sin_BTK = sin_BTK[:, :, None, :]
    cos_BTK = cos_BTK[:, :, None, :]
    return jnp.concatenate([x1 * cos_BTK - x2 * sin_BTK, x2 * cos_BTK + x1 * sin_BTK], axis=-1).astype(x_BTHK.dtype)


def compute_mrope_pos_embeddings(
    position_ids_ZBT: jax.Array, head_dim: int, rope_theta: float, mrope_section: tuple[int, ...]
) -> tuple[jax.Array, jax.Array]:
    """Compute M-RoPE positional embeddings with interleaved frequency layout.

    Args:
        position_ids_ZBT: (3, batch, seq_len)

    Returns:
        sin_BTK, cos_BTK: (batch, seq_len, head_dim // 2)
    """
    dim = head_dim
    inv_freq_K = 1.0 / (rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    freqs_ZBTK = jnp.einsum("ZBT,K->ZBTK", position_ids_ZBT.astype(jnp.float32), inv_freq_K)

    freqs_BTK = freqs_ZBTK[0]
    h_indices = np.arange(1, mrope_section[1] * 3, 3)
    w_indices = np.arange(2, mrope_section[2] * 3, 3)
    freqs_BTK = freqs_BTK.at[:, :, h_indices].set(freqs_ZBTK[1][:, :, h_indices])
    freqs_BTK = freqs_BTK.at[:, :, w_indices].set(freqs_ZBTK[2][:, :, w_indices])

    return jnp.sin(freqs_BTK), jnp.cos(freqs_BTK)


def get_rope_index(
    input_ids,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    spatial_merge_size: int = 2,
    image_token_id: int = 0,
    video_token_id: int = 0,
    vision_start_token_id: int = 0,
):
    """Compute 3D position IDs for M-RoPE (numpy, non-JIT).

    Returns:
        position_ids: np.ndarray (3, batch, seq_len)
        rope_deltas: np.ndarray (batch, 1)
    """
    input_ids_np = np.asarray(input_ids)
    if attention_mask is None:
        attention_mask_np = np.ones_like(input_ids_np)
    else:
        attention_mask_np = np.asarray(attention_mask)

    if video_grid_thw is not None:
        video_grid_thw_arr = np.asarray(video_grid_thw)
        split_thw = []
        for row in video_grid_thw_arr:
            for _ in range(int(row[0])):
                split_thw.append([1, int(row[1]), int(row[2])])
        video_grid_thw_list = split_thw
    else:
        video_grid_thw_list = None

    image_grid_thw_list = np.asarray(image_grid_thw).tolist() if image_grid_thw is not None else None

    batch_size, seq_len = input_ids_np.shape
    position_ids = np.zeros((3, batch_size, seq_len), dtype=np.int64)
    image_index, video_index = 0, 0
    mrope_position_deltas = []

    for i in range(batch_size):
        ids = input_ids_np[i][attention_mask_np[i] == 1]
        vision_start_indices = np.where(ids == vision_start_token_id)[0]
        vision_tokens = ids[vision_start_indices + 1]
        image_nums = int(np.sum(vision_tokens == image_token_id))
        video_nums = int(np.sum(vision_tokens == video_token_id))
        input_tokens = ids.tolist()
        llm_pos_ids_list: list[np.ndarray] = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums

        for _ in range(image_nums + video_nums):
            ed_image = input_tokens.index(image_token_id, st) if (image_token_id in input_tokens[st:] and remain_images > 0) else len(input_tokens) + 1
            ed_video = input_tokens.index(video_token_id, st) if (video_token_id in input_tokens[st:] and remain_videos > 0) else len(input_tokens) + 1

            if ed_image < ed_video:
                t, h, w = image_grid_thw_list[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grid_thw_list[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = t, h // spatial_merge_size, w // spatial_merge_size
            text_len = ed - st
            st_idx = int(llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0

            text_pos = np.tile(np.arange(text_len).reshape(1, -1), (3, 1)) + st_idx
            llm_pos_ids_list.append(text_pos)

            t_idx = np.repeat(np.arange(llm_grid_t), llm_grid_h * llm_grid_w)
            h_idx = np.tile(np.repeat(np.arange(llm_grid_h), llm_grid_w), llm_grid_t)
            w_idx = np.tile(np.arange(llm_grid_w), llm_grid_t * llm_grid_h)
            vision_pos = np.stack([t_idx, h_idx, w_idx]) + text_len + st_idx
            llm_pos_ids_list.append(vision_pos)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = int(llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            text_pos = np.tile(np.arange(text_len).reshape(1, -1), (3, 1)) + st_idx
            llm_pos_ids_list.append(text_pos)

        if llm_pos_ids_list:
            llm_positions = np.concatenate(llm_pos_ids_list, axis=1)
        else:
            llm_positions = np.zeros((3, 0), dtype=np.int64)

        position_ids[:, i, attention_mask_np[i] == 1] = llm_positions
        mrope_position_deltas.append(int(llm_positions.max() + 1 - seq_len) if llm_positions.size > 0 else 0)

    return position_ids, np.array(mrope_position_deltas).reshape(-1, 1)


class TextMLP(nnx.Module):
    def __init__(self, cfg: Qwen3VLConfig, *, rngs: nnx.Rngs):
        linear = partial(nnx.Linear, use_bias=False, rngs=rngs, dtype=cfg.dtype)
        self.gate_proj = linear(cfg.emb_dim, cfg.mlp_dim)
        self.up_proj = linear(cfg.emb_dim, cfg.mlp_dim)
        self.down_proj = linear(cfg.mlp_dim, cfg.emb_dim)

    def __call__(self, hidden_BTD: jax.Array) -> jax.Array:
        return self.down_proj(nnx.silu(self.gate_proj(hidden_BTD)) * self.up_proj(hidden_BTD))


class TextMoEFeedForward(nnx.Module):
    """Sparse MoE MLP mirroring the Qwen3 MoE structure."""

    def __init__(self, cfg: Qwen3VLConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        E, D, F = cfg.num_experts, cfg.emb_dim, cfg.moe_intermediate_size
        init = nnx.initializers.lecun_normal()
        self.gate_proj = nnx.Param(init(rngs.params(), (E, D, F)))
        self.up_proj = nnx.Param(init(rngs.params(), (E, D, F)))
        self.down_proj = nnx.Param(init(rngs.params(), (E, F, D)))
        self.router = nnx.Linear(D, E, use_bias=False, rngs=rngs, dtype=cfg.dtype)

    def __call__(self, hidden_BTD: jax.Array) -> tuple[jax.Array, jax.Array]:
        cfg = self.cfg
        router_logits_BTE = self.router(hidden_BTD)
        probs_BTE = jax.nn.softmax(router_logits_BTE.astype(jnp.float32), axis=-1)
        topk_weights_BTk, topk_idx_BTk = jax.lax.top_k(probs_BTE, cfg.num_experts_per_tok)
        if cfg.norm_topk_prob:
            topk_weights_BTk = topk_weights_BTk / jnp.clip(
                jnp.sum(topk_weights_BTk, axis=-1, keepdims=True), min=1e-9
            )
        topk_weights_BTk = topk_weights_BTk.astype(probs_BTE.dtype)

        gate_BTEF = jnp.einsum("BTD,EDF->BTEF", hidden_BTD, self.gate_proj[...])
        up_BTEF = jnp.einsum("BTD,EDF->BTEF", hidden_BTD, self.up_proj[...])
        expert_hidden_BTEF = nnx.silu(gate_BTEF) * up_BTEF
        expert_out_BTED = jnp.einsum("BTEF,EFD->BTED", expert_hidden_BTEF, self.down_proj[...])

        B, T = hidden_BTD.shape[:2]
        flat_out = expert_out_BTED.reshape(B * T, cfg.num_experts, cfg.emb_dim)
        flat_idx = topk_idx_BTk.reshape(B * T, cfg.num_experts_per_tok)
        gathered = jnp.take_along_axis(flat_out, flat_idx[..., None], axis=1)
        gathered = gathered.reshape(B, T, cfg.num_experts_per_tok, cfg.emb_dim)
        merged_BTD = jnp.sum(gathered * topk_weights_BTk[..., None], axis=-2)

        expert_mask_BTkE = jax.nn.one_hot(topk_idx_BTk, cfg.num_experts, dtype=probs_BTE.dtype)
        tokens_per_expert = jnp.mean(expert_mask_BTkE, axis=(0, 1))
        router_prob_per_expert_E = jnp.mean(probs_BTE, axis=(0, 1))
        aux_loss = jnp.sum(tokens_per_expert * router_prob_per_expert_E) * cfg.num_experts
        return merged_BTD, aux_loss


class TextAttention(nnx.Module):
    def __init__(self, cfg: Qwen3VLConfig, *, rngs: nnx.Rngs):
        linear = partial(nnx.Linear, use_bias=False, rngs=rngs, dtype=cfg.dtype)
        self.q_proj = linear(cfg.emb_dim, cfg.num_heads * cfg.head_dim)
        self.k_proj = linear(cfg.emb_dim, cfg.num_kv_heads * cfg.head_dim)
        self.v_proj = linear(cfg.emb_dim, cfg.num_kv_heads * cfg.head_dim)
        self.o_proj = linear(cfg.num_heads * cfg.head_dim, cfg.emb_dim)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps, rngs=rngs)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps, rngs=rngs)
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.scale = cfg.head_dim**-0.5
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads

    def __call__(self, hidden_BTD: jax.Array, sin_BTK: jax.Array, cos_BTK: jax.Array, mask: jax.Array | None) -> jax.Array:
        B, T, _ = hidden_BTD.shape
        q_BTHK = self.q_norm(self.q_proj(hidden_BTD).reshape(B, T, self.num_heads, self.head_dim))
        k_BTGK = self.k_norm(self.k_proj(hidden_BTD).reshape(B, T, self.num_kv_heads, self.head_dim))
        v_BTGK = self.v_proj(hidden_BTD).reshape(B, T, self.num_kv_heads, self.head_dim)

        q_BTHK = apply_rope(q_BTHK, sin_BTK, cos_BTK)
        k_BTGK = apply_rope(k_BTGK, sin_BTK, cos_BTK)

        q_BTGRK = q_BTHK.reshape(B, T, self.num_kv_heads, self.n_rep, self.head_dim)
        logits_BTSGR = jnp.einsum("BTGRK,BSGK->BTSGR", q_BTGRK, k_BTGK) * self.scale

        if mask is not None:
            attn_mask = mask[:, :, :, None, None]
            logits_BTSGR = jnp.where(attn_mask, logits_BTSGR, _mask_value(logits_BTSGR.dtype))

        weights_BTSGR = jax.nn.softmax(logits_BTSGR.astype(jnp.float32), axis=2).astype(logits_BTSGR.dtype)
        attn_BTGRK = jnp.einsum("BTSGR,BSGK->BTGRK", weights_BTSGR, v_BTGK)
        attn_BTHK = attn_BTGRK.reshape(B, T, self.num_heads, self.head_dim)
        return self.o_proj(attn_BTHK.reshape(B, T, self.num_heads * self.head_dim))


class TextDecoderLayer(nnx.Module):
    def __init__(self, cfg: Qwen3VLConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(cfg.emb_dim, cfg.norm_eps, rngs=rngs)
        self.attn = TextAttention(cfg, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim, cfg.norm_eps, rngs=rngs)
        self.is_moe = cfg.is_moe_layer(layer_idx)
        self.mlp = TextMoEFeedForward(cfg, rngs=rngs) if self.is_moe else TextMLP(cfg, rngs=rngs)

    def __call__(self, hidden_BTD: jax.Array, sin_BTK: jax.Array, cos_BTK: jax.Array, mask: jax.Array | None) -> tuple[jax.Array, jax.Array]:
        hidden_BTD = hidden_BTD + self.attn(self.input_layernorm(hidden_BTD), sin_BTK, cos_BTK, mask)
        if self.is_moe:
            ff_out_BTD, aux_loss = self.mlp(self.post_attention_layernorm(hidden_BTD))
        else:
            ff_out_BTD = self.mlp(self.post_attention_layernorm(hidden_BTD))
            aux_loss = jnp.array(0.0, dtype=jnp.float32)
        hidden_BTD = hidden_BTD + ff_out_BTD
        return hidden_BTD, aux_loss


class TextModel(nnx.Module):
    def __init__(self, cfg: Qwen3VLConfig, *, rngs: nnx.Rngs):
        self.embedder = nnx.Embed(
            num_embeddings=cfg.vocab_size, features=cfg.emb_dim, dtype=cfg.dtype, rngs=rngs
        )
        self.layers = nnx.List([TextDecoderLayer(cfg, layer_idx=i, rngs=rngs) for i in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.emb_dim, cfg.norm_eps, rngs=rngs)


class Qwen3VL(nnx.Module):
    def __init__(self, cfg: Qwen3VLConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.vision = VisionModel(cfg.vision, rngs=rngs)
        self.text = TextModel(cfg, rngs=rngs)
        self.lm_head = nnx.Linear(cfg.emb_dim, cfg.vocab_size, use_bias=False, rngs=rngs, dtype=cfg.dtype)

    def __call__(
        self,
        token_ids_BT: jax.Array,
        attention_mask_BT: jax.Array,
        position_ids_ZBT: jax.Array | np.ndarray | None = None,
        pixel_values: jax.Array | None = None,
        image_grid_thw: jax.Array | None = None,
    ) -> jax.Array:
        cfg = self.cfg

        image_features_ND = None
        deepstack_features = None
        visual_pos_mask_BT = None
        if pixel_values is not None and image_grid_thw is not None:
            image_features_ND, deepstack_features = self.vision(pixel_values, image_grid_thw)

        inputs_embeds_BTD = self.text.embedder(token_ids_BT)

        if image_features_ND is not None:
            image_mask_BT = token_ids_BT == cfg.image_token_id
            visual_pos_mask_BT = image_mask_BT
            batch_idx, seq_idx = jnp.where(image_mask_BT)
            inputs_embeds_BTD = inputs_embeds_BTD.at[batch_idx, seq_idx].set(
                image_features_ND.astype(inputs_embeds_BTD.dtype)
            )

        if position_ids_ZBT is None:
            if image_grid_thw is not None:
                position_ids_ZBT, _ = get_rope_index(
                    token_ids_BT,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask_BT,
                    spatial_merge_size=cfg.vision.spatial_merge_size,
                    image_token_id=cfg.image_token_id,
                    video_token_id=cfg.video_token_id,
                    vision_start_token_id=cfg.vision_start_token_id,
                )
            else:
                positions_BT = jnp.cumsum(attention_mask_BT, axis=-1) - 1
                positions_BT = jnp.where(attention_mask_BT, positions_BT, 0)
                position_ids_ZBT = jnp.stack([positions_BT, positions_BT, positions_BT], axis=0)

        position_ids_ZBT = jnp.asarray(position_ids_ZBT)
        text_position_ids_BT = position_ids_ZBT[0]

        sin_BTK, cos_BTK = compute_mrope_pos_embeddings(position_ids_ZBT, cfg.head_dim, cfg.rope_theta, cfg.mrope_section)
        sin_BTK = sin_BTK.astype(cfg.dtype)
        cos_BTK = cos_BTK.astype(cfg.dtype)

        causal_mask_BTS = text_position_ids_BT[:, :, None] >= text_position_ids_BT[:, None, :]
        padding_mask_B1T = attention_mask_BT[:, None, :].astype(jnp.bool_)
        final_mask_BTS = causal_mask_BTS & padding_mask_B1T

        hidden_BTD = inputs_embeds_BTD
        aux_losses = []
        for layer_idx, layer in enumerate(self.text.layers):
            hidden_BTD, aux_loss = layer(hidden_BTD, sin_BTK, cos_BTK, final_mask_BTS)
            aux_losses.append(aux_loss)
            if deepstack_features is not None and layer_idx < len(deepstack_features):
                hidden_BTD = _deepstack_process(hidden_BTD, visual_pos_mask_BT, deepstack_features[layer_idx])

        logits_BTV = self.lm_head(self.text.final_norm(hidden_BTD))
        if self.cfg.num_experts > 0:
            total_aux = jnp.sum(jnp.stack(aux_losses)) if aux_losses else jnp.array(0.0, dtype=jnp.float32)
            return logits_BTV, total_aux
        return logits_BTV


def _deepstack_process(
    hidden_BTD: jax.Array, visual_pos_mask_BT: jax.Array, visual_embeds_ND: jax.Array
) -> jax.Array:
    """Add visual embeddings to hidden states at visual token positions."""
    batch_idx, seq_idx = jnp.where(visual_pos_mask_BT)
    current_vals = hidden_BTD[batch_idx, seq_idx]
    new_vals = current_vals + visual_embeds_ND.astype(current_vals.dtype)
    return hidden_BTD.at[batch_idx, seq_idx].set(new_vals)
