"""Qwen3-VL composite model: vision encoder + text decoder with M-RoPE and DeepStack."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .config import Qwen3VLConfig
from .vision import VisionModel

_K_MASK: float = float(jnp.finfo(jnp.float32).min)


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float, *, rngs: nnx.Rngs):
        self.scale = nnx.Param(jnp.ones(dim))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype
        variance = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return (self.scale[...] * x).astype(dtype)


def apply_rope(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    """Apply rotary position embedding.

    Args:
        x: (batch, seq_len, num_heads, head_dim)
        sin, cos: (batch, seq_len, head_dim // 2)
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    sin = sin[:, :, None, :]
    cos = cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def compute_mrope_pos_embeddings(
    position_ids: jax.Array, head_dim: int, rope_theta: float, mrope_section: tuple[int, ...]
) -> tuple[jax.Array, jax.Array]:
    """Compute M-RoPE positional embeddings with interleaved frequency layout.

    Args:
        position_ids: (3, batch, seq_len)
        head_dim: int
        rope_theta: float
        mrope_section: tuple of 3 ints summing to head_dim // 2

    Returns:
        sin, cos: (batch, seq_len, head_dim // 2)
    """
    dim = head_dim
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    freqs = jnp.einsum("dbs,h->dbsh", position_ids.astype(jnp.float32), inv_freq)

    freqs_t = freqs[0]
    h_indices = np.arange(1, mrope_section[1] * 3, 3)
    w_indices = np.arange(2, mrope_section[2] * 3, 3)
    freqs_t = freqs_t.at[:, :, h_indices].set(freqs[1][:, :, h_indices])
    freqs_t = freqs_t.at[:, :, w_indices].set(freqs[2][:, :, w_indices])

    return jnp.sin(freqs_t), jnp.cos(freqs_t)


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
        linear = partial(nnx.Linear, use_bias=False, rngs=rngs, dtype=jnp.float32)
        self.gate_proj = linear(cfg.emb_dim, cfg.mlp_dim)
        self.up_proj = linear(cfg.emb_dim, cfg.mlp_dim)
        self.down_proj = linear(cfg.mlp_dim, cfg.emb_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(nnx.silu(self.gate_proj(x)) * self.up_proj(x))


class TextAttention(nnx.Module):
    def __init__(self, cfg: Qwen3VLConfig, *, rngs: nnx.Rngs):
        linear = partial(nnx.Linear, use_bias=False, rngs=rngs, dtype=jnp.float32)
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

    def __call__(self, x: jax.Array, sin: jax.Array, cos: jax.Array, mask: jax.Array | None) -> jax.Array:
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, t, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(b, t, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).reshape(b, t, self.num_kv_heads, self.head_dim)

        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        q_gqa = q.reshape(b, t, self.num_kv_heads, self.n_rep, self.head_dim)
        attn = jnp.einsum("BTKGH,BSKH->BTSKG", q_gqa, k, precision=jax.lax.Precision.HIGHEST) * self.scale

        if mask is not None:
            attn_mask = mask[:, :, :, None, None]
            attn = jnp.where(attn_mask, attn, jnp.full_like(attn, _K_MASK))

        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=2).astype(attn.dtype)
        out = jnp.einsum("BTSKG,BSKH->BTKGH", attn, v, precision=jax.lax.Precision.HIGHEST)
        out = out.reshape(b, t, self.num_heads, self.head_dim)
        return self.o_proj(out.reshape(b, t, self.num_heads * self.head_dim))


class TextDecoderLayer(nnx.Module):
    def __init__(self, cfg: Qwen3VLConfig, *, rngs: nnx.Rngs):
        self.input_layernorm = RMSNorm(cfg.emb_dim, cfg.norm_eps, rngs=rngs)
        self.attn = TextAttention(cfg, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim, cfg.norm_eps, rngs=rngs)
        self.mlp = TextMLP(cfg, rngs=rngs)

    def __call__(self, x: jax.Array, sin: jax.Array, cos: jax.Array, mask: jax.Array | None) -> jax.Array:
        x = x + self.attn(self.input_layernorm(x), sin, cos, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TextModel(nnx.Module):
    def __init__(self, cfg: Qwen3VLConfig, *, rngs: nnx.Rngs):
        self.embedder = nnx.Embed(
            num_embeddings=cfg.vocab_size, features=cfg.emb_dim, dtype=jnp.float32, rngs=rngs
        )
        self.layers = nnx.List([TextDecoderLayer(cfg, rngs=rngs) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.emb_dim, cfg.norm_eps, rngs=rngs)


class Qwen3VL(nnx.Module):
    def __init__(self, cfg: Qwen3VLConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.vision = VisionModel(cfg.vision, rngs=rngs)
        self.text = TextModel(cfg, rngs=rngs)
        self.lm_head = nnx.Linear(cfg.emb_dim, cfg.vocab_size, use_bias=False, rngs=rngs, dtype=jnp.float32)

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array | np.ndarray | None = None,
        pixel_values: jax.Array | None = None,
        image_grid_thw: jax.Array | None = None,
    ) -> jax.Array:
        cfg = self.cfg

        # 1. Vision encoding
        image_features = None
        deepstack_features = None
        visual_pos_mask = None
        if pixel_values is not None and image_grid_thw is not None:
            image_features, deepstack_features = self.vision(pixel_values, image_grid_thw)

        # 2. Text embedding
        inputs_embeds = self.text.embedder(input_ids)

        # 3. Scatter image features into embeddings
        if image_features is not None:
            image_mask = input_ids == cfg.image_token_id
            visual_pos_mask = image_mask
            batch_idx, seq_idx = jnp.where(image_mask)
            inputs_embeds = inputs_embeds.at[batch_idx, seq_idx].set(
                image_features.astype(inputs_embeds.dtype)
            )

        # 4. Compute position IDs if not provided
        if position_ids is None:
            if image_grid_thw is not None:
                position_ids, _ = get_rope_index(
                    input_ids,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                    spatial_merge_size=cfg.vision.spatial_merge_size,
                    image_token_id=cfg.image_token_id,
                    video_token_id=cfg.video_token_id,
                    vision_start_token_id=cfg.vision_start_token_id,
                )
            else:
                positions = jnp.cumsum(attention_mask, axis=-1) - 1
                positions = jnp.where(attention_mask, positions, 0)
                position_ids = jnp.stack([positions, positions, positions], axis=0)

        position_ids = jnp.asarray(position_ids)
        text_position_ids = position_ids[0]

        # 5. M-RoPE
        sin, cos = compute_mrope_pos_embeddings(position_ids, cfg.head_dim, cfg.rope_theta, cfg.mrope_section)

        # 6. Standard causal mask based on sequential position (not M-RoPE position IDs)
        seq_len = inputs_embeds.shape[1]
        seq_pos = jnp.arange(seq_len)
        causal_mask = seq_pos[:, None] >= seq_pos[None, :]
        padding_mask = attention_mask[:, None, :].astype(jnp.bool_)
        final_mask = causal_mask[None, :, :] & padding_mask

        # 7. Text decoder with deepstack
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.text.layers):
            hidden_states = layer(hidden_states, sin, cos, final_mask)
            if deepstack_features is not None and layer_idx < len(deepstack_features):
                hidden_states = _deepstack_process(hidden_states, visual_pos_mask, deepstack_features[layer_idx])

        # 8. Final norm + LM head
        return self.lm_head(self.text.final_norm(hidden_states))


def _deepstack_process(
    hidden_states: jax.Array, visual_pos_mask: jax.Array, visual_embeds: jax.Array
) -> jax.Array:
    """Add visual embeddings to hidden states at visual token positions."""
    batch_idx, seq_idx = jnp.where(visual_pos_mask)
    current_vals = hidden_states[batch_idx, seq_idx]
    new_vals = current_vals + visual_embeds.astype(current_vals.dtype)
    return hidden_states.at[batch_idx, seq_idx].set(new_vals)
