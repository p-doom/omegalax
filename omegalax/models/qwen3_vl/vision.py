"""Qwen3-VL vision encoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .config import Qwen3VLVisionConfig


class LayerNorm(nnx.Module):
    """Standard LayerNorm (weight + bias)."""

    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.scale = nnx.Param(jnp.ones(dim))
        self.bias = nnx.Param(jnp.zeros(dim))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        normed = (x - mean) * jax.lax.rsqrt(var + self.eps)
        return self.scale[...] * normed + self.bias[...]


class VisionPatchEmbed(nnx.Module):
    """Conv3D patch embedding, represented as a linear layer over flattened patches."""

    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        in_features = config.in_channels * config.temporal_patch_size * config.patch_size**2
        self.proj = nnx.Linear(in_features, config.hidden_size, use_bias=True, rngs=rngs, dtype=config.dtype)
        self.in_features = in_features

    def __call__(self, pixels: jax.Array) -> jax.Array:
        flat = pixels.reshape(-1, self.in_features)
        return self.proj(flat)


class VisionMLP(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=True, rngs=rngs, dtype=config.dtype)
        self.fc2 = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=True, rngs=rngs, dtype=config.dtype)
        self.act_fn = lambda x: jax.nn.gelu(x, approximate=True)

    def __call__(self, hidden_ND: jax.Array) -> jax.Array:
        return self.fc2(self.act_fn(self.fc1(hidden_ND)))


def _rotate_half(x: jax.Array) -> jax.Array:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(
    q_NHK: jax.Array, k_NHK: jax.Array, cos_NK: jax.Array, sin_NK: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Apply 2D rotary embeddings to vision query/key.

    Args:
        q_NHK, k_NHK: (seq_len, num_heads, head_dim)
        cos_NK, sin_NK: (seq_len, head_dim)
    """
    orig_dtype = q_NHK.dtype
    q_NHK, k_NHK = q_NHK.astype(jnp.float32), k_NHK.astype(jnp.float32)
    cos_NK = cos_NK[:, None, :].astype(jnp.float32)
    sin_NK = sin_NK[:, None, :].astype(jnp.float32)
    q_rot_NHK = (q_NHK * cos_NK) + (_rotate_half(q_NHK) * sin_NK)
    k_rot_NHK = (k_NHK * cos_NK) + (_rotate_half(k_NHK) * sin_NK)
    return q_rot_NHK.astype(orig_dtype), k_rot_NHK.astype(orig_dtype)


class VisionAttention(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nnx.Linear(config.hidden_size, config.hidden_size * 3, use_bias=True, rngs=rngs, dtype=config.dtype)
        self.proj = nnx.Linear(config.hidden_size, config.hidden_size, use_bias=True, rngs=rngs, dtype=config.dtype)

    def __call__(self, hidden_ND: jax.Array, cu_seqlens: list[int], cos_NK: jax.Array, sin_NK: jax.Array) -> jax.Array:
        N = hidden_ND.shape[0]
        qkv = self.qkv(hidden_ND).reshape(N, 3, self.num_heads, self.head_dim)
        q_NHK, k_NHK, v_NHK = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q_NHK, k_NHK = apply_rotary_pos_emb_vision(q_NHK, k_NHK, cos_NK, sin_NK)

        outputs = []
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            q_c, k_c, v_c = q_NHK[start:end], k_NHK[start:end], v_NHK[start:end]
            logits_HNN = jnp.einsum("NHK,MHK->HNM", q_c, k_c) * self.scale
            weights_HNN = jax.nn.softmax(logits_HNN.astype(jnp.float32), axis=-1).astype(q_c.dtype)
            out_NHK = jnp.einsum("HNM,MHK->NHK", weights_HNN, v_c)
            outputs.append(out_NHK)

        result_ND = jnp.concatenate(outputs, axis=0).reshape(N, -1)
        return self.proj(result_ND)


class VisionBlock(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.norm1 = LayerNorm(config.hidden_size, eps=1e-6, rngs=rngs)
        self.norm2 = LayerNorm(config.hidden_size, eps=1e-6, rngs=rngs)
        self.attn = VisionAttention(config, rngs=rngs)
        self.mlp = VisionMLP(config, rngs=rngs)

    def __call__(self, hidden_ND: jax.Array, cu_seqlens: list[int], cos_NK: jax.Array, sin_NK: jax.Array) -> jax.Array:
        hidden_ND = hidden_ND + self.attn(self.norm1(hidden_ND), cu_seqlens, cos_NK, sin_NK)
        hidden_ND = hidden_ND + self.mlp(self.norm2(hidden_ND))
        return hidden_ND


class VisionPatchMerger(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, use_postshuffle_norm: bool = False, rngs: nnx.Rngs):
        hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.hidden_size = hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = LayerNorm(norm_dim, eps=1e-6, rngs=rngs)
        self.fc1 = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs, dtype=config.dtype)
        self.fc2 = nnx.Linear(hidden_size, config.out_hidden_size, use_bias=True, rngs=rngs, dtype=config.dtype)

    def __call__(self, hidden_ND: jax.Array) -> jax.Array:
        if self.use_postshuffle_norm:
            normed = self.norm(hidden_ND.reshape(-1, self.hidden_size))
        else:
            normed = self.norm(hidden_ND).reshape(-1, self.hidden_size)
        # FIXME (f.srambical):  we should probably approximate the gelu for increased throughput,
        # even if that deviates from huggingface numerics
        return self.fc2(jax.nn.gelu(self.fc1(normed), approximate=False))


class VisionModel(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = VisionPatchEmbed(config, rngs=rngs)
        self.pos_embed = nnx.Embed(
            num_embeddings=config.num_position_embeddings, features=config.hidden_size, dtype=config.dtype, rngs=rngs
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_dim = head_dim // 2
        self.rotary_theta = 10000.0
        self.blocks = nnx.List([VisionBlock(config, rngs=rngs) for _ in range(config.depth)])
        self.merger = VisionPatchMerger(config, use_postshuffle_norm=False, rngs=rngs)
        self.deepstack_mergers = nnx.List(
            [VisionPatchMerger(config, use_postshuffle_norm=True, rngs=rngs) for _ in config.deepstack_visual_indexes]
        )
        self.deepstack_visual_indexes = config.deepstack_visual_indexes

    def _compute_rotary_pos_emb(self, grid_thw_list: list[list[int]]) -> jax.Array:
        merge_size = self.spatial_merge_size
        max_hw = max(max(int(h), int(w)) for _, h, w in grid_thw_list)

        inv_freq = 1.0 / (self.rotary_theta ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim))
        seq = jnp.arange(max_hw, dtype=jnp.float32)
        freq_table = jnp.outer(seq, inv_freq)

        all_pos_ids = []
        for num_frames, height, width in grid_thw_list:
            num_frames, height, width = int(num_frames), int(height), int(width)
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = np.arange(merged_h)
            block_cols = np.arange(merged_w)
            intra_row = np.arange(merge_size)
            intra_col = np.arange(merge_size)
            row_idx = (block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None])
            col_idx = (block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :])
            row_idx = np.broadcast_to(row_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)
            col_idx = np.broadcast_to(col_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)
            coords = np.stack([row_idx, col_idx], axis=-1)
            if num_frames > 1:
                coords = np.tile(coords, (num_frames, 1))
            all_pos_ids.append(coords)

        pos_ids = jnp.array(np.concatenate(all_pos_ids, axis=0))
        embeddings = freq_table[pos_ids]
        return embeddings.reshape(pos_ids.shape[0], -1)

    def _interpolate_pos_embed(self, grid_thw_list: list[list[int]]) -> jax.Array:
        merge_size = self.spatial_merge_size
        pos_weight_VD = self.pos_embed.embedding[...]
        n = self.num_grid_per_side

        all_pos = []
        for t, h, w in grid_thw_list:
            t, h, w = int(t), int(h), int(w)
            h_idxs = jnp.linspace(0, n - 1, h)
            w_idxs = jnp.linspace(0, n - 1, w)

            h_floor = jnp.floor(h_idxs).astype(jnp.int32)
            w_floor = jnp.floor(w_idxs).astype(jnp.int32)
            h_ceil = jnp.minimum(h_floor + 1, n - 1)
            w_ceil = jnp.minimum(w_floor + 1, n - 1)
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            idx_ff = (h_floor[:, None] * n + w_floor[None, :]).reshape(-1)
            idx_fc = (h_floor[:, None] * n + w_ceil[None, :]).reshape(-1)
            idx_cf = (h_ceil[:, None] * n + w_floor[None, :]).reshape(-1)
            idx_cc = (h_ceil[:, None] * n + w_ceil[None, :]).reshape(-1)

            w_ff = ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1)
            w_fc = ((1 - dh)[:, None] * dw[None, :]).reshape(-1)
            w_cf = (dh[:, None] * (1 - dw)[None, :]).reshape(-1)
            w_cc = (dh[:, None] * dw[None, :]).reshape(-1)

            pos_ND = (
                pos_weight_VD[idx_ff] * w_ff[:, None]
                + pos_weight_VD[idx_fc] * w_fc[:, None]
                + pos_weight_VD[idx_cf] * w_cf[:, None]
                + pos_weight_VD[idx_cc] * w_cc[:, None]
            )
            pos_ND = jnp.tile(pos_ND, (t, 1))
            pos_ND = pos_ND.reshape(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            pos_ND = pos_ND.transpose(0, 1, 3, 2, 4, 5).reshape(-1, pos_ND.shape[-1])
            all_pos.append(pos_ND)

        return jnp.concatenate(all_pos, axis=0)

    def __call__(
        self, pixel_values: jax.Array, grid_thw: jax.Array
    ) -> tuple[jax.Array, list[jax.Array]]:
        grid_thw_list = grid_thw.tolist() if hasattr(grid_thw, "tolist") else [[int(v) for v in row] for row in grid_thw]

        hidden_ND = self.patch_embed(pixel_values)
        pos_embeds_ND = self._interpolate_pos_embed(grid_thw_list)
        hidden_ND = hidden_ND + pos_embeds_ND

        rotary_emb_NK = self._compute_rotary_pos_emb(grid_thw_list)
        emb_NK = jnp.concatenate([rotary_emb_NK, rotary_emb_NK], axis=-1)
        cos_NK, sin_NK = jnp.cos(emb_NK), jnp.sin(emb_NK)
        cos_NK = cos_NK.astype(self.config.dtype)
        sin_NK = sin_NK.astype(self.config.dtype)

        cu_seqlens = [0]
        for t_val, h_val, w_val in grid_thw_list:
            t_val, h_val, w_val = int(t_val), int(h_val), int(w_val)
            for _ in range(t_val):
                cu_seqlens.append(cu_seqlens[-1] + h_val * w_val)

        deepstack_features: list[jax.Array] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_ND = blk(hidden_ND, cu_seqlens, cos_NK, sin_NK)
            if layer_num in self.deepstack_visual_indexes:
                idx = list(self.deepstack_visual_indexes).index(layer_num)
                deepstack_features.append(self.deepstack_mergers[idx](hidden_ND))

        merged_ND = self.merger(hidden_ND)
        return merged_ND, deepstack_features
