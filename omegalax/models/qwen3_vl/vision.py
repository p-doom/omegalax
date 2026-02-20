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
        x_norm = (x - mean) * jax.lax.rsqrt(var + self.eps)
        return self.scale[...] * x_norm + self.bias[...]


class VisionPatchEmbed(nnx.Module):
    """Conv3D patch embedding, represented as a linear layer over flattened patches."""

    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        in_features = config.in_channels * config.temporal_patch_size * config.patch_size**2
        self.proj = nnx.Linear(in_features, config.hidden_size, use_bias=True, rngs=rngs, dtype=jnp.float32)
        self.in_features = in_features

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape(-1, self.in_features)
        return self.proj(x)


class VisionMLP(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=True, rngs=rngs, dtype=jnp.float32)
        self.fc2 = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=True, rngs=rngs, dtype=jnp.float32)
        self.act_fn = lambda x: jax.nn.gelu(x, approximate=True)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(self.act_fn(self.fc1(x)))


def _rotate_half(x: jax.Array) -> jax.Array:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(
    q: jax.Array, k: jax.Array, cos: jax.Array, sin: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Apply 2D rotary embeddings to vision query/key.

    Uses full-size cos/sin (head_dim), matching HF's convention where
    emb = cat([freqs, freqs]) before taking cos/sin.

    Args:
        q, k: (seq_len, num_heads, head_dim)
        cos, sin: (seq_len, head_dim)
    """
    orig_dtype = q.dtype
    q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    cos = cos[:, None, :].astype(jnp.float32)
    sin = sin[:, None, :].astype(jnp.float32)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot.astype(orig_dtype), k_rot.astype(orig_dtype)


class VisionAttention(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nnx.Linear(config.hidden_size, config.hidden_size * 3, use_bias=True, rngs=rngs, dtype=jnp.float32)
        self.proj = nnx.Linear(config.hidden_size, config.hidden_size, use_bias=True, rngs=rngs, dtype=jnp.float32)

    def __call__(self, x: jax.Array, cu_seqlens: list[int], cos: jax.Array, sin: jax.Array) -> jax.Array:
        seq_len = x.shape[0]
        qkv = self.qkv(x).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        outputs = []
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            q_c, k_c, v_c = q[start:end], k[start:end], v[start:end]
            attn = jnp.einsum("snh,tnh->nst", q_c, k_c) * self.scale
            attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q_c.dtype)
            out = jnp.einsum("nst,tnh->snh", attn, v_c)
            outputs.append(out)

        result = jnp.concatenate(outputs, axis=0).reshape(seq_len, -1)
        return self.proj(result)


class VisionBlock(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.norm1 = LayerNorm(config.hidden_size, eps=1e-6, rngs=rngs)
        self.norm2 = LayerNorm(config.hidden_size, eps=1e-6, rngs=rngs)
        self.attn = VisionAttention(config, rngs=rngs)
        self.mlp = VisionMLP(config, rngs=rngs)

    def __call__(self, x: jax.Array, cu_seqlens: list[int], cos: jax.Array, sin: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x), cu_seqlens, cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionPatchMerger(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, use_postshuffle_norm: bool = False, rngs: nnx.Rngs):
        hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.hidden_size = hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = LayerNorm(norm_dim, eps=1e-6, rngs=rngs)
        self.fc1 = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs, dtype=jnp.float32)
        self.fc2 = nnx.Linear(hidden_size, config.out_hidden_size, use_bias=True, rngs=rngs, dtype=jnp.float32)

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape(-1, self.hidden_size))
        else:
            x = self.norm(x).reshape(-1, self.hidden_size)
        # FIXME (f.srambical):  we should probably approximate the gelu for increased throughput,
        # even if that deviates from huggingface numerics
        return self.fc2(jax.nn.gelu(self.fc1(x), approximate=False))


class VisionModel(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = VisionPatchEmbed(config, rngs=rngs)
        self.pos_embed = nnx.Embed(
            num_embeddings=config.num_position_embeddings, features=config.hidden_size, dtype=jnp.float32, rngs=rngs
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
        pos_weight = self.pos_embed.embedding[...]
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

            pos = (
                pos_weight[idx_ff] * w_ff[:, None]
                + pos_weight[idx_fc] * w_fc[:, None]
                + pos_weight[idx_cf] * w_cf[:, None]
                + pos_weight[idx_cc] * w_cc[:, None]
            )
            pos = jnp.tile(pos, (t, 1))
            pos = pos.reshape(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            pos = pos.transpose(0, 1, 3, 2, 4, 5).reshape(-1, pos.shape[-1])
            all_pos.append(pos)

        return jnp.concatenate(all_pos, axis=0)

    def __call__(
        self, pixel_values: jax.Array, grid_thw: jax.Array
    ) -> tuple[jax.Array, list[jax.Array]]:
        """Forward pass for vision encoder.

        Args:
            pixel_values: flat pixel tensor
            grid_thw: (num_images, 3) - temporal, height, width per image

        Returns:
            merged_features: (total_merged_tokens, out_hidden_size)
            deepstack_features: list of (total_merged_tokens, out_hidden_size)
        """
        grid_thw_list = grid_thw.tolist() if hasattr(grid_thw, "tolist") else [[int(v) for v in row] for row in grid_thw]

        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = self._interpolate_pos_embed(grid_thw_list)
        hidden_states = hidden_states + pos_embeds

        rotary_emb = self._compute_rotary_pos_emb(grid_thw_list)
        emb = jnp.concatenate([rotary_emb, rotary_emb], axis=-1)
        cos, sin = jnp.cos(emb), jnp.sin(emb)

        # cu_seqlens from grid_thw
        cu_seqlens = [0]
        for t_val, h_val, w_val in grid_thw_list:
            t_val, h_val, w_val = int(t_val), int(h_val), int(w_val)
            for _ in range(t_val):
                cu_seqlens.append(cu_seqlens[-1] + h_val * w_val)

        deepstack_features: list[jax.Array] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens, cos, sin)
            if layer_num in self.deepstack_visual_indexes:
                idx = list(self.deepstack_visual_indexes).index(layer_num)
                deepstack_features.append(self.deepstack_mergers[idx](hidden_states))

        merged = self.merger(hidden_states)
        return merged, deepstack_features
