"""Qwen3.5 Vision Encoder.

Implements the ViT-style vision encoder with 3-D patch embedding,
rotary position embeddings, spatial merge, and bilinear position
embedding interpolation.
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from .config import Qwen3_5VisionConfig
from .norms import LayerNorm
from .rope import apply_vision_rope, generate_vision_rope

_K_MASK: float = float(jnp.finfo(jnp.float32).min)


class VisionPatchEmbed(nnx.Module):
    """3-D Conv patch embedding (temporal, H, W)."""

    def __init__(self, cfg: Qwen3_5VisionConfig, *, rngs: nnx.Rngs):
        k = (cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
        self.proj = nnx.Conv(
            in_features=cfg.in_channels,
            out_features=cfg.hidden_size,
            kernel_size=k,
            strides=k,
            use_bias=True,
            rngs=rngs,
        )
        self.in_channels = cfg.in_channels
        self.temporal_patch_size = cfg.temporal_patch_size
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.hidden_size

    @jax.named_scope("vision_patch_embed")
    def __call__(self, x: jax.Array) -> jax.Array:
        """Args: x — flattened pixel patches (num_patches, C * tp * p * p)."""
        x = x.reshape(-1, self.temporal_patch_size, self.patch_size, self.patch_size, self.in_channels)
        x = self.proj(x)  # (N, 1, 1, 1, embed_dim)
        return x.reshape(-1, self.embed_dim)


class VisionMLP(nnx.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig, *, rngs: nnx.Rngs):
        linear = partial(nnx.Linear, use_bias=True, rngs=rngs)
        self.fc1 = linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = linear(cfg.intermediate_size, cfg.hidden_size)

    @jax.named_scope("vision_mlp")
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(nnx.gelu(self.fc1(x), approximate=True))


class VisionAttention(nnx.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig, *, rngs: nnx.Rngs):
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nnx.Linear(cfg.hidden_size, cfg.hidden_size * 3, use_bias=True, rngs=rngs)
        self.proj = nnx.Linear(cfg.hidden_size, cfg.hidden_size, use_bias=True, rngs=rngs)

    @jax.named_scope("vision_attention")
    def __call__(
        self,
        x: jax.Array,
        cu_seqlens: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
    ) -> jax.Array:
        """Args:
            x: (seq_len, dim)
            cu_seqlens: (num_seqs + 1,) cumulative lengths
            cos, sin: (seq_len, head_dim)
        """
        seq_len = x.shape[0]
        qkv = self.qkv(x).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (S, H, D)

        q, k = apply_vision_rope(q, k, cos, sin)

        # Process each sub-sequence independently
        num_seqs = cu_seqlens.shape[0] - 1
        outputs = jnp.zeros_like(q.reshape(seq_len, -1))

        def _attn_chunk(start, end):
            qi = q[start:end]  # (L, H, D)
            ki = k[start:end]
            vi = v[start:end]
            # (H, L, D) @ (H, D, L) → (H, L, L)
            qi_t = qi.transpose(1, 0, 2)
            ki_t = ki.transpose(1, 0, 2)
            vi_t = vi.transpose(1, 0, 2)
            attn_w = jnp.matmul(qi_t, ki_t.transpose(0, 2, 1)) * self.scale
            attn_w = jax.nn.softmax(attn_w.astype(jnp.float32), axis=-1).astype(attn_w.dtype)
            out = jnp.matmul(attn_w, vi_t)  # (H, L, D)
            return out.transpose(1, 0, 2).reshape(-1, self.num_heads * self.head_dim)

        # Static unrolling is not possible; use lax.fori_loop with dynamic_update_slice.
        def body_fn(i, out):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            length = end - start
            chunk_out = _attn_chunk(start, end)
            return jax.lax.dynamic_update_slice(out, chunk_out, (start, 0))

        outputs = jax.lax.fori_loop(0, num_seqs, body_fn, outputs)
        return self.proj(outputs)


class VisionBlock(nnx.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig, *, rngs: nnx.Rngs):
        self.norm1 = LayerNorm(cfg.hidden_size, cfg.norm_eps, rngs=rngs)
        self.norm2 = LayerNorm(cfg.hidden_size, cfg.norm_eps, rngs=rngs)
        self.attn = VisionAttention(cfg, rngs=rngs)
        self.mlp = VisionMLP(cfg, rngs=rngs)

    def __call__(self, x, cu_seqlens, cos, sin):
        x = x + self.attn(self.norm1(x), cu_seqlens, cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionPatchMerger(nnx.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig, *, rngs: nnx.Rngs):
        merged_dim = cfg.hidden_size * (cfg.spatial_merge_size ** 2)
        self.norm = LayerNorm(cfg.hidden_size, cfg.norm_eps, rngs=rngs)
        self.fc1 = nnx.Linear(merged_dim, merged_dim, use_bias=True, rngs=rngs)
        self.fc2 = nnx.Linear(merged_dim, cfg.out_hidden_size, use_bias=True, rngs=rngs)

    @jax.named_scope("vision_merger")
    def __call__(self, x: jax.Array, merge_size: int) -> jax.Array:
        merged_dim = x.shape[-1] * merge_size * merge_size
        x = self.norm(x)
        x = x.reshape(-1, merged_dim)
        return self.fc2(nnx.gelu(self.fc1(x), approximate=True))


class VisionModel(nnx.Module):
    """Full Qwen3.5 vision encoder."""

    def __init__(self, cfg: Qwen3_5VisionConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.patch_embed = VisionPatchEmbed(cfg, rngs=rngs)
        self.pos_embed = nnx.Embed(
            num_embeddings=cfg.num_position_embeddings, features=cfg.hidden_size, rngs=rngs,
        )
        self.num_grid_per_side = int(cfg.num_position_embeddings ** 0.5)
        head_dim = cfg.hidden_size // cfg.num_heads
        self.rotary_half_dim = head_dim // 2
        self.blocks = nnx.List([VisionBlock(cfg, rngs=rngs) for _ in range(cfg.depth)])
        self.merger = VisionPatchMerger(cfg, rngs=rngs)

    def _rot_pos_emb(self, grid_thw: jax.Array) -> jax.Array:
        """Build per-token 2-D rotary embeddings from grid info."""
        merge_size = self.cfg.spatial_merge_size
        max_hw = int(jnp.max(grid_thw[:, 1:]))
        freq_table = generate_vision_rope(max_hw, self.rotary_half_dim * 2)

        all_pos_ids = []
        for idx in range(grid_thw.shape[0]):
            t, h, w = int(grid_thw[idx, 0]), int(grid_thw[idx, 1]), int(grid_thw[idx, 2])
            merged_h, merged_w = h // merge_size, w // merge_size

            # Compute full-resolution positions within merged blocks
            block_rows = jnp.arange(merged_h)
            block_cols = jnp.arange(merged_w)
            intra_row = jnp.arange(merge_size)
            intra_col = jnp.arange(merge_size)

            row_idx = (block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None])
            col_idx = (block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :])

            row_idx = jnp.broadcast_to(row_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)
            col_idx = jnp.broadcast_to(col_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)

            coords = jnp.stack([row_idx, col_idx], axis=-1)
            if t > 1:
                coords = jnp.tile(coords, (t, 1))
            all_pos_ids.append(coords)

        pos_ids = jnp.concatenate(all_pos_ids, axis=0)
        embeddings = freq_table[pos_ids]  # (total, 2, half_dim)
        return embeddings.reshape(embeddings.shape[0], -1)  # (total, head_dim)

    def _fast_pos_embed_interpolate(self, grid_thw: jax.Array) -> jax.Array:
        """Bilinear position embedding interpolation."""
        merge_size = self.cfg.spatial_merge_size
        pos_weight = self.pos_embed.embedding[...]
        n = self.num_grid_per_side

        all_embeds = []
        for idx in range(grid_thw.shape[0]):
            t, h, w = int(grid_thw[idx, 0]), int(grid_thw[idx, 1]), int(grid_thw[idx, 2])
            h_idxs = jnp.linspace(0, n - 1, h)
            w_idxs = jnp.linspace(0, n - 1, w)

            h_floor = jnp.floor(h_idxs).astype(jnp.int32)
            w_floor = jnp.floor(w_idxs).astype(jnp.int32)
            h_ceil = jnp.clip(h_floor + 1, min=0, max=n - 1)
            w_ceil = jnp.clip(w_floor + 1, min=0, max=n - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # Compute 4 corners (bilinear)
            idx00 = (h_floor[:, None] * n + w_floor[None, :]).reshape(-1)
            idx01 = (h_floor[:, None] * n + w_ceil[None, :]).reshape(-1)
            idx10 = (h_ceil[:, None] * n + w_floor[None, :]).reshape(-1)
            idx11 = (h_ceil[:, None] * n + w_ceil[None, :]).reshape(-1)

            w00 = ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1)
            w01 = ((1 - dh)[:, None] * dw[None, :]).reshape(-1)
            w10 = (dh[:, None] * (1 - dw)[None, :]).reshape(-1)
            w11 = (dh[:, None] * dw[None, :]).reshape(-1)

            embed = (
                pos_weight[idx00] * w00[:, None]
                + pos_weight[idx01] * w01[:, None]
                + pos_weight[idx10] * w10[:, None]
                + pos_weight[idx11] * w11[:, None]
            )
            # Repeat for temporal frames, then permute for spatial merge
            embed = jnp.tile(embed, (t, 1))  # (t*h*w, D)
            embed = embed.reshape(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            embed = embed.transpose(0, 1, 3, 2, 4, 5).reshape(-1, embed.shape[-1])
            all_embeds.append(embed)

        return jnp.concatenate(all_embeds, axis=0)

    @jax.named_scope("vision_model")
    def __call__(self, hidden_states: jax.Array, grid_thw: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            hidden_states: flattened pixel patches
            grid_thw: (num_images, 3) — temporal, height, width
        Returns:
            merged_features: (num_merged_tokens, out_hidden_size)
        """
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self._fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_emb = self._rot_pos_emb(grid_thw)
        emb = jnp.concatenate([rotary_emb, rotary_emb], axis=-1)
        cos, sin = jnp.cos(emb), jnp.sin(emb)

        cu_seqlens = jnp.concatenate([
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(
                jnp.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
            ).astype(jnp.int32),
        ])

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens, cos, sin)

        return self.merger(hidden_states, self.cfg.spatial_merge_size)
