"""Qwen3.5 Vision Encoder.

Implements the ViT-style vision encoder with 3-D patch embedding,
rotary position embeddings, spatial merge, and bilinear position
embedding interpolation.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard
from flax import nnx
from tokamax import dot_product_attention
from tokamax._src.ops.attention.base import Mask

from omegalax.models.shard_config import ShardConfig
from .config import Qwen3_5VisionConfig
from .norms import LayerNorm
from .rope import apply_vision_rope


def _token_spatial_coords(
    grid_thw: jax.Array, merge_size: int, total_tokens: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map each vision token to its (row, col) in the original spatial grid.

    Returns:
        row_coord, col_coord, image_id: each int32 of shape
        ``(total_tokens,)``.
    """
    tokens_per_image = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    cu_tokens = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(tokens_per_image).astype(jnp.int32)]
    )

    tok_idx = jnp.arange(total_tokens, dtype=jnp.int32)
    image_id = jnp.searchsorted(cu_tokens[1:], tok_idx, side="right")
    local_idx = tok_idx - cu_tokens[image_id]

    h = grid_thw[image_id, 1]
    w = grid_thw[image_id, 2]
    spatial_idx = local_idx % (h * w)

    merge_sq = merge_size * merge_size
    merged_w = w // merge_size
    group_idx = spatial_idx // merge_sq
    intra_idx = spatial_idx % merge_sq

    block_r = group_idx // merged_w
    block_c = group_idx % merged_w
    intra_r = intra_idx // merge_size
    intra_c = intra_idx % merge_size

    row_coord = block_r * merge_size + intra_r
    col_coord = block_c * merge_size + intra_c
    return row_coord, col_coord, image_id

wp = nnx.with_partitioning


class VisionPatchEmbed(nnx.Module):
    """3-D Conv patch embedding (temporal, H, W)."""

    def __init__(self, cfg: Qwen3_5VisionConfig, hidden_shd: P, *, rngs: nnx.Rngs):
        k = (cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
        conv_init = nnx.initializers.lecun_normal()
        self.proj = nnx.Conv(
            in_features=cfg.in_channels,
            out_features=cfg.hidden_size,
            kernel_size=k,
            strides=k,
            use_bias=True,
            rngs=rngs,
            kernel_init=wp(conv_init, (None, None, None, None, "hidden")),
        )
        self.in_channels = cfg.in_channels
        self.temporal_patch_size = cfg.temporal_patch_size
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.hidden_size
        self.hidden_shd = hidden_shd

    @jax.named_scope("vision_patch_embed")
    def __call__(self, pixels: jax.Array) -> jax.Array:
        """
        Args:
            pixels: flattened pixel patches (num_patches, C * tp * p * p).
        """
        patches = pixels.reshape(-1, self.temporal_patch_size, self.patch_size, self.patch_size, self.in_channels)
        embedded = self.proj(patches)
        return reshard(embedded.reshape(-1, self.embed_dim), self.hidden_shd)


class VisionMLP(nnx.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig, hidden_shd: P, ff_shd: P, *, rngs: nnx.Rngs):
        init = nnx.initializers.lecun_normal()
        self.fc1 = nnx.Linear(
            cfg.hidden_size,
            cfg.intermediate_size,
            use_bias=True,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=wp(init, (None, "hidden")),
        )
        self.fc2 = nnx.Linear(
            cfg.intermediate_size,
            cfg.hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=wp(init, ("hidden", None)),
        )
        self.hidden_shd = hidden_shd
        self.ff_shd = ff_shd

    @jax.named_scope("vision_mlp")
    def __call__(self, hidden_ND: jax.Array) -> jax.Array:
        ff_NF = self.fc1(hidden_ND, out_sharding=self.ff_shd)
        ff_NF = reshard(nnx.gelu(ff_NF, approximate=True), self.ff_shd)
        out_ND = self.fc2(ff_NF, out_sharding=self.hidden_shd)
        return reshard(out_ND, self.hidden_shd)


class VisionAttention(nnx.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig, hidden_shd: P, heads_shd: P, *, rngs: nnx.Rngs):
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.scale = self.head_dim ** -0.5
        init = nnx.initializers.lecun_normal()
        qkv_init = wp(init, (None, "hidden"))
        self.qkv = nnx.Linear(
            cfg.hidden_size,
            cfg.hidden_size * 3,
            use_bias=True,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=qkv_init,
        )
        self.hidden_shd = hidden_shd
        self.heads_shd = heads_shd
        self.proj = nnx.Linear(
            cfg.hidden_size,
            cfg.hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=qkv_init,
        )

    @jax.named_scope("vision_attention")
    def __call__(
        self,
        hidden_ND: jax.Array,
        cu_seqlens: jax.Array,
        cos_NK: jax.Array,
        sin_NK: jax.Array,
    ) -> jax.Array:
        hidden_ND = reshard(hidden_ND, self.hidden_shd)
        N = hidden_ND.shape[0]
        qkv = self.qkv(hidden_ND, out_sharding=self.hidden_shd).reshape(N, 3, self.num_heads, self.head_dim)
        q_NHK = reshard(qkv[:, 0], self.heads_shd)
        k_NHK = reshard(qkv[:, 1], self.heads_shd)
        v_NHK = reshard(qkv[:, 2], self.heads_shd)

        q_NHK, k_NHK = apply_vision_rope(q_NHK, k_NHK, cos_NK, sin_NK)

        _BLOCK = 128
        N_padded = (N + _BLOCK - 1) // _BLOCK * _BLOCK
        pad_n = N_padded - N

        if pad_n > 0:
            q_NHK = jnp.pad(q_NHK, ((0, pad_n), (0, 0), (0, 0)))
            k_NHK = jnp.pad(k_NHK, ((0, pad_n), (0, 0), (0, 0)))
            v_NHK = jnp.pad(v_NHK, ((0, pad_n), (0, 0), (0, 0)))

        seg_ids = jnp.searchsorted(cu_seqlens[1:], jnp.arange(N_padded), side="right")
        k_start = cu_seqlens[jnp.minimum(seg_ids, cu_seqlens.shape[0] - 2)]
        k_end = cu_seqlens[jnp.minimum(seg_ids + 1, cu_seqlens.shape[0] - 1)]
        is_pad = jnp.arange(N_padded) >= N
        k_start = jnp.where(is_pad, N_padded, k_start)
        k_end = jnp.where(is_pad, N_padded, k_end)

        attn_NHK = dot_product_attention(
            q_NHK[None], k_NHK[None], v_NHK[None],
            mask=Mask(k_start=k_start, k_end=k_end),
            scale=self.scale, is_causal=False, implementation="mosaic",
        )
        outputs_ND = attn_NHK[0, :N].reshape(N, -1)

        out_ND = self.proj(outputs_ND, out_sharding=self.hidden_shd)
        return reshard(out_ND, self.hidden_shd)


class VisionBlock(nnx.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig, hidden_shd: P, ff_shd: P, heads_shd: P, *, rngs: nnx.Rngs):
        self.norm1 = LayerNorm(cfg.hidden_size, 1e-6, rngs=rngs)
        self.norm2 = LayerNorm(cfg.hidden_size, 1e-6, rngs=rngs)
        self.attn = VisionAttention(cfg, hidden_shd=hidden_shd, heads_shd=heads_shd, rngs=rngs)
        self.mlp = VisionMLP(cfg, hidden_shd=hidden_shd, ff_shd=ff_shd, rngs=rngs)
        self.hidden_shd = hidden_shd

    @partial(jax.remat, static_argnums=0)
    def __call__(self, hidden_ND, cu_seqlens, cos_NK, sin_NK):
        hidden_ND = reshard(hidden_ND, self.hidden_shd)
        hidden_ND = reshard(hidden_ND + self.attn(self.norm1(hidden_ND), cu_seqlens, cos_NK, sin_NK), self.hidden_shd)
        hidden_ND = reshard(hidden_ND + self.mlp(self.norm2(hidden_ND)), self.hidden_shd)
        return hidden_ND


class VisionPatchMerger(nnx.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig, hidden_shd: P, ff_shd: P, *, rngs: nnx.Rngs):
        merged_dim = cfg.hidden_size * (cfg.spatial_merge_size ** 2)
        self.norm = LayerNorm(cfg.hidden_size, 1e-6, rngs=rngs)
        init = nnx.initializers.lecun_normal()
        self.fc1 = nnx.Linear(
            merged_dim,
            merged_dim,
            use_bias=True,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=wp(init, (None, None)),
        )
        self.fc2 = nnx.Linear(
            merged_dim,
            cfg.out_hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=wp(init, (None, "hidden")),
        )
        self.hidden_shd = hidden_shd
        self.ff_shd = ff_shd

    @jax.named_scope("vision_merger")
    def __call__(self, hidden_ND: jax.Array, merge_size: int) -> jax.Array:
        hidden_ND = reshard(hidden_ND, self.hidden_shd)
        merged_dim = hidden_ND.shape[-1] * merge_size * merge_size
        normed = self.norm(hidden_ND)
        normed = jax.lax.reshape(
            normed,
            (normed.shape[0] // (merge_size * merge_size), merged_dim),
            out_sharding=self.hidden_shd,
        )
        ff_NF = self.fc1(normed, out_sharding=self.ff_shd)
        ff_NF = reshard(nnx.gelu(ff_NF, approximate=True), self.ff_shd)
        out_ND = self.fc2(ff_NF, out_sharding=self.hidden_shd)
        return reshard(out_ND, self.hidden_shd)


class VisionModel(nnx.Module):
    """Full Qwen3.5 vision encoder."""

    def __init__(self, cfg: Qwen3_5VisionConfig, shd_cfg: ShardConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.hidden_shd = P(shd_cfg.act_btd[0], shd_cfg.act_btd[2])
        self.ff_shd = P(shd_cfg.act_btd[0], shd_cfg.act_btf[2])
        self.heads_shd = P(shd_cfg.act_btd[0], shd_cfg.act_btnh[2], None)
        self.patch_embed = VisionPatchEmbed(cfg, hidden_shd=self.hidden_shd, rngs=rngs)
        pos_init = nnx.initializers.normal(stddev=0.02)
        self.pos_embed = nnx.Embed(
            num_embeddings=cfg.num_position_embeddings,
            features=cfg.hidden_size,
            rngs=rngs,
            dtype=cfg.dtype,
            embedding_init=wp(pos_init, (None, "hidden")),
        )
        self.num_grid_per_side = int(cfg.num_position_embeddings ** 0.5)
        head_dim = cfg.hidden_size // cfg.num_heads
        self.rotary_half_dim = head_dim // 2
        self.blocks = nnx.List(
            [VisionBlock(cfg, hidden_shd=self.hidden_shd, ff_shd=self.ff_shd, heads_shd=self.heads_shd, rngs=rngs) for _ in range(cfg.depth)]
        )
        self.merger = VisionPatchMerger(cfg, hidden_shd=self.hidden_shd, ff_shd=self.ff_shd, rngs=rngs)

    def _rot_pos_emb(self, grid_thw: jax.Array, total_tokens: int) -> jax.Array:
        """Build per-token 2-D rotary embeddings from grid info."""
        row, col, _ = _token_spatial_coords(grid_thw, self.cfg.spatial_merge_size, total_tokens)
        inv_freq = 1.0 / (
            10000.0
            ** (jnp.arange(0, self.rotary_half_dim, 2, dtype=jnp.float32) / self.rotary_half_dim)
        )
        row_emb = row[:, None].astype(jnp.float32) * inv_freq[None, :]
        col_emb = col[:, None].astype(jnp.float32) * inv_freq[None, :]
        return jnp.concatenate([row_emb, col_emb], axis=-1)

    def _fast_pos_embed_interpolate(self, grid_thw: jax.Array, total_tokens: int) -> jax.Array:
        """Bilinear position embedding interpolation."""
        row, col, img_id = _token_spatial_coords(grid_thw, self.cfg.spatial_merge_size, total_tokens)
        pos_weight_VD = self.pos_embed.embedding[...]
        n = self.num_grid_per_side

        h = grid_thw[img_id, 1].astype(jnp.float32)
        w = grid_thw[img_id, 2].astype(jnp.float32)

        h_idx = row.astype(jnp.float32) * (n - 1) / jnp.maximum(h - 1.0, 1.0)
        w_idx = col.astype(jnp.float32) * (n - 1) / jnp.maximum(w - 1.0, 1.0)

        h_floor = jnp.floor(h_idx).astype(jnp.int32)
        w_floor = jnp.floor(w_idx).astype(jnp.int32)
        h_ceil = jnp.minimum(h_floor + 1, n - 1)
        w_ceil = jnp.minimum(w_floor + 1, n - 1)
        dh = h_idx - h_floor.astype(jnp.float32)
        dw = w_idx - w_floor.astype(jnp.float32)

        idx_ff = h_floor * n + w_floor
        idx_fc = h_floor * n + w_ceil
        idx_cf = h_ceil * n + w_floor
        idx_cc = h_ceil * n + w_ceil

        w_ff = (1.0 - dh) * (1.0 - dw)
        w_fc = (1.0 - dh) * dw
        w_cf = dh * (1.0 - dw)
        w_cc = dh * dw

        return (
            pos_weight_VD[idx_ff] * w_ff[:, None]
            + pos_weight_VD[idx_fc] * w_fc[:, None]
            + pos_weight_VD[idx_cf] * w_cf[:, None]
            + pos_weight_VD[idx_cc] * w_cc[:, None]
        )

    @jax.named_scope("vision_model")
    def __call__(self, pixel_values: jax.Array, grid_thw: jax.Array) -> jax.Array:
        hidden_ND = self.patch_embed(pixel_values)
        total_tokens: int = hidden_ND.shape[0]

        pos_embeds_ND = self._fast_pos_embed_interpolate(grid_thw, total_tokens)
        hidden_ND = reshard(hidden_ND + pos_embeds_ND, self.hidden_shd)

        rotary_emb_NK = self._rot_pos_emb(grid_thw, total_tokens)
        emb_NK = jnp.concatenate([rotary_emb_NK, rotary_emb_NK], axis=-1)
        cos_NK, sin_NK = jnp.cos(emb_NK), jnp.sin(emb_NK)
        cos_NK = cos_NK.astype(self.cfg.dtype)
        sin_NK = sin_NK.astype(self.cfg.dtype)

        cu_seqlens = jnp.concatenate([
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(
                jnp.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0], out_sharding=P())
            ).astype(jnp.int32),
        ])

        for blk in self.blocks:
            hidden_ND = blk(hidden_ND, cu_seqlens, cos_NK, sin_NK)

        return self.merger(hidden_ND, self.cfg.spatial_merge_size)
