"""Qwen3-VL vision encoder."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P, reshard

from jax._src.cudnn.fused_attention_stablehlo import (
    MaskType as _CuDnnMaskType,
    dot_product_attention as _cudnn_dot_product_attention,
)
from tokamax._src.ops.attention.api import IMPLEMENTATIONS as _ATTN_IMPLS
from tokamax._src.ops.attention.base import Mask

from omegalax.models.shard_config import ShardConfig
from .config import Qwen3VLVisionConfig


def _cudnn_packed_vision_attention(
    q_NHK: jax.Array,
    k_NHK: jax.Array,
    v_NHK: jax.Array,
    cu_seqlens: jax.Array,
    scale: float,
) -> jax.Array:
    """Run vision attention via cuDNN's packed (THD) kernel.

    All image tokens are concatenated along the sequence dim. ``cu_seqlens``
    describes per-image segment boundaries; cuDNN uses these to skip
    cross-segment tiles entirely rather than materializing a full [T, S] mask.

    Args:
        q_NHK, k_NHK, v_NHK: (N, num_heads, head_dim) with N == cu_seqlens[-1].
        cu_seqlens: int32, shape (M+1,). Static length at trace time.
        scale: attention logits scale.
    Returns:
        (N, num_heads, head_dim)
    """
    cu = cu_seqlens.astype(jnp.int32)
    q_offsets = cu[None]
    kv_offsets = cu[None]
    seqlens = jnp.diff(cu)[None]

    out = _cudnn_dot_product_attention(
        q_NHK[None], k_NHK[None], v_NHK[None],
        q_seqlen=seqlens, kv_seqlen=seqlens,
        q_offsets=q_offsets, kv_offsets=kv_offsets,
        scale=scale, mask_type=_CuDnnMaskType.NO_MASK,
        qkv_layout="BTNH",
    )
    return out[0]


def _token_spatial_coords(
    image_grid: jax.Array, merge_size: int, total_tokens: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map each vision token to its (row, col) in the original spatial grid.

    Args:
        image_grid: int32 array of shape ``(num_images, 3)`` with ``(t, h, w)``
            per image.
        merge_size: spatial merge factor (typically 2).
        total_tokens: total number of vision tokens across all images
            (``sum(t*h*w)``).  Must be a **static** Python int (known from
            array shapes at trace time).

    Returns:
        row_coord, col_coord, image_id — each int32 of shape
        ``(total_tokens,)``.
    """
    tokens_per_image = image_grid[:, 0] * image_grid[:, 1] * image_grid[:, 2]
    cu_tokens = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(tokens_per_image).astype(jnp.int32)]
    )

    tok_idx = jnp.arange(total_tokens, dtype=jnp.int32)
    image_id = jnp.searchsorted(cu_tokens[1:], tok_idx, side="right")
    local_idx = tok_idx - cu_tokens[image_id]

    h = image_grid[image_id, 1]
    w = image_grid[image_id, 2]
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


class LayerNorm(nnx.Module):
    """Standard LayerNorm (weight + bias)."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        rngs: nnx.Rngs,
        sharding: tuple[str | None, ...] = ("hidden",),
        param_dtype: Any = jnp.float32,
    ):
        self.scale = nnx.Param(jnp.ones(dim, dtype=param_dtype), sharding=sharding)
        self.bias = nnx.Param(jnp.zeros(dim, dtype=param_dtype), sharding=sharding)
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        normed = (x - mean) * jax.lax.rsqrt(var + self.eps)
        return self.scale[...] * normed + self.bias[...]


class VisionPatchEmbed(nnx.Module):
    """Conv3D patch embedding, represented as a linear layer over flattened patches."""

    def __init__(self, config: Qwen3VLVisionConfig, hidden_shd: P, *, rngs: nnx.Rngs):
        in_features = config.in_channels * config.temporal_patch_size * config.patch_size**2
        init = nnx.initializers.lecun_normal()
        self.proj = nnx.Linear(
            in_features,
            config.hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=config.dtype,
            kernel_init=wp(init, (None, "hidden")),
        )
        self.in_features = in_features
        self.hidden_shd = hidden_shd

    def __call__(self, pixels: jax.Array) -> jax.Array:
        flat = pixels.reshape(-1, self.in_features)
        out_ND = self.proj(flat, out_sharding=self.hidden_shd)
        return out_ND


class VisionMLP(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, hidden_shd: P, ff_shd: P, *, rngs: nnx.Rngs):
        init = nnx.initializers.lecun_normal()
        self.fc1 = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=True,
            rngs=rngs,
            dtype=config.dtype,
            kernel_init=wp(init, (None, "hidden")),
        )
        self.fc2 = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=config.dtype,
            kernel_init=wp(init, ("hidden", None)),
        )
        self.hidden_shd = hidden_shd
        self.ff_shd = ff_shd

    def __call__(self, hidden_ND: jax.Array) -> jax.Array:
        ff_NF = self.fc1(hidden_ND, out_sharding=self.ff_shd)
        ff_NF = jax.nn.gelu(ff_NF, approximate=True)
        out_ND = self.fc2(ff_NF, out_sharding=self.hidden_shd)
        return out_ND


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
    def __init__(self, config: Qwen3VLVisionConfig, hidden_shd: P, heads_shd: P, *, rngs: nnx.Rngs):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5
        init = nnx.initializers.lecun_normal()
        qkv_init = wp(init, (None, "hidden"))
        self.qkv = nnx.Linear(
            config.hidden_size,
            config.hidden_size * 3,
            use_bias=True,
            rngs=rngs,
            dtype=config.dtype,
            kernel_init=qkv_init,
        )
        self.proj = nnx.Linear(
            config.hidden_size,
            config.hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=config.dtype,
            kernel_init=qkv_init,
        )
        self.hidden_shd = hidden_shd
        self.heads_shd = heads_shd
        object.__setattr__(self, "_q_sharding", None)
        object.__setattr__(self, "_q_sharding_spec", P(None, *heads_shd))
        object.__setattr__(self, "_attn_backend", "mosaic_gpu")
        object.__setattr__(self, "_attn_kind", "vision")

    def __call__(self, hidden_ND: jax.Array, cu_seqlens: jax.Array, cos_NK: jax.Array, sin_NK: jax.Array) -> jax.Array:
        N = hidden_ND.shape[0]
        qkv_shd = P(self.hidden_shd[0], None, self.heads_shd[1], self.heads_shd[2])
        qkv = jax.lax.reshape(
            self.qkv(hidden_ND, out_sharding=self.hidden_shd),
            (N, 3, self.num_heads, self.head_dim),
            out_sharding=qkv_shd,
        )
        q_NHK = reshard(qkv[:, 0], self.heads_shd)
        k_NHK = reshard(qkv[:, 1], self.heads_shd)
        v_NHK = reshard(qkv[:, 2], self.heads_shd)

        q_NHK, k_NHK = apply_rotary_pos_emb_vision(q_NHK, k_NHK, cos_NK, sin_NK)

        if self._attn_backend == "cudnn_packed":
            attn_NHK = _cudnn_packed_vision_attention(
                q_NHK, k_NHK, v_NHK, cu_seqlens, self.scale,
            )
            outputs_ND = attn_NHK.reshape(N, -1)
        else:
            _BLOCK = 128  # tokamax mosaic block alignment
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
            mask = Mask(k_start=k_start, k_end=k_end)

            attn_NHK = _ATTN_IMPLS[self._attn_backend](
                q_NHK[None], k_NHK[None], v_NHK[None],
                mask=mask, logits_scale=self.scale,
                q_sharding=self._q_sharding,
            )
            outputs_ND = attn_NHK[0, :N].reshape(N, -1)

        out_ND = self.proj(outputs_ND, out_sharding=self.hidden_shd)
        return out_ND


class VisionBlock(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, hidden_shd: P, ff_shd: P, heads_shd: P, *, rngs: nnx.Rngs):
        self.norm1 = LayerNorm(config.hidden_size, eps=1e-6, rngs=rngs)
        self.norm2 = LayerNorm(config.hidden_size, eps=1e-6, rngs=rngs)
        self.attn = VisionAttention(config, hidden_shd=hidden_shd, heads_shd=heads_shd, rngs=rngs)
        self.mlp = VisionMLP(config, hidden_shd=hidden_shd, ff_shd=ff_shd, rngs=rngs)
        self.hidden_shd = hidden_shd

    @partial(jax.remat, static_argnums=0)
    def __call__(self, hidden_ND: jax.Array, cu_seqlens: jax.Array, cos_NK: jax.Array, sin_NK: jax.Array) -> jax.Array:
        hidden_ND = hidden_ND + self.attn(self.norm1(hidden_ND), cu_seqlens, cos_NK, sin_NK)
        hidden_ND = hidden_ND + self.mlp(self.norm2(hidden_ND))
        return hidden_ND


class VisionPatchMerger(nnx.Module):
    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        hidden_shd: P,
        ff_shd: P,
        *,
        use_postshuffle_norm: bool = False,
        rngs: nnx.Rngs,
    ):
        hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.hidden_size = hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = LayerNorm(norm_dim, eps=1e-6, rngs=rngs)
        init = nnx.initializers.lecun_normal()
        self.fc1 = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=config.dtype,
            kernel_init=wp(init, (None, None)),
        )
        self.fc2 = nnx.Linear(
            hidden_size,
            config.out_hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=config.dtype,
            kernel_init=wp(init, (None, "hidden")),
        )
        self.hidden_shd = hidden_shd
        self.ff_shd = ff_shd

    def __call__(self, hidden_ND: jax.Array) -> jax.Array:
        new_sizes = (hidden_ND.shape[0] * hidden_ND.shape[1] // self.hidden_size, self.hidden_size)
        if self.use_postshuffle_norm:
            normed = self.norm(jax.lax.reshape(hidden_ND, new_sizes, out_sharding=self.hidden_shd))
        else:
            normed = jax.lax.reshape(self.norm(hidden_ND), new_sizes, out_sharding=self.hidden_shd)
        ff_NF = self.fc1(normed, out_sharding=self.ff_shd)
        ff_NF = jax.nn.gelu(ff_NF, approximate=True)
        out_ND = self.fc2(ff_NF, out_sharding=self.hidden_shd)
        return out_ND


class VisionModel(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, shd_cfg: ShardConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.hidden_shd = P(shd_cfg.act_btd[0], shd_cfg.act_btd[2])
        self.ff_shd = P(shd_cfg.act_btd[0], shd_cfg.act_btf[2])
        self.heads_shd = P(shd_cfg.act_btd[0], shd_cfg.act_btnh[2], None)
        self.patch_embed = VisionPatchEmbed(config, hidden_shd=self.hidden_shd, rngs=rngs)
        pos_init = nnx.initializers.normal(stddev=0.02)
        self.pos_embed = nnx.Embed(
            num_embeddings=config.num_position_embeddings,
            features=config.hidden_size,
            dtype=config.dtype,
            rngs=rngs,
            embedding_init=wp(pos_init, (None, "hidden")),
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_dim = head_dim // 2
        self.rotary_theta = 10000.0
        self.blocks = nnx.List(
            [VisionBlock(config, hidden_shd=self.hidden_shd, ff_shd=self.ff_shd, heads_shd=self.heads_shd, rngs=rngs) for _ in range(config.depth)]
        )
        self.merger = VisionPatchMerger(
            config,
            hidden_shd=self.hidden_shd,
            ff_shd=self.ff_shd,
            use_postshuffle_norm=False,
            rngs=rngs,
        )
        self.deepstack_mergers = nnx.List(
            [
                VisionPatchMerger(
                    config,
                    hidden_shd=self.hidden_shd,
                    ff_shd=self.ff_shd,
                    use_postshuffle_norm=True,
                    rngs=rngs,
                )
                for _ in config.deepstack_visual_indexes
            ]
        )
        self.deepstack_visual_indexes = config.deepstack_visual_indexes

    def _compute_rotary_pos_emb(self, image_grid: jax.Array, total_tokens: int) -> jax.Array:
        row, col, _ = _token_spatial_coords(image_grid, self.spatial_merge_size, total_tokens)
        inv_freq = 1.0 / (
            self.rotary_theta
            ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim)
        )
        row_emb = row[:, None].astype(jnp.float32) * inv_freq[None, :]
        col_emb = col[:, None].astype(jnp.float32) * inv_freq[None, :]
        return jnp.concatenate([row_emb, col_emb], axis=-1)

    def _interpolate_pos_embed(self, image_grid: jax.Array, total_tokens: int) -> jax.Array:
        row, col, img_id = _token_spatial_coords(image_grid, self.spatial_merge_size, total_tokens)
        pos_weight_VD = self.pos_embed.embedding[...]
        n = self.num_grid_per_side

        h = image_grid[img_id, 1].astype(jnp.float32)
        w = image_grid[img_id, 2].astype(jnp.float32)

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

    def __call__(
        self, pixel_values: jax.Array, image_grid: jax.Array, vision_cu_seqlens: jax.Array
    ) -> tuple[jax.Array, list[jax.Array]]:
        hidden_ND = self.patch_embed(pixel_values)
        total_tokens: int = hidden_ND.shape[0]

        pos_embeds_ND = self._interpolate_pos_embed(image_grid, total_tokens)
        hidden_ND = hidden_ND + pos_embeds_ND

        rotary_emb_NK = self._compute_rotary_pos_emb(image_grid, total_tokens)
        emb_NK = jnp.concatenate([rotary_emb_NK, rotary_emb_NK], axis=-1)
        cos_NK, sin_NK = jnp.cos(emb_NK), jnp.sin(emb_NK)
        cos_NK = cos_NK.astype(self.config.dtype)
        sin_NK = sin_NK.astype(self.config.dtype)

        cu_seqlens = vision_cu_seqlens.astype(jnp.int32)

        deepstack_features: list[jax.Array] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_ND = blk(hidden_ND, cu_seqlens, cos_NK, sin_NK)
            if layer_num in self.deepstack_visual_indexes:
                idx = list(self.deepstack_visual_indexes).index(layer_num)
                deepstack_features.append(self.deepstack_mergers[idx](hidden_ND))

        merged_ND = self.merger(hidden_ND)
        return merged_ND, deepstack_features
