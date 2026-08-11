"""Qwen3-VL vision encoder."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P, reshard

from jax._src.cudnn.fused_attention_stablehlo import (
    MaskType as _CuDnnMaskType,
    dot_product_attention as _cudnn_dot_product_attention,
)

from omegalax.models.shard_config import ShardConfig
from .config import Qwen3VLVisionConfig


def _cudnn_batched_vision_attention_local(
    q_BSHK: jax.Array,
    k_BSHK: jax.Array,
    v_BSHK: jax.Array,
    seqlens_BM: jax.Array,
    offsets_BM1: jax.Array,
    scale: float,
) -> jax.Array:
    """cuDNN batched packed (THD) attention on a device-local shard.

    ``seqlens_BM``/``offsets_BM1`` give the per-image segment boundaries within
    each sample; cuDNN skips cross-segment tiles rather than materializing a full
    ``[S, S]`` mask, so the backward stays fused (dQ/dK/dV only, no dense
    ``[H, S, S]`` score matrix).
    """
    # force bfloat16 - cuDNN flash attention only supports fp16/bf16/fp8
    orig_dtype = q_BSHK.dtype
    out = _cudnn_dot_product_attention(
        q_BSHK.astype(jnp.bfloat16),
        k_BSHK.astype(jnp.bfloat16),
        v_BSHK.astype(jnp.bfloat16),
        q_seqlen=seqlens_BM.astype(jnp.int32),
        kv_seqlen=seqlens_BM.astype(jnp.int32),
        q_offsets=offsets_BM1.astype(jnp.int32),
        kv_offsets=offsets_BM1.astype(jnp.int32),
        scale=scale,
        mask_type=_CuDnnMaskType.NO_MASK,
        qkv_layout="BTNH",
    )
    return out.astype(orig_dtype)


def _cudnn_batched_vision_attention(
    q_BSHK: jax.Array,
    k_BSHK: jax.Array,
    v_BSHK: jax.Array,
    seqlens_BM: jax.Array,
    offsets_BM1: jax.Array,
    scale: float,
) -> jax.Array:
    """Batched packed vision attention, sharded on **batch** (and num_head).

    The ViT has no batch dim natively; per-sample padding gives us a real
    ``[B, S, num_heads, head_dim]`` batch (each sample a contiguous S-row). We
    run cuDNN inside a ``shard_map`` over the batch (dp/fsdp) and head (tp) axes,
    so each device computes attention only for its own samples/heads over the
    full ``S`` sequence — exactly correct for packed block-diagonal attention
    (no cross-device dependency), memory-safe (per-device
    ``[B_local, S, H_local, K]`` → cost ``Σ n_i²``, never a global ``[N, N]``),
    and fused (cuDNN's own fwd+bwd runs locally).

    ``shard_map`` (not cuDNN's ``custom_partitioning``) is required because under
    an all-Explicit mesh the partitioner leaves an unlowered ``Sharding`` custom
    call ("No registered implementation for custom call to Sharding" on CUDA).
    Sharding **batch**, not the sequence axis, keeps every image's tokens on one
    device — fixing both that error and the old token-shard correctness hazard.

    Args:
        q/k/v_BSHK: ``(B, S, num_heads, head_dim)`` (BTNH), batch sharded over
            (dp,fsdp), heads over tp.
        seqlens_BM: int32 ``(B, M)`` per-image token counts, batch-sharded.
        offsets_BM1: int32 ``(B, M+1)`` per-sample cumulative offsets (0..S),
            batch-sharded.
        scale: attention logits scale.
    """
    sharding = jax.typeof(q_BSHK).sharding
    batch_axis = sharding.spec[0]
    head_axis = sharding.spec[2]
    if batch_axis is None and head_axis is None:
        return _cudnn_batched_vision_attention_local(
            q_BSHK, k_BSHK, v_BSHK, seqlens_BM, offsets_BM1, scale
        )

    q_spec = P(batch_axis, None, head_axis, None)
    m_spec = P(batch_axis, None)
    return jax.shard_map(
        lambda q, k, v, sl, off: _cudnn_batched_vision_attention_local(q, k, v, sl, off, scale),
        mesh=sharding.mesh,
        in_specs=(q_spec, q_spec, q_spec, m_spec, m_spec),
        out_specs=q_spec,
        check_vma=False,
    )(q_BSHK, k_BSHK, v_BSHK, seqlens_BM, offsets_BM1)


def _token_spatial_coords(
    image_grid: jax.Array, merge_size: int, total_tokens: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map each vision token to its (row, col) in the original spatial grid.

    Args:
        image_grid: int32 array of shape ``(num_images, 3)`` with ``(t, h, w)``
            per image.  Expected to be replicated across the mesh; callers
            should ``reshard`` before invoking this.
        merge_size: spatial merge factor (typically 2).
        total_tokens: total number of vision tokens across all images
            (``sum(t*h*w)``).  Must be a **static** Python int (known from
            array shapes at trace time).

    Returns:
        row_coord, col_coord, image_id — each int32 of shape
        ``(total_tokens,)`` and replicated.
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
    ):
        self.scale = nnx.Param(jnp.ones(dim, dtype=jnp.float32), sharding=sharding)
        self.bias = nnx.Param(jnp.zeros(dim, dtype=jnp.float32), sharding=sharding)
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        mean = jnp.mean(x_f32, axis=-1, keepdims=True)
        var = jnp.var(x_f32, axis=-1, keepdims=True)
        normed = jnp.astype((x_f32 - mean) * jax.lax.rsqrt(var + self.eps), dtype)
        scale = jnp.astype(self.scale[...], dtype)
        bias = jnp.astype(self.bias[...], dtype)
        return scale * normed + bias


class VisionPatchEmbed(nnx.Module):
    """Conv3D patch embedding, represented as a linear layer over flattened patches."""

    def __init__(self, cfg: Qwen3VLVisionConfig, hidden_shd: P, *, rngs: nnx.Rngs):
        in_features = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size**2
        init = nnx.initializers.lecun_normal()
        self.proj = nnx.Linear(
            in_features,
            cfg.hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
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
            param_dtype=config.param_dtype,
            kernel_init=wp(init, (None, "hidden")),
        )
        self.fc2 = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
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
            param_dtype=config.param_dtype,
            kernel_init=qkv_init,
        )
        self.proj = nnx.Linear(
            config.hidden_size,
            config.hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=qkv_init,
        )
        self.hidden_shd = hidden_shd
        self.heads_shd = heads_shd
        object.__setattr__(self, "_q_sharding", None)
        object.__setattr__(self, "_q_sharding_spec", P(None, *heads_shd))

    def __call__(
        self,
        hidden_ND: jax.Array,
        seqlens_BM: jax.Array,
        offsets_BM1: jax.Array,
        cos_NK: jax.Array,
        sin_NK: jax.Array,
    ) -> jax.Array:
        N = hidden_ND.shape[0]
        B = seqlens_BM.shape[0]
        S = N // B
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

        # Reshape the flat [B*S, H, K] sequence into a real batch [B, S, H, K].
        # Per-sample padding guarantees each sample occupies a contiguous S-block,
        # so this split moves the (dp,fsdp) sharding from the token axis onto the
        # batch axis (collective-free) and shards heads over tp — the exact layout
        # cuDNN's fused fwd+bwd partitioner accepts.
        attn_shd = P(self.heads_shd[0], None, self.heads_shd[1], self.heads_shd[2])
        q_BSHK = jax.lax.reshape(
            q_NHK, (B, S, self.num_heads, self.head_dim), out_sharding=attn_shd
        )
        k_BSHK = jax.lax.reshape(
            k_NHK, (B, S, self.num_heads, self.head_dim), out_sharding=attn_shd
        )
        v_BSHK = jax.lax.reshape(
            v_NHK, (B, S, self.num_heads, self.head_dim), out_sharding=attn_shd
        )

        attn_BSHK = _cudnn_batched_vision_attention(
            q_BSHK, k_BSHK, v_BSHK, seqlens_BM, offsets_BM1, self.scale
        )

        attn_NHK = jax.lax.reshape(
            attn_BSHK, (N, self.num_heads, self.head_dim), out_sharding=self.heads_shd
        )
        outputs_ND = attn_NHK.reshape(N, -1)

        out_ND = self.proj(outputs_ND, out_sharding=self.hidden_shd)
        return out_ND


# Single source of truth for vision-block remat; also drives perf.py's HFU recompute term (trained tower only).
VISION_BLOCK_REMAT = True


class VisionBlock(nnx.Module):
    def __init__(
        self, cfg: Qwen3VLVisionConfig, hidden_shd: P, ff_shd: P, heads_shd: P, *, rngs: nnx.Rngs
    ):
        self.norm1 = LayerNorm(cfg.hidden_size, eps=1e-6, rngs=rngs)
        self.norm2 = LayerNorm(cfg.hidden_size, eps=1e-6, rngs=rngs)
        self.attn = VisionAttention(cfg, hidden_shd=hidden_shd, heads_shd=heads_shd, rngs=rngs)
        self.mlp = VisionMLP(cfg, hidden_shd=hidden_shd, ff_shd=ff_shd, rngs=rngs)
        self.hidden_shd = hidden_shd

    @(partial(jax.remat, static_argnums=0) if VISION_BLOCK_REMAT else (lambda f: f))
    def __call__(
        self,
        hidden_ND: jax.Array,
        seqlens_BM: jax.Array,
        offsets_BM1: jax.Array,
        cos_NK: jax.Array,
        sin_NK: jax.Array,
    ) -> jax.Array:
        hidden_ND = hidden_ND + self.attn(
            self.norm1(hidden_ND), seqlens_BM, offsets_BM1, cos_NK, sin_NK
        )
        hidden_ND = hidden_ND + self.mlp(self.norm2(hidden_ND))
        return hidden_ND


class VisionPatchMerger(nnx.Module):
    def __init__(
        self,
        cfg: Qwen3VLVisionConfig,
        hidden_shd: P,
        ff_shd: P,
        *,
        use_postshuffle_norm: bool = False,
        rngs: nnx.Rngs,
    ):
        hidden_size = cfg.hidden_size * (cfg.spatial_merge_size**2)
        self.hidden_size = hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = hidden_size if use_postshuffle_norm else cfg.hidden_size
        self.norm = LayerNorm(norm_dim, eps=1e-6, rngs=rngs)
        init = nnx.initializers.lecun_normal()
        self.fc1 = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
            kernel_init=wp(init, (None, None)),
        )
        self.fc2 = nnx.Linear(
            hidden_size,
            cfg.out_hidden_size,
            use_bias=True,
            rngs=rngs,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
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
    def __init__(self, cfg: Qwen3VLVisionConfig, shd_cfg: ShardConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.spatial_merge_size = cfg.spatial_merge_size
        self.hidden_shd = P(shd_cfg.act_btd[0], shd_cfg.act_btd[2])
        self.ff_shd = P(shd_cfg.act_btd[0], shd_cfg.act_btf[2])
        self.heads_shd = P(shd_cfg.act_btd[0], shd_cfg.act_btnh[2], None)
        self.patch_embed = VisionPatchEmbed(cfg, hidden_shd=self.hidden_shd, rngs=rngs)
        pos_init = nnx.initializers.normal(stddev=0.02)
        self.pos_embed = nnx.Embed(
            num_embeddings=cfg.num_position_embeddings,
            features=cfg.hidden_size,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
            rngs=rngs,
            embedding_init=wp(pos_init, (None, "hidden")),
        )
        self.num_grid_per_side = int(cfg.num_position_embeddings**0.5)
        head_dim = cfg.hidden_size // cfg.num_heads
        self.rotary_dim = head_dim // 2
        self.rotary_theta = 10000.0
        self.blocks = nnx.List(
            [
                VisionBlock(
                    cfg,
                    hidden_shd=self.hidden_shd,
                    ff_shd=self.ff_shd,
                    heads_shd=self.heads_shd,
                    rngs=rngs,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.merger = VisionPatchMerger(
            cfg,
            hidden_shd=self.hidden_shd,
            ff_shd=self.ff_shd,
            use_postshuffle_norm=False,
            rngs=rngs,
        )
        self.deepstack_mergers = nnx.List(
            [
                VisionPatchMerger(
                    cfg,
                    hidden_shd=self.hidden_shd,
                    ff_shd=self.ff_shd,
                    use_postshuffle_norm=True,
                    rngs=rngs,
                )
                for _ in cfg.deepstack_visual_indexes
            ]
        )
        self.deepstack_visual_indexes = cfg.deepstack_visual_indexes

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
        self,
        pixel_values: jax.Array,
        image_grid: jax.Array,
        vision_cu_seqlens: jax.Array,
        batch_size: int = 1,
    ) -> tuple[jax.Array, list[jax.Array]]:
        # ``vision_cu_seqlens`` is retained for API compatibility (callers still
        # pass it); the batched attention derives its own per-sample segment
        # metadata from ``image_grid`` + ``batch_size`` below.
        del vision_cu_seqlens
        image_grid = reshard(image_grid, P())
        # Per-image token counts as a real batch [B, M] (whole image = one
        # attention segment), plus per-sample cumulative offsets [B, M+1] (0..S).
        # Each sample's M image slots (real + dummy padding) sum to S tokens.
        M = image_grid.shape[0] // batch_size
        grid_BM3 = image_grid.reshape(batch_size, M, 3)
        seqlens_BM = (grid_BM3[:, :, 0] * grid_BM3[:, :, 1] * grid_BM3[:, :, 2]).astype(jnp.int32)
        offsets_BM1 = jnp.concatenate(
            [
                jnp.zeros((batch_size, 1), jnp.int32),
                jnp.cumsum(seqlens_BM, axis=1, dtype=jnp.int32),
            ],
            axis=1,
        )
        # Batch-shard the segment metadata (over the same (dp,fsdp) axes the
        # batch dim uses) so the attention shard_map's per-device batch slice
        # gets its matching [B_local, M] / [B_local, M+1] rows.
        seqlens_BM = reshard(seqlens_BM, P(self.heads_shd[0], None))
        offsets_BM1 = reshard(offsets_BM1, P(self.heads_shd[0], None))

        hidden_ND = self.patch_embed(pixel_values)
        total_tokens: int = hidden_ND.shape[0]
        assert total_tokens % batch_size == 0, (
            f"vision tokens {total_tokens} not divisible by batch_size {batch_size}; "
            "per-sample vision padding (max_vision_patches_per_sample) is required "
            "for the batched vision-attention layout."
        )

        pos_embeds_ND = self._interpolate_pos_embed(image_grid, total_tokens)
        pos_embeds_ND = reshard(pos_embeds_ND, self.hidden_shd)
        hidden_ND = hidden_ND + pos_embeds_ND

        rotary_emb_NK = self._compute_rotary_pos_emb(image_grid, total_tokens)
        rotary_emb_NK = reshard(rotary_emb_NK, P(self.hidden_shd[0], None))
        emb_NK = jnp.concatenate([rotary_emb_NK, rotary_emb_NK], axis=-1)
        cos_NK, sin_NK = jnp.cos(emb_NK), jnp.sin(emb_NK)
        cos_NK = cos_NK.astype(self.cfg.dtype)
        sin_NK = sin_NK.astype(self.cfg.dtype)

        deepstack_features: list[jax.Array] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_ND = blk(hidden_ND, seqlens_BM, offsets_BM1, cos_NK, sin_NK)
            if layer_num in self.deepstack_visual_indexes:
                idx = list(self.deepstack_visual_indexes).index(layer_num)
                deepstack_features.append(self.deepstack_mergers[idx](hidden_ND))

        merged_ND = self.merger(hidden_ND)
        return merged_ND, deepstack_features
