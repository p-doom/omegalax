"""Qwen3.5 Vision Encoder.

Implements the ViT-style vision encoder with 3-D patch embedding,
rotary position embeddings, spatial merge, and bilinear position
embedding interpolation.
"""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard
from flax import nnx

from jax._src.cudnn.fused_attention_stablehlo import (
    MaskType as _CuDnnMaskType,
    dot_product_attention as _cudnn_dot_product_attention,
)

from omegalax.models.remat_policy import resolve_remat_policy
from omegalax.models.shard_config import ShardConfig
from .config import Qwen3_5VisionConfig
from .norms import LayerNorm
from .rope import apply_vision_rope


def _cudnn_batched_vision_attention_local(
    q_BSHK: jax.Array,
    k_BSHK: jax.Array,
    v_BSHK: jax.Array,
    seqlens_BM: jax.Array,
    offsets_BM1: jax.Array,
    scale: float,
) -> jax.Array:
    """cuDNN batched packed (THD) attention on a device-local shard.

    ``seqlens_BM``/``offsets_BM1`` give the per-image segment boundaries WITHIN each
    sample; cuDNN skips cross-segment tiles rather than materializing a full
    ``[S, S]`` mask, so the backward stays fused (dQ/dK/dV only, never a dense
    ``[H, S, S]`` score matrix).
    """
    orig_dtype = q_BSHK.dtype
    out = _cudnn_dot_product_attention(
        # cuDNN flash attention only supports fp16/bf16/fp8.
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
    """Batched packed vision attention, sharded on **batch** (and head).

    The ViT has no batch dim natively. Per-sample vision padding
    (``--batched_vision_padding``) gives every sample its own contiguous ``S``-row
    block, so the flat sequence reshapes to a real ``[B, S, num_heads, head_dim]``
    and the (dp,fsdp) sharding moves off the token axis onto the batch axis.

    Sharding batch rather than the packed token axis is what makes this correct:
    every image's tokens stay on one device, so a device's segment offsets always
    describe tokens it actually holds. Sharding the packed axis put shard
    boundaries at arbitrary tokens, mid-image.

    ``shard_map`` (not cuDNN's ``custom_partitioning``) is required because under an
    all-Explicit mesh the partitioner cannot emit a valid sharding rule for this
    custom call -- it fails with "dim mapping can't have a factor of size 1 if there
    are multiple factors" on a mesh mixing size-1 axes with a real one, even when
    every operand is replicated. ``shard_map`` makes the region device-local, so no
    sharding rule is needed at all.

    Args:
        q/k/v_BSHK: ``(B, S, num_heads, head_dim)`` (BTNH), batch over (dp,fsdp),
            heads over tp.
        seqlens_BM: int32 ``(B, M)`` per-image token counts, batch-sharded.
        offsets_BM1: int32 ``(B, M+1)`` per-sample cumulative offsets (0..S).
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
    grid_thw: jax.Array, merge_size: int, total_tokens: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map each vision token to its (row, col) in the original spatial grid.

    Args:
        grid_thw: per-image (t, h, w). Expected to be REPLICATED across the mesh;
            callers must ``reshard`` before invoking this. The cumulative token
            offsets below are global, so a mesh-sharded leading axis is both
            wrong and, under an Explicit mesh, a hard ShardingTypeError on the
            concat against the replicated leading zero.

    Returns:
        row_coord, col_coord, image_id: each int32 of shape
        ``(total_tokens,)`` and replicated.
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
        N = pixels.shape[0]
        patches = pixels.reshape(
            N, self.temporal_patch_size, self.patch_size, self.patch_size, self.in_channels
        )
        embedded = self.proj(patches)
        return jax.lax.reshape(embedded, (N, self.embed_dim), out_sharding=self.hidden_shd)


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
        ff_NF = nnx.gelu(ff_NF, approximate=True)
        out_ND = self.fc2(ff_NF, out_sharding=self.hidden_shd)
        return out_ND


class VisionAttention(nnx.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig, hidden_shd: P, heads_shd: P, *, rngs: nnx.Rngs):
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.scale = self.head_dim**-0.5
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
        object.__setattr__(self, "_q_sharding", None)
        object.__setattr__(self, "_q_sharding_spec", P(None, *heads_shd))
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
        seqlens_BM: jax.Array,
        offsets_BM1: jax.Array,
        cos_NK: jax.Array,
        sin_NK: jax.Array,
    ) -> jax.Array:
        N = hidden_ND.shape[0]
        qkv = self.qkv(hidden_ND, out_sharding=self.hidden_shd).reshape(
            N, 3, self.num_heads, self.head_dim
        )
        q_NHK = reshard(qkv[:, 0], self.heads_shd)
        k_NHK = reshard(qkv[:, 1], self.heads_shd)
        v_NHK = reshard(qkv[:, 2], self.heads_shd)

        q_NHK, k_NHK = apply_vision_rope(q_NHK, k_NHK, cos_NK, sin_NK)

        # Split the flat [B*S, H, K] sequence into a real batch [B, S, H, K]. Each
        # sample owns a contiguous S-block (per-sample vision padding), so this
        # reshape moves the (dp,fsdp) sharding from the token axis onto the batch
        # axis collective-free, and heads onto tp -- the layout cuDNN's fused
        # fwd+bwd accepts. B == 1 reproduces the old single packed sequence.
        B = seqlens_BM.shape[0]
        S = N // B
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


class VisionBlock(nnx.Module):
    def __init__(
        self, cfg: Qwen3_5VisionConfig, hidden_shd: P, ff_shd: P, heads_shd: P, *, rngs: nnx.Rngs
    ):
        self.norm1 = LayerNorm(cfg.hidden_size, 1e-6, rngs=rngs)
        self.norm2 = LayerNorm(cfg.hidden_size, 1e-6, rngs=rngs)
        self.attn = VisionAttention(cfg, hidden_shd=hidden_shd, heads_shd=heads_shd, rngs=rngs)
        self.mlp = VisionMLP(cfg, hidden_shd=hidden_shd, ff_shd=ff_shd, rngs=rngs)
        self.hidden_shd = hidden_shd

        self._remat_policy = resolve_remat_policy(cfg.remat_policy)

    def __call__(self, hidden_ND, seqlens_BM, offsets_BM1, cos_NK, sin_NK):
        # Inline nnx.remat on the UNBOUND method (no static_argnums): nnx
        # functionalizes ``self`` via split/merge, so it must not be static.
        # Building the transform inline keeps graphdefs equal across fresh
        # instances (stable hash -> one trace, not one per instance).
        return nnx.remat(type(self)._impl, policy=self._remat_policy)(
            self, hidden_ND, seqlens_BM, offsets_BM1, cos_NK, sin_NK
        )

    def _impl(self, hidden_ND, seqlens_BM, offsets_BM1, cos_NK, sin_NK):
        hidden_ND = hidden_ND + self.attn(
            self.norm1(hidden_ND), seqlens_BM, offsets_BM1, cos_NK, sin_NK
        )
        hidden_ND = hidden_ND + self.mlp(self.norm2(hidden_ND))
        return hidden_ND


class VisionPatchMerger(nnx.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig, hidden_shd: P, ff_shd: P, *, rngs: nnx.Rngs):
        merged_dim = cfg.hidden_size * (cfg.spatial_merge_size**2)
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
        merged_dim = hidden_ND.shape[-1] * merge_size * merge_size
        normed = self.norm(hidden_ND)
        normed = jax.lax.reshape(
            normed,
            (normed.shape[0] // (merge_size * merge_size), merged_dim),
            out_sharding=self.hidden_shd,
        )
        ff_NF = self.fc1(normed, out_sharding=self.ff_shd)
        ff_NF = nnx.gelu(ff_NF, approximate=True)
        out_ND = self.fc2(ff_NF, out_sharding=self.hidden_shd)
        return out_ND


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
        self.num_grid_per_side = int(cfg.num_position_embeddings**0.5)
        head_dim = cfg.hidden_size // cfg.num_heads
        self.rotary_half_dim = head_dim // 2
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
            cfg, hidden_shd=self.hidden_shd, ff_shd=self.ff_shd, rngs=rngs
        )

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
        row, col, img_id = _token_spatial_coords(
            grid_thw, self.cfg.spatial_merge_size, total_tokens
        )
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
    def __call__(
        self,
        pixel_values: jax.Array,
        grid_thw: jax.Array,
        cu_seqlens: jax.Array | None = None,
        batch_size: int = 1,
    ) -> jax.Array:
        # ``cu_seqlens`` is accepted for API compatibility (callers still pass the
        # collator's global packed offsets) but unused: the batched attention derives
        # its own PER-SAMPLE segment metadata from grid_thw + batch_size below.
        # ``batch_size=1`` treats the whole flat sequence as one sample, which is
        # exactly the legacy global-bucket packed layout.
        del cu_seqlens

        # Under an all-Explicit mesh JAX never inserts an implicit reshard, so every
        # operand of a binary op / concat must already agree. `grid_thw` arrives
        # sharded on its leading (image) axis, which makes the cumulative token
        # offsets in `_token_spatial_coords` both mis-sharded and semantically wrong
        # (a cumsum used as GLOBAL offsets). Replicate it, then push the derived
        # per-token tables back onto the token-sharded layout `hidden_ND` uses.
        grid_thw = reshard(grid_thw, P())

        # Per-image token counts as a real batch [B, M] (one image == one attention
        # segment), plus per-sample cumulative offsets [B, M+1] running 0..S. Each
        # sample's M image slots (real + dummy padding) sum to exactly S tokens.
        M = grid_thw.shape[0] // batch_size
        grid_BM3 = grid_thw.reshape(batch_size, M, 3)
        seqlens_BM = (grid_BM3[:, :, 0] * grid_BM3[:, :, 1] * grid_BM3[:, :, 2]).astype(jnp.int32)
        offsets_BM1 = jnp.concatenate(
            [
                jnp.zeros((batch_size, 1), jnp.int32),
                jnp.cumsum(seqlens_BM, axis=1, dtype=jnp.int32),
            ],
            axis=1,
        )
        # Batch-shard the segment metadata over the same axes the batch dim uses, so
        # the attention shard_map's per-device batch slice gets its matching rows.
        seqlens_BM = reshard(seqlens_BM, P(self.heads_shd[0], None))
        offsets_BM1 = reshard(offsets_BM1, P(self.heads_shd[0], None))

        hidden_ND = self.patch_embed(pixel_values)
        total_tokens: int = hidden_ND.shape[0]
        assert total_tokens % batch_size == 0, (
            f"vision tokens {total_tokens} not divisible by batch_size {batch_size}; "
            "per-sample vision padding (--batched_vision_padding with "
            "max_vision_patches_per_sample) is required for the batched layout."
        )

        pos_embeds_ND = self._fast_pos_embed_interpolate(grid_thw, total_tokens)
        pos_embeds_ND = reshard(pos_embeds_ND, self.hidden_shd)
        hidden_ND = hidden_ND + pos_embeds_ND

        rotary_emb_NK = self._rot_pos_emb(grid_thw, total_tokens)
        rotary_emb_NK = reshard(rotary_emb_NK, P(self.hidden_shd[0], None))
        emb_NK = jnp.concatenate([rotary_emb_NK, rotary_emb_NK], axis=-1)
        cos_NK, sin_NK = jnp.cos(emb_NK), jnp.sin(emb_NK)
        cos_NK = cos_NK.astype(self.cfg.dtype)
        sin_NK = sin_NK.astype(self.cfg.dtype)

        for blk in self.blocks:
            hidden_ND = blk(hidden_ND, seqlens_BM, offsets_BM1, cos_NK, sin_NK)

        return self.merger(hidden_ND, self.cfg.spatial_merge_size)
