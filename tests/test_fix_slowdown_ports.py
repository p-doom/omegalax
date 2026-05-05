"""GPU verification of fix-slowdown ports to qwen3 and qwen3_5.

Covers four layered checks:

  A. cuDNN packed vision attention vs. pure-JAX block-diagonal reference,
     using unequal-segment cu_seqlens.
  B. qwen3 text-attn backend selector: forward outputs are equivalent under
     mosaic_gpu and cudnn.
  C. qwen3_5 padding-no-op: VLM forward with unpadded vs collator-padded
     vision arrays must match at non-padded token positions.
  D. JIT stability: under the same padded budget, two forwards with
     different real-image counts trigger only one compilation.

Requires GPU with cuDNN (head_dim=64 used to satisfy the cuDNN kernel).
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx

from omegalax.data.collator_qwen3 import _pad_vision_arrays, _compute_vision_cu_seqlens
from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.qwen3.config import Qwen3Config
from omegalax.models.qwen3.model import Qwen3
from omegalax.models.qwen3_5.config import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from omegalax.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from omegalax.models.qwen3_5.vision import _cudnn_packed_vision_attention
from omegalax.models.sharding_runtime import set_attn_backend


def _block_diag_reference(
    q_NHK: jax.Array,
    k_NHK: jax.Array,
    v_NHK: jax.Array,
    cu_seqlens: np.ndarray,
    scale: float,
) -> jax.Array:
    """Pure-JAX block-diagonal masked SDPA reference (fp32).

    Each segment defined by cu_seqlens attends only to itself.
    """
    N, H, K = q_NHK.shape
    q = q_NHK.astype(jnp.float32)
    k = k_NHK.astype(jnp.float32)
    v = v_NHK.astype(jnp.float32)
    seg = np.zeros(N, dtype=np.int32)
    for i in range(len(cu_seqlens) - 1):
        seg[int(cu_seqlens[i]):int(cu_seqlens[i + 1])] = i
    seg_j = jnp.asarray(seg)
    mask_NN = seg_j[:, None] == seg_j[None, :]  # block-diagonal
    logits_HNN = jnp.einsum("nhk,mhk->hnm", q, k) * scale
    neg_inf = jnp.finfo(logits_HNN.dtype).min
    logits_HNN = jnp.where(mask_NN[None], logits_HNN, neg_inf)
    weights_HNN = jax.nn.softmax(logits_HNN, axis=-1)
    out_HNK = jnp.einsum("hnm,mhk->nhk", weights_HNN, v)
    return out_HNK


# --------------------------------------------------------------------------- #
# A. cuDNN packed attention correctness
# --------------------------------------------------------------------------- #


class CuDnnPackedVisionAttentionTest(absltest.TestCase):
    """Validates the cuDNN packed kernel + cu_seqlens path used by both VLMs.

    Uses unequal segment sizes — the precise case the `Mask(k_start, k_end)`
    path used to handle on the old branch.
    """

    def test_unequal_segments_match_reference(self):
        rng = np.random.RandomState(0)
        # Three "images" of unequal token count.
        seg_sizes = [16, 8, 24]
        N = sum(seg_sizes)
        H, K = 4, 64  # head_dim=64 satisfies cuDNN
        q = rng.randn(N, H, K).astype(np.float32) * 0.1
        k = rng.randn(N, H, K).astype(np.float32) * 0.1
        v = rng.randn(N, H, K).astype(np.float32) * 0.1
        cu = np.concatenate([[0], np.cumsum(seg_sizes)]).astype(np.int32)

        q_jax = jnp.asarray(q, dtype=jnp.bfloat16)
        k_jax = jnp.asarray(k, dtype=jnp.bfloat16)
        v_jax = jnp.asarray(v, dtype=jnp.bfloat16)
        cu_jax = jnp.asarray(cu)

        scale = 1.0 / (K ** 0.5)
        out_cudnn = _cudnn_packed_vision_attention(q_jax, k_jax, v_jax, cu_jax, scale)
        out_ref = _block_diag_reference(q_jax, k_jax, v_jax, cu, scale)

        np.testing.assert_allclose(
            np.asarray(out_cudnn, dtype=np.float32),
            np.asarray(out_ref, dtype=np.float32),
            rtol=2e-2, atol=2e-2,
        )


# --------------------------------------------------------------------------- #
# B. qwen3 text-attn backend selector
# --------------------------------------------------------------------------- #


def _qwen3_test_cfg() -> Qwen3Config:
    return Qwen3Config(
        num_layers=2,
        vocab_size=128,
        emb_dim=128,
        mlp_dim=256,
        num_heads=2,
        head_dim=64,  # cuDNN-compatible
        num_kv_heads=2,
        rope_theta=10_000,
        rope_scaling_factor=None,
        local_rope_theta=None,
        norm_eps=1e-6,
        tie_word_embeddings=False,
    )


class Qwen3TextAttnBackendSwapTest(absltest.TestCase):

    def test_mosaic_gpu_and_cudnn_outputs_match(self):
        cfg = _qwen3_test_cfg()
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model = Qwen3(cfg=cfg, rngs=nnx.Rngs(params=0))

        rng = np.random.RandomState(7)
        token_ids = jnp.asarray(rng.randint(1, cfg.vocab_size, size=(1, 32), dtype=np.int32))
        segment_ids = jnp.ones_like(token_ids)

        # Default backend ("mosaic_gpu") set by attention.__init__.
        h_default, _ = model(token_ids, segment_ids, None, jnp.array(0, dtype=jnp.int32))
        out_default = np.asarray(h_default, dtype=np.float32)

        set_attn_backend(model, text_backend="cudnn")
        h_cudnn, _ = model(token_ids, segment_ids, None, jnp.array(0, dtype=jnp.int32))
        out_cudnn = np.asarray(h_cudnn, dtype=np.float32)

        np.testing.assert_allclose(out_cudnn, out_default, rtol=3e-2, atol=3e-2)


# --------------------------------------------------------------------------- #
# Shared qwen3_5 VLM test fixtures
# --------------------------------------------------------------------------- #


def _qwen3_5_vlm_test_cfg() -> Qwen3_5Config:
    """Tiny qwen3.5 VLM with cuDNN-compatible vision head_dim."""
    text = Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        rms_norm_eps=1e-6,
        layer_types=("full_attention", "full_attention"),
        rope_theta=10_000,
        partial_rotary_factor=0.25,
        mrope_section=(8, 4, 4),  # sums to 16 = head_dim * partial_rotary_factor
        intermediate_size=256,
        # MoE off (num_experts=0 -> dense MLP)
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=128,
        intermediate_size=256,
        num_heads=2,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
        out_hidden_size=128,
        num_position_embeddings=64,
    )
    return Qwen3_5Config(
        vision_config=vision,
        text_config=text,
        image_token_id=2,
        video_token_id=3,
        vision_start_token_id=4,
        vision_end_token_id=5,
    )


def _make_vlm_inputs(cfg: Qwen3_5Config, real_grids, max_images=None, max_patches=None):
    """Build (tokens, segment_ids, pixel_values, image_grid_thw, cu_seqlens, position_ids).

    `real_grids` is a list of [t, h, w]. Tokens have one image-token per merged
    pixel for each grid. Optionally pads pixel_values/grid_thw via the same
    padder used by VLMSFTCollator.
    """
    ms = cfg.vision_config.spatial_merge_size
    ms2 = ms * ms
    real_grid = np.array(real_grids, dtype=np.int32)
    real_patches = int(np.sum(real_grid[:, 0] * real_grid[:, 1] * real_grid[:, 2]))

    img_tokens_per_image = [int(t * h * w // ms2) for t, h, w in real_grids]
    total_img_tokens = sum(img_tokens_per_image)
    tokens = np.full((1, total_img_tokens + 4), 7, dtype=np.int32)
    tokens[0, 0] = 1  # bos
    tokens[0, 1:1 + total_img_tokens] = cfg.image_token_id

    patch_dim = (
        cfg.vision_config.in_channels
        * cfg.vision_config.patch_size
        * cfg.vision_config.patch_size
        * cfg.vision_config.temporal_patch_size
    )
    rng = np.random.RandomState(123)
    pv = rng.randn(real_patches, patch_dim).astype(np.float32)
    grid = real_grid

    if max_images is not None:
        pv, grid, cu = _pad_vision_arrays(
            pv, grid,
            merge_size=ms,
            max_patches=max_patches,
            max_images=max_images,
        )
    else:
        cu = _compute_vision_cu_seqlens(grid)

    seg = np.ones_like(tokens)
    pos = np.broadcast_to(
        np.arange(tokens.shape[1], dtype=np.int32)[None, None, :],
        (3, *tokens.shape),
    ).copy()
    return (
        jnp.asarray(tokens),
        jnp.asarray(seg),
        jnp.asarray(pv, dtype=jnp.bfloat16),
        jnp.asarray(grid, dtype=jnp.int32),
        jnp.asarray(cu, dtype=jnp.int32),
        jnp.asarray(pos, dtype=jnp.int32),
    )


# --------------------------------------------------------------------------- #
# C. qwen3_5 padding-no-op
# --------------------------------------------------------------------------- #


class Qwen3_5PaddingNoOpTest(absltest.TestCase):
    """Padded-vs-unpadded outputs at non-pad text positions must match."""

    def test_padded_and_unpadded_match_at_real_positions(self):
        cfg = _qwen3_5_vlm_test_cfg()
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model = Qwen3_5ForConditionalGeneration(cfg=cfg, rngs=nnx.Rngs(params=0))
        # Use cudnn for text-attn so the test isn't constrained by mosaic_gpu's
        # seq-len tiling (the fusion+padding logic is what we want to validate).
        set_attn_backend(model, text_backend="cudnn")

        # Real batch: two images of unequal grid.
        ms = cfg.vision_config.spatial_merge_size
        real_grids = [[1, 4, 4], [1, 2, 6]]

        # Same image content, but pad the vision arrays to a larger budget.
        # Pick a budget achievable by the padder (max_patches multiple of ms*ms).
        real_patches = sum(t * h * w for t, h, w in real_grids)
        max_images = 4
        max_patches = real_patches + (max_images - len(real_grids)) * ms * ms

        toks_a, seg_a, pv_a, grid_a, cu_a, pos_a = _make_vlm_inputs(cfg, real_grids)
        toks_b, seg_b, pv_b, grid_b, cu_b, pos_b = _make_vlm_inputs(
            cfg, real_grids, max_images=max_images, max_patches=max_patches,
        )
        np.testing.assert_array_equal(np.asarray(toks_a), np.asarray(toks_b))

        h_a, _ = model(toks_a, seg_a, None, jnp.array(0, dtype=jnp.int32),
                       pixel_values=pv_a, image_grid_thw=grid_a,
                       vision_cu_seqlens=cu_a, position_ids_ZBT=pos_a)
        h_b, _ = model(toks_b, seg_b, None, jnp.array(0, dtype=jnp.int32),
                       pixel_values=pv_b, image_grid_thw=grid_b,
                       vision_cu_seqlens=cu_b, position_ids_ZBT=pos_b)

        out_a = np.asarray(h_a, dtype=np.float32)
        out_b = np.asarray(h_b, dtype=np.float32)
        # Position (0, seq_len-1) is the fusion's fill sink — by contract it
        # holds a pad token whose loss is masked. Padded fusion overwrites it
        # with zero; unpadded leaves the embedding intact. Exclude that
        # position from the comparison.
        np.testing.assert_allclose(
            out_b[:, :-1], out_a[:, :-1], rtol=3e-2, atol=3e-2,
        )


# --------------------------------------------------------------------------- #
# D. JIT-stability: padded budget should not recompile when real counts vary
# --------------------------------------------------------------------------- #


class Qwen3_5JitStabilityTest(absltest.TestCase):
    """Static-padded vision shapes must not retrigger compilation when real
    image counts vary but the padded budget is fixed.

    We exercise the vision encoder directly, since the text shape is held
    constant by max_length padding in the real collator (orthogonal concern).
    """

    def test_vision_encoder_no_recompile_across_real_counts(self):
        cfg = _qwen3_5_vlm_test_cfg()
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model = Qwen3_5ForConditionalGeneration(cfg=cfg, rngs=nnx.Rngs(params=0))

        ms = cfg.vision_config.spatial_merge_size
        ms2 = ms * ms
        max_images = 4
        # Both batches padded to identical (max_images, max_patches).
        # Pick max_patches large enough for either real grid.
        max_patches = 64

        def padded_vision(real_grids):
            real_patches = int(sum(t * h * w for t, h, w in real_grids))
            patch_dim = (
                cfg.vision_config.in_channels
                * cfg.vision_config.patch_size
                * cfg.vision_config.patch_size
                * cfg.vision_config.temporal_patch_size
            )
            rng = np.random.RandomState(len(real_grids))
            pv = rng.randn(real_patches, patch_dim).astype(np.float32)
            grid = np.array(real_grids, dtype=np.int32)
            pv, grid, cu = _pad_vision_arrays(
                pv, grid, merge_size=ms,
                max_patches=max_patches, max_images=max_images,
            )
            return (
                jnp.asarray(pv, dtype=jnp.bfloat16),
                jnp.asarray(grid, dtype=jnp.int32),
                jnp.asarray(cu, dtype=jnp.int32),
            )

        # Side-effect counter: increments only on Python re-tracing.
        trace_count = [0]

        def vision_inner(model, pv, grid, cu):
            trace_count[0] += 1
            return model.vision(pv, grid, cu)

        vision_fwd = nnx.jit(vision_inner)

        # Sanity: choose grids whose patch counts are <= max_patches and
        # whose image counts are <= max_images.
        a = padded_vision([[1, ms, ms]])                     # 1 image, 4 patches
        b = padded_vision([[1, ms, ms], [1, ms, 2 * ms]])    # 2 images, 4+8=12 patches
        c = padded_vision([[1, 2 * ms, 2 * ms]])              # 1 image, 16 patches
        # All three must produce identical PADDED shapes.
        self.assertEqual(a[0].shape, b[0].shape)
        self.assertEqual(a[1].shape, b[1].shape)
        self.assertEqual(a[2].shape, b[2].shape)
        self.assertEqual(a[0].shape, c[0].shape)

        _ = vision_fwd(model, *a)
        jax.tree.map(lambda x: x.block_until_ready(), _)
        self.assertEqual(trace_count[0], 1, "Sanity: first call should trace once.")

        _ = vision_fwd(model, *b)
        jax.tree.map(lambda x: x.block_until_ready(), _)
        _ = vision_fwd(model, *c)
        jax.tree.map(lambda x: x.block_until_ready(), _)
        self.assertEqual(
            trace_count[0], 1,
            "Recompile triggered by changing real image counts under static padding "
            f"(trace_count={trace_count[0]}).",
        )


if __name__ == "__main__":
    absltest.main()
