"""GPU check: the vision splice is invariant to per-process vision padding.

With ``process_count>1`` each collator pads its own local block and
``jax.make_array_from_process_local_data`` concatenates those blocks, so the
global ``pixel_values`` interleaves real and padding rows
(``[b0_real|b0_pad|b1_real|b1_pad|...]``) instead of holding all real rows in a
prefix the way a single-process batch does.

The two layouts carry the same images in the same order, so the model must
produce the same hidden states from either. This runs both through the real
Qwen3.5 VLM forward on one GPU — no multi-process launch needed, since the
interleaved global array can simply be built by hand.

Requires GPU with cuDNN (head_dim=64 used to satisfy the cuDNN kernel).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx

from omegalax.data.collator_qwen3 import _compute_vision_cu_seqlens, _pad_vision_arrays
from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.qwen3_5.config import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from omegalax.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from omegalax.models.sharding_runtime import set_attn_backend

MAX_PATCHES_PER_SAMPLE = 64
MAX_IMAGES_PER_SAMPLE = 2
SEQ_LEN = 24


def _vlm_test_cfg() -> Qwen3_5Config:
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
        mrope_section=(8, 4, 4),
        intermediate_size=256,
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


def _patch_dim(cfg: Qwen3_5Config) -> int:
    v = cfg.vision_config
    return v.in_channels * v.patch_size * v.patch_size * v.temporal_patch_size


def _sample_pixels(cfg: Qwen3_5Config, grids, seed: int) -> np.ndarray:
    patches = int(sum(t * h * w for t, h, w in grids))
    return np.random.RandomState(seed).randn(patches, _patch_dim(cfg)).astype(np.float32)


def _pad_block(cfg, grids, pixels, max_patches, max_images):
    """One collator's output: real rows first, that block's padding after them."""
    real_rows = pixels.shape[0]
    padded_pv, padded_grid, _ = _pad_vision_arrays(
        pixels,
        np.array(grids, dtype=np.int32).reshape(len(grids), 3),
        merge_size=cfg.vision_config.spatial_merge_size,
        max_patches=max_patches,
        max_images=max_images,
    )
    valid = np.zeros(padded_pv.shape[0], dtype=np.int32)
    valid[:real_rows] = 1
    return padded_pv, padded_grid, valid


def _to_jax(pixel_values, grid, valid):
    return (
        jnp.asarray(pixel_values, dtype=jnp.bfloat16),
        jnp.asarray(grid, dtype=jnp.int32),
        jnp.asarray(_compute_vision_cu_seqlens(np.asarray(grid)), dtype=jnp.int32),
        jnp.asarray(valid, dtype=jnp.int32),
    )


class MultiProcessVisionLayoutTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.cfg = _vlm_test_cfg()
        self.per_sample_grids = [
            [[1, 2, 2]],
            [[1, 2, 4], [1, 2, 2]],
            [[1, 4, 4]],
            [[1, 2, 2], [1, 4, 2]],
        ]
        ms2 = self.cfg.vision_config.spatial_merge_size**2
        self.pixels = [
            _sample_pixels(self.cfg, grids, seed=i)
            for i, grids in enumerate(self.per_sample_grids)
        ]
        self.tokens_per_sample = [
            int(sum(t * h * w for t, h, w in grids) // ms2) for grids in self.per_sample_grids
        ]

        batch = len(self.per_sample_grids)
        token_ids = np.full((batch, SEQ_LEN), 7, dtype=np.int32)
        token_ids[:, 0] = 1
        for row, n_tokens in enumerate(self.tokens_per_sample):
            token_ids[row, 1 : 1 + n_tokens] = self.cfg.image_token_id
        self.token_ids = jnp.asarray(token_ids)
        self.segment_ids = jnp.ones_like(self.token_ids)
        self.position_ids = jnp.asarray(
            np.broadcast_to(
                np.arange(SEQ_LEN, dtype=np.int32)[None, None, :], (3, batch, SEQ_LEN)
            ).copy()
        )

        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            self.model = Qwen3_5ForConditionalGeneration(cfg=self.cfg, rngs=nnx.Rngs(params=0))
        set_attn_backend(self.model, text_backend="cudnn")

    def _interleaved(self):
        blocks = [
            _pad_block(
                self.cfg,
                grids,
                pixels,
                MAX_PATCHES_PER_SAMPLE,
                MAX_IMAGES_PER_SAMPLE,
            )
            for grids, pixels in zip(self.per_sample_grids, self.pixels)
        ]
        return _to_jax(
            np.concatenate([b[0] for b in blocks], axis=0),
            np.concatenate([b[1] for b in blocks], axis=0),
            np.concatenate([b[2] for b in blocks], axis=0),
        )

    def _compact(self):
        n = len(self.per_sample_grids)
        grids = [g for sample in self.per_sample_grids for g in sample]
        pixels = np.concatenate(self.pixels, axis=0)
        return _to_jax(
            *_pad_block(
                self.cfg,
                grids,
                pixels,
                MAX_PATCHES_PER_SAMPLE * n,
                MAX_IMAGES_PER_SAMPLE * n,
            )
        )

    def _forward(self, vision, with_mask: bool):
        pixel_values, grid, cu, valid = vision
        hidden, _ = self.model(
            self.token_ids,
            self.segment_ids,
            None,
            jnp.array(0, dtype=jnp.int32),
            pixel_values=pixel_values,
            image_grid_thw=grid,
            vision_cu_seqlens=cu,
            position_ids_ZBT=self.position_ids,
            vision_patch_valid=valid if with_mask else None,
        )
        return np.asarray(hidden, dtype=np.float32)

    def test_interleaved_layout_matches_single_block_layout(self):
        compact = self._forward(self._compact(), with_mask=True)
        interleaved = self._forward(self._interleaved(), with_mask=True)
        np.testing.assert_allclose(interleaved, compact, rtol=3e-2, atol=3e-2)

    def test_positional_splice_diverges_on_the_interleaved_layout(self):
        """Without the mask the splice reverts to the buggy positional pairing."""
        compact = self._forward(self._compact(), with_mask=True)
        legacy = self._forward(self._interleaved(), with_mask=False)
        max_delta = float(np.max(np.abs(legacy - compact)))
        self.assertGreater(
            max_delta,
            1e-2,
            "positional pairing should mis-route embeddings on the interleaved layout",
        )

    def test_single_block_layout_is_unaffected_by_the_mask(self):
        with_mask = self._forward(self._compact(), with_mask=True)
        without_mask = self._forward(self._compact(), with_mask=False)
        np.testing.assert_array_equal(with_mask, without_mask)


if __name__ == "__main__":
    absltest.main()
