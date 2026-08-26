"""Every sample's image tokens must receive that sample's own vision embeddings.

The collator pads each process's vision arrays locally and
``make_array_from_process_local_data`` concatenates the per-process blocks, so
the padding lands *between* samples in the global array. Pairing embedding rows
with image tokens positionally therefore reads off the end of an earlier block
and corrupts every sample after the first — invisibly, since train and val share
the path. These tests pin the block-stride mapping that makes the pairing exact.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from omegalax.data.collator_qwen3 import _compute_vision_cu_seqlens, _pad_vision_arrays
from omegalax.models.vision_splice import vision_scatter_index
from omegalax.vlm import api as vlm_api

MERGE_SIZE = 2
MS2 = MERGE_SIZE * MERGE_SIZE
IMAGE_TOKEN_ID = 999
SEQ_LEN = 40
FEAT_DIM = 1


def _sample_block(tag: int, real_patches: int, budget_patches: int, budget_images: int):
    """One collator-padded sample: patch rows tagged ``tag*1000 + merged_row``."""
    if real_patches:
        pixel_values = np.repeat(np.arange(real_patches // MS2) + tag * 1000, MS2).astype(
            np.float32
        )[:, None]
        grid = np.array([[1, MERGE_SIZE, real_patches // MERGE_SIZE]], dtype=np.int32)
    else:
        pixel_values = np.zeros((0, FEAT_DIM), dtype=np.float32)
        grid = np.zeros((0, 3), dtype=np.int32)
    padded_pv, _, _ = _pad_vision_arrays(
        pixel_values, grid, MERGE_SIZE, budget_patches, budget_images
    )
    return padded_pv


def _splice(pixel_values: np.ndarray, image_token_counts: list[int]) -> np.ndarray:
    """Run the model's splice and return, per sample, the tags its tokens got."""
    embeds = pixel_values.reshape(-1, MS2, FEAT_DIM).mean(axis=1)
    token_ids = np.zeros((len(image_token_counts), SEQ_LEN), dtype=np.int32)
    for row, count in enumerate(image_token_counts):
        token_ids[row, 5 : 5 + count] = IMAGE_TOKEN_ID

    mask = jnp.asarray(token_ids == IMAGE_TOKEN_ID)
    batch_idx, seq_idx = vision_scatter_index(mask, embeds.shape[0])
    spliced = (
        jnp.zeros((len(image_token_counts), SEQ_LEN, FEAT_DIM))
        .at[batch_idx, seq_idx]
        .set(jnp.asarray(embeds), mode="drop")
    )
    return np.asarray(spliced)[:, :, 0]


class VisionSpliceTest(absltest.TestCase):
    def test_each_sample_gets_its_own_embeddings_across_process_blocks(self):
        """The four-process case: one sample per process, differently sized images.

        Under the old positional pairing sample 0 was the only correct one; every
        later sample read the previous process's padding, shifted by the padding
        accumulated ahead of it.
        """
        budget_patches, budget_images = 64, 2
        real_patches = [32, 48, 16, 60]
        # make_array_from_process_local_data concatenates the process-local blocks.
        pixel_values = np.concatenate(
            [
                _sample_block(tag, patches, budget_patches, budget_images)
                for tag, patches in enumerate(real_patches)
            ],
            axis=0,
        )
        counts = [p // MS2 for p in real_patches]

        got = _splice(pixel_values, counts)

        for sample, count in enumerate(counts):
            want = [sample * 1000 + j for j in range(count)]
            self.assertEqual(
                got[sample, 5 : 5 + count].astype(int).tolist(),
                want,
                f"sample {sample} did not receive its own image embeddings, in order",
            )

    def test_text_only_sample_consumes_its_block_without_shifting_the_rest(self):
        """A text-only row still owns a block; the image rows must not slide into it."""
        budget_patches, budget_images = 32, 2
        pixel_values = np.concatenate(
            [
                _sample_block(0, 0, budget_patches, budget_images),  # text-only
                _sample_block(1, 16, budget_patches, budget_images),
                _sample_block(2, 24, budget_patches, budget_images),
            ],
            axis=0,
        )

        got = _splice(pixel_values, [0, 4, 6])

        self.assertEqual(got[1, 5:9].astype(int).tolist(), [1000, 1001, 1002, 1003])
        self.assertEqual(got[2, 5:11].astype(int).tolist(), [2000, 2001, 2002, 2003, 2004, 2005])

    def test_rejects_embeddings_that_do_not_divide_into_per_sample_blocks(self):
        mask = jnp.zeros((4, SEQ_LEN), dtype=bool).at[:, 5:7].set(True)
        with self.assertRaisesRegex(ValueError, "padded per sample"):
            vision_scatter_index(mask, 10)


class VisionSpliceModelTest(absltest.TestCase):
    """The whole forward, not just the index: a sample must not care who it rides with."""

    BUDGET_PATCHES = 24
    BUDGET_IMAGES = 2
    SEQ = 24

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if jax.default_backend() != "gpu":
            return
        cfg = vlm_api.resolve_config("qwen3-vl-smoke")
        # fp32 throughout, so a mis-spliced embedding cannot hide under bf16 noise.
        cfg = dataclasses.replace(
            cfg,
            dtype=jnp.float32,
            vision=dataclasses.replace(cfg.vision, dtype=jnp.float32),
        )
        cls.model, cls.cfg = vlm_api.init_model(
            cfg, jax.random.key(0), tp_size=1, fsdp_size=1, dp_size=1
        )
        cls.feat_dim = (
            cfg.vision.temporal_patch_size * cfg.vision.in_channels * cfg.vision.patch_size**2
        )

    def _sample(self, grid_thw, seed):
        cfg = self.cfg
        t, h, w = grid_thw
        n_patches = t * h * w
        pixel_values = np.random.RandomState(seed).randn(n_patches, self.feat_dim)
        grid = np.array([[t, h, w]], dtype=np.int32)
        token_ids = np.full(self.SEQ, 7, dtype=np.int32)
        token_ids[2] = cfg.vision_start_token_id
        merge = cfg.vision.spatial_merge_size
        token_ids[3 : 3 + n_patches // (merge * merge)] = cfg.image_token_id
        return pixel_values.astype(np.float32), grid, token_ids

    def _forward(self, samples):
        from omegalax.models.qwen3_vl.model import get_rope_index

        cfg = self.cfg
        merge = cfg.vision.spatial_merge_size
        pixel_values, grids, token_ids = zip(*samples)
        token_ids_BT = np.stack(token_ids)
        attention_mask_BT = np.ones_like(token_ids_BT)
        # position_ids come from the real grid, exactly as the collator builds them.
        position_ids, _ = get_rope_index(
            token_ids_BT,
            image_grid_thw=np.concatenate(grids),
            attention_mask=attention_mask_BT,
            spatial_merge_size=merge,
            image_token_id=cfg.image_token_id,
            video_token_id=cfg.video_token_id,
            vision_start_token_id=cfg.vision_start_token_id,
        )
        padded = [
            _pad_vision_arrays(pv, g, merge, self.BUDGET_PATCHES, self.BUDGET_IMAGES)[:2]
            for pv, g in zip(pixel_values, grids)
        ]
        grid_all = np.concatenate([g for _, g in padded])
        hidden, _ = vlm_api.forward(
            self.model,
            jnp.asarray(token_ids_BT),
            0,
            cfg,
            attention_mask_BT=jnp.asarray(attention_mask_BT),
            position_ids_ZBT=jnp.asarray(position_ids.astype(np.int32)),
            pixel_values=jnp.asarray(np.concatenate([pv for pv, _ in padded])),
            image_grid_thw=jnp.asarray(grid_all),
            vision_cu_seqlens=jnp.asarray(_compute_vision_cu_seqlens(grid_all)),
        )
        return np.asarray(hidden)

    @absltest.skipUnless(jax.default_backend() == "gpu", "vision attention is cuDNN-only")
    def test_hidden_states_do_not_depend_on_batch_position(self):
        """Each row of a batched forward must equal that sample forwarded alone.

        Differently sized images mean differently sized padding blocks, which is
        precisely what a positional splice smears across sample boundaries.
        """
        samples = [self._sample((1, 4, 4), seed=1), self._sample((1, 2, 6), seed=2)]
        batched = self._forward(samples)

        for row, sample in enumerate(samples):
            solo = self._forward([sample])[0]
            np.testing.assert_allclose(
                batched[row],
                solo,
                atol=1e-4,
                rtol=1e-4,
                err_msg=f"sample {row} changed when batched with a differently sized image",
            )


if __name__ == "__main__":
    absltest.main()
