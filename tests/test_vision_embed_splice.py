"""Vision-embedding splice under multi-process vision padding.

Each process's collator pads its own local block of ``pixel_values`` and appends
the dummy rows at the end of that block, so
``jax.make_array_from_process_local_data`` produces the global layout::

    [b0_real | b0_pad | b1_real | b1_pad | b2_real | b2_pad | b3_real | b3_pad]

while the global image-token mask is gapless. Pairing the k-th merged embedding
with the k-th ``True`` of that mask therefore only holds for batch index 0.

Spinning up four JAX processes on CPU is impractical, so these tests build the
interleaved global arrays directly, with a distinct value per sample, and drive
the pure splice helpers.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from omegalax.data.collator_qwen3 import (
    _compute_vision_cu_seqlens,
    _pad_vision_arrays,
    padded_vision_shape,
)
from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.qwen3_vl.model import _deepstack_process
from omegalax.models.vision_splice import image_embed_destinations, merged_embed_valid

MERGE_SIZE = 2
MS2 = MERGE_SIZE * MERGE_SIZE
PAD_CODE = -999.0
TEXT_CODE = 0.0
EMB_DIM = 3


def _sample_code(sample: int, token: int) -> float:
    """Distinct value per (sample, image-token) pair, readable back from output."""
    return float(100 * (sample + 1) + token)


class _Fixture:
    """Interleaved global batch for ``real_per_sample`` merged tokens per sample.

    ``slots_per_block`` is the per-process merged-embedding budget, so each
    sample contributes ``real`` real embeddings followed by
    ``slots_per_block - real`` padding embeddings.
    """

    def __init__(self, real_per_sample: list[int], slots_per_block: int, seq_len: int = 12):
        self.real_per_sample = real_per_sample
        self.slots_per_block = slots_per_block
        self.batch = len(real_per_sample)
        self.seq_len = seq_len
        self.num_embeds = self.batch * slots_per_block

        image_mask = np.zeros((self.batch, seq_len), dtype=bool)
        self.token_positions: list[list[int]] = []
        for sample, real in enumerate(real_per_sample):
            start = 1 + sample
            positions = list(range(start, start + real))
            assert positions[-1] < seq_len if positions else True
            image_mask[sample, positions] = True
            self.token_positions.append(positions)
        self.image_mask_BT = jnp.asarray(image_mask)

        embeds = np.full((self.num_embeds, EMB_DIM), PAD_CODE, dtype=np.float32)
        patch_valid = np.zeros(self.num_embeds * MS2, dtype=np.int32)
        for sample, real in enumerate(real_per_sample):
            block = sample * slots_per_block
            for token in range(real):
                embeds[block + token, :] = _sample_code(sample, token)
            patch_valid[block * MS2 : (block + real) * MS2] = 1
        self.embeds_ND = jnp.asarray(embeds)
        self.vision_patch_valid = jnp.asarray(patch_valid)

        self.inputs_embeds_BTD = jnp.full((self.batch, seq_len, EMB_DIM), TEXT_CODE)

    def valid_N(self):
        return merged_embed_valid(self.vision_patch_valid, self.num_embeds)

    def splice(self, batch_idx, seq_idx, base=None):
        base = self.inputs_embeds_BTD if base is None else base
        return base.at[batch_idx, seq_idx].set(self.embeds_ND, mode="drop")

    def legacy_destinations(self):
        """The pre-fix splice: positional pairing against the row-major mask."""
        return image_embed_destinations(self.image_mask_BT, self.num_embeds, None)

    def fixed_destinations(self):
        return image_embed_destinations(self.image_mask_BT, self.num_embeds, self.valid_N())

    def read_token(self, spliced, sample: int, token: int) -> float:
        return float(np.asarray(spliced)[sample, self.token_positions[sample][token], 0])


class _MeshTestCase(absltest.TestCase):
    """The splice helpers reshard to a replicated layout, which needs a mesh."""

    def setUp(self):
        super().setUp()
        self.enterContext(mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1))


class MergedEmbedValidTest(_MeshTestCase):
    def test_per_patch_mask_reduces_to_per_embedding_mask(self):
        fx = _Fixture([2, 3, 1, 2], slots_per_block=4)
        valid = np.asarray(fx.valid_N())
        expected = np.zeros(fx.num_embeds, dtype=bool)
        for sample, real in enumerate(fx.real_per_sample):
            expected[sample * 4 : sample * 4 + real] = True
        np.testing.assert_array_equal(valid, expected)

    def test_none_passes_through(self):
        self.assertIsNone(merged_embed_valid(None, 8))


class InterleavedPaddingSpliceTest(_MeshTestCase):
    """The four-process layout: [real|pad] per sample, concatenated."""

    def setUp(self):
        super().setUp()
        self.fx = _Fixture([2, 3, 1, 2], slots_per_block=4)

    def test_legacy_splice_misassigns_every_sample_after_the_first(self):
        spliced = self.fx.splice(*self.fx.legacy_destinations())
        for token in range(self.fx.real_per_sample[0]):
            self.assertEqual(
                self.fx.read_token(spliced, 0, token),
                _sample_code(0, token),
                "sample 0 is the one index the positional splice gets right",
            )

        wrong = 0
        total = 0
        for sample in range(1, self.fx.batch):
            for token in range(self.fx.real_per_sample[sample]):
                total += 1
                if self.fx.read_token(spliced, sample, token) != _sample_code(sample, token):
                    wrong += 1
        self.assertEqual(wrong, total, "every image token past batch index 0 must be mis-assigned")

    def test_legacy_splice_writes_padding_embeddings_into_real_tokens(self):
        spliced = np.asarray(self.fx.splice(*self.fx.legacy_destinations()))
        got_padding = [
            (sample, token)
            for sample in range(self.fx.batch)
            for token in range(self.fx.real_per_sample[sample])
            if spliced[sample, self.fx.token_positions[sample][token], 0] == PAD_CODE
        ]
        self.assertGreater(len(got_padding), 0)

    def test_fixed_splice_gives_every_token_its_own_sample_embedding(self):
        spliced = self.fx.splice(*self.fx.fixed_destinations())
        for sample in range(self.fx.batch):
            for token in range(self.fx.real_per_sample[sample]):
                self.assertEqual(
                    self.fx.read_token(spliced, sample, token),
                    _sample_code(sample, token),
                    f"sample {sample} image token {token} got the wrong embedding",
                )

    def test_fixed_splice_never_writes_a_padding_embedding(self):
        spliced = np.asarray(self.fx.splice(*self.fx.fixed_destinations()))
        self.assertEqual(int(np.sum(spliced == PAD_CODE)), 0)

    def test_fixed_splice_leaves_non_image_tokens_untouched(self):
        spliced = np.asarray(self.fx.splice(*self.fx.fixed_destinations()))
        mask = np.asarray(self.fx.image_mask_BT)
        np.testing.assert_array_equal(spliced[~mask], np.full(((~mask).sum(), EMB_DIM), TEXT_CODE))

    def test_deepstack_adds_at_the_same_destinations(self):
        destinations = self.fx.fixed_destinations()
        hidden = jnp.zeros((self.fx.batch, self.fx.seq_len, EMB_DIM))
        added = np.asarray(_deepstack_process(hidden, destinations, self.fx.embeds_ND))
        spliced = np.asarray(self.fx.splice(*destinations))
        np.testing.assert_array_equal(added, spliced)


class SingleProcessEquivalenceTest(_MeshTestCase):
    """One process pads once, after all its samples, so the layout has no gaps.

    The fix must be a no-op there: with an all-real prefix the destination of
    embedding k is the k-th image token, exactly what the old code computed.
    """

    def _single_process_fixture(self):
        real = [2, 3, 1, 2]
        fx = _Fixture(real, slots_per_block=4)
        embeds = np.full((fx.num_embeds, EMB_DIM), PAD_CODE, dtype=np.float32)
        patch_valid = np.zeros(fx.num_embeds * MS2, dtype=np.int32)
        slot = 0
        for sample, count in enumerate(real):
            for token in range(count):
                embeds[slot, :] = _sample_code(sample, token)
                slot += 1
        patch_valid[: slot * MS2] = 1
        fx.embeds_ND = jnp.asarray(embeds)
        fx.vision_patch_valid = jnp.asarray(patch_valid)
        return fx

    def test_destinations_identical_with_and_without_the_mask(self):
        fx = self._single_process_fixture()
        legacy_b, legacy_s = fx.legacy_destinations()
        fixed_b, fixed_s = fx.fixed_destinations()
        real_total = sum(fx.real_per_sample)
        np.testing.assert_array_equal(
            np.asarray(fixed_b)[:real_total], np.asarray(legacy_b)[:real_total]
        )
        np.testing.assert_array_equal(
            np.asarray(fixed_s)[:real_total], np.asarray(legacy_s)[:real_total]
        )

    def test_spliced_output_is_bit_identical(self):
        fx = self._single_process_fixture()
        np.testing.assert_array_equal(
            np.asarray(fx.splice(*fx.fixed_destinations())),
            np.asarray(fx.splice(*fx.legacy_destinations())),
        )

    def test_output_is_correct_in_both_paths(self):
        fx = self._single_process_fixture()
        spliced = fx.splice(*fx.fixed_destinations())
        for sample in range(fx.batch):
            for token in range(fx.real_per_sample[sample]):
                self.assertEqual(fx.read_token(spliced, sample, token), _sample_code(sample, token))


class UnpaddedBatchTest(_MeshTestCase):
    """No vision padding configured: every embedding is real."""

    def test_all_valid_mask_matches_positional_pairing(self):
        fx = _Fixture([2, 2, 2, 2], slots_per_block=2)
        legacy_b, legacy_s = fx.legacy_destinations()
        fixed_b, fixed_s = fx.fixed_destinations()
        np.testing.assert_array_equal(np.asarray(fixed_b), np.asarray(legacy_b))
        np.testing.assert_array_equal(np.asarray(fixed_s), np.asarray(legacy_s))


class VisionPaddingBudgetTest(absltest.TestCase):
    """``_pad_vision_arrays`` must accept every in-budget batch."""

    def _grid(self, images):
        return np.array(
            [[1, MERGE_SIZE, MERGE_SIZE] for _ in range(images)], dtype=np.int32
        ).reshape(images, 3)

    def _pixels(self, patches):
        return np.arange(patches * 2, dtype=np.float32).reshape(patches, 2)

    def test_full_image_budget_with_spare_patches_does_not_raise(self):
        max_images, max_patches = 6, 64
        grid = self._grid(max_images)
        pixels = self._pixels(max_images * MS2)
        padded_pv, padded_grid, padded_cu = _pad_vision_arrays(
            pixels,
            grid,
            merge_size=MERGE_SIZE,
            max_patches=max_patches,
            max_images=max_images,
        )
        patch_rows, grid_rows = padded_vision_shape(MERGE_SIZE, max_patches, max_images)
        self.assertEqual(padded_pv.shape[0], patch_rows)
        self.assertEqual(padded_grid.shape[0], grid_rows)
        self.assertEqual(int(padded_cu[-1]), patch_rows)

    def test_exactly_full_on_both_budgets_does_not_raise(self):
        max_images, max_patches = 4, 4 * MS2
        padded_pv, padded_grid, _ = _pad_vision_arrays(
            self._pixels(max_patches),
            self._grid(max_images),
            merge_size=MERGE_SIZE,
            max_patches=max_patches,
            max_images=max_images,
        )
        patch_rows, grid_rows = padded_vision_shape(MERGE_SIZE, max_patches, max_images)
        self.assertEqual(padded_pv.shape[0], patch_rows)
        self.assertEqual(padded_grid.shape[0], grid_rows)

    def test_padded_shapes_are_static_across_real_counts(self):
        max_images, max_patches = 6, 64
        shapes = set()
        for images in range(0, max_images + 1):
            pv, grid, cu = _pad_vision_arrays(
                self._pixels(images * MS2),
                self._grid(images),
                merge_size=MERGE_SIZE,
                max_patches=max_patches,
                max_images=max_images,
            )
            shapes.add((pv.shape, grid.shape, cu.shape))
        self.assertEqual(len(shapes), 1, f"padded shapes must not depend on real counts: {shapes}")

    def test_padding_rows_conserve_the_grid_patch_invariant(self):
        pv, grid, cu = _pad_vision_arrays(
            self._pixels(3 * MS2),
            self._grid(3),
            merge_size=MERGE_SIZE,
            max_patches=64,
            max_images=6,
        )
        self.assertEqual(pv.shape[0], int(np.sum(grid[:, 0] * grid[:, 1] * grid[:, 2])))
        np.testing.assert_array_equal(cu, _compute_vision_cu_seqlens(grid))

    def test_real_rows_come_first_so_a_prefix_mask_describes_validity(self):
        real_patches = 3 * MS2
        pixels = self._pixels(real_patches)
        pv, _, _ = _pad_vision_arrays(
            pixels,
            self._grid(3),
            merge_size=MERGE_SIZE,
            max_patches=64,
            max_images=6,
        )
        np.testing.assert_array_equal(pv[:real_patches], pixels)
        np.testing.assert_array_equal(pv[real_patches:], 0.0)

    def test_over_budget_still_raises(self):
        with self.assertRaisesRegex(ValueError, "exceeds padding budget"):
            _pad_vision_arrays(
                self._pixels(8 * MS2),
                self._grid(8),
                merge_size=MERGE_SIZE,
                max_patches=64,
                max_images=6,
            )


if __name__ == "__main__":
    absltest.main()
