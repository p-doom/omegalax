"""Correctness tests for sequence packing in the VLM SFT pipeline.

The cardinal sin of packing is cross-segment leakage: a token of one packed
sub-sequence attending to, or being trained to predict into, another. These
tests prove that is impossible:

* ``PackedForwardEquivalenceTest`` — the load-bearing test. Hidden states of a
  PACKED row equal, segment-by-segment, the hidden states of the SAME sequences
  run UNPACKED (one per row). A negative control runs the identical packed row
  WITHOUT the segment mask (naive full-attention pack) and shows it differs by a
  large margin, so the test genuinely catches leakage.
* ``PackedLossEquivalenceTest`` — total masked loss and gradients are identical
  whether sequences are packed or run one-per-row, for num_loss_tiles in {1, 4}.
* ``PackedCollatorTest`` — segment_ids, per-segment position_ids (incl. image
  mRoPE), and the boundary-zeroed loss mask are constructed correctly, and image
  features are concatenated in token order.
* ``SequencePackerTest`` — the greedy next-fit grouping is correct (no overflow,
  no dropped/duplicated records, edge cases) and checkpoints round-trip.

The model tests use a tiny random Qwen3-VL on CPU with the ``xla`` attention
backend (the ``mosaic_gpu`` production backend enforces the same ``k_start``
mask; see ``VisionTowerPackingEquivalenceTest`` for the GPU vision path).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx

from omegalax.models.qwen3_vl.config import make_vl_config
from omegalax.models.qwen3_vl.model import get_rope_index
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.trainers.loss import chunked_cross_entropy_loss
from omegalax.vlm import api as vlm_api

_VOCAB = 1024
_MAX_LEN = 24
# Token ids kept clear of the smoke config's special ids (image=2, video=3,
# vision_start=4) so text-only sequences contain no accidental vision tokens.
_TEXT_LO = 10


def _build_smoke_model(seed: int = 0):
    cfg = make_vl_config("qwen3-vl-smoke")
    model, cfg = vlm_api.init_model(cfg, jax.random.key(seed), tp_size=1, fsdp_size=1, dp_size=1)
    set_attn_backend(model, text_backend="xla")
    return model, cfg


def _text_seqs(lengths, seed=0):
    rng = np.random.RandomState(seed)
    return [rng.randint(_TEXT_LO, _VOCAB, size=L).astype(np.int32) for L in lengths]


def _pack_row(seqs, max_len=_MAX_LEN):
    """Concatenate text sequences into one packed row's arrays."""
    T = max_len
    ids = np.zeros((1, T), np.int32)
    attn = np.zeros((1, T), np.int32)
    seg = np.zeros((1, T), np.int32)
    pos = np.zeros((3, 1, T), np.int32)
    off = 0
    for i, s in enumerate(seqs, start=1):
        L = len(s)
        ids[0, off : off + L] = s
        attn[0, off : off + L] = 1
        seg[0, off : off + L] = i
        pos[:, 0, off : off + L] = np.arange(L)
        off += L
    return ids, attn, seg, pos


def _unpacked_row(seq, max_len=_MAX_LEN):
    T = max_len
    L = len(seq)
    ids = np.zeros((1, T), np.int32)
    ids[0, :L] = seq
    attn = np.zeros((1, T), np.int32)
    attn[0, :L] = 1
    pos = np.zeros((3, 1, T), np.int32)
    pos[:, 0, :L] = np.arange(L)
    return ids, attn, pos


class PackedForwardEquivalenceTest(absltest.TestCase):
    """Packed hidden states equal unpacked, and a naive (unmasked) pack does not."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model, cls.cfg = _build_smoke_model(seed=0)

    def _hidden(self, ids, attn, pos, seg=None):
        seg_arr = None if seg is None else jnp.asarray(seg)
        h, _ = self.model(
            jnp.asarray(ids),
            jnp.asarray(attn),
            position_ids_ZBT=jnp.asarray(pos),
            segment_ids_BT=seg_arr,
        )
        return np.asarray(h[0])

    def test_packed_equals_unpacked_no_leakage(self):
        seqs = _text_seqs([7, 5, 4], seed=1)
        ids, attn, seg, pos = _pack_row(seqs)
        h_packed = self._hidden(ids, attn, pos, seg=seg)
        h_naive = self._hidden(ids, attn, pos, seg=None)  # full causal over the pack

        off = 0
        max_pack = 0.0
        max_naive = 0.0
        for s in seqs:
            L = len(s)
            u_ids, u_attn, u_pos = _unpacked_row(s)
            h_u = self._hidden(u_ids, u_attn, u_pos)[:L]
            max_pack = max(max_pack, float(np.abs(h_packed[off : off + L] - h_u).max()))
            max_naive = max(max_naive, float(np.abs(h_naive[off : off + L] - h_u).max()))
            off += L

        # Packed == unpacked (bit-identical in practice for the xla reference).
        self.assertLess(max_pack, 1e-3, f"packed vs unpacked hidden diff too large: {max_pack}")
        # Negative control: the naive full-attention pack leaks across segments
        # and must differ substantially, proving the test would catch leakage.
        self.assertGreater(
            max_naive,
            0.1,
            f"naive unmasked pack did not differ from unpacked ({max_naive}); "
            "the equivalence test is not meaningful.",
        )
        self.assertGreater(max_naive, 100 * max(max_pack, 1e-9))

    def test_single_sequence_pack_is_identity(self):
        (s,) = _text_seqs([9], seed=2)
        ids, attn, seg, pos = _pack_row([s])
        h_packed = self._hidden(ids, attn, pos, seg=seg)[: len(s)]
        u_ids, u_attn, u_pos = _unpacked_row(s)
        h_u = self._hidden(u_ids, u_attn, u_pos)[: len(s)]
        self.assertLess(float(np.abs(h_packed - h_u).max()), 1e-3)

    def test_pack_exactly_fills_max_length(self):
        # Two segments summing to exactly max_length (no padding segment).
        seqs = _text_seqs([_MAX_LEN - 8, 8], seed=3)
        self.assertEqual(sum(len(s) for s in seqs), _MAX_LEN)
        ids, attn, seg, pos = _pack_row(seqs)
        self.assertTrue(bool((seg > 0).all()))  # no padding
        h_packed = self._hidden(ids, attn, pos, seg=seg)
        off = 0
        for s in seqs:
            L = len(s)
            u_ids, u_attn, u_pos = _unpacked_row(s)
            h_u = self._hidden(u_ids, u_attn, u_pos)[:L]
            self.assertLess(float(np.abs(h_packed[off : off + L] - h_u).max()), 1e-3)
            off += L

    def test_mixed_short_and_long_segments(self):
        seqs = _text_seqs([1, 11, 2], seed=4)
        ids, attn, seg, pos = _pack_row(seqs)
        h_packed = self._hidden(ids, attn, pos, seg=seg)
        off = 0
        for s in seqs:
            L = len(s)
            u_ids, u_attn, u_pos = _unpacked_row(s)
            h_u = self._hidden(u_ids, u_attn, u_pos)[:L]
            self.assertLess(float(np.abs(h_packed[off : off + L] - h_u).max()), 1e-3)
            off += L


class PackedLossEquivalenceTest(absltest.TestCase):
    """Loss and gradients are identical packed vs one-per-row."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model, cls.cfg = _build_smoke_model(seed=5)

    def _loss_mask(self, seqs):
        """A deterministic per-token loss mask; first token of each segment 0
        (boundary label-mask), as the packed collator produces."""
        masks = []
        for s in seqs:
            m = np.ones(len(s), np.int32)
            m[0] = 0  # boundary: first token is never a supervised target
            masks.append(m)
        return masks

    def _packed_arrays(self, seqs, masks):
        T = _MAX_LEN
        ids = np.zeros((1, T), np.int32)
        attn = np.zeros((1, T), np.int32)
        seg = np.zeros((1, T), np.int32)
        pos = np.zeros((3, 1, T), np.int32)
        lm = np.zeros((1, T), np.int32)
        off = 0
        for i, (s, m) in enumerate(zip(seqs, masks), start=1):
            L = len(s)
            ids[0, off : off + L] = s
            attn[0, off : off + L] = 1
            seg[0, off : off + L] = i
            pos[:, 0, off : off + L] = np.arange(L)
            lm[0, off : off + L] = m
            off += L
        return ids, attn, seg, pos, lm

    def _unpacked_batch(self, seqs, masks):
        K, T = len(seqs), _MAX_LEN
        ids = np.zeros((K, T), np.int32)
        attn = np.zeros((K, T), np.int32)
        pos = np.zeros((3, K, T), np.int32)
        lm = np.zeros((K, T), np.int32)
        for k, (s, m) in enumerate(zip(seqs, masks)):
            L = len(s)
            ids[k, :L] = s
            attn[k, :L] = 1
            pos[:, k, :L] = np.arange(L)
            lm[k, :L] = m
        return ids, attn, pos, lm

    def _loss_fn(self, ids, attn, pos, lm, num_tiles, seg=None):
        ids_j = jnp.asarray(ids)
        lm_j = jnp.asarray(lm)
        seg_j = None if seg is None else jnp.asarray(seg)

        def loss(model):
            h, _ = model(
                ids_j,
                jnp.asarray(attn),
                position_ids_ZBT=jnp.asarray(pos),
                segment_ids_BT=seg_j,
            )
            return chunked_cross_entropy_loss(
                h, model.output_weight(), ids_j, lm_j, num_tiles=num_tiles
            )

        return loss

    def _value_and_grad(self, loss_fn):
        # tokamax's attention lowers through shard_map, which cannot be eagerly
        # differentiated; jit the value_and_grad exactly as the real trainer does.
        @nnx.jit
        def vg(model):
            return nnx.value_and_grad(loss_fn)(model)

        return vg(self.model)

    def test_loss_and_grad_equivalence(self):
        seqs = _text_seqs([7, 6, 5], seed=6)
        masks = self._loss_mask(seqs)
        p_ids, p_attn, p_seg, p_pos, p_lm = self._packed_arrays(seqs, masks)
        u_ids, u_attn, u_pos, u_lm = self._unpacked_batch(seqs, masks)

        for num_tiles in (1, 4):
            packed_loss_fn = self._loss_fn(p_ids, p_attn, p_pos, p_lm, num_tiles, seg=p_seg)
            unpacked_loss_fn = self._loss_fn(u_ids, u_attn, u_pos, u_lm, num_tiles, seg=None)

            l_packed, g_packed = self._value_and_grad(packed_loss_fn)
            l_unpacked, g_unpacked = self._value_and_grad(unpacked_loss_fn)
            self.assertAlmostEqual(
                float(l_packed),
                float(l_unpacked),
                places=4,
                msg=f"loss mismatch (num_tiles={num_tiles})",
            )

            lp = jax.tree.leaves(nnx.state(g_packed))
            lu = jax.tree.leaves(nnx.state(g_unpacked))
            max_diff = max(float(jnp.abs(a - b).max()) for a, b in zip(lp, lu) if a.size)
            grad_norm = float(
                jnp.sqrt(sum(jnp.sum(jnp.asarray(a, jnp.float32) ** 2) for a in lp))
            )
            self.assertLess(
                max_diff,
                1e-2 * max(grad_norm, 1e-6),
                f"grad mismatch (num_tiles={num_tiles}): max_diff={max_diff} norm={grad_norm}",
            )

    def test_num_loss_tiles_invariant(self):
        seqs = _text_seqs([7, 6, 5], seed=6)
        masks = self._loss_mask(seqs)
        p_ids, p_attn, p_seg, p_pos, p_lm = self._packed_arrays(seqs, masks)
        l1, _ = self._value_and_grad(self._loss_fn(p_ids, p_attn, p_pos, p_lm, 1, seg=p_seg))
        l4, _ = self._value_and_grad(self._loss_fn(p_ids, p_attn, p_pos, p_lm, 4, seg=p_seg))
        self.assertAlmostEqual(float(l1), float(l4), places=4)


class PackedCollatorTest(absltest.TestCase):
    """PackedVLMSFTCollator metadata: segment_ids, position reset, loss mask, vision."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from transformers import AutoImageProcessor, AutoTokenizer

        from omegalax.data.collator_qwen3 import PACK_EXAMPLES_KEY, PackedVLMSFTCollator

        cls.PACK_KEY = PACK_EXAMPLES_KEY
        cls.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        cls.ip = AutoImageProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", use_fast=False)
        cls.max_len = 4096
        cls.collator = PackedVLMSFTCollator(cls.tok, cls.max_len, cls.ip)
        cls.image_pad_id = cls.tok.convert_tokens_to_ids("<|image_pad|>")

    @staticmethod
    def _text_ex(user, asst):
        return {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": asst},
            ]
        }

    def _img_ex(self, asst):
        from PIL import Image

        img = Image.fromarray(
            (np.random.RandomState(0).rand(56, 56, 3) * 255).astype(np.uint8)
        )
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "describe"},
                    ],
                },
                {"role": "assistant", "content": asst},
            ]
        }

    def test_segment_ids_and_boundary_loss_mask(self):
        exs = [self._text_ex("hello there", "hi"), self._text_ex("what is 2+2", "four")]
        pack = {self.PACK_KEY: exs}
        batch = self.collator([pack])
        seg = batch["segment_ids_BT"][0]
        attn = batch["attention_mask_BT"][0]
        lm = batch["loss_mask_BT"][0]

        # segment ids: strictly 1,2 over real tokens, 0 on padding, contiguous.
        real = attn == 1
        self.assertEqual(sorted(set(seg[real].tolist())), [1, 2])
        self.assertTrue(bool((seg[~real] == 0).all()))
        # segment starts: first index of each distinct positive segment.
        starts = [0]
        for j in range(1, self.max_len):
            if seg[j] != seg[j - 1] and seg[j] != 0:
                starts.append(j)
        for j in starts:
            self.assertEqual(int(lm[j]), 0, f"loss mask must be 0 at segment start {j}")
        # There is supervised signal somewhere (assistant tokens).
        self.assertGreater(int(lm.sum()), 0)

    def test_position_ids_reset_per_segment(self):
        exs = [self._text_ex("alpha beta gamma", "delta"), self._text_ex("one two", "three")]
        pack = {self.PACK_KEY: exs}
        batch = self.collator([pack])
        seg = batch["segment_ids_BT"][0]
        pos = batch["position_ids_ZBT"][:, 0, :]  # (3, T)

        # Every segment (and the padding block) starts at mRoPE position 0.
        prev = None
        for j in range(self.max_len):
            if seg[j] != prev:
                self.assertTrue(
                    bool((pos[:, j] == 0).all()),
                    f"segment starting at {j} must reset positions to 0, got {pos[:, j]}",
                )
                prev = seg[j]

        # A naive whole-row rope index would NOT reset — confirm the reset matters.
        ids_row = batch["token_ids_BT"]
        naive_pos, _ = get_rope_index(
            ids_row,
            image_grid_thw=None,
            attention_mask=batch["attention_mask_BT"],
            spatial_merge_size=self.ip.merge_size,
            image_token_id=self.tok.convert_tokens_to_ids("<|image_pad|>"),
            video_token_id=self.tok.convert_tokens_to_ids("<|video_pad|>"),
            vision_start_token_id=self.tok.convert_tokens_to_ids("<|vision_start|>"),
        )
        # second segment start position under packing is 0 but continuous would be >0
        seg2_start = int(np.argmax(seg == 2))
        self.assertEqual(int(pos[0, seg2_start]), 0)
        self.assertGreater(int(naive_pos[0, 0, seg2_start]), 0)

    def test_vision_features_concatenated_in_token_order(self):
        # Pack an image example then a text example. Image features / grid must be
        # accumulated in row-major token order and align with <|image_pad|> tokens.
        exs = [self._img_ex("a cat"), self._text_ex("hi", "yo")]
        pack = {self.PACK_KEY: exs}
        batch = self.collator([pack])
        grid = batch["image_grid_thw"]
        self.assertEqual(grid.shape[0], 1)  # one image
        # number of <|image_pad|> tokens == merged vision tokens of that image
        t, h, w = [int(x) for x in grid[0]]
        ms = self.ip.merge_size
        expected_tokens = t * (h // ms) * (w // ms)
        n_image_pad = int((batch["token_ids_BT"] == self.image_pad_id).sum())
        self.assertEqual(n_image_pad, expected_tokens)
        # pixel rows == sum of t*h*w patches
        self.assertEqual(batch["pixel_values"].shape[0], t * h * w)
        # image tokens live inside segment 1 only
        seg = batch["segment_ids_BT"][0]
        img_positions = np.where(batch["token_ids_BT"][0] == self.image_pad_id)[0]
        self.assertTrue(bool((seg[img_positions] == 1).all()))

    def test_position_ids_match_per_example_standalone(self):
        # Packing must concatenate per-example positions, not recompute globally.
        ex1 = self._text_ex("alpha beta gamma", "delta")
        ex2 = self._text_ex("one two", "three")
        batch = self.collator([{self.PACK_KEY: [ex1, ex2]}])
        seg = batch["segment_ids_BT"][0]
        pos = batch["position_ids_ZBT"][:, 0, :]
        for seg_id, ex in ((1, ex1), (2, ex2)):
            enc = self.collator._encode_one(ex)
            L = enc["input_ids"].shape[0]
            idx = np.where(seg == seg_id)[0]
            self.assertEqual(len(idx), L)
            np.testing.assert_array_equal(pos[:, idx], enc["position_ids_3L"])


class SequencePackerTest(absltest.TestCase):
    """Greedy next-fit grouping correctness + checkpoint round-trip."""

    def _pack(self, lengths, max_length):
        import grain

        from omegalax.data.collator_qwen3 import PACK_EXAMPLES_KEY
        from omegalax.data.packing import MEASURED_LENGTH_KEY, SequencePackIterDataset

        records = [
            {"id": i, MEASURED_LENGTH_KEY: int(L), "source_ids": 0}
            for i, L in enumerate(lengths)
        ]
        ds = grain.MapDataset.source(records).to_iter_dataset()
        packed = SequencePackIterDataset(ds, max_length=max_length)
        return [[e["id"] for e in p[PACK_EXAMPLES_KEY]] for p in packed]

    def test_next_fit_grouping(self):
        # lengths 5,5,3,10,2 with capacity 10 -> [5,5],[3],[10],[2]
        packs = self._pack([5, 5, 3, 10, 2], max_length=10)
        self.assertEqual(packs, [[0, 1], [2], [3], [4]])

    def test_no_pack_overflows_and_all_records_present(self):
        rng = np.random.RandomState(0)
        lengths = rng.randint(1, 17, size=200).tolist()
        max_length = 16
        packs = self._pack(lengths, max_length)
        seen = []
        for p in packs:
            total = sum(lengths[i] for i in p)
            self.assertLessEqual(total, max_length)
            self.assertGreater(len(p), 0)
            seen.extend(p)
        # every record present exactly once, in order
        self.assertEqual(seen, list(range(len(lengths))))

    def test_exact_fit_and_single_long_record(self):
        packs = self._pack([16, 8, 8, 16], max_length=16)
        self.assertEqual(packs, [[0], [1, 2], [3]])

    def test_record_longer_than_max_length_raises(self):
        with self.assertRaises(ValueError):
            self._pack([5, 20, 3], max_length=16)

    def test_missing_length_key_raises(self):
        import grain

        from omegalax.data.packing import SequencePackIterDataset

        ds = grain.MapDataset.source([{"id": 0}]).to_iter_dataset()
        with self.assertRaises(KeyError):
            list(SequencePackIterDataset(ds, max_length=16))

    def test_checkpoint_roundtrip(self):
        import grain

        from omegalax.data.collator_qwen3 import PACK_EXAMPLES_KEY
        from omegalax.data.packing import MEASURED_LENGTH_KEY, SequencePackIterDataset

        rng = np.random.RandomState(1)
        lengths = rng.randint(1, 9, size=60).tolist()
        records = [{"id": i, MEASURED_LENGTH_KEY: int(L)} for i, L in enumerate(lengths)]
        ds = grain.MapDataset.source(records).to_iter_dataset()
        packed = SequencePackIterDataset(ds, max_length=16)

        def ids(pack):
            return [e["id"] for e in pack[PACK_EXAMPLES_KEY]]

        it = iter(packed)
        first = [ids(next(it)) for _ in range(4)]
        state = it.get_state()
        rest_a = [ids(p) for p in it]

        it2 = iter(packed)
        it2.set_state(state)
        rest_b = [ids(p) for p in it2]
        self.assertEqual(rest_a, rest_b)

        # Full run equals first-4 + resumed-rest (no dropped/duplicated packs).
        full = [ids(p) for p in iter(packed)]
        self.assertEqual(full, first + rest_a)


@absltest.skipIf(
    not any(d.platform == "gpu" for d in jax.devices()),
    "vision tower uses the cuDNN packed kernel; requires a GPU",
)
class VisionTowerPackingEquivalenceTest(absltest.TestCase):
    """Full multimodal forward: a packed [image-segment, text-segment] row equals
    the two segments run unpacked. Runs only where a GPU is present."""

    def test_image_segment_equivalence(self):
        model, cfg = _build_smoke_model(seed=7)
        set_attn_backend(model, text_backend="mosaic_gpu")
        ms = cfg.vision.spatial_merge_size
        img_tok = cfg.image_token_id
        vstart = cfg.vision_start_token_id
        grid = np.array([[1, 2 * ms, 2 * ms]], np.int32)  # one image
        n_img = 1 * (grid[0, 1] // ms) * (grid[0, 2] // ms)
        patch_feat = (
            cfg.vision.temporal_patch_size
            * cfg.vision.in_channels
            * cfg.vision.patch_size
            * cfg.vision.patch_size
        )
        n_patches = int(grid[0, 0] * grid[0, 1] * grid[0, 2])
        rng = np.random.RandomState(0)
        pv = rng.randn(n_patches, patch_feat).astype(np.float32)
        cu = np.array([0, n_patches], np.int32)

        seg1 = np.array([vstart] + [img_tok] * n_img + [30, 31], np.int32)
        seg2 = np.array([40, 41, 42], np.int32)
        L1, L2 = len(seg1), len(seg2)
        T = _MAX_LEN

        def rope(ids, grid_arg):
            p, _ = get_rope_index(
                ids[None],
                image_grid_thw=grid_arg,
                attention_mask=np.ones((1, len(ids)), np.int32),
                spatial_merge_size=ms,
                image_token_id=img_tok,
                video_token_id=cfg.video_token_id,
                vision_start_token_id=vstart,
            )
            return p[:, 0, :]

        # packed row
        ids = np.zeros((1, T), np.int32)
        attn = np.zeros((1, T), np.int32)
        seg = np.zeros((1, T), np.int32)
        pos = np.zeros((3, 1, T), np.int32)
        ids[0, :L1] = seg1
        ids[0, L1 : L1 + L2] = seg2
        attn[0, : L1 + L2] = 1
        seg[0, :L1] = 1
        seg[0, L1 : L1 + L2] = 2
        pos[:, 0, :L1] = rope(seg1, grid)
        pos[:, 0, L1 : L1 + L2] = rope(seg2, None)
        h_packed, _ = model(
            jnp.asarray(ids),
            jnp.asarray(attn),
            position_ids_ZBT=jnp.asarray(pos),
            pixel_values=jnp.asarray(pv),
            image_grid_thw=jnp.asarray(grid),
            vision_cu_seqlens=jnp.asarray(cu),
            segment_ids_BT=jnp.asarray(seg),
        )
        h_packed = np.asarray(h_packed[0])

        # unpacked image segment
        ids1 = np.zeros((1, T), np.int32)
        ids1[0, :L1] = seg1
        attn1 = np.zeros((1, T), np.int32)
        attn1[0, :L1] = 1
        pos1 = np.zeros((3, 1, T), np.int32)
        pos1[:, 0, :L1] = rope(seg1, grid)
        h_u1, _ = model(
            jnp.asarray(ids1),
            jnp.asarray(attn1),
            position_ids_ZBT=jnp.asarray(pos1),
            pixel_values=jnp.asarray(pv),
            image_grid_thw=jnp.asarray(grid),
            vision_cu_seqlens=jnp.asarray(cu),
        )
        h_u1 = np.asarray(h_u1[0])[:L1]

        # unpacked text segment
        ids2, attn2, pos2 = _unpacked_row(seg2)
        h_u2, _ = model(jnp.asarray(ids2), jnp.asarray(attn2), position_ids_ZBT=jnp.asarray(pos2))
        h_u2 = np.asarray(h_u2[0])[:L2]

        self.assertLess(float(np.abs(h_packed[:L1] - h_u1).max()), 5e-2)
        self.assertLess(float(np.abs(h_packed[L1 : L1 + L2] - h_u2).max()), 5e-2)


if __name__ == "__main__":
    absltest.main()
