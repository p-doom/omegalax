"""Tests for SFT collators: loss-mask correctness, multi-turn, and overflow checks."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import dataclasses
import pickle
from unittest import mock

import ml_dtypes
import numpy as np
from absl.testing import absltest
from PIL import Image
from renderers import Qwen3RendererConfig
from transformers import AutoImageProcessor, AutoTokenizer

from omegalax.data import collator_qwen3
from omegalax.data.collator_qwen3 import (
    Qwen3RendererEncoder,
    TextSFTCollator,
    VLMSFTCollator,
    make_message_length_fn,
)


def _make_tokenizer():
    """Use a small, fast tokenizer available offline or from HF cache."""
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)


def _runs(mask):
    """Contiguous [start, end) spans where mask is nonzero."""
    m = np.asarray(mask, dtype=bool)
    edges = np.diff(np.concatenate(([False], m, [False])).astype(np.int8))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True))


class TextSFTCollatorTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = _make_tokenizer()
        self.max_length = 128
        self.collator = TextSFTCollator(self.tokenizer, max_length=self.max_length)

    def test_output_keys_and_shapes(self):
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            },
        ]
        batch = self.collator(examples)
        self.assertIn("token_ids_BT", batch)
        self.assertIn("attention_mask_BT", batch)
        self.assertIn("loss_mask_BT", batch)
        self.assertEqual(batch["token_ids_BT"].shape, (1, self.max_length))
        self.assertEqual(batch["attention_mask_BT"].shape, (1, self.max_length))
        self.assertEqual(batch["loss_mask_BT"].shape, (1, self.max_length))

    def test_loss_mask_zero_on_padding(self):
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "Say X"},
                    {"role": "assistant", "content": "X"},
                ]
            },
        ]
        batch = self.collator(examples)
        attn = batch["attention_mask_BT"][0]
        mask = batch["loss_mask_BT"][0]
        # Where attention is 0 (padding), loss_mask must also be 0
        self.assertTrue(np.all(mask[attn == 0] == 0))

    def test_loss_mask_only_on_assistant_tokens(self):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
        batch = self.collator([{"messages": messages}])
        mask = batch["loss_mask_BT"][0].astype(bool)
        # The whole supervised set, decoded: the assistant body (Qwen3's template
        # prefixes the final turn with the empty think block) plus the stop token,
        # and nothing from the user turn or the ChatML headers.
        self.assertEqual(
            self.tokenizer.decode(batch["token_ids_BT"][0][mask].tolist()),
            "<think>\n\n</think>\n\nThe answer is 4.<|im_end|>",
        )

    def test_multiturn_masks_all_assistant_spans(self):
        """Historical assistant turns are supervised too, one contiguous run each."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am fine, thanks."},
        ]
        batch = self.collator([{"messages": messages}])
        ids = batch["token_ids_BT"][0]
        mask = batch["loss_mask_BT"][0]
        runs = _runs(mask)
        # The think block lands on the FINAL assistant turn only, so the two runs
        # differ in shape; a single supervised-token sum would not see one go missing.
        self.assertEqual(
            [self.tokenizer.decode(ids[s:e].tolist()) for s, e in runs],
            ["Hello!<|im_end|>", "<think>\n\n</think>\n\nI am fine, thanks.<|im_end|>"],
        )
        self.assertEqual(int(np.sum(mask)), sum(e - s for s, e in runs))

    def test_batch_size(self):
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "A"},
                    {"role": "assistant", "content": "B"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "C"},
                    {"role": "assistant", "content": "D"},
                ]
            },
        ]
        batch = self.collator(examples)
        self.assertEqual(batch["token_ids_BT"].shape[0], 2)

    def test_dtypes_are_int32(self):
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "X"},
                    {"role": "assistant", "content": "Y"},
                ]
            },
        ]
        batch = self.collator(examples)
        for key in ("token_ids_BT", "attention_mask_BT", "loss_mask_BT"):
            self.assertEqual(batch[key].dtype, np.int32, f"{key} dtype mismatch")

    def test_raises_on_overflow(self):
        collator = TextSFTCollator(self.tokenizer, max_length=8)
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "Tell me a story in many words."},
                    {
                        "role": "assistant",
                        "content": "This answer is intentionally too long for the tiny max length.",
                    },
                ]
            },
        ]
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            collator(examples)


class VLMSFTCollatorTest(absltest.TestCase):
    """Tests for the VLM SFT collator with real images."""

    def setUp(self):
        super().setUp()
        self.tokenizer = _make_tokenizer()
        self.image_processor = AutoImageProcessor.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            use_fast=False,  # force the numpy codepath
        )
        self.max_length = 256
        self.collator = VLMSFTCollator(
            self.tokenizer,
            max_length=self.max_length,
            image_processor=self.image_processor,
        )

    def test_text_only_example(self):
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            },
        ]
        batch = self.collator(examples)
        self.assertIn("token_ids_BT", batch)
        self.assertEqual(batch["token_ids_BT"].shape, (1, self.max_length))
        # Vision arrays are always emitted (as empty placeholders for text-only
        # batches) so the JIT pytree structure stays constant and never recompiles.
        self.assertIn("pixel_values", batch)
        self.assertEqual(batch["pixel_values"].shape[0], 0)
        self.assertIn("image_grid_thw", batch)
        self.assertEqual(batch["image_grid_thw"].shape, (0, 3))

    def test_multimodal_example(self):
        from PIL import Image

        img = Image.new("RGB", (200, 200), color=(100, 150, 200))
        examples = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "Describe."},
                        ],
                    },
                    {"role": "assistant", "content": "A solid color image."},
                ]
            },
        ]
        batch = self.collator(examples)
        self.assertIn("token_ids_BT", batch)
        self.assertIn("pixel_values", batch)
        self.assertIn("image_grid_thw", batch)
        self.assertEqual(batch["token_ids_BT"].shape, (1, self.max_length))
        self.assertEqual(batch["image_grid_thw"].shape[1], 3)

        image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        token_ids = batch["token_ids_BT"][0]
        n_pad = int(np.sum(token_ids == image_pad_id))
        grid = batch["image_grid_thw"][0]
        expected_pads = int(grid[0]) * (int(grid[1]) // 2) * (int(grid[2]) // 2)
        self.assertEqual(n_pad, expected_pads)

    def test_pixel_values_dtype_is_bf16_by_default(self):
        from PIL import Image

        img = Image.new("RGB", (100, 100), color=(10, 20, 30))
        examples = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "Describe."},
                        ],
                    },
                    {"role": "assistant", "content": "ok."},
                ]
            },
        ]
        # Default: pixel_values are downcast to bf16 (the vision patch embed
        # computes in bf16 anyway), halving the input buffer with no numerical
        # change vs the implicit fp32->bf16 cast inside the Linear.
        self.assertEqual(self.collator(examples)["pixel_values"].dtype, ml_dtypes.bfloat16)

        # The override is honored, e.g. for a full-fp32-compute run.
        fp32_collator = VLMSFTCollator(
            self.tokenizer,
            max_length=self.max_length,
            image_processor=self.image_processor,
            pixel_values_dtype=np.float32,
        )
        self.assertEqual(fp32_collator(examples)["pixel_values"].dtype, np.float32)

    def test_loss_mask_on_assistant_only(self):
        from PIL import Image

        img = Image.new("RGB", (100, 100), color=(50, 50, 50))
        examples = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "What?"},
                        ],
                    },
                    {"role": "assistant", "content": "Nothing special."},
                ]
            },
        ]
        batch = self.collator(examples)
        token_ids = batch["token_ids_BT"][0]
        mask = batch["loss_mask_BT"][0].astype(bool)
        self.assertEqual(
            self.tokenizer.decode(token_ids[mask].tolist()), "Nothing special.<|im_end|>"
        )
        # The 64 image pads are the positive control for mask leakage: the
        # pre-renderers encoder supervised every one of them.
        is_pad = token_ids == self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.assertEqual(int(np.sum(is_pad)), 64)
        self.assertEqual(int(np.sum(mask[is_pad])), 0)

    def test_raises_on_overflow(self):
        from PIL import Image

        img = Image.new("RGB", (200, 200), color=(100, 150, 200))
        collator = VLMSFTCollator(
            self.tokenizer,
            max_length=8,
            image_processor=self.image_processor,
        )
        examples = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "Describe."},
                        ],
                    },
                    {"role": "assistant", "content": "A solid color image."},
                ]
            },
        ]
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            collator(examples)

    def test_heterogeneous_batch_text_and_multimodal(self):
        """Batches that mix text-only and multimodal samples must collate cleanly.

        This is the fundamental requirement that lets data mixing pull
        instruction-tuning text into a VLM run for catastrophic-forgetting
        mitigation. Vision tensors come from the image-having sample only;
        text-only contributes zero patches and zero image-pad tokens.
        """
        from PIL import Image

        img = Image.new("RGB", (100, 100), color=(120, 30, 200))
        examples = [
            # Sample 0: text-only (would come from instruction-tuning data).
            {
                "messages": [
                    {"role": "user", "content": "Quick question."},
                    {"role": "assistant", "content": "Quick answer."},
                ]
            },
            # Sample 1: multimodal (would come from VLM training data).
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "Describe."},
                        ],
                    },
                    {"role": "assistant", "content": "A solid color image."},
                ]
            },
        ]
        batch = self.collator(examples)

        self.assertEqual(batch["token_ids_BT"].shape, (2, self.max_length))
        # Vision keys present (because sample 1 has an image), with zero
        # contribution from sample 0.
        self.assertIn("pixel_values", batch)
        self.assertIn("image_grid_thw", batch)
        self.assertIn("vision_cu_seqlens", batch)
        self.assertIn("position_ids_ZBT", batch)

        # Sample 0 has no <|image_pad|> tokens, sample 1 has exactly the count
        # implied by image_grid_thw — the alignment that lets the row-major
        # image-token scatter in the model forward put the right embedding
        # into the right position in a heterogeneous batch.
        image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        n_pad_sample_0 = int(np.sum(batch["token_ids_BT"][0] == image_pad_id))
        n_pad_sample_1 = int(np.sum(batch["token_ids_BT"][1] == image_pad_id))
        self.assertEqual(n_pad_sample_0, 0)
        self.assertEqual(batch["image_grid_thw"].shape, (1, 3))
        grid = batch["image_grid_thw"][0]
        expected_pads = int(grid[0]) * (int(grid[1]) // 2) * (int(grid[2]) // 2)
        self.assertEqual(n_pad_sample_1, expected_pads)


class TemplateParityTest(absltest.TestCase):
    """What we train on must be what the serving stack renders.

    The module docstring promised byte-parity with ``apply_chat_template`` and
    nothing asserted it; the deleted ``BuildChatMLTextTest`` held the only such
    checks. Text models were additionally getting ``Qwen3VLRendererConfig``,
    which silently drops the ``<think>\\n\\n</think>\\n\\n`` block.
    """

    PROBE = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]

    def test_text_collator_renders_the_tokenizers_own_template(self):
        tokenizer = _make_tokenizer()
        collator = TextSFTCollator(tokenizer, max_length=256)
        batch = collator([{"messages": self.PROBE}])
        n = int(batch["attention_mask_BT"][0].sum())
        rendered = tokenizer.decode(batch["token_ids_BT"][0][:n].tolist())
        self.assertEqual(
            rendered,
            tokenizer.apply_chat_template(self.PROBE, tokenize=False, add_generation_prompt=False),
        )

    def test_text_collator_supervises_the_think_block(self):
        """The concrete regression the VL-config-on-text-model bug caused."""
        tokenizer = _make_tokenizer()
        collator = TextSFTCollator(tokenizer, max_length=256)
        batch = collator([{"messages": self.PROBE}])
        mask = batch["loss_mask_BT"][0].astype(bool)
        supervised = tokenizer.decode(batch["token_ids_BT"][0][mask].tolist())
        self.assertIn("<think>", supervised)
        self.assertEqual(supervised, "b<|im_end|><think>\n\n</think>\n\nd<|im_end|>")

    def test_vl_config_on_a_text_model_is_rejected(self):
        """The self-check must catch exactly the config that shipped."""
        from renderers import Qwen3VLRendererConfig as _VLCfg

        from omegalax.data.collator_qwen3 import assert_text_template_parity

        with self.assertRaisesRegex(ValueError, "does not reproduce the chat template"):
            assert_text_template_parity(_make_tokenizer(), _VLCfg())

    def test_text_collator_rejects_a_vlm_renderer_config(self):
        from renderers import Qwen3VLRendererConfig as _VLCfg

        with self.assertRaisesRegex(TypeError, "text renderer config"):
            TextSFTCollator(_make_tokenizer(), max_length=256, renderer_config=_VLCfg())

    def test_explicit_text_config_still_checks_template_parity(self):
        tokenizer = _make_tokenizer()
        with mock.patch.object(tokenizer, "apply_chat_template", return_value="mismatch"):
            with self.assertRaisesRegex(ValueError, "does not reproduce the chat template"):
                TextSFTCollator(
                    tokenizer,
                    max_length=256,
                    renderer_config=Qwen3RendererConfig(),
                )

    def test_vlm_collator_matches_the_hf_processor_byte_for_byte(self):
        """Vision-placeholder expansion parity, incl. the image-pad count."""
        from PIL import Image
        from transformers import AutoProcessor

        tokenizer = _make_tokenizer()
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", use_fast=False)
        image = Image.new("RGB", (112, 112), (128, 64, 32))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "what?"},
                ],
            },
            {"role": "assistant", "content": "a square"},
        ]
        encoder = Qwen3RendererEncoder(
            tokenizer,
            AutoImageProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", use_fast=False),
            None,
        )
        ours = encoder.encode(messages)
        reference = processor(
            images=[image],
            text=[processor.apply_chat_template(messages, tokenize=False)],
            return_tensors="np",
        )
        np.testing.assert_array_equal(
            np.asarray(ours["input_ids"]), np.asarray(reference["input_ids"][0])
        )
        np.testing.assert_allclose(
            ours["pixel_values"].astype(np.float32),
            np.asarray(reference["pixel_values"]).astype(np.float32),
        )


class VLMSFTCollatorRendererContractTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = mock.Mock(pad_token_id=0)

    def test_rejects_a_missing_image_processor_at_construction(self):
        with self.assertRaisesRegex(ValueError, "image_processor is required"):
            VLMSFTCollator(self.tokenizer, max_length=8, image_processor=None)

    def test_renderer_uses_the_supplied_image_processor(self):
        image_processor = mock.Mock(
            temporal_patch_size=2,
            image_mean=(0.5, 0.5, 0.5),
            patch_size=14,
        )
        rendered = mock.sentinel.renderer
        with mock.patch.object(collator_qwen3, "Qwen3VLRenderer", return_value=rendered) as factory:
            collator = VLMSFTCollator(
                self.tokenizer,
                max_length=8,
                image_processor=image_processor,
            )

            self.assertIs(collator._encoder.renderer, rendered)

        processor = factory.call_args.kwargs["processor"]
        self.assertIs(processor.image_processor, image_processor)

    def test_rejects_a_text_renderer_config(self):
        image_processor = mock.Mock(
            temporal_patch_size=2,
            image_mean=(0.5, 0.5, 0.5),
            patch_size=14,
        )
        with self.assertRaisesRegex(TypeError, "Qwen3-VL renderer config"):
            VLMSFTCollator(
                self.tokenizer,
                max_length=8,
                image_processor=image_processor,
                renderer_config=Qwen3RendererConfig(),
            )


class MessageLengthFnTest(absltest.TestCase):
    """``make_message_length_fn`` is the record builders' measure fn.

    It had no coverage, and shipped as an unbounded ``__call__``/``encode``
    mutual recursion: ``Qwen3RendererEncoder.encode`` self-called via
    ``self(...)``, which ``_MessageLengthFn.__call__`` overrides with a
    different signature and calls ``encode`` from. Every chunk-index /
    length-measurement job died with ``RecursionError``.
    """

    def setUp(self):
        super().setUp()
        self.tokenizer = _make_tokenizer()
        self.image_processor = AutoImageProcessor.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", use_fast=False
        )

    def test_measures_a_single_text_message(self):
        fn = make_message_length_fn(self.tokenizer, self.image_processor)
        out = fn({"role": "user", "content": "hello world"})
        self.assertEqual(out["length"], 7)
        self.assertEqual(out["vision_tokens"], 0)
        self.assertEqual(out["num_images"], 0)

    def test_rejects_an_image_processor_without_merge_size(self):
        with self.assertRaisesRegex(ValueError, "merge_size must be a positive integer"):
            make_message_length_fn(self.tokenizer, mock.Mock(spec=[]))

    def test_rejects_an_invalid_merge_size(self):
        for merge_size in (0, -1, True, 1.5, "2"):
            with self.subTest(merge_size=merge_size):
                with self.assertRaisesRegex(ValueError, "merge_size must be a positive integer"):
                    make_message_length_fn(
                        self.tokenizer,
                        mock.Mock(merge_size=merge_size),
                    )

    def test_lengths_are_additive_at_message_boundaries(self):
        """The property the chunk index depends on: sum(per-message) == full."""
        fn = make_message_length_fn(self.tokenizer, self.image_processor)
        conversation = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "turn 0"},
            {"role": "assistant", "content": "12 -8 0 ; +LMB -LMB"},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "TERMINATE"},
        ]
        per_message = sum(fn(m)["length"] for m in conversation)
        full = len(fn.encode(conversation)["input_ids"])
        self.assertEqual(per_message, full)

    def test_raises_when_the_renderer_drops_the_images(self):
        """A renderer that stops emitting image mm_items used to measure a VLM turn
        as text: the absent ``image_grid_thw`` was substituted with an empty grid,
        so the record was indexed at its text length with the images still in it.

        The double strips ``multi_modal_data`` off a real render, which is what a
        ``renderers`` regression looks like from here -- the encoder emits
        ``pixel_values``/``image_grid_thw`` only when that field carries items.
        """
        image_processor = AutoImageProcessor.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", use_fast=False
        )
        fn = make_message_length_fn(self.tokenizer, image_processor)
        message = {
            "role": "user",
            "content": [{"type": "image", "image": Image.new("RGB", (64, 64))}],
        }
        self.assertEqual(fn(message)["num_images"], 1)

        real = collator_qwen3.build_training_sample

        def drop_images(*args, **kwargs):
            return dataclasses.replace(real(*args, **kwargs), multi_modal_data=None)

        with mock.patch.object(collator_qwen3, "build_training_sample", drop_images):
            with self.assertRaisesRegex(ValueError, "produced no image items"):
                fn(message)

    def test_survives_pickling_with_a_live_renderer(self):
        """Shipped to ``spawn`` workers via the pool initializer."""
        fn = make_message_length_fn(self.tokenizer, self.image_processor)
        _ = fn.renderer
        restored = pickle.loads(pickle.dumps(fn))
        self.assertIsNone(restored._renderer)
        self.assertEqual(restored({"role": "user", "content": "hello world"})["length"], 7)


if __name__ == "__main__":
    absltest.main()
