"""Tests for SFT collators: loss-mask correctness, multi-turn, and overflow checks."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest

import ml_dtypes
import numpy as np
from transformers import AutoTokenizer

from transformers import AutoImageProcessor

from omegalax.data.collator_qwen3 import (
    TextSFTCollator,
    VLMSFTCollator,
)
from omegalax.data.qwen3_encoding import (
    build_chatml_text as _build_chatml_text,
    encode_qwen_messages,
)


def _make_tokenizer():
    """Use a small, fast tokenizer available offline or from HF cache."""
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)


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
        examples = [{"messages": messages}]
        batch = self.collator(examples)
        mask = batch["loss_mask_BT"][0]
        # At least some tokens should be supervised
        self.assertGreater(np.sum(mask), 0)
        # Supervised tokens should be fewer than non-padding tokens
        attn = batch["attention_mask_BT"][0]
        self.assertLess(np.sum(mask), np.sum(attn))

    def test_multiturn_masks_all_assistant_spans(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am fine, thanks."},
        ]
        examples = [{"messages": messages}]
        batch = self.collator(examples)
        mask = batch["loss_mask_BT"][0]
        self.assertGreater(np.sum(mask), 0)

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


class StructuralLossMaskTest(absltest.TestCase):
    """Tests for the structural assistant loss mask returned by encode_qwen_messages."""

    def setUp(self):
        super().setUp()
        self.tokenizer = _make_tokenizer()
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _encode(self, messages):
        encoded = encode_qwen_messages(messages, tokenizer=self.tokenizer)
        return encoded["input_ids"], encoded["loss_mask"]

    def test_mask_matches_input_length(self):
        ids, mask = self._encode(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        )
        self.assertEqual(len(ids), len(mask))

    def test_single_turn(self):
        ids, mask = self._encode(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        )
        self.assertGreater(np.sum(mask), 0)
        # User tokens should not be supervised
        self.assertLess(np.sum(mask), len(ids))

    def test_multi_turn(self):
        _, mask = self._encode(
            [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
                {"role": "user", "content": "C"},
                {"role": "assistant", "content": "D"},
            ]
        )
        # Should have supervised tokens from both assistant turns
        self.assertGreater(np.sum(mask), 0)

    def test_no_assistant(self):
        _, mask = self._encode(
            [
                {"role": "user", "content": "Hello"},
            ]
        )
        self.assertEqual(np.sum(mask), 0)

    def test_mask_includes_assistant_im_end(self):
        ids, mask = self._encode(
            [
                {"role": "user", "content": "X"},
                {"role": "assistant", "content": "Y"},
            ]
        )
        # Assistant <|im_end|> should be supervised so the model learns to terminate.
        # User <|im_end|> should not be supervised.
        im_end_positions = np.where(ids == self._im_end_id)[0]
        # First <|im_end|> is from user turn (not supervised)
        self.assertEqual(mask[im_end_positions[0]], 0, "User <|im_end|> should not be supervised")
        # Second <|im_end|> is from assistant turn (supervised)
        self.assertEqual(mask[im_end_positions[1]], 1, "Assistant <|im_end|> should be supervised")

    def test_header_tokens_not_supervised(self):
        # The <|im_start|>assistant\n header must never be supervised — only the
        # content and the terminating <|im_end|>.
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        ids, mask = self._encode(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A longer answer here."},
            ]
        )
        for pos in np.where(ids == im_start_id)[0]:
            self.assertEqual(mask[pos], 0, "<|im_start|> must not be supervised")
            # role token and its trailing newline (the next two) are header, not content
            self.assertEqual(mask[pos + 1], 0, "role token must not be supervised")
            self.assertEqual(mask[pos + 2], 0, "header newline must not be supervised")


class ChatMLLeakageTest(absltest.TestCase):
    """Literal ChatML markers in user/context text must not corrupt the loss mask.

    Regression for run lq3fgwvd (qwen3vl8b_fft_ds_v3_...): user/context text
    containing literal ``<|im_start|>`` injected a spurious special token, which
    broke the 1:1 start/end pairing of the old token-scanning mask so that later
    user / image tokens were marked supervised (train/supervised_tokens spiked,
    loss collapsed). The structural per-turn mask is immune to this.
    """

    def setUp(self):
        super().setUp()
        self.tokenizer = _make_tokenizer()
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _supervised_count(self, messages):
        return int(np.sum(encode_qwen_messages(messages, tokenizer=self.tokenizer)["loss_mask"]))

    def test_literal_marker_in_user_text_keeps_supervised_count(self):
        answer = "The scoring runs in three stages."
        clean = [
            {"role": "user", "content": "Trace the teacher scoring mechanism."},
            {"role": "assistant", "content": answer},
        ]
        poisoned = [
            {
                "role": "user",
                "content": (
                    "Trace the teacher scoring mechanism.\n"
                    "Chat format: Qwen-style <|im_start|> / <|im_end|> boundaries"
                ),
            },
            {"role": "assistant", "content": answer},
        ]
        clean_sup = self._supervised_count(clean)
        pois_sup = self._supervised_count(poisoned)
        self.assertGreater(clean_sup, 0)
        # The assistant answer is identical and the poison lives entirely in the
        # user turn, so the supervised-token count must be unchanged.
        self.assertEqual(pois_sup, clean_sup)

    def test_only_final_assistant_im_end_supervised_with_injected_markers(self):
        messages = [
            {
                "role": "user",
                "content": "boundaries look like <|im_start|>assistant and <|im_end|>",
            },
            {"role": "assistant", "content": "Understood."},
        ]
        encoded = encode_qwen_messages(messages, tokenizer=self.tokenizer)
        ids, mask = encoded["input_ids"], encoded["loss_mask"]
        self.assertGreater(int(np.sum(mask)), 0)
        # Only the assistant turn's terminating <|im_end|> (the last one) is
        # supervised; every earlier <|im_end|> — the injected one and the user's
        # own closer — must be masked out.
        im_end_positions = np.where(ids == self._im_end_id)[0]
        self.assertEqual(mask[im_end_positions[-1]], 1)
        for pos in im_end_positions[:-1]:
            self.assertEqual(mask[pos], 0)


class BuildChatMLTextTest(absltest.TestCase):
    """Tests for _build_chatml_text ChatML output format."""

    def setUp(self):
        super().setUp()
        self.tokenizer = _make_tokenizer()

    def test_text_only_single_turn(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = _build_chatml_text(messages, image_grids=[], merge_size=2)
        expected = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>\n"
        self.assertEqual(result, expected)

    def test_text_only_multi_turn(self):
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"},
        ]
        result = _build_chatml_text(messages, image_grids=[], merge_size=2)
        expected = (
            "<|im_start|>user\nA<|im_end|>\n"
            "<|im_start|>assistant\nB<|im_end|>\n"
            "<|im_start|>user\nC<|im_end|>\n"
            "<|im_start|>assistant\nD<|im_end|>\n"
        )
        self.assertEqual(result, expected)

    def test_with_system_prompt(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = _build_chatml_text(messages, image_grids=[], merge_size=2)
        expected = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nHi!<|im_end|>\n"
        )
        self.assertEqual(result, expected)

    def test_image_tokens_inserted(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe."},
                ],
            },
            {"role": "assistant", "content": "A cat."},
        ]
        grid = (1, 8, 8)
        merge_size = 2
        n_tokens = 1 * (8 // 2) * (8 // 2)  # = 16

        result = _build_chatml_text(messages, image_grids=[grid], merge_size=merge_size)
        self.assertIn("<|vision_start|>", result)
        self.assertIn("<|vision_end|>", result)
        self.assertEqual(result.count("<|image_pad|>"), n_tokens)

    def test_multi_image(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "Compare."},
                ],
            },
        ]
        grids = [(1, 4, 4), (1, 8, 8)]
        merge_size = 2
        n1 = 1 * (4 // 2) * (4 // 2)  # = 4
        n2 = 1 * (8 // 2) * (8 // 2)  # = 16

        result = _build_chatml_text(messages, image_grids=grids, merge_size=merge_size)
        self.assertEqual(result.count("<|image_pad|>"), n1 + n2)
        self.assertEqual(result.count("<|vision_start|>"), 2)
        self.assertEqual(result.count("<|vision_end|>"), 2)

    def test_encodes_correctly(self):
        """Verify that tokenizer.encode on our ChatML text produces valid token IDs."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is this?"},
                ],
            },
            {"role": "assistant", "content": "A photo."},
        ]
        grid = (1, 4, 4)
        text = _build_chatml_text(messages, image_grids=[grid], merge_size=2)
        ids = self.tokenizer.encode(text, add_special_tokens=False)

        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        self.assertEqual(ids.count(im_start_id), 2)
        self.assertEqual(ids.count(im_end_id), 2)
        n_expected = 1 * (4 // 2) * (4 // 2)  # = 4
        self.assertEqual(ids.count(image_pad_id), n_expected)

    def test_per_block_encoding_matches_full_encoding(self):
        """encode_qwen_messages ids must equal a single full tokenizer.encode.

        Guards the additive property that structural per-turn masking relies on:
        ``<|im_start|>``/``<|im_end|>`` are hard BPE split points, so concatenating
        the per-block token ids must reproduce ``tokenizer.encode`` of the whole
        ChatML string exactly — even when the text embeds literal ChatML markers.
        """
        conversations = [
            [
                {"role": "user", "content": "Hello there"},
                {"role": "assistant", "content": "General Kenobi"},
            ],
            [
                {"role": "system", "content": "sys prompt"},
                {"role": "user", "content": "boundary <|im_start|>x<|im_end|> test"},
                {"role": "assistant", "content": "ok <|im_end|> done"},
            ],
        ]
        for messages in conversations:
            text = _build_chatml_text(messages, image_grids=[], merge_size=2)
            full = np.asarray(self.tokenizer.encode(text, add_special_tokens=False), dtype=np.int32)
            got = encode_qwen_messages(messages, tokenizer=self.tokenizer)["input_ids"]
            np.testing.assert_array_equal(got, full)


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
        mask = batch["loss_mask_BT"][0]
        self.assertGreater(np.sum(mask), 0)
        attn = batch["attention_mask_BT"][0]
        self.assertLess(np.sum(mask), np.sum(attn))

    def test_literal_chatml_in_user_text_does_not_supervise_image_pads(self):
        """Regression (run lq3fgwvd): literal <|im_start|> in a user turn that also
        carries an image must never flip that image's pad tokens to supervised."""
        from PIL import Image

        img = Image.new("RGB", (100, 100), color=(50, 50, 50))
        poison_text = "Chat format: Qwen-style <|im_start|> / <|im_end|> boundaries"
        answer = "Nothing special."

        def _messages(user_text):
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": user_text},
                    ],
                },
                {"role": "assistant", "content": answer},
            ]

        image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        poisoned = self.collator([{"messages": _messages(poison_text)}])
        token_ids = poisoned["token_ids_BT"][0]
        mask = poisoned["loss_mask_BT"][0]

        n_pad = int(np.sum(token_ids == image_pad_id))
        self.assertGreater(n_pad, 0, "sanity: the image should produce pad tokens")
        # Not a single image pad token may be supervised.
        self.assertEqual(int(np.sum(mask[token_ids == image_pad_id])), 0)

        # And the assistant answer is supervised exactly as in the clean case —
        # the poison in the user turn changes nothing about supervision.
        clean = self.collator([{"messages": _messages("Describe the image.")}])
        self.assertGreater(int(np.sum(mask)), 0)
        self.assertEqual(int(np.sum(mask)), int(np.sum(clean["loss_mask_BT"][0])))

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


if __name__ == "__main__":
    absltest.main()
