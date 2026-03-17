"""Tests for SFT collators: loss-mask correctness, multi-turn, and overflow checks."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest

import numpy as np
from transformers import AutoTokenizer

from transformers import AutoImageProcessor

from omegalax.data.collator_qwen3 import TextSFTCollator, VLMSFTCollator, _build_assistant_loss_mask, _build_chatml_text


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
            {"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]},
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
            {"messages": [
                {"role": "user", "content": "Say X"},
                {"role": "assistant", "content": "X"},
            ]},
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
            {"messages": [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
            ]},
            {"messages": [
                {"role": "user", "content": "C"},
                {"role": "assistant", "content": "D"},
            ]},
        ]
        batch = self.collator(examples)
        self.assertEqual(batch["token_ids_BT"].shape[0], 2)

    def test_dtypes_are_int32(self):
        examples = [
            {"messages": [
                {"role": "user", "content": "X"},
                {"role": "assistant", "content": "Y"},
            ]},
        ]
        batch = self.collator(examples)
        for key in ("token_ids_BT", "attention_mask_BT", "loss_mask_BT"):
            self.assertEqual(batch[key].dtype, np.int32, f"{key} dtype mismatch")

    def test_raises_on_overflow(self):
        collator = TextSFTCollator(self.tokenizer, max_length=8)
        examples = [
            {"messages": [
                {"role": "user", "content": "Tell me a story in many words."},
                {"role": "assistant", "content": "This answer is intentionally too long for the tiny max length."},
            ]},
        ]
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            collator(examples)


class BuildAssistantLossMaskTest(absltest.TestCase):
    """Direct tests for the token-scanning loss mask builder."""

    def setUp(self):
        super().setUp()
        self.tokenizer = _make_tokenizer()
        self._im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._assistant_token_id = self.tokenizer.encode("assistant", add_special_tokens=False)[0]

    def _apply_and_mask(self, messages):
        result = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
        )
        ids = np.array(result["input_ids"], dtype=np.int32)
        mask = _build_assistant_loss_mask(
            ids, self._im_start_id, self._im_end_id,
            self._assistant_token_id,
        )
        return ids, mask

    def test_single_turn(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        ids, mask = self._apply_and_mask(messages)
        self.assertGreater(np.sum(mask), 0)
        # User tokens should not be supervised
        self.assertLess(np.sum(mask), len(ids))

    def test_multi_turn(self):
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"},
        ]
        _, mask = self._apply_and_mask(messages)
        # Should have supervised tokens from both assistant turns
        self.assertGreater(np.sum(mask), 0)

    def test_no_assistant(self):
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        _, mask = self._apply_and_mask(messages)
        self.assertEqual(np.sum(mask), 0)

    def test_mask_excludes_im_end(self):
        messages = [
            {"role": "user", "content": "X"},
            {"role": "assistant", "content": "Y"},
        ]
        ids, mask = self._apply_and_mask(messages)
        # <|im_end|> tokens should not be supervised
        im_end_positions = np.where(ids == self._im_end_id)[0]
        for pos in im_end_positions:
            self.assertEqual(mask[pos], 0, f"<|im_end|> at position {pos} should not be masked")


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
        expected = (
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nHi!<|im_end|>\n"
        )
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
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe."},
            ]},
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
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "Compare."},
            ]},
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
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "What is this?"},
            ]},
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


class VLMSFTCollatorTest(absltest.TestCase):
    """Tests for the VLM SFT collator with real images."""

    def setUp(self):
        super().setUp()
        self.tokenizer = _make_tokenizer()
        self.image_processor = AutoImageProcessor.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", use_fast=False, # force the numpy codepath
        )
        self.max_length = 256
        self.collator = VLMSFTCollator(
            self.tokenizer, max_length=self.max_length,
            image_processor=self.image_processor,
        )

    def test_text_only_example(self):
        examples = [
            {"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]},
        ]
        batch = self.collator(examples)
        self.assertIn("token_ids_BT", batch)
        self.assertEqual(batch["token_ids_BT"].shape, (1, self.max_length))
        self.assertNotIn("pixel_values", batch)
        self.assertNotIn("image_grid_thw", batch)

    def test_multimodal_example(self):
        from PIL import Image
        img = Image.new("RGB", (200, 200), color=(100, 150, 200))
        examples = [
            {"messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe."},
                ]},
                {"role": "assistant", "content": "A solid color image."},
            ]},
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

    def test_loss_mask_on_assistant_only(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100), color=(50, 50, 50))
        examples = [
            {"messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "What?"},
                ]},
                {"role": "assistant", "content": "Nothing special."},
            ]},
        ]
        batch = self.collator(examples)
        mask = batch["loss_mask_BT"][0]
        self.assertGreater(np.sum(mask), 0)
        attn = batch["attention_mask_BT"][0]
        self.assertLess(np.sum(mask), np.sum(attn))

    def test_raises_on_overflow(self):
        from PIL import Image

        img = Image.new("RGB", (200, 200), color=(100, 150, 200))
        collator = VLMSFTCollator(
            self.tokenizer,
            max_length=8,
            image_processor=self.image_processor,
        )
        examples = [
            {"messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe."},
                ]},
                {"role": "assistant", "content": "A solid color image."},
            ]},
        ]
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            collator(examples)


if __name__ == "__main__":
    absltest.main()
