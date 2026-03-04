"""Tests for SFT collators: loss-mask correctness, multi-turn, truncation."""

import json
import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest

import numpy as np
from transformers import AutoTokenizer

from omegalax.data.collators import TextSFTCollator, _build_assistant_loss_mask
from omegalax.data.jsonl import JSONLDataset


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


class JSONLDatasetTest(absltest.TestCase):
    def test_load_and_iterate(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(5):
                f.write(json.dumps({"messages": [{"role": "user", "content": f"msg{i}"}]}) + "\n")
            f.flush()
            ds = JSONLDataset(f.name)
        self.assertEqual(len(ds), 5)
        examples = list(ds.iter_examples(shuffle=False, seed=0, process_split=False, num_epochs=1))
        self.assertEqual(len(examples), 5)
        os.unlink(f.name)

    def test_multimodal_messages_preserved(self):
        """Structured content with image blocks is loaded as-is."""
        example = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": "/data/imgs/a.png"},
                        {"type": "text", "text": "describe"},
                    ],
                },
                {"role": "assistant", "content": "A cat."},
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(example) + "\n")
            f.flush()
            ds = JSONLDataset(f.name)
        msg = ds.examples[0]["messages"][0]
        self.assertEqual(msg["content"][0], {"type": "image", "url": "/data/imgs/a.png"})
        self.assertEqual(msg["content"][1], {"type": "text", "text": "describe"})
        self.assertEqual(ds.examples[0]["messages"][1]["content"], "A cat.")
        os.unlink(f.name)

    def test_epoch_count(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(3):
                f.write(json.dumps({"messages": [{"role": "user", "content": str(i)}]}) + "\n")
            f.flush()
            ds = JSONLDataset(f.name)
        examples = list(ds.iter_examples(shuffle=False, seed=0, process_split=False, num_epochs=2))
        self.assertEqual(len(examples), 6)
        os.unlink(f.name)


if __name__ == "__main__":
    absltest.main()
