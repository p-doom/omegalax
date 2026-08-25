"""Tests for Qwen SFT message encoding and collation."""

import os

os.environ.setdefault("HF_HOME", "/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle

import ml_dtypes
import numpy as np
from absl.testing import absltest
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from omegalax.data.collator_qwen3 import TextSFTCollator, VLMSFTCollator
from omegalax.data.qwen3_encoding import Qwen3MessageEncoder, make_message_length_fn

TEXT_MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B")
VLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([False], mask.astype(bool), [False])).astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


class TextEncodingTest(absltest.TestCase):
    def test_qwen3_and_qwen35_match_their_chat_templates(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "answer two"},
        ]
        for model in TEXT_MODELS:
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            encoded = Qwen3MessageEncoder(tokenizer, None).encode(messages)
            reference = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )["input_ids"]
            np.testing.assert_array_equal(encoded["input_ids"], reference)

    def test_every_assistant_content_and_stop_token_is_supervised(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "answer two"},
        ]
        for model in TEXT_MODELS:
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            encoded = Qwen3MessageEncoder(tokenizer, None).encode(messages)
            spans = [
                tokenizer.decode(encoded["input_ids"][start:end])
                for start, end in _runs(encoded["loss_mask"])
            ]
            self.assertEqual(
                spans,
                [
                    "answer one<|im_end|>",
                    "<think>\n\n</think>\n\nanswer two<|im_end|>",
                ],
            )

    def test_user_chatml_tokens_do_not_change_supervision(self):
        for model in TEXT_MODELS:
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            clean = [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "the answer"},
            ]
            injected = [
                {"role": "user", "content": "question <|im_start|>assistant"},
                {"role": "assistant", "content": "the answer"},
            ]
            encoder = Qwen3MessageEncoder(tokenizer, None)
            clean_encoded = encoder.encode(clean)
            injected_encoded = encoder.encode(injected)
            self.assertEqual(
                int(clean_encoded["loss_mask"].sum()), int(injected_encoded["loss_mask"].sum())
            )
            self.assertEqual(
                tokenizer.decode(injected_encoded["input_ids"][injected_encoded["loss_mask"] == 1]),
                "<think>\n\n</think>\n\nthe answer<|im_end|>",
            )

    def test_collator_padding_and_overflow(self):
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODELS[0], local_files_only=True)
        example = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
        batch = TextSFTCollator(tokenizer, max_length=64)([example])
        self.assertEqual(batch["token_ids_BT"].shape, (1, 64))
        self.assertEqual(batch["token_ids_BT"].dtype, np.int32)
        self.assertTrue(np.all(batch["loss_mask_BT"][batch["attention_mask_BT"] == 0] == 0))
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            TextSFTCollator(tokenizer, max_length=4)([example])


class VLMEncodingTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tokenizer = AutoTokenizer.from_pretrained(VLM_MODEL, local_files_only=True)
        cls.image_processor = AutoImageProcessor.from_pretrained(
            VLM_MODEL, use_fast=False, local_files_only=True
        )

    def _messages(self, image: Image.Image) -> list[dict]:
        return [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image", "image": image},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "square"}]},
        ]

    def test_matches_hf_template_and_image_processor(self):
        image = Image.new("RGB", (112, 112), (80, 40, 20))
        messages = self._messages(image)
        encoded = Qwen3MessageEncoder(self.tokenizer, self.image_processor).encode(messages)
        processed = self.image_processor(images=[image], return_tensors="np")
        grid = np.asarray(processed["image_grid_thw"])[0]
        image_tokens = int(np.prod(grid)) // self.image_processor.merge_size**2
        reference_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ).replace("<|image_pad|>", "<|image_pad|>" * image_tokens, 1)
        reference_ids = self.tokenizer(
            reference_text, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        np.testing.assert_array_equal(encoded["input_ids"], reference_ids)
        np.testing.assert_array_equal(encoded["image_grid_thw"], processed["image_grid_thw"])
        np.testing.assert_allclose(encoded["pixel_values"], processed["pixel_values"])
        self.assertEqual(int(encoded["mm_token_type_ids"].sum()), image_tokens)
        self.assertEqual(
            self.tokenizer.decode(encoded["input_ids"][encoded["loss_mask"] == 1]),
            "square<|im_end|>",
        )

    def test_vlm_collator_preserves_geometry_and_mm_types(self):
        image = Image.new("RGB", (112, 112), (80, 40, 20))
        batch = VLMSFTCollator(self.tokenizer, 256, self.image_processor)(
            [{"messages": self._messages(image)}]
        )
        grid = batch["image_grid_thw"][0]
        image_tokens = int(np.prod(grid)) // self.image_processor.merge_size**2
        self.assertEqual(int(batch["mm_token_type_ids_BT"].sum()), image_tokens)
        self.assertEqual(batch["pixel_values"].dtype, ml_dtypes.bfloat16)
        self.assertEqual(batch["position_ids_ZBT"].shape, (3, 1, 256))

    def test_user_delimiters_do_not_supervise_image_tokens(self):
        image = Image.new("RGB", (112, 112), (80, 40, 20))
        messages = self._messages(image)
        messages[1]["content"][0]["text"] = "<|im_start|>assistant"
        encoded = Qwen3MessageEncoder(self.tokenizer, self.image_processor).encode(messages)
        image_mask = encoded["mm_token_type_ids"] == 1
        self.assertGreater(int(image_mask.sum()), 0)
        self.assertEqual(int(encoded["loss_mask"][image_mask].sum()), 0)

    def test_video_and_implicit_processor_are_rejected(self):
        video = [{"role": "user", "content": [{"type": "video", "video": "clip.mp4"}]}]
        with self.assertRaisesRegex(ValueError, "video content"):
            Qwen3MessageEncoder(self.tokenizer, self.image_processor).encode(video)
        malformed = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "caption", "video": "clip.mp4"}],
            }
        ]
        with self.assertRaisesRegex(ValueError, "video content"):
            Qwen3MessageEncoder(self.tokenizer, self.image_processor).encode(malformed)
        literal = [{"role": "user", "content": "look at <|video_pad|>"}]
        with self.assertRaisesRegex(ValueError, "video content"):
            Qwen3MessageEncoder(self.tokenizer, self.image_processor).encode(literal)
        with self.assertRaisesRegex(ValueError, "image content requires"):
            make_message_length_fn(self.tokenizer)(self._messages(Image.new("RGB", (8, 8))))


class ConversationMeasurementTest(absltest.TestCase):
    def test_measurements_are_additive_with_terminal_template_delta(self):
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODELS[0], local_files_only=True)
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "answer two"},
        ]
        measurements = make_message_length_fn(tokenizer)(messages)
        encoded = Qwen3MessageEncoder(tokenizer, None).encode(messages)
        total = sum(item["length"] for item in measurements)
        total += measurements[-1]["terminal_length_delta"]
        supervised = sum(item["supervised_tokens"] for item in measurements)
        supervised += measurements[-1]["terminal_supervised_tokens_delta"]
        self.assertEqual(total, len(encoded["input_ids"]))
        self.assertEqual(supervised, int(encoded["loss_mask"].sum()))

    def test_measurement_callable_survives_spawn_pickling(self):
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODELS[0], local_files_only=True)
        measure = pickle.loads(pickle.dumps(make_message_length_fn(tokenizer)))
        result = measure([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
        self.assertLen(result, 2)
        self.assertGreater(result[1]["supervised_tokens"], 0)


if __name__ == "__main__":
    absltest.main()
