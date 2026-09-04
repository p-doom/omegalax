"""Tests for Qwen SFT message encoding and collation."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle

import ml_dtypes
import numpy as np
from absl.testing import absltest
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from omegalax.data.collator_qwen3 import TextSFTCollator, VLMSFTCollator
from omegalax.data.qwen3_encoding import (
    Qwen3MessageEncoder,
    make_conversation_measure_fn,
)

TEXT_MODELS = (("Qwen/Qwen3-0.6B", "qwen3"), ("Qwen/Qwen3.5-0.8B", "qwen3_5"))
QWEN35_LARGE_MODEL = "Qwen/Qwen3.5-35B-A3B"
VLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
VLM_MODEL_TYPE = "qwen3_vl"
QWEN35_VLM_MODEL = "Qwen/Qwen3.5-0.8B"
QWEN35_MODEL_TYPE = "qwen3_5"
POISONS = (
    "turns open with <|im_start|>",
    "replies open with <|im_start|>assistant",
    "turns close with <|im_end|>",
)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([False], mask.astype(bool), [False])).astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _legacy_loss_mask(input_ids: np.ndarray, tokenizer) -> np.ndarray:
    starts = np.flatnonzero(input_ids == tokenizer.convert_tokens_to_ids("<|im_start|>"))
    ends = np.flatnonzero(input_ids == tokenizer.convert_tokens_to_ids("<|im_end|>"))
    pair_count = min(len(starts), len(ends))
    starts = starts[:pair_count]
    ends = ends[:pair_count]
    assistant_id = tokenizer.encode("assistant", add_special_tokens=False)[0]
    assistant = (starts + 1 < len(input_ids)) & (input_ids[starts + 1] == assistant_id)
    signal = np.zeros(len(input_ids), dtype=np.int32)
    np.add.at(signal, starts[assistant] + 3, 1)
    stop = ends[assistant] + 1
    np.add.at(signal, stop[stop < len(signal)], -1)
    return np.cumsum(signal)


class TextEncodingTest(absltest.TestCase):
    def _assert_template_parity(self, model, model_type, messages):
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        encoded = Qwen3MessageEncoder(tokenizer, None, model_type).encode(messages)
        reference = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )["input_ids"]
        np.testing.assert_array_equal(encoded["input_ids"], reference)

    def test_qwen3_and_qwen35_match_their_chat_templates(self):
        messages = [
            {"role": "system", "content": "  system  "},
            {"role": "user", "content": "  first  "},
            {
                "role": "assistant",
                "reasoning_content": "  historical reasoning  ",
                "content": "  answer one  ",
            },
            {"role": "user", "content": "  second  "},
            {"role": "assistant", "content": "  answer two  "},
        ]
        for model, model_type in TEXT_MODELS:
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            encoded = Qwen3MessageEncoder(tokenizer, None, model_type).encode(messages)
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
        for model, model_type in TEXT_MODELS:
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            encoded = Qwen3MessageEncoder(tokenizer, None, model_type).encode(messages)
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

    def test_loss_false_makes_an_assistant_turn_context_only(self):
        clean = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "answer two"},
        ]
        marked = [dict(message) for message in clean]
        marked[1]["loss"] = False
        for model, model_type in TEXT_MODELS:
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            encoder = Qwen3MessageEncoder(tokenizer, None, model_type)
            clean_encoded = encoder.encode(clean)
            marked_encoded = encoder.encode(marked)
            np.testing.assert_array_equal(marked_encoded["input_ids"], clean_encoded["input_ids"])
            measurement = encoder.measure(marked[:2])["message_measurements"][1]
            self.assertEqual(measurement["supervised_tokens"], 0)
            spans = [
                tokenizer.decode(marked_encoded["input_ids"][start:end])
                for start, end in _runs(marked_encoded["loss_mask"])
            ]
            self.assertEqual(spans, ["<think>\n\n</think>\n\nanswer two<|im_end|>"])

    def test_loss_defaults_true(self):
        clean = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        marked = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer", "loss": True},
        ]
        model, model_type = TEXT_MODELS[0]
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        encoder = Qwen3MessageEncoder(tokenizer, None, model_type)
        clean_encoded = encoder.encode(clean)
        marked_encoded = encoder.encode(marked)
        np.testing.assert_array_equal(marked_encoded["input_ids"], clean_encoded["input_ids"])
        np.testing.assert_array_equal(marked_encoded["loss_mask"], clean_encoded["loss_mask"])

    def test_loss_must_be_boolean(self):
        model, model_type = TEXT_MODELS[0]
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer", "loss": 0},
        ]
        with self.assertRaisesRegex(ValueError, "loss must be a boolean"):
            Qwen3MessageEncoder(tokenizer, None, model_type).encode(messages)

    def test_loss_is_rejected_on_non_assistant_turns(self):
        model, model_type = TEXT_MODELS[0]
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        messages = [
            {"role": "user", "content": "question", "loss": False},
            {"role": "assistant", "content": "answer"},
        ]
        with self.assertRaisesRegex(ValueError, "only on assistant turns"):
            Qwen3MessageEncoder(tokenizer, None, model_type).encode(messages)

    def test_explicit_reasoning_matches_chat_templates(self):
        messages = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "reasoning_content": "reasoning",
                "content": "answer",
            },
        ]
        for model, model_type in TEXT_MODELS:
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            encoded = Qwen3MessageEncoder(tokenizer, None, model_type).encode(messages)
            reference = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )["input_ids"]
            np.testing.assert_array_equal(encoded["input_ids"], reference)

    def test_consecutive_assistant_reasoning_matches_chat_templates_and_masks(self):
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "reasoning_content": "first", "content": "answer one"},
            {"role": "assistant", "reasoning_content": "second", "content": "answer two"},
        ]
        for model, model_type in TEXT_MODELS:
            self._assert_template_parity(model, model_type, messages)
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            encoded = Qwen3MessageEncoder(tokenizer, None, model_type).encode(messages)
            spans = [
                tokenizer.decode(encoded["input_ids"][start:end])
                for start, end in _runs(encoded["loss_mask"])
            ]
            self.assertEqual(
                spans,
                [
                    "<think>\nfirst\n</think>\n\nanswer one<|im_end|>",
                    "<think>\nsecond\n</think>\n\nanswer two<|im_end|>",
                ],
            )

    def test_qwen3_reasoning_and_whitespace_edges_match_chat_template(self):
        cases = [
            [{"role": "assistant", "reasoning_content": "hidden", "content": "answer"}],
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "reasoning_content": "reason", "content": "\n\nanswer"},
            ],
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "<think>\nreason\n</think>\n\nanswer"},
            ],
            [
                {"role": "user", "content": "question"},
                {"role": "system", "content": "later system"},
                {"role": "assistant", "content": "answer"},
            ],
        ]
        for messages in cases:
            self._assert_template_parity("Qwen/Qwen3-0.6B", "qwen3", messages)

    def test_qwen35_reasoning_edges_match_both_template_variants(self):
        cases = [
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "reasoning_content": "reason", "content": "\n\nanswer"},
            ],
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "<think>\nreason\n</think>\n\nanswer"},
            ],
            [
                {"role": "system", "content": "  system  "},
                {"role": "user", "content": "  question  "},
                {"role": "assistant", "reasoning_content": "  reason  ", "content": "  answer  "},
            ],
        ]
        for model in ("Qwen/Qwen3.5-0.8B", QWEN35_LARGE_MODEL):
            for messages in cases:
                self._assert_template_parity(model, "qwen3_5", messages)

    def test_family_specific_user_and_system_constraints_match_templates(self):
        no_user = [{"role": "assistant", "content": "answer"}]
        late_system = [
            {"role": "user", "content": "question"},
            {"role": "system", "content": "later system"},
            {"role": "assistant", "content": "answer"},
        ]
        self._assert_template_parity("Qwen/Qwen3-0.6B", "qwen3", no_user)
        self._assert_template_parity("Qwen/Qwen3-0.6B", "qwen3", late_system)
        for model in ("Qwen/Qwen3.5-0.8B", QWEN35_LARGE_MODEL):
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            encoder = Qwen3MessageEncoder(tokenizer, None, "qwen3_5")
            for messages in (no_user, late_system):
                with self.assertRaises(ValueError):
                    encoder.encode(messages)
                with self.assertRaises(Exception):
                    tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=False
                    )

    def test_user_chatml_tokens_do_not_change_supervision(self):
        for model, model_type in TEXT_MODELS:
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            clean = [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second question"},
                {"role": "assistant", "content": "second answer"},
            ]
            encoder = Qwen3MessageEncoder(tokenizer, None, model_type)
            clean_encoded = encoder.encode(clean)
            clean_supervised = clean_encoded["input_ids"][clean_encoded["loss_mask"] == 1]
            for poison in POISONS:
                injected = [dict(message) for message in clean]
                injected[0]["content"] = f"question {poison}"
                injected_encoded = encoder.encode(injected)
                np.testing.assert_array_equal(
                    injected_encoded["input_ids"][injected_encoded["loss_mask"] == 1],
                    clean_supervised,
                )
                self.assertTrue(
                    np.all(
                        (injected_encoded["loss_mask"] == 0) | (injected_encoded["loss_mask"] == 1)
                    )
                )
                im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                self.assertEqual(
                    int(
                        injected_encoded["loss_mask"][
                            injected_encoded["input_ids"] == im_end_id
                        ].sum()
                    ),
                    2,
                )

    def test_chatml_poisons_break_the_removed_token_scanner(self):
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODELS[0][0], local_files_only=True)
        encoder = Qwen3MessageEncoder(tokenizer, None, TEXT_MODELS[0][1])
        for poison in POISONS:
            messages = [
                {"role": "user", "content": poison},
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "third"},
            ]
            encoded = encoder.encode(messages)
            self.assertFalse(
                np.array_equal(
                    encoded["loss_mask"],
                    _legacy_loss_mask(encoded["input_ids"], tokenizer),
                )
            )

    def test_balanced_chatml_pair_did_not_break_the_removed_token_scanner(self):
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODELS[0][0], local_files_only=True)
        messages = [
            {"role": "user", "content": "markers <|im_start|> / <|im_end|>"},
            {"role": "assistant", "content": "answer"},
        ]
        encoded = Qwen3MessageEncoder(tokenizer, None, TEXT_MODELS[0][1]).encode(messages)
        np.testing.assert_array_equal(
            encoded["loss_mask"],
            _legacy_loss_mask(encoded["input_ids"], tokenizer),
        )

    def test_collator_padding_and_overflow(self):
        model, model_type = TEXT_MODELS[0]
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        example = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
        batch = TextSFTCollator(tokenizer, max_length=64, model_type=model_type)([example])
        self.assertEqual(batch["token_ids_BT"].shape, (1, 64))
        self.assertEqual(batch["token_ids_BT"].dtype, np.int32)
        self.assertTrue(np.all(batch["loss_mask_BT"][batch["attention_mask_BT"] == 0] == 0))
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            TextSFTCollator(tokenizer, max_length=4, model_type=model_type)([example])

    def test_text_collator_rejects_zero_supervision(self):
        model, model_type = TEXT_MODELS[0]
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        example = {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "context", "loss": False},
            ]
        }
        with self.assertRaisesRegex(ValueError, "must contain supervised tokens"):
            TextSFTCollator(tokenizer, max_length=64, model_type=model_type)([example])

    def test_collator_does_not_impose_a_conversation_role_sequence(self):
        model, model_type = TEXT_MODELS[0]
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        messages = [
            {"role": "system", "content": "first context"},
            {"role": "system", "content": "second context"},
            {"role": "user", "content": "first input"},
            {"role": "user", "content": "second input"},
            {"role": "assistant", "content": "first target"},
            {"role": "assistant", "content": "second target"},
        ]
        batch = TextSFTCollator(tokenizer, 256, model_type)([{"messages": messages}])
        self.assertGreater(int(batch["loss_mask_BT"].sum()), 0)

    def test_video_placeholder_is_rejected(self):
        model, model_type = TEXT_MODELS[0]
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        messages = [{"role": "user", "content": "look at <|video_pad|>"}]
        with self.assertRaisesRegex(ValueError, "video content"):
            Qwen3MessageEncoder(tokenizer, None, model_type).encode(messages)


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
        encoded = Qwen3MessageEncoder(self.tokenizer, self.image_processor, VLM_MODEL_TYPE).encode(
            messages
        )
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
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.assertEqual(int(np.sum(encoded["input_ids"] == image_token_id)), image_tokens)
        self.assertEqual(
            self.tokenizer.decode(encoded["input_ids"][encoded["loss_mask"] == 1]),
            "square<|im_end|>",
        )

    def test_qwen3_vl_role_and_reasoning_behavior_matches_chat_template(self):
        cases = [
            [{"role": "assistant", "reasoning_content": "ignored", "content": "answer"}],
            [
                {"role": "user", "content": "question"},
                {"role": "system", "content": "ignored system"},
                {"role": "assistant", "content": "answer"},
            ],
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "reasoning_content": "ignored", "content": "first"},
                {"role": "assistant", "content": "<think>\nkept\n</think>\n\nsecond"},
            ],
        ]
        encoder = Qwen3MessageEncoder(
            self.tokenizer, self.image_processor, VLM_MODEL_TYPE
        )
        for messages in cases:
            encoded = encoder.encode(messages)
            reference = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )["input_ids"]
            np.testing.assert_array_equal(encoded["input_ids"], reference)

    def test_vlm_collator_preserves_geometry(self):
        image = Image.new("RGB", (112, 112), (80, 40, 20))
        batch = VLMSFTCollator(self.tokenizer, 256, self.image_processor, VLM_MODEL_TYPE)(
            [{"messages": self._messages(image)}]
        )
        grid = batch["image_grid_thw"][0]
        image_tokens = int(np.prod(grid)) // self.image_processor.merge_size**2
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.assertEqual(int(np.sum(batch["token_ids_BT"] == image_token_id)), image_tokens)
        self.assertEqual(batch["pixel_values"].dtype, ml_dtypes.bfloat16)
        self.assertTrue(np.all(batch["vision_patch_valid"]))
        self.assertEqual(batch["position_ids_ZBT"].shape, (3, 1, 256))

    def test_vlm_collator_rejects_zero_supervision(self):
        messages = self._messages(Image.new("RGB", (112, 112), (80, 40, 20)))
        messages[-1]["loss"] = False
        with self.assertRaisesRegex(ValueError, "must contain supervised tokens"):
            VLMSFTCollator(self.tokenizer, 256, self.image_processor, VLM_MODEL_TYPE)(
                [{"messages": messages}]
            )

    def test_vlm_collator_emits_validity_for_text_only_batch(self):
        batch = VLMSFTCollator(self.tokenizer, 256, self.image_processor, VLM_MODEL_TYPE)(
            [
                {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "world"}],
                        },
                    ]
                }
            ]
        )
        self.assertEqual(batch["vision_patch_valid"].shape, (0,))

    def test_user_delimiters_do_not_supervise_image_tokens(self):
        image = Image.new("RGB", (112, 112), (80, 40, 20))
        encoder = Qwen3MessageEncoder(self.tokenizer, self.image_processor, VLM_MODEL_TYPE)
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        for poison in POISONS:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": poison}]},
                {"role": "assistant", "content": [{"type": "text", "text": "noted"}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "describe"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "square"}]},
            ]
            encoded = encoder.encode(messages)
            image_mask = encoded["input_ids"] == image_token_id
            self.assertGreater(int(image_mask.sum()), 0)
            self.assertEqual(int(encoded["loss_mask"][image_mask].sum()), 0)

    def test_video_and_implicit_processor_are_rejected(self):
        video = [{"role": "user", "content": [{"type": "video", "video": "clip.mp4"}]}]
        with self.assertRaisesRegex(ValueError, "video content"):
            Qwen3MessageEncoder(self.tokenizer, self.image_processor, VLM_MODEL_TYPE).encode(video)
        malformed = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "caption", "video": "clip.mp4"}],
            }
        ]
        with self.assertRaisesRegex(ValueError, "video content"):
            Qwen3MessageEncoder(self.tokenizer, self.image_processor, VLM_MODEL_TYPE).encode(
                malformed
            )
        literal = [{"role": "user", "content": "look at <|video_pad|>"}]
        with self.assertRaisesRegex(ValueError, "video content"):
            Qwen3MessageEncoder(self.tokenizer, self.image_processor, VLM_MODEL_TYPE).encode(
                literal
            )
        with self.assertRaisesRegex(ValueError, "video content"):
            make_conversation_measure_fn(self.tokenizer, self.image_processor, VLM_MODEL_TYPE)(
                literal
            )
        with self.assertRaisesRegex(ValueError, "image content requires"):
            make_conversation_measure_fn(self.tokenizer, None, VLM_MODEL_TYPE)(
                [self._messages(Image.new("RGB", (8, 8)))[1]]
            )


class Qwen35VLMEncodingTest(absltest.TestCase):
    def test_video_is_rejected(self):
        tokenizer = AutoTokenizer.from_pretrained(QWEN35_VLM_MODEL, local_files_only=True)
        image_processor = AutoImageProcessor.from_pretrained(
            QWEN35_VLM_MODEL, use_fast=False, local_files_only=True
        )
        messages = [{"role": "user", "content": [{"type": "video", "video": "clip.mp4"}]}]
        with self.assertRaisesRegex(ValueError, "video content"):
            Qwen3MessageEncoder(tokenizer, image_processor, QWEN35_MODEL_TYPE).encode(messages)

    def test_template_processor_mask_and_positions_match(self):
        tokenizer = AutoTokenizer.from_pretrained(QWEN35_VLM_MODEL, local_files_only=True)
        image_processor = AutoImageProcessor.from_pretrained(
            QWEN35_VLM_MODEL, use_fast=False, local_files_only=True
        )
        image = Image.new("RGB", (112, 112), (80, 40, 20))
        messages = [
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

        encoded = Qwen3MessageEncoder(tokenizer, image_processor, QWEN35_MODEL_TYPE).encode(
            messages
        )
        processed = image_processor(images=[image], return_tensors="np")
        grid = np.asarray(processed["image_grid_thw"], dtype=np.int32)[0]
        image_tokens = int(np.prod(grid)) // image_processor.merge_size**2
        reference_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ).replace("<|image_pad|>", "<|image_pad|>" * image_tokens, 1)
        reference_ids = tokenizer(
            reference_text, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]

        np.testing.assert_array_equal(encoded["input_ids"], reference_ids)
        np.testing.assert_array_equal(encoded["image_grid_thw"], processed["image_grid_thw"])
        np.testing.assert_allclose(encoded["pixel_values"], processed["pixel_values"])
        self.assertEqual(
            tokenizer.decode(encoded["input_ids"][encoded["loss_mask"] == 1]),
            "<think>\n\n</think>\n\nsquare<|im_end|>",
        )

        batch = VLMSFTCollator(tokenizer, 256, image_processor, QWEN35_MODEL_TYPE)(
            [{"messages": messages}]
        )
        valid = batch["attention_mask_BT"][0].astype(bool)
        token_ids = batch["token_ids_BT"][0, valid]
        position_ids = batch["position_ids_ZBT"][:, 0, valid]
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_positions = np.flatnonzero(token_ids == image_token_id)
        self.assertLen(image_positions, image_tokens)
        start = int(image_positions[0])
        end = int(image_positions[-1]) + 1
        np.testing.assert_array_equal(image_positions, np.arange(start, end))

        t, h, w = (int(value) for value in grid)
        merged_h = h // image_processor.merge_size
        merged_w = w // image_processor.merge_size
        vision_positions = np.stack(
            [
                np.repeat(np.arange(t), merged_h * merged_w),
                np.tile(np.repeat(np.arange(merged_h), merged_w), t),
                np.tile(np.arange(merged_w), t * merged_h),
            ]
        )
        tail_start = start + int(vision_positions.max()) + 1
        expected_positions = np.concatenate(
            [
                np.tile(np.arange(start), (3, 1)),
                vision_positions + start,
                np.tile(np.arange(len(token_ids) - end) + tail_start, (3, 1)),
            ],
            axis=1,
        )
        np.testing.assert_array_equal(position_ids, expected_positions)


class ConversationMeasurementTest(absltest.TestCase):
    def test_every_prepared_span_matches_the_encoder(self):
        conversations = [
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer one"},
                {"role": "user", "content": "second"},
                {
                    "role": "assistant",
                    "reasoning_content": "reasoning",
                    "content": "answer two",
                },
            ],
            [
                {"role": "user", "content": POISONS[0]},
                {"role": "assistant", "content": "answer"},
            ],
            [
                {"role": "user", "content": POISONS[2]},
                {"role": "assistant", "content": POISONS[1]},
            ],
        ]
        for model, model_type in (*TEXT_MODELS, (VLM_MODEL, VLM_MODEL_TYPE)):
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            prepare = make_conversation_measure_fn(tokenizer, None, model_type)
            for messages in conversations:
                prepared = prepare(messages)
                encoder = Qwen3MessageEncoder(tokenizer, None, model_type)
                for start in range(len(messages)):
                    for end in range(start + 1, len(messages) + 1):
                        try:
                            measurement = prepared(start, end)
                            encoded = encoder.encode(messages[start:end])
                        except ValueError:
                            continue
                        self.assertEqual(measurement["length"], len(encoded["input_ids"]))
                        self.assertEqual(
                            measurement["supervised_tokens"], int(encoded["loss_mask"].sum())
                        )

    def test_measurement_callable_survives_spawn_pickling(self):
        model, model_type = TEXT_MODELS[0]
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        prepare = pickle.loads(
            pickle.dumps(make_conversation_measure_fn(tokenizer, None, model_type))
        )
        result = prepare(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        )(0, 2)
        self.assertGreater(result["supervised_tokens"], 0)


if __name__ == "__main__":
    absltest.main()
