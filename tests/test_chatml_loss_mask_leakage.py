"""Regression: literal ChatML markers in user text must not move the loss mask.

The pre-``renderers`` encoder built the assistant loss mask by scanning the final
token stream for ``<|im_start|>`` / ``<|im_end|>`` pairs in sequence order. User
or context text that embeds literal ChatML markers -- a screen note describing
the chat format -- injects spurious special tokens, breaks the 1:1 pairing and
flips later user turns, INCLUDING image pad tokens, to supervised. Symptom in run
``lq3fgwvd``: ``train/supervised_tokens`` spiking at anomalously low loss.

Lifted from upstream ``fix/chatml-loss-mask-leakage`` (``4e5b705``), which never
merged here -- it repairs ``encode_qwen_messages``, a function this branch
deleted when ``renderers`` took over message-to-token conversion. ``renderers``
masks structurally per message and is immune by construction, but that is a
property we can lose on a renderers bump, so it is pinned here rather than
assumed. Verified non-vacuous: the three leakage tests FAIL against the scanning
encoder at ``b3f32c0``, the revision this branch replaced (supervised count
15 vs 8, a supervised user ``<|im_end|>``, and all 64 image pads supervised).
``PerMessageEncodingAdditivityTest`` passes on both -- it pins a premise, not a
bug.

``test_renderers_loss_mask_gate`` pins the mask on clean conversations; this file
only covers what adversarial marker text does to it.
"""

import os

os.environ.setdefault("HF_HOME", "/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from absl.testing import absltest
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from omegalax.data.collator_qwen3 import (
    Qwen3RendererEncoder,
    VLMSFTCollator,
    make_message_length_fn,
    resolve_text_renderer_config,
)

TEXT_MODEL = "Qwen/Qwen3-0.6B"
VL_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

#: Prose describing chat markup, in the one shape that actually breaks a scanning
#: mask: an UNMATCHED ``<|im_start|>`` followed by the literal word ``assistant``.
#: The scanner zips starts to ends by index, so an unmatched start shifts every
#: later pairing, and the ``assistant`` word makes the scanner open a supervised
#: span there. A balanced ``<|im_start|> ... <|im_end|>`` pair leaves the zip
#: intact and leaks nothing -- upstream's poison string was balanced, which is why
#: three of its four regression tests passed against the very encoder they were
#: written to catch. Do not "simplify" this string.
POISON = "the screen shows <|im_start|>assistant in the log"


def _encode(tokenizer, messages):
    """``messages -> (input_ids, loss_mask)``, as ``TextSFTCollator`` builds them."""
    encoder = Qwen3RendererEncoder(tokenizer, None, resolve_text_renderer_config(None))
    encoded = encoder.encode(messages)
    return encoded["input_ids"], encoded["loss_mask"]


def _measure_ids(tokenizer, messages):
    """``messages -> input_ids`` through the renderer the chunk index measures with."""
    return Qwen3RendererEncoder(tokenizer, None, None).encode(messages)["input_ids"]


class ChatMLLeakageTextTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, trust_remote_code=True)
        self.im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def test_literal_marker_in_user_text_keeps_supervised_count(self):
        answer = "The scoring runs in three stages."
        question = "Trace the teacher scoring mechanism."
        _, clean_mask = _encode(
            self.tokenizer,
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        )
        _, poisoned_mask = _encode(
            self.tokenizer,
            [
                {"role": "user", "content": f"{question}\n{POISON}"},
                {"role": "assistant", "content": answer},
            ],
        )
        clean_supervised = int(clean_mask.sum())
        self.assertGreater(clean_supervised, 0)
        # The assistant answer is identical and the poison lives entirely in the
        # user turn, so the supervised-token count must be unchanged.
        self.assertEqual(int(poisoned_mask.sum()), clean_supervised)

    def test_only_final_assistant_im_end_supervised_with_injected_markers(self):
        answer = "Understood."
        ids, mask = _encode(
            self.tokenizer,
            [
                {"role": "user", "content": POISON},
                {"role": "assistant", "content": answer},
            ],
        )
        im_start_positions = np.where(ids == self.im_start_id)[0]
        im_end_positions = np.where(ids == self.im_end_id)[0]
        self.assertLen(im_start_positions, 3, "sanity: the poison must inject an <|im_start|>")
        self.assertLen(im_end_positions, 2)
        # Only the assistant turn's terminating <|im_end|> (the last one) is
        # supervised; every earlier one -- the injected marker and the user's own
        # closer -- must be masked out.
        self.assertEqual(mask[im_end_positions[-1]], 1)
        for pos in im_end_positions[:-1]:
            self.assertEqual(mask[pos], 0)
        # Mirror failure: the answer must not be DROPPED from supervision either.
        # ``endswith`` because Qwen3's template prefixes the final assistant turn
        # with an empty ``<think>\n\n</think>\n\n`` block, which is supervised too.
        supervised = self.tokenizer.decode(ids[mask == 1].tolist())
        self.assertEndsWith(supervised, f"{answer}<|im_end|>")


class ChatMLLeakageVLMTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = AutoTokenizer.from_pretrained(VL_MODEL, trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(VL_MODEL, use_fast=False)
        self.collator = VLMSFTCollator(self.tokenizer, 4096, self.image_processor)
        self.image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def _messages(self, user_text):
        # Text BEFORE the image: a scanning mask opens its bogus span at the
        # injected marker and runs forward, so pads only leak when they follow it.
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": Image.new("RGB", (100, 100), color=(50, 50, 50))},
                ],
            },
            {"role": "assistant", "content": "Nothing special."},
        ]

    def test_literal_chatml_in_user_text_does_not_supervise_image_pads(self):
        poisoned = self.collator([{"messages": self._messages(POISON)}])
        token_ids = poisoned["token_ids_BT"][0]
        mask = poisoned["loss_mask_BT"][0]

        is_pad = token_ids == self.image_pad_id
        self.assertGreater(int(np.sum(is_pad)), 0, "sanity: the image must produce pad tokens")
        self.assertEqual(int(np.sum(mask[is_pad])), 0, "image pad tokens must never be supervised")

        # The assistant answer is supervised exactly as in the clean case -- the
        # poison in the user turn changes nothing about supervision.
        clean = self.collator([{"messages": self._messages("Describe the image.")}])
        self.assertGreater(int(np.sum(mask)), 0)
        self.assertEqual(int(np.sum(mask)), int(np.sum(clean["loss_mask_BT"][0])))


class PerMessageEncodingAdditivityTest(absltest.TestCase):
    """Assert the premise ``make_message_length_fn`` documents rather than trusting it.

    ``<|im_start|>`` / ``<|im_end|>`` are registered specials and therefore hard
    BPE split points, so encoding each message alone and concatenating must
    reproduce the full-sequence ids exactly -- even when the text embeds literal
    ChatML markers. Every chunk-index length is a per-message measurement; if
    this does not hold, they are all wrong and examples overflow at train time.

    Pinned on the default (Qwen3-VL) renderer because that is the only config
    ``make_message_length_fn`` is ever constructed with. It does NOT hold for
    ``Qwen3RendererConfig``, whose template appends ``<think>\\n\\n</think>\\n\\n``
    to the FINAL assistant turn only: that is turn-position-dependent, so a
    message encoded alone renders differently than the same message mid-sequence.
    """

    CONVERSATIONS = (
        [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "General Kenobi"},
        ],
        [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "boundary <|im_start|>x<|im_end|> test"},
            {"role": "assistant", "content": "ok <|im_end|> done"},
        ],
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"},
        ],
    )

    def setUp(self):
        super().setUp()
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, trust_remote_code=True)

    def test_concatenated_per_message_ids_equal_full_sequence_ids(self):
        for messages in self.CONVERSATIONS:
            full = _measure_ids(self.tokenizer, messages)
            per_message = np.concatenate([_measure_ids(self.tokenizer, [m]) for m in messages])
            np.testing.assert_array_equal(per_message, full)

    def test_measured_lengths_sum_to_the_full_sequence_length(self):
        measure = make_message_length_fn(self.tokenizer)
        for messages in self.CONVERSATIONS:
            full = _measure_ids(self.tokenizer, messages)
            self.assertEqual(sum(measure(m)["length"] for m in messages), len(full))


if __name__ == "__main__":
    absltest.main()
