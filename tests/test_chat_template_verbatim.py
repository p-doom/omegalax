"""Golden train/serve-identity test for the verbatim CUA chat template.

Training renders conversations with ``build_chatml_text`` (raw ChatML,
assistant content verbatim, generation continues from
``<|im_start|>assistant\\n``). Serving renders with the checkpoint's HF chat
template. The stock Qwen3-VL-Thinking template strips ``<think>...</think>``
from prior assistant turns and force-opens a think block in the generation
prompt -- a silent train/serve mismatch that broke evals for weeks. This test
pins byte-identity between ``apply_chat_template`` under the verbatim template
and the training-side renderer, and guards its own sensitivity by asserting
the stock template FAILS the same check.

Image placeholders: the HF template emits a single ``<|image_pad|>`` per image
(the processor later expands it to the grid's token count), while
``build_chatml_text`` performs that expansion itself from ``image_grids``.
The two agree exactly when each grid is ``(1, merge_size, merge_size)`` --
one merged vision token -- which is what this test uses; structural identity
of the ``<|vision_start|>...<|vision_end|>`` wrapper is asserted separately.
"""

import json
import os
import tempfile
from pathlib import Path

# Stay offline: everything must come from the local HF cache.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, AutoTokenizer

from omegalax.data.chat_template import (
    VERBATIM_CHAT_TEMPLATE_PATH,
    copy_tokenizer_assets,
    load_verbatim_chat_template,
    write_chat_template,
)
from omegalax.data.qwen3_encoding import build_chatml_text

MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"

THINK_TEXT = "The menu is closed; open it before selecting the tool."
BARE_ACTION = "scroll(0, -3)"

# System prompt; user turns with one image each; one user turn with a text
# block; assistant turns both with <think>...</think>\naction and bare-action.
MESSAGES = [
    {"role": "system", "content": "You are a computer-use agent."},
    {"role": "user", "content": [{"type": "image"}]},
    {"role": "assistant", "content": f"<think>\n{THINK_TEXT}\n</think>\nclick(412, 88)"},
    {"role": "user", "content": [{"type": "image"}]},
    {"role": "assistant", "content": BARE_ACTION},
    {"role": "user", "content": [{"type": "text", "text": "keep going"}]},
]
NUM_IMAGES = 2


class VerbatimChatTemplateTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.snapshot_dir = snapshot_download(MODEL_ID)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.snapshot_dir)
        cls.merge_size = int(AutoImageProcessor.from_pretrained(cls.snapshot_dir).merge_size)
        cls.verbatim_template = load_verbatim_chat_template()

    def _training_render(self) -> str:
        # (1, merge, merge) grids => exactly one <|image_pad|> per image,
        # matching the template's pre-expansion placeholder (see module doc).
        grids = [(1, self.merge_size, self.merge_size)] * NUM_IMAGES
        return build_chatml_text(MESSAGES, grids, self.merge_size)

    def test_verbatim_template_matches_training_render(self):
        """apply_chat_template(verbatim) is byte-identical to build_chatml_text."""
        rendered = self.tokenizer.apply_chat_template(
            MESSAGES,
            chat_template=self.verbatim_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        expected = self._training_render() + "<|im_start|>assistant\n"
        self.assertEqual(rendered, expected)
        # Assistant content survives verbatim; generation prompt is bare.
        self.assertIn(f"<think>\n{THINK_TEXT}\n</think>\nclick(412, 88)", rendered)
        self.assertTrue(rendered.endswith("<|im_end|>\n<|im_start|>assistant\n"))
        # Image placeholders keep the stock structural wrapper.
        self.assertEqual(rendered.count("<|vision_start|><|image_pad|><|vision_end|>"), NUM_IMAGES)

    def test_stock_template_fails_training_identity(self):
        """Sensitivity guard: the STOCK template must NOT pass the same check."""
        stock = self.tokenizer.apply_chat_template(
            MESSAGES,
            tokenize=False,
            add_generation_prompt=True,
        )
        expected = self._training_render() + "<|im_start|>assistant\n"
        self.assertNotEqual(stock, expected)
        # ... and for the two known reasons:
        # (a) it force-opens a think block in the generation prompt;
        self.assertTrue(stock.endswith("<|im_start|>assistant\n<think>\n"))
        # (b) it strips <think>...</think> from prior assistant turns.
        self.assertNotIn(THINK_TEXT, stock)

    def test_verbatim_asset_diff_against_stock_is_minimal(self):
        """The asset differs from stock only in the strip logic and the tail."""
        stock_path = Path(self.snapshot_dir) / "chat_template.json"
        stock = json.loads(stock_path.read_text())["chat_template"]
        verbatim = self.verbatim_template
        self.assertIn("'</think>'", stock)
        self.assertNotIn("</think>", verbatim)
        self.assertNotIn("reasoning_content", verbatim)
        self.assertIn("<|im_start|>assistant\\n<think>\\n", stock)
        self.assertNotIn("<think>", verbatim)
        # Multimodal handling stays byte-identical to stock.
        macro = stock[: stock.index("{%- if tools %}")]
        self.assertIn(macro, verbatim)

    def test_exported_checkpoint_serves_verbatim_template(self):
        """End-to-end: assets written by the export path serve the verbatim template.

        Mirrors scripts/export_to_hf.py::_write_serving_assets, then reloads
        the tokenizer from the exported dir WITHOUT passing chat_template=,
        i.e. exactly what sglang/transformers would pick up.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            copy_tokenizer_assets(self.snapshot_dir, tmpdir)
            write_chat_template(tmpdir, self.verbatim_template)

            served = json.loads((Path(tmpdir) / "chat_template.json").read_text())
            self.assertEqual(served["chat_template"], self.verbatim_template)
            tok_cfg = json.loads((Path(tmpdir) / "tokenizer_config.json").read_text())
            self.assertEqual(tok_cfg["chat_template"], self.verbatim_template)

            exported_tokenizer = AutoTokenizer.from_pretrained(tmpdir)
            rendered = exported_tokenizer.apply_chat_template(
                MESSAGES,
                tokenize=False,
                add_generation_prompt=True,
            )
            self.assertEqual(rendered, self._training_render() + "<|im_start|>assistant\n")

    def test_asset_file_is_packaged_next_to_encoding_code(self):
        self.assertTrue(VERBATIM_CHAT_TEMPLATE_PATH.exists())
        self.assertEqual(VERBATIM_CHAT_TEMPLATE_PATH.parent.name, "data")


if __name__ == "__main__":
    absltest.main()
