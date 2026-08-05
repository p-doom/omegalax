"""Gate: the ``renderers`` default loss mask IS our supervision target.

Locks in the one property the whole adoption rests on. We pass NO
``role_to_mask``, so ``build_training_sample`` returns the renderer's
``sampled_mask`` verbatim. That default must be, for EVERY assistant turn
including historical ones: the assistant content plus ``<|im_end|>``, excluding
the 3-token ``<|im_start|>assistant\\n`` header and the trailing ``\\n``.

If a renderers bump changes this, the mask silently shifts and every SFT run
trains on the wrong spans — so this test, not a comment, is the contract.
"""

import os

os.environ.setdefault("HF_HOME", "/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface")

import numpy as np
from absl.testing import absltest
from renderers import Qwen3VLRendererConfig, build_training_sample, create_renderer
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def _runs(mask):
    """Contiguous [start, end) spans where mask is True."""
    m = np.asarray(mask, dtype=bool)
    edges = np.diff(np.concatenate(([False], m, [False])).astype(np.int8))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True))


class RenderersLossMaskGateTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        # Qwen3VLRendererConfig() is passed EXPLICITLY: auto-resolution is an
        # exact match on tokenizer.name_or_path and raises for any VLM missing
        # from MODEL_RENDERER_MAP — i.e. every fine-tuned Phase-B export.
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.renderer = create_renderer(self.tok, Qwen3VLRendererConfig())
        self.im_start = self.tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.header = self.tok.encode("assistant\n", add_special_tokens=False)

    def test_every_assistant_turn_is_supervised_over_exactly_content_plus_im_end(self):
        contents = ["12 -8 0 ; +LMB -LMB", "0 0 0 ; type(\"hello\")", "TERMINATE"]
        messages = [{"role": "system", "content": "sys"}]
        for i, c in enumerate(contents):
            messages.append({"role": "user", "content": f"turn {i}"})
            messages.append({"role": "assistant", "content": c})

        sample = build_training_sample(self.renderer, messages)  # NO role_to_mask
        ids = np.asarray(sample.token_ids)
        runs = _runs(sample.loss_mask)

        # (a) historical turns too — one supervised run per assistant turn.
        self.assertEqual(len(runs), len(contents), f"expected {len(contents)} runs, got {runs}")

        for (start, end), content in zip(runs, contents, strict=True):
            # (b) header excluded: the 3 tokens before the run are
            # <|im_start|> + "assistant" + "\n", and none of them is trained.
            self.assertEqual(len(self.header) + 1, 3, "header must be exactly 3 tokens")
            self.assertEqual(int(ids[start - len(self.header) - 1]), self.im_start)
            self.assertEqual([int(t) for t in ids[start - len(self.header) : start]], self.header)

            # <|im_end|> IS the last supervised token (the stop target).
            self.assertEqual(int(ids[end - 1]), self.im_end)

            # trailing "\n" excluded.
            self.assertEqual(self.tok.decode([int(ids[end])]), "\n")

            # the span decodes to exactly the content plus the stop token.
            self.assertEqual(
                self.tok.decode([int(t) for t in ids[start:end]]), content + "<|im_end|>"
            )

        # nothing outside the assistant bodies is supervised.
        self.assertEqual(int(np.sum(sample.loss_mask)), sum(e - s for s, e in runs))


if __name__ == "__main__":
    absltest.main()
