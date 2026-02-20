"""End-to-end correctness test for Qwen3-30B-A3B against HuggingFace."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, Qwen3MoeForCausalLM

from omegalax.text import api
from omegalax.models.qwen3.moe.params_moe import create_qwen3_moe_from_safe_tensors

jax.config.update("jax_default_matmul_precision", "highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PROMPT = "Why is the sky blue instead of another color like purple?"
RTOL = 1e-5
ATOL = 1e-4


class Qwen3_30B_A3B_Test(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_path = snapshot_download(MODEL_ID)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_path)
        cls.pad_id = cls.tokenizer.pad_token_id or 0

        cls.cfg = api.registry.build_config(MODEL_ID)
        cls.jax_model = create_qwen3_moe_from_safe_tensors(cls.model_path, MODEL_ID)

        cls.hf_model = Qwen3MoeForCausalLM.from_pretrained(
            cls.model_path, torch_dtype=torch.float32,
        ).eval()

    def _tokenize(self, texts: list[str]):
        chat_texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": t}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for t in texts
        ]
        return self.tokenizer(chat_texts, return_tensors="pt", padding=True, padding_side="left")

    def _jax_prefill_logits(self, input_ids: torch.Tensor) -> np.ndarray:
        tokens = jnp.asarray(np.array(input_ids.cpu(), dtype=np.int32))
        logits, _ = api.forward(self.jax_model, tokens, self.pad_id, self.cfg)
        return np.asarray(logits, dtype=np.float32)

    def test_prefill_logits_match_hf(self):
        inputs = self._tokenize([PROMPT])
        with torch.no_grad():
            hf_logits = self.hf_model(**inputs).logits.cpu().numpy()
        jax_logits = self._jax_prefill_logits(inputs["input_ids"])
        mask = inputs["attention_mask"].numpy().astype(bool)

        jax_masked = jax_logits[mask]
        hf_masked = hf_logits[mask]
        max_abs_diff = np.max(np.abs(jax_masked - hf_masked))
        max_rel_diff = np.max(np.abs(jax_masked - hf_masked) / np.clip(np.abs(hf_masked), 1e-8, None))
        print(f"\n  max_abs_diff = {max_abs_diff:.6e}")
        print(f"  max_rel_diff = {max_rel_diff:.6e}")

        np.testing.assert_allclose(jax_masked, hf_masked, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    absltest.main()
