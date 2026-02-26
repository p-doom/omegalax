"""End-to-end correctness test for Qwen3.5-397B-A17B against HuggingFace."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration as HFModel,
)

from omegalax.models.qwen3_5.config import make_config
from omegalax.models.qwen3_5.params import create_qwen3_5_from_safetensors

jax.config.update("jax_default_matmul_precision", "highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

MODEL_ID = "Qwen/Qwen3.5-397B-A17B"
PROMPT = "Why is the sky blue instead of another color like purple?"
RTOL = 1e-5
ATOL = 1e-4


class Qwen3_5RealTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_path = snapshot_download(MODEL_ID)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_path)
        cls.pad_id = cls.tokenizer.pad_token_id or 0

        cls.jax_cfg = make_config(MODEL_ID)
        cls.jax_model, _ = create_qwen3_5_from_safetensors(
            cls.model_path,
            MODEL_ID,
            tp_size=1,
            fsdp_size=1,
        )

        cls.hf_model = HFModel.from_pretrained(
            cls.model_path, torch_dtype=torch.float32,
        ).eval()

    def _tokenize(self, texts: list[str]):
        chat_texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": t}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for t in texts
        ]
        return self.tokenizer(chat_texts, return_tensors="pt", padding=True, padding_side="left")

    def _jax_prefill_logits(self, tokens_np: np.ndarray) -> np.ndarray:
        token_ids_BT = jnp.asarray(tokens_np)
        segment_ids_BT = (token_ids_BT != self.pad_id).astype(jnp.int32)
        logits_BTV, _ = self.jax_model(
            token_ids_BT, segment_ids_BT, None, jnp.array(0, dtype=jnp.int32),
        )
        return np.asarray(logits_BTV, dtype=np.float32)

    def test_prefill_logits_match_hf(self):
        inputs = self._tokenize([PROMPT])
        with torch.no_grad():
            hf_logits_BTV = self.hf_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            ).logits.cpu().numpy()

        jax_logits_BTV = self._jax_prefill_logits(
            np.array(inputs["input_ids"].cpu(), dtype=np.int32),
        )

        mask = inputs["attention_mask"].numpy().astype(bool)
        jax_masked = jax_logits_BTV[mask]
        hf_masked = hf_logits_BTV[mask]
        max_abs_diff = np.max(np.abs(jax_masked - hf_masked))
        max_rel_diff = np.max(np.abs(jax_masked - hf_masked) / np.clip(np.abs(hf_masked), 1e-8, None))
        print(f"\n  max_abs_diff = {max_abs_diff:.6e}")
        print(f"  max_rel_diff = {max_rel_diff:.6e}")

        np.testing.assert_allclose(jax_masked, hf_masked, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    absltest.main()
