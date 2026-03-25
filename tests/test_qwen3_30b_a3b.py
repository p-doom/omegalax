"""End-to-end correctness test for Qwen3-30B-A3B against HuggingFace."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ["USE_HUB_KERNELS"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, Qwen3MoeForCausalLM

from tests.logits_assert import assert_logits_close
from tests.real_weights import requires_real_weights
from omegalax.text import api
from omegalax.models.qwen3.loader import create_qwen3_from_safetensors

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PROMPT = "Why is the sky blue instead of another color like purple?"


@requires_real_weights
class Qwen3_30B_A3B_Test(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model_path = snapshot_download(MODEL_ID)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_path)
        cls.pad_id = cls.tokenizer.pad_token_id or 0

        cls.cfg = api.registry.build_config(MODEL_ID)
        cls.jax_model = create_qwen3_from_safetensors(
            cls.model_path,
            MODEL_ID,
            tp_size=1,
            fsdp_size=1,
        )

        cls.hf_model = Qwen3MoeForCausalLM.from_pretrained(
            cls.model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
        ).to(cls.device).eval()

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
        toks = self.tokenizer(chat_texts, return_tensors="pt", padding=True, padding_side="left")
        return {k: v.to(self.device) for k, v in toks.items()}

    def _jax_prefill_logits(self, input_ids: torch.Tensor) -> np.ndarray:
        token_ids_BT = jnp.asarray(np.array(input_ids.cpu(), dtype=np.int32))
        segment_ids_BT = 1 * (token_ids_BT != self.pad_id)
        hidden_BTD, _ = self.jax_model(token_ids_BT, segment_ids_BT, None, jnp.array(0, dtype=jnp.int32))
        logits_BTV = self.jax_model.lm_head(hidden_BTD)
        return np.asarray(logits_BTV, dtype=np.float32)

    def test_prefill_logits_match_hf(self):
        inputs = self._tokenize([PROMPT])
        with torch.no_grad():
            hf_logits_BTV = self.hf_model(**inputs).logits.float().cpu().numpy()
        jax_logits_BTV = self._jax_prefill_logits(inputs["input_ids"])
        mask = inputs["attention_mask"].cpu().numpy().astype(bool)

        jax_masked = jax_logits_BTV[mask]
        hf_masked = hf_logits_BTV[mask]
        assert_logits_close(self, jax_masked, hf_masked, top1_min_match=0.8)


if __name__ == "__main__":
    absltest.main()
