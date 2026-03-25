"""End-to-end correctness test for Qwen3.5-0.8B (dense) against HuggingFace."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ["USE_HUB_KERNELS"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration as HFModel,
)

from tests.logits_assert import assert_logits_close
from tests.real_weights import requires_real_weights
from omegalax.models.qwen3_5.config import make_config
from omegalax.models.qwen3_5.params import create_qwen3_5_from_safetensors

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

MODEL_ID = "Qwen/Qwen3.5-0.8B"
PROMPT = "Why is the sky blue instead of another color like purple?"


@requires_real_weights
class Qwen3_5_0_8B_Test(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            cls.model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
        ).to(cls.device).eval()

    def _tokenize(self, texts: list[str]):
        chat_texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": t}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for t in texts
        ]
        toks = self.tokenizer(chat_texts, return_tensors="pt", padding=True, padding_side="left")
        return {k: v.to(self.device) for k, v in toks.items()}

    def _jax_prefill_logits(self, tokens_np: np.ndarray) -> np.ndarray:
        token_ids_BT = jnp.asarray(tokens_np)
        segment_ids_BT = (token_ids_BT != self.pad_id).astype(jnp.int32)
        hidden_BTD, _ = self.jax_model(
            token_ids_BT, segment_ids_BT, None, jnp.array(0, dtype=jnp.int32),
        )
        logits_BTV = self.jax_model.lm_head(hidden_BTD)
        return np.asarray(logits_BTV, dtype=np.float32)

    def test_prefill_logits_match_hf(self):
        inputs = self._tokenize([PROMPT])
        with torch.no_grad():
            hf_logits_BTV = self.hf_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            ).logits.float().cpu().numpy()

        jax_logits_BTV = self._jax_prefill_logits(
            np.array(inputs["input_ids"].cpu(), dtype=np.int32),
        )

        mask = inputs["attention_mask"].cpu().numpy().astype(bool)
        jax_masked = jax_logits_BTV[mask]
        hf_masked = hf_logits_BTV[mask]
        assert_logits_close(self, jax_masked, hf_masked, top1_min_match=0.8)


if __name__ == "__main__":
    absltest.main()
