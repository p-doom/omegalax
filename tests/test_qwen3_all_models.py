"""Forward-pass numerical equivalence across all registered Qwen3 variants.

# FIXME (f.srambical)
Tolerance rationale (float32, CPU, highest matmul precision):
  - Per-matmul BLAS divergence between XLA and PyTorch: 6-13 ULPs
  - Per-layer compound divergence (attention + MLP + residual): ~5 ULPs
  - Accumulated through N layers: grows roughly as sqrt(N)
  - Empirically measured on Qwen3-0.6B (28 layers):
      max_abs_diff = 8.77e-5, p99.9 = 4.2e-5
  - rtol=1e-5 tightens the check for medium/large logit values (~5-10x
    tighter than the previous 1e-4); atol=1e-4 provides a floor for
    near-zero logits where absolute BLAS error dominates.
"""

import gc
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest, parameterized
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from omegalax.text import api
from omegalax.models.qwen3.params import create_qwen3_from_safetensors

jax.config.update("jax_default_matmul_precision", "highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

RTOL = 1e-5
ATOL = 1e-4
PROMPT = "Why is the sky blue instead of another color like purple?"

_ALL_MODELS = [
    *api.list_qwen3_dense_model_ids(),
    *api.list_qwen3_moe_model_ids(),
]


def _make_params():
    return [
        dict(
            testcase_name=mid.split("/")[-1].replace(".", "_").replace("-", "_"),
            model_id=mid,
        )
        for mid in _ALL_MODELS
    ]


class Qwen3AllModelsTest(parameterized.TestCase):

    @parameterized.named_parameters(_make_params())
    def test_prefill_logits_match_hf(self, model_id):
        model_path = snapshot_download(model_id)

        cfg = api.registry.build_config(model_id)
        jax_model = create_qwen3_from_safetensors(
            model_path,
            model_id,
            tp_size=1,
            fsdp_size=1,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        hf_cfg = AutoConfig.from_pretrained(model_path)
        hf_cfg.tie_word_embeddings = cfg.tie_word_embeddings
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, config=hf_cfg, torch_dtype=torch.float32,
        ).eval()

        pad_id = tokenizer.pad_token_id or 0
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(chat_text, return_tensors="pt", padding=True)
        token_ids_BT = jnp.asarray(np.array(inputs["input_ids"].cpu(), dtype=np.int32))

        with torch.no_grad():
            hf_logits_BTV = hf_model(**inputs).logits.cpu().numpy()

        jax_logits_BTV, _ = api.forward(jax_model, token_ids_BT, pad_id, cfg)
        jax_logits_BTV = np.asarray(jax_logits_BTV, dtype=np.float32)

        mask = inputs["attention_mask"].numpy().astype(bool)
        j, h = jax_logits_BTV[mask], hf_logits_BTV[mask]
        abs_diff = np.abs(j - h)
        max_abs = float(np.max(abs_diff))
        mean_abs = float(np.mean(abs_diff))
        print(
            f"\n  {model_id}: max_abs_diff={max_abs:.6e}, "
            f"mean_abs_diff={mean_abs:.6e}, "
            f"headroom={ATOL / max(max_abs, 1e-30):.1f}x"
        )

        np.testing.assert_allclose(j, h, rtol=RTOL, atol=ATOL)

        del hf_model, jax_model, jax_logits_BTV, hf_logits_BTV
        gc.collect()


if __name__ == "__main__":
    absltest.main()
