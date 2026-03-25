"""End-to-end correctness test for Qwen3-0.6B against HuggingFace."""

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ["USE_HUB_KERNELS"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import safetensors
import torch

from absl.testing import absltest
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

from tests.logits_assert import assert_logits_close
from tests.real_weights import requires_real_weights
from omegalax.distributed.mesh import mesh_rules_for
from omegalax.text import api
from omegalax.models.params_utils import map_to_bonsai_key
from omegalax.models.qwen3 import loader as qwen3_loader


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "Why is the sky blue instead of another color like purple?"


def _flatten_leaf_keys(tree: dict, prefix: str = "") -> list[str]:
    keys: list[str] = []
    for k, v in tree.items():
        dotted = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            keys.extend(_flatten_leaf_keys(v, dotted))
        else:
            keys.append(dotted)
    return keys


def _get_value(tree: dict, dotted_key: str):
    node = tree
    for token in dotted_key.split("."):
        key = int(token) if token.isdigit() else token
        node = node[key]
    return node


@requires_real_weights
class Qwen3MappingTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.cfg = api.registry.build_config(MODEL_ID)
        cls.model_path = snapshot_download(MODEL_ID)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_path)
        hf_cfg = AutoConfig.from_pretrained(cls.model_path)
        hf_cfg.tie_word_embeddings = cls.cfg.tie_word_embeddings
        cls.hf_model = Qwen3ForCausalLM.from_pretrained(
            cls.model_path, config=hf_cfg, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to(cls.device).eval()
        cls.pad_id = cls.tokenizer.pad_token_id or 0

        cls.jax_model = qwen3_loader.create_qwen3_from_safetensors(
            cls.model_path,
            MODEL_ID,
            tp_size=1,
            fsdp_size=1,
        )

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

    def test_parameter_mapping_is_complete(self):
        # HF -> JAX mapping should cover every HF tensor key.
        mapping = qwen3_loader._get_key_mapping()
        unmapped: list[str] = []
        for f in Path(self.model_path).glob("*.safetensors"):
            with safetensors.safe_open(f, framework="numpy") as sf:
                for torch_key in sf.keys():
                    jax_key, _ = map_to_bonsai_key(mapping, torch_key)
                    if jax_key is None:
                        unmapped.append(torch_key)
        if unmapped:
            self.fail(f"Unmapped HF parameter keys: {unmapped}")

        # All JAX leaves should be populated and match abstract shapes.
        model_cls = api.registry.get_model_cls(self.cfg.variant)
        with mesh_rules_for(tp_size=1, fsdp_size=1):
            _, abs_state = nnx.split(nnx.eval_shape(lambda: model_cls(self.cfg, rngs=nnx.Rngs(params=0))))
        abs_dict = nnx.to_pure_dict(abs_state)
        _, loaded_state = nnx.split(self.jax_model)
        loaded_dict = nnx.to_pure_dict(loaded_state)

        abs_keys = set(_flatten_leaf_keys(abs_dict))
        loaded_keys = set(_flatten_leaf_keys(loaded_dict))
        self.assertSetEqual(abs_keys, loaded_keys)

        for key in abs_keys:
            abs_val = _get_value(abs_dict, key)
            loaded_val = _get_value(loaded_dict, key)
            self.assertEqual(abs_val.shape, loaded_val.shape, f"Shape mismatch at {key}")
            self.assertIsInstance(loaded_val, (jax.Array, np.ndarray))

    def test_prefill_logits_match_hf(self):
        inputs = self._tokenize([PROMPT])
        with torch.no_grad():
            hf_logits_BTV = self.hf_model(**inputs).logits.float().cpu().numpy()
        jax_logits_BTV = self._jax_prefill_logits(inputs["input_ids"])
        mask = inputs["attention_mask"].cpu().numpy().astype(bool)
        assert_logits_close(self, jax_logits_BTV, hf_logits_BTV, mask, top1_min_match=0.8)

    def test_prefill_logits_match_hf_batched(self):
        inputs = self._tokenize([PROMPT, "Who am I?"])
        with torch.no_grad():
            hf_logits_BTV = self.hf_model(**inputs).logits.float().cpu().numpy()
        jax_logits_BTV = self._jax_prefill_logits(inputs["input_ids"])
        mask = inputs["attention_mask"].cpu().numpy().astype(bool)
        assert_logits_close(self, jax_logits_BTV, hf_logits_BTV, mask, top1_min_match=0.8)

    def test_round_trip_preserves_prefill_logits(self):
        inputs = self._tokenize([PROMPT])
        baseline_BTV = self._jax_prefill_logits(inputs["input_ids"])

        graph_def, state = nnx.split(self.jax_model)
        pure_state = nnx.to_pure_dict(state)
        restored = nnx.merge(graph_def, pure_state)
        restored_token_ids_BT = jnp.asarray(np.array(inputs["input_ids"].cpu()), dtype=jnp.int32)
        segment_ids_BT = 1 * (restored_token_ids_BT != self.pad_id)
        restored_hidden_BTD, _ = restored(restored_token_ids_BT, segment_ids_BT, None, jnp.array(0, dtype=jnp.int32))
        restored_logits_BTV = restored.lm_head(restored_hidden_BTD)
        restored_logits_BTV = np.asarray(restored_logits_BTV)

        np.testing.assert_allclose(restored_logits_BTV, baseline_BTV, rtol=0, atol=0)


if __name__ == "__main__":
    absltest.main()
