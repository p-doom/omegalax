"""End-to-end correctness test for Qwen3-0.6B against HuggingFace."""

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import safetensors
import torch
from absl.testing import absltest
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

from omegalax.text import api
from omegalax.models.params_utils import map_to_bonsai_key
from omegalax.models.qwen3.dense import params_dense


jax.config.update("jax_default_matmul_precision", "highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "Why is the sky blue instead of another color like purple?"
RTOL = 1e-5
ATOL = 1e-4


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


class Qwen3MappingTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        super().setUpClass()
        cls.cfg = api.registry.build_config(MODEL_ID)
        cls.model_path = snapshot_download(MODEL_ID)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_path)
        hf_cfg = AutoConfig.from_pretrained(cls.model_path)
        hf_cfg.tie_word_embeddings = cls.cfg.tie_word_embeddings
        cls.hf_model = Qwen3ForCausalLM.from_pretrained(
            cls.model_path, config=hf_cfg, torch_dtype=torch.float32
        ).eval()
        cls.pad_id = cls.tokenizer.pad_token_id or 0

        cls.jax_model = params_dense.create_qwen3_dense_from_safe_tensors(cls.model_path, MODEL_ID)

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

    def test_parameter_mapping_is_complete(self):
        # HF -> JAX mapping should cover every HF tensor key.
        mapping = params_dense._get_key_and_transform_mapping(self.cfg)
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
            hf_logits = self.hf_model(**inputs).logits.cpu().numpy()
        jax_logits = self._jax_prefill_logits(inputs["input_ids"])
        mask = inputs["attention_mask"].numpy().astype(bool)
        np.testing.assert_allclose(jax_logits[mask], hf_logits[mask], rtol=RTOL, atol=ATOL)

    def test_prefill_logits_match_hf_batched(self):
        inputs = self._tokenize([PROMPT, "Who am I?"])
        with torch.no_grad():
            hf_logits = self.hf_model(**inputs).logits.cpu().numpy()
        jax_logits = self._jax_prefill_logits(inputs["input_ids"])
        mask = inputs["attention_mask"].numpy().astype(bool)
        np.testing.assert_allclose(jax_logits[mask], hf_logits[mask], rtol=RTOL, atol=ATOL)

    def test_round_trip_preserves_prefill_logits(self):
        inputs = self._tokenize([PROMPT])
        baseline = self._jax_prefill_logits(inputs["input_ids"])

        graph_def, state = nnx.split(self.jax_model)
        pure_state = nnx.to_pure_dict(state)
        restored = nnx.merge(graph_def, pure_state)
        restored_tokens = jnp.asarray(np.array(inputs["input_ids"]), dtype=jnp.int32)
        restored_logits, _ = api.forward(restored, restored_tokens, self.pad_id, self.cfg)
        restored_logits = np.asarray(restored_logits)

        np.testing.assert_allclose(restored_logits, baseline, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    absltest.main()
