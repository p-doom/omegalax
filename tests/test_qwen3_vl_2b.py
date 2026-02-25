"""End-to-end correctness test for Qwen3-VL-2B against HuggingFace."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import safetensors
import torch
from absl.testing import absltest
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from omegalax.models.params_utils import map_to_bonsai_key
from omegalax.models.qwen3_vl import create_qwen3_vl_from_safetensors
from omegalax.models.qwen3_vl.loader import _get_non_expert_mapping
from omegalax.models.qwen3_vl.config import make_vl_config_from_hf
from omegalax.models.params_utils import load_hf_config

jax.config.update("jax_default_matmul_precision", "highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
PROMPT = "Why is the sky blue?"
TEXT_RTOL = 1e-4
TEXT_ATOL = 5e-4
IMAGE_RTOL = 1e-3
IMAGE_ATOL = 5e-3


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


class Qwen3VLMappingTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        super().setUpClass()
        cls.model_path = snapshot_download(MODEL_ID)
        cls.processor = AutoProcessor.from_pretrained(cls.model_path)
        hf_cfg = AutoConfig.from_pretrained(cls.model_path)
        cls.hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
            cls.model_path, config=hf_cfg, torch_dtype=torch.float32, attn_implementation="eager"
        ).eval()
        cls.pad_id = cls.processor.tokenizer.pad_token_id or 0

        hf_cfg_dict = load_hf_config(cls.model_path)
        cls.cfg = make_vl_config_from_hf(hf_cfg_dict)
        cls.jax_model, _ = create_qwen3_vl_from_safetensors(cls.model_path, MODEL_ID)

    def test_parameter_mapping_is_complete(self):
        """All HF keys should be mapped; all JAX leaves should be populated."""
        mapping = _get_non_expert_mapping()
        unmapped: list[str] = []
        for f in Path(self.model_path).glob("*.safetensors"):
            with safetensors.safe_open(f, framework="numpy") as sf:
                for torch_key in sf.keys():
                    jax_key, _ = map_to_bonsai_key(mapping, torch_key)
                    if jax_key is None:
                        unmapped.append(torch_key)
        if unmapped:
            self.fail(f"Unmapped HF parameter keys ({len(unmapped)}):\n" + "\n".join(sorted(unmapped)))

        from omegalax.models.qwen3_vl.model import Qwen3VL
        _, abs_state = nnx.split(nnx.eval_shape(lambda: Qwen3VL(self.cfg, rngs=nnx.Rngs(params=0))))
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

    def test_text_only_prefill_logits_match_hf(self):
        """Text-only forward (no images) should match HF model."""
        messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor.tokenizer(text, return_tensors="pt", padding=True)

        with torch.no_grad():
            hf_logits_BTV = self.hf_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits.cpu().numpy()

        token_ids_BT = jnp.asarray(np.array(inputs["input_ids"].cpu(), dtype=np.int32))
        attention_mask_BT = jnp.asarray(np.array(inputs["attention_mask"].cpu(), dtype=np.int32))
        jax_logits_BTV = np.asarray(self.jax_model(token_ids_BT, attention_mask_BT), dtype=np.float32)

        mask = inputs["attention_mask"].numpy().astype(bool)
        np.testing.assert_allclose(jax_logits_BTV[mask], hf_logits_BTV[mask], rtol=TEXT_RTOL, atol=TEXT_ATOL)

    def test_text_only_prefill_logits_batched(self):
        """Batched text-only forward should match HF model."""
        prompts = [PROMPT, "Who am I?"]
        texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": p}]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        inputs = self.processor.tokenizer(texts, return_tensors="pt", padding=True, padding_side="left")

        with torch.no_grad():
            hf_logits_BTV = self.hf_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits.cpu().numpy()

        token_ids_BT = jnp.asarray(np.array(inputs["input_ids"].cpu(), dtype=np.int32))
        attention_mask_BT = jnp.asarray(np.array(inputs["attention_mask"].cpu(), dtype=np.int32))
        jax_logits_BTV = np.asarray(self.jax_model(token_ids_BT, attention_mask_BT), dtype=np.float32)

        mask = inputs["attention_mask"].numpy().astype(bool)
        np.testing.assert_allclose(jax_logits_BTV[mask], hf_logits_BTV[mask], rtol=TEXT_RTOL, atol=TEXT_ATOL)

    def test_image_prefill_logits_match_hf(self):
        """Image+text forward should match HF model."""
        from PIL import Image

        img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt", padding=True)

        with torch.no_grad():
            hf_logits_BTV = self.hf_model(**inputs).logits.cpu().numpy()

        token_ids_BT = jnp.asarray(np.array(inputs["input_ids"].cpu(), dtype=np.int32))
        attention_mask_BT = jnp.asarray(np.array(inputs["attention_mask"].cpu(), dtype=np.int32))
        pixel_values_jax = jnp.asarray(inputs["pixel_values"].cpu().numpy())
        image_grid_thw_jax = jnp.asarray(inputs["image_grid_thw"].cpu().numpy())

        jax_logits_BTV = np.asarray(
            self.jax_model(
                token_ids_BT,
                attention_mask_BT,
                pixel_values=pixel_values_jax,
                image_grid_thw=image_grid_thw_jax,
            ),
            dtype=np.float32,
        )

        mask = inputs["attention_mask"].numpy().astype(bool)
        np.testing.assert_allclose(jax_logits_BTV[mask], hf_logits_BTV[mask], rtol=IMAGE_RTOL, atol=IMAGE_ATOL)


if __name__ == "__main__":
    absltest.main()
