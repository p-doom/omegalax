"""Parameter storage dtypes for the Qwen3.5 text model."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
from absl.testing import absltest
from flax import nnx

from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.qwen3_5.config import make_config
from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM


class Qwen3_5ParamDtypeTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1))

    def test_default_text_parameters_use_expected_storage_dtypes(self):
        cfg = make_config("qwen3.5-smoke").text_config
        model = Qwen3_5ForCausalLM(cfg, rngs=nnx.Rngs(0))

        linear_layer = model.text.layers[0]
        attention_layer = model.text.layers[3]

        self.assertEqual(cfg.dtype, jnp.bfloat16)
        self.assertEqual(cfg.param_dtype, jnp.float32)
        self.assertEqual(model.text.embedder.embedding.dtype, jnp.float32)
        self.assertEqual(model.lm_head.kernel.dtype, jnp.float32)
        self.assertEqual(attention_layer.attn.q_proj.kernel.dtype, jnp.float32)
        self.assertEqual(attention_layer.attn.o_proj.kernel.dtype, jnp.float32)
        self.assertEqual(linear_layer.linear_attn.in_proj_qkv.kernel.dtype, jnp.float32)
        self.assertEqual(linear_layer.linear_attn.out_proj.kernel.dtype, jnp.float32)
        self.assertEqual(linear_layer.mlp.gate_proj.dtype, jnp.float32)
        self.assertEqual(linear_layer.mlp.down_proj.dtype, jnp.float32)
        self.assertEqual(linear_layer.mlp.shared_expert.gate_proj.kernel.dtype, jnp.float32)

        self.assertEqual(linear_layer.input_layernorm.weight.dtype, jnp.float32)
        self.assertEqual(attention_layer.attn.q_norm.weight.dtype, jnp.float32)
        self.assertEqual(linear_layer.linear_attn.norm.weight.dtype, jnp.float32)
        self.assertEqual(linear_layer.linear_attn.conv_weight.dtype, jnp.float32)
        self.assertEqual(linear_layer.linear_attn.dt_bias.dtype, jnp.float32)
        self.assertEqual(linear_layer.linear_attn.A_log.dtype, jnp.float32)


if __name__ == "__main__":
    absltest.main()
