import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from absl.testing import absltest
from flax import nnx

from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.params_utils import flatten_pure_state
from omegalax.models.qwen3_5.config import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
    make_config,
)
from omegalax.models.qwen3_5.model import MLP, Qwen3_5ForConditionalGeneration
from omegalax.models.qwen3_5.params import qwen3_5_to_hf_config_dict


def _flat_params(model):
    return flatten_pure_state(nnx.to_pure_dict(nnx.state(model, nnx.Param)))


def _small_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        text_config=Qwen3_5TextConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=("linear_attention",),
            partial_rotary_factor=0.5,
            mrope_section=(2, 1, 1),
            intermediate_size=32,
            linear_conv_kernel_dim=3,
            linear_key_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_value_head_dim=4,
        ),
        vision_config=Qwen3_5VisionConfig(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=1,
            temporal_patch_size=1,
            spatial_merge_size=1,
            in_channels=1,
            out_hidden_size=16,
            num_position_embeddings=4,
        ),
    )


class Qwen3_5ParamDtypeTest(absltest.TestCase):
    def test_hf_config_exports_compute_dtype_not_internal_param_dtype(self):
        hf_config = qwen3_5_to_hf_config_dict(make_config("qwen3.5-smoke"))

        self.assertEqual(hf_config["text_config"]["dtype"], "bfloat16")
        self.assertNotIn("param_dtype", hf_config["text_config"])
        self.assertNotIn("param_dtype", hf_config["vision_config"])

    def test_abstract_model_skeleton_has_only_fp32_parameters(self):
        cfg = make_config("qwen3.5-smoke")
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model = nnx.eval_shape(
                lambda: Qwen3_5ForConditionalGeneration(cfg, rngs=nnx.Rngs(params=0))
            )
        params = _flat_params(model)

        expected = {
            "lm_head.kernel",
            "text.embedder.embedding",
            "text.layers.0.linear_attn.conv_weight",
            "text.layers.0.linear_attn.in_proj_qkv.kernel",
            "text.layers.0.mlp.gate_proj",
            "text.layers.0.mlp.shared_expert.gate_proj.kernel",
            "text.layers.3.attn.q_proj.kernel",
            "vision.blocks.0.attn.qkv.kernel",
            "vision.blocks.0.norm1.weight",
            "vision.patch_embed.proj.kernel",
            "vision.pos_embed.embedding",
        }
        self.assertEmpty(expected - params.keys())
        self.assertEqual(
            {value.dtype for value in params.values()},
            {jnp.dtype(jnp.float32)},
        )

    def test_dense_forward_uses_bf16_dot_operands_and_output(self):
        cfg = make_config("qwen3.5-smoke-dense").text_config
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            mlp = MLP(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.shd_cfg,
                dtype=cfg.dtype,
                param_dtype=cfg.param_dtype,
                rngs=nnx.Rngs(0),
            )
            hidden = jnp.ones((1, 2, cfg.hidden_size), dtype=jnp.bfloat16)
            jaxpr = jax.make_jaxpr(mlp)(hidden).jaxpr
            dots = [
                equation for equation in jaxpr.eqns if equation.primitive.name == "dot_general"
            ]

        self.assertLen(dots, 3)
        self.assertTrue(
            all(
                tuple(value.aval.dtype for value in equation.invars)
                == (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.bfloat16))
                for equation in dots
            )
        )
        self.assertEqual(mlp(hidden).dtype, jnp.bfloat16)
        self.assertEqual(
            {value.dtype for value in _flat_params(mlp).values()},
            {jnp.dtype(jnp.float32)},
        )

    def test_orbax_restore_preserves_fp32_model_state(self):
        cfg = _small_config()
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model = Qwen3_5ForConditionalGeneration(cfg, rngs=nnx.Rngs(0))
            model.lm_head.kernel[...] = jnp.full_like(model.lm_head.kernel[...], 0.25)
            state = nnx.state(model, nnx.Param)

            restored_model = Qwen3_5ForConditionalGeneration(cfg, rngs=nnx.Rngs(2))
            abstract = jax.tree.map(
                lambda value: jax.ShapeDtypeStruct(
                    value.shape, value.dtype, sharding=value.sharding
                ),
                nnx.state(restored_model, nnx.Param),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = ocp.PyTreeCheckpointer()
            checkpoint_path = Path(tmpdir) / "model"
            checkpointer.save(checkpoint_path, args=ocp.args.PyTreeSave(state))
            restored = checkpointer.restore(
                checkpoint_path,
                args=ocp.args.PyTreeRestore(abstract),
            )
            nnx.update(restored_model, restored)

        self.assertEqual(
            {value.dtype for value in _flat_params(restored_model).values()},
            {jnp.dtype(jnp.float32)},
        )
        self.assertTrue(jnp.all(restored_model.lm_head.kernel[...] == 0.25))


if __name__ == "__main__":
    absltest.main()
