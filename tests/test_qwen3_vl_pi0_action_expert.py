"""Tests for the Qwen3-VL pi0 action expert training/export path."""

import dataclasses
import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open

from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.qwen3_vl.config import make_vl_config, with_pi0_action_expert
from omegalax.models.qwen3_vl.model import Qwen3VL
from omegalax.models.qwen3_vl.params import (
    export_qwen3_vl_pi0_action_expert_to_safetensors,
)
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.trainers import vlm as vlm_trainer


def _pi0_smoke_config():
    cfg = make_vl_config("qwen3-vl-smoke")
    cfg = with_pi0_action_expert(
        cfg,
        enabled=True,
        action_width=16,
        action_mlp_size=32,
    )
    return dataclasses.replace(cfg, dtype=jnp.float32, param_dtype=jnp.float32)


class Qwen3VLPi0ActionExpertTest(absltest.TestCase):

    def test_forward_routes_action_tokens(self):
        cfg = _pi0_smoke_config()
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model = Qwen3VL(cfg, rngs=nnx.Rngs(0))
        set_attn_backend(model, text_backend="xla")

        token_ids_BT = jnp.asarray([[11, 12, 13, 14, 15, 16, 17, 18]], dtype=jnp.int32)
        attention_mask_BT = jnp.ones_like(token_ids_BT)
        action_mask_BT = jnp.asarray([[0, 0, 0, 1, 1, 1, 0, 0]], dtype=jnp.int32)

        hidden_BTD, aux = model(
            token_ids_BT,
            attention_mask_BT,
            action_expert_mask_BT=action_mask_BT,
        )

        self.assertEqual(hidden_BTD.shape, (1, 8, cfg.emb_dim))
        self.assertEqual(aux.shape, ())
        self.assertFalse(np.isnan(np.asarray(hidden_BTD)).any())

    def test_sft_train_step_uses_loss_mask_as_action_mask(self):
        cfg = _pi0_smoke_config()
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model = Qwen3VL(cfg, rngs=nnx.Rngs(0))
        set_attn_backend(model, text_backend="xla")

        train_cfg = vlm_trainer.TrainConfig(
            batch_size=1,
            seq_len=8,
            num_steps=1,
            learning_rate=1e-4,
            pi0_action_expert_enabled=True,
            pi0_action_width=16,
            pi0_action_mlp_size=32,
            num_loss_tiles=1,
        )
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            optimizer = vlm_trainer.build_optimizer(model, 1e-4, train_cfg)
        step = vlm_trainer.make_sft_train_step(cfg, pad_id=0, num_loss_tiles=1)
        batch = {
            "token_ids_BT": jnp.asarray([[11, 12, 13, 14, 15, 16, 17, 18]], dtype=jnp.int32),
            "attention_mask_BT": jnp.ones((1, 8), dtype=jnp.int32),
            "loss_mask_BT": jnp.asarray([[0, 0, 0, 1, 1, 1, 0, 0]], dtype=jnp.int32),
        }

        _, metrics = step(optimizer, batch)

        self.assertGreater(float(metrics["supervised_tokens"]), 0.0)
        self.assertFalse(np.isnan(float(metrics["loss"])))

    def test_sglang_export_key_shapes(self):
        cfg = _pi0_smoke_config()
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model = Qwen3VL(cfg, rngs=nnx.Rngs(0))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_qwen3_vl_pi0_action_expert_to_safetensors(model, cfg, tmpdir)
            self.assertEqual(
                path.name,
                "qwen3_vl_pi0_action_expert_w16_m32_l2.safetensors",
            )
            with safe_open(path, framework="numpy") as sf:
                keys = set(sf.keys())
                self.assertLen(keys, 2 + cfg.num_layers * 8)
                self.assertEqual(sf.get_tensor("action_input_proj.weight").shape, (16, cfg.emb_dim))
                self.assertEqual(sf.get_tensor("action_output_proj.weight").shape, (cfg.emb_dim, 16))

                q_size = cfg.num_heads * cfg.head_dim
                kv_size = cfg.num_kv_heads * cfg.head_dim
                for layer_idx in range(cfg.num_layers):
                    prefix = f"layers.{layer_idx}.action_expert"
                    self.assertEqual(sf.get_tensor(f"{prefix}.input_layernorm.weight").shape, (16,))
                    self.assertEqual(sf.get_tensor(f"{prefix}.post_attention_layernorm.weight").shape, (16,))
                    self.assertEqual(sf.get_tensor(f"{prefix}.q_norm.weight").shape, (cfg.head_dim,))
                    self.assertEqual(sf.get_tensor(f"{prefix}.k_norm.weight").shape, (cfg.head_dim,))
                    self.assertEqual(sf.get_tensor(f"{prefix}.qkv_proj.weight").shape, (q_size + 2 * kv_size, 16))
                    self.assertEqual(sf.get_tensor(f"{prefix}.o_proj.weight").shape, (16, q_size))
                    self.assertEqual(sf.get_tensor(f"{prefix}.gate_up_proj.weight").shape, (64, 16))
                    self.assertEqual(sf.get_tensor(f"{prefix}.down_proj.weight").shape, (16, 32))


if __name__ == "__main__":
    absltest.main()
