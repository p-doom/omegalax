"""Smoke test for Qwen3.5 GatedDeltaNet XLA backend against HuggingFace."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from flax import nnx
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5TextConfig as HFTextConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet as HFGatedDeltaNet,
)

from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.qwen3_5.config import Qwen3_5TextConfig
from omegalax.models.qwen3_5.deltanet import GatedDeltaNet
from omegalax.models.shard_config import ShardConfig

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _hf_config() -> HFTextConfig:
    return HFTextConfig(
        dtype="float32",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        layer_types=["linear_attention"],
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        tie_word_embeddings=False,
    )


def _jax_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        layer_types=("linear_attention",),
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        shd_cfg=ShardConfig.no_sharding(),
        dtype=jnp.float32,
    )


def _to_jax(tensor: torch.Tensor):
    return jnp.asarray(tensor.detach().cpu().float().numpy(), dtype=jnp.float32)


def _copy_linear(hf_linear: torch.nn.Linear, jax_linear: nnx.Linear) -> None:
    jax_linear.kernel[...] = _to_jax(hf_linear.weight).T


def _copy_hf_weights(hf_module: HFGatedDeltaNet, jax_module: GatedDeltaNet) -> None:
    _copy_linear(hf_module.in_proj_qkv, jax_module.in_proj_qkv)
    _copy_linear(hf_module.in_proj_z, jax_module.in_proj_z)
    _copy_linear(hf_module.in_proj_b, jax_module.in_proj_b)
    _copy_linear(hf_module.in_proj_a, jax_module.in_proj_a)
    _copy_linear(hf_module.out_proj, jax_module.out_proj)
    jax_module.conv_weight[...] = _to_jax(hf_module.conv1d.weight.squeeze(1))
    jax_module.dt_bias[...] = _to_jax(hf_module.dt_bias)
    jax_module.A_log[...] = _to_jax(hf_module.A_log)
    jax_module.norm.weight[...] = _to_jax(hf_module.norm.weight)


def _hidden_states() -> torch.Tensor:
    data = np.arange(2 * 5 * 16, dtype=np.float32).reshape(2, 5, 16)
    data = data / 50.0 - 1.0
    return torch.tensor(data, dtype=torch.float32)


def _assert_allclose_with_stats(testcase: absltest.TestCase, actual, expected) -> None:
    actual_np = np.asarray(actual, dtype=np.float32)
    expected_np = np.asarray(expected, dtype=np.float32)
    testcase.assertEqual(actual_np.shape, expected_np.shape)
    testcase.assertTrue(np.isfinite(actual_np).all(), "JAX output contains non-finite values.")
    testcase.assertTrue(np.isfinite(expected_np).all(), "HF output contains non-finite values.")

    abs_diff = np.abs(actual_np - expected_np)
    max_expected = max(float(np.abs(expected_np).max()), 1e-12)
    max_abs_diff = float(abs_diff.max())
    max_rel_diff = max_abs_diff / max_expected
    np.testing.assert_allclose(
        actual_np,
        expected_np,
        atol=1e-4,
        rtol=1e-4,
        err_msg=(
            "Qwen3.5 GatedDeltaNet XLA smoke mismatch against HF "
            f"(shape={actual_np.shape}, max_abs_diff={max_abs_diff:.6e}, "
            f"max_rel_diff={max_rel_diff:.6e}, atol=1e-4, rtol=1e-4, "
            "B=2, T=5, D=16, masked tokens in second row)."
        ),
    )
    print(
        "TEST PASSED: Qwen3.5 GatedDeltaNet XLA backend matches HF smoke reference "
        f"(shape={actual_np.shape}, max_abs_diff={max_abs_diff:.6e}, "
        f"max_rel_diff={max_rel_diff:.6e}, atol=1e-4, rtol=1e-4).",
        flush=True,
    )


class Qwen3_5DeltaNetXLAHFSmokeTest(absltest.TestCase):
    def test_forward_matches_hf_with_padding_mask(self):
        torch.manual_seed(0)
        np.random.seed(0)

        hf_module = HFGatedDeltaNet(_hf_config(), layer_idx=0).eval()
        hidden_states = _hidden_states()
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
            ],
            dtype=torch.float32,
        )

        with torch.no_grad():
            hf_out = hf_module(hidden_states, attention_mask=attention_mask)

        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            jax_module = GatedDeltaNet(_jax_config(), rngs=nnx.Rngs(0))
            _copy_hf_weights(hf_module, jax_module)
            jax_out = jax_module(
                jnp.asarray(hidden_states.numpy(), dtype=jnp.float32),
                jnp.asarray(attention_mask.numpy(), dtype=jnp.float32),
            )

        _assert_allclose_with_stats(
            self,
            jax_out,
            hf_out.detach().cpu().float().numpy(),
        )


if __name__ == "__main__":
    absltest.main()
