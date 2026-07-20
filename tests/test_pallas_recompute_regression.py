"""Regression comparisons for the Pallas custom-VJP recomputation change."""

from __future__ import annotations

import os
import sys
from unittest import mock

from absl import flags
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from omegalax.models.qwen3_5.kernels import pallas_triton as recompute_kernel
from omegalax.trainers import pretrain as pretrain_trainer
from tests import reference_pallas_triton_no_recompute as no_recompute_kernel
from tests import reference_pallas_triton_origin_main as origin_main_kernel
from tests import test_pretrain_statepassing_pallas as integration_helpers


_PALLAS_MODULE = "omegalax.models.qwen3_5.kernels.pallas_triton"


def _has_cuda_device() -> bool:
    try:
        return any(device.platform == "gpu" for device in jax.devices())
    except Exception:
        return False


def _make_kernel_inputs():
    rng = np.random.RandomState(0)
    q = jnp.asarray(rng.randn(1, 128, 1, 64).astype(np.float32) * 0.1, jnp.bfloat16)
    k = jnp.asarray(rng.randn(1, 128, 1, 64).astype(np.float32) * 0.1, jnp.bfloat16)
    v = jnp.asarray(rng.randn(1, 128, 1, 64).astype(np.float32) * 0.1, jnp.bfloat16)
    a = jnp.asarray(rng.randn(1, 128, 1).astype(np.float32) * 0.5)
    g = -jnp.exp(a) * jax.nn.softplus(a)
    beta = jax.nn.sigmoid(jnp.asarray(rng.randn(1, 128, 1).astype(np.float32) * 0.5))
    return q, k, v, g, beta


def _kernel_value_and_grad(fn, inputs):
    def loss_fn(q, k, v, g, beta):
        out = fn(q, k, v, g, beta)
        return jnp.sum(out.astype(jnp.float32) ** 2), out

    return jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True))(*inputs)


@absltest.skipIf(not _has_cuda_device(), "requires CUDA GPU")
class PallasRecomputeRegressionTest(absltest.TestCase):
    def test_first_segment_matches_origin_main_outputs_and_gradients(self):
        inputs = _make_kernel_inputs()
        (main_loss, main_out), main_grads = _kernel_value_and_grad(
            origin_main_kernel.chunk_gated_delta_rule_pallas, inputs
        )
        (current_loss, current_out), current_grads = _kernel_value_and_grad(
            recompute_kernel.chunk_gated_delta_rule_pallas, inputs
        )

        np.testing.assert_allclose(current_loss, main_loss, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(current_out, main_out, rtol=1e-5, atol=1e-5)
        for current_grad, main_grad in zip(current_grads, main_grads, strict=True):
            np.testing.assert_allclose(current_grad, main_grad, rtol=1e-5, atol=1e-5)

    def test_tiny_train_step_matches_no_recompute_with_and_without_statepassing(self):
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["pytest"])
        previous_backend = os.environ.get("OMEGALAX_DELTANET_KERNEL")
        os.environ["OMEGALAX_DELTANET_KERNEL"] = "pallas"
        try:
            for pass_gdn_state in (False, True):
                with self.subTest(pass_gdn_state=pass_gdn_state):
                    baseline_state, baseline_metrics = self._run_tiny_train_step(
                        no_recompute_kernel, pass_gdn_state
                    )
                    current_state, current_metrics = self._run_tiny_train_step(
                        recompute_kernel, pass_gdn_state
                    )
                    self._assert_train_results_close(
                        current_state,
                        current_metrics,
                        baseline_state,
                        baseline_metrics,
                    )
        finally:
            if previous_backend is None:
                os.environ.pop("OMEGALAX_DELTANET_KERNEL", None)
            else:
                os.environ["OMEGALAX_DELTANET_KERNEL"] = previous_backend

    def _run_tiny_train_step(self, kernel_module, pass_gdn_state):
        jax.clear_caches()
        cfg = integration_helpers._hybrid_config()
        train_cfg = pretrain_trainer.TrainConfig(
            seed=0,
            batch_size=2,
            seq_len=128,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )
        token_ids = (np.arange(256, dtype=np.int32) % 255 + 1).reshape(1, 2, 128)
        batch = {
            "token_ids_BCT": token_ids,
            "attention_mask_BCT": np.ones((1, 2, 128), dtype=np.int32),
            "loss_mask_BCT": np.ones((1, 2, 128), dtype=np.int32),
            "chunk_idx_BC": np.asarray([[0, 1]], dtype=np.int32),
            "reset_state_BC": np.asarray([[True, False]], dtype=np.bool_),
            "metadata": {"doc_ids": ["pallas-recompute-regression"]},
        }
        with mock.patch.dict(sys.modules, {_PALLAS_MODULE: kernel_module}):
            optimizer, metrics = pretrain_trainer.run_pretrain(
                cfg,
                train_cfg,
                integration_helpers._batch_iterator(batch),
                pretrain_mode=pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
                pass_gdn_state=pass_gdn_state,
                log_every=0,
                tp_size=1,
                fsdp_size=1,
                dp_size=1,
                bptt_chunks=2,
                pass_rope_positions=True,
                pass_conv_state=True,
                text_attn_backend="mosaic_gpu",
            )
        return nnx.state(optimizer), metrics

    def _assert_train_results_close(
        self,
        current_state,
        current_metrics,
        baseline_state,
        baseline_metrics,
    ):
        self.assertEqual(jax.tree.structure(current_state), jax.tree.structure(baseline_state))
        for current_leaf, baseline_leaf in zip(
            jax.tree.leaves(current_state),
            jax.tree.leaves(baseline_state),
            strict=True,
        ):
            np.testing.assert_allclose(current_leaf, baseline_leaf, rtol=1e-5, atol=1e-5)
        for key in (
            "loss",
            "nll",
            "grad_norm",
            "segment0_nll",
            "segment1_nll",
            "boundary_nll",
            "iid_comparable_nll",
        ):
            np.testing.assert_allclose(
                current_metrics[key], baseline_metrics[key], rtol=1e-5, atol=1e-5
            )


if __name__ == "__main__":
    absltest.main()
