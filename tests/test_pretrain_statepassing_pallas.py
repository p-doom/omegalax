"""GPU integration test for the production Statepassing Pallas path."""

from __future__ import annotations

import os

from absl import flags
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from omegalax.models.qwen3_5.config import Qwen3_5TextConfig
from omegalax.models.shard_config import ShardConfig
from omegalax.trainers import pretrain as pretrain_trainer


def _has_cuda_device() -> bool:
    try:
        return any(device.platform == "gpu" for device in jax.devices())
    except Exception:
        return False


def _hybrid_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        rms_norm_eps=1e-6,
        layer_types=("linear_attention", "full_attention"),
        rope_theta=10_000,
        partial_rotary_factor=0.25,
        mrope_section=(4, 2, 2),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=64,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_value_head_dim=64,
        intermediate_size=128,
        shd_cfg=ShardConfig.no_sharding(),
        dtype=jnp.bfloat16,
    )


def _batch_iterator(batch):
    while True:
        yield {
            key: value.copy() if hasattr(value, "copy") else value for key, value in batch.items()
        }


@absltest.skipIf(not _has_cuda_device(), "requires CUDA GPU")
class StatepassingPallasIntegrationTest(absltest.TestCase):
    def test_hybrid_train_step_uses_carried_gdn_state(self):
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["pytest"])
        previous_backend = os.environ.get("OMEGALAX_DELTANET_KERNEL")
        os.environ["OMEGALAX_DELTANET_KERNEL"] = "pallas"
        try:
            cfg = _hybrid_config()
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
                "metadata": {"doc_ids": ["hybrid-pallas"]},
            }
            run_kwargs = {
                "pretrain_mode": pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
                "log_every": 0,
                "tp_size": 1,
                "fsdp_size": 1,
                "dp_size": 1,
                "bptt_chunks": 2,
                "pass_rope_positions": True,
                "pass_conv_state": True,
                "text_attn_backend": "mosaic_gpu",
            }

            _, stateful_metrics = pretrain_trainer.run_pretrain(
                cfg,
                train_cfg,
                _batch_iterator(batch),
                pass_gdn_state=True,
                **run_kwargs,
            )
            _, stateless_metrics = pretrain_trainer.run_pretrain(
                cfg,
                train_cfg,
                _batch_iterator(batch),
                pass_gdn_state=False,
                **run_kwargs,
            )
        finally:
            if previous_backend is None:
                os.environ.pop("OMEGALAX_DELTANET_KERNEL", None)
            else:
                os.environ["OMEGALAX_DELTANET_KERNEL"] = previous_backend

        self.assertTrue(np.isfinite(stateful_metrics["nll"]))
        self.assertTrue(np.isfinite(stateful_metrics["grad_norm"]))
        self.assertGreater(stateful_metrics["grad_norm"], 0.0)
        self.assertEqual(stateful_metrics["supervised_tokens"], 255.0)
        self.assertNotAlmostEqual(
            stateful_metrics["segment1_nll"],
            stateless_metrics["segment1_nll"],
            places=5,
        )


if __name__ == "__main__":
    absltest.main()
