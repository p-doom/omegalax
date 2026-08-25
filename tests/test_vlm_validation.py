from __future__ import annotations

import dataclasses
import os
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from omegalax.models.qwen3_vl import make_vl_config
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.trainers import vlm


class VLMValidationTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cfg = make_vl_config("qwen3-vl-smoke")
        cfg = dataclasses.replace(
            cfg,
            dtype=jnp.float32,
            vision=dataclasses.replace(cfg.vision, dtype=jnp.float32),
        )
        cls.model, cls.cfg = vlm.vlm_api.init_model(
            cfg,
            jax.random.key(11),
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )
        set_attn_backend(cls.model, text_backend="xla")
        cls.eval_step = staticmethod(vlm.make_sft_eval_step(cls.cfg, num_loss_tiles=1))

    def _batch(self, row: slice = slice(None)):
        token_ids = jnp.array(
            [[11, 12, 13, 14], [21, 22, 23, 24]],
            dtype=jnp.int32,
        )
        return {
            "token_ids_BT": token_ids[row],
            "attention_mask_BT": jnp.ones_like(token_ids[row]),
            "loss_mask_BT": jnp.array(
                [[0, 0, 0, 1], [0, 1, 1, 1]],
                dtype=jnp.int32,
            )[row],
        }

    def test_eval_returns_token_weighted_ce_sum_for_asymmetric_masks(self):
        full_sum, full_count, full_aux = self.eval_step(self.model, self._batch())
        first_sum, first_count, first_aux = self.eval_step(self.model, self._batch(slice(0, 1)))
        second_sum, second_count, second_aux = self.eval_step(self.model, self._batch(slice(1, 2)))

        self.assertEqual(float(first_count), 1.0)
        self.assertEqual(float(second_count), 3.0)
        self.assertEqual(float(full_count), 4.0)
        np.testing.assert_allclose(full_sum, first_sum + second_sum, rtol=0.0, atol=1e-6)
        self.assertNotAlmostEqual(
            float(full_sum / full_count),
            float((first_sum / first_count + second_sum / second_count) / 2.0),
            places=5,
        )
        np.testing.assert_array_equal(np.asarray([full_aux, first_aux, second_aux]), 0.0)

    def test_eval_rejects_batch_without_supervised_targets(self):
        batch = self._batch()
        batch["loss_mask_BT"] = jnp.zeros_like(batch["loss_mask_BT"])

        with self.assertRaisesRegex(ValueError, "no supervised next-token targets"):
            self.eval_step(self.model, batch)

    def test_eval_rejects_invalid_metrics(self):
        cases = (
            ((1.0, float("nan"), 0.0), FloatingPointError, "count is non-finite"),
            ((1.0, float("inf"), 0.0), FloatingPointError, "count is non-finite"),
            ((1.0, 1.5, 0.0), ValueError, "count must be an integer"),
            ((float("nan"), 1.0, 0.0), FloatingPointError, "CE loss sum is non-finite"),
            ((1.0, 1.0, float("inf")), FloatingPointError, "auxiliary loss is non-finite"),
        )
        for metrics, error_type, message in cases:
            with (
                self.subTest(metrics=metrics),
                mock.patch.object(vlm.nnx, "jit", side_effect=lambda fn: fn),
                mock.patch.object(
                    vlm.vlm_api,
                    "forward",
                    return_value=(object(), jnp.asarray(metrics[2])),
                ),
                mock.patch.object(
                    vlm,
                    "chunked_cross_entropy_loss_sum",
                    return_value=tuple(jnp.asarray(value) for value in metrics[:2]),
                ),
                self.assertRaisesRegex(error_type, message),
            ):
                model = mock.Mock()
                model.output_weight.return_value = object()
                cfg = mock.Mock()
                cfg.shd_cfg.logits_btv = None
                vlm.make_sft_eval_step(cfg)(model, self._batch())


if __name__ == "__main__":
    absltest.main()
