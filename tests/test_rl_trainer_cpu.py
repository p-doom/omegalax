"""Trainer on an FSDP-2 mesh of forced CPU host devices: no compiles after warmup, the host
optimizer placement takes the same steps as the device placement, and a power-of-two loss scale
is undone exactly."""

import logging
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import numpy as np
from absl.testing import absltest

from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.rl.rollout import Sample
from omegalax.rl.trainer import TrainConfig, Trainer
from omegalax.rl.weights import sub_mesh
from omegalax.text import api as text_api


class _Compiles(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.names = []

    def emit(self, record):
        if "Compiling" in record.getMessage():
            self.names.append(record.getMessage().split(" with ")[0])


class FsdpTrainerTest(absltest.TestCase):
    def _run(self, host_optimizer: bool, loss_scale: float = 1.0):
        with jax.set_mesh(sub_mesh(jax.devices()[:2], 1, 2, 1)):
            model, cfg = text_api.init_model("qwen3-smoke", jax.random.key(0), tp_size=1, fsdp_size=2, dp_size=1)
        set_attn_backend(model, "xla")
        trainer = Trainer(model, cfg, TrainConfig(
            lr=1e-3, weight_decay=0.1, adam_b1=0.9, adam_b2=0.98, adam_eps=1e-8, max_grad_norm=1.0,
            eps_low=0.2, eps_high=0.28, tokens_per_microbatch=256, length_buckets=(32, 64),
            logit_tile_tokens=64, pad_id=0, host_optimizer=host_optimizer, loss_scale=loss_scale,
        ))
        rng = np.random.default_rng(0)
        samples = [
            Sample(0, rng.integers(1, 100, L).tolist(), 4, [0] * 4 + [1] * (L - 4), [-1.0] * L, [0] * L)
            for L in rng.integers(10, 60, 24)
        ]
        advantages = list(rng.normal(size=24))
        counter = _Compiles()
        logging.getLogger("jax").addHandler(counter)
        jax.config.update("jax_log_compiles", True)
        try:
            trainer.warmup()
            norms, compiles = [], []
            for _ in range(3):
                counter.names.clear()
                norms.append(trainer.step(samples, advantages)["train/grad_norm"])
                compiles.append(list(counter.names))
        finally:
            jax.config.update("jax_log_compiles", False)
            logging.getLogger("jax").removeHandler(counter)
        return norms, compiles, jax.device_get(trainer.params)

    def test_both_placements_step_identically_without_recompiling(self):
        device_norms, device_compiles, device_params = self._run(host_optimizer=False)
        host_norms, host_compiles, host_params = self._run(host_optimizer=True)
        self.assertEqual(device_compiles, [[], [], []])
        # The host placement reshards its updated master onto the trainer mesh once.
        self.assertTrue(all("_multi_slice" in c for c in host_compiles[0]), host_compiles[0])
        self.assertEqual(host_compiles[1:], [[], []])
        np.testing.assert_allclose(device_norms, host_norms, rtol=1e-5)
        for a, b in zip(jax.tree.leaves(device_params), jax.tree.leaves(host_params)):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)

    def test_loss_scale_is_exact(self):
        norms, _, params = self._run(host_optimizer=False)
        scaled_norms, _, scaled_params = self._run(host_optimizer=False, loss_scale=2.0**16)
        np.testing.assert_array_equal(norms, scaled_norms)
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(scaled_params)):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    absltest.main()
