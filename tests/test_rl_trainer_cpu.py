"""Trainer on an FSDP-2 mesh of forced CPU host devices: no compiles after warmup, the host
optimizer placement takes the same steps as the device placement, and Linear weight gradients are
float32 sums across devices."""

import logging
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax.sharding import NamedSharding, PartitionSpec as P

from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.rl.rollout import Sample
from omegalax.rl.trainer import TrainConfig, Trainer, _linear_dot
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
    def _run(self, host_optimizer: bool):
        with jax.set_mesh(sub_mesh(jax.devices()[:2], 1, 2, 1)):
            model, cfg = text_api.init_model("qwen3-smoke", jax.random.key(0), tp_size=1, fsdp_size=2, dp_size=1)
        set_attn_backend(model, "xla")
        trainer = Trainer(model, cfg, TrainConfig(
            lr=1e-3, weight_decay=0.1, adam_b1=0.9, adam_b2=0.98, adam_eps=1e-8, max_grad_norm=1.0,
            eps_low=0.2, eps_high=0.28, tokens_per_microbatch=256, length_buckets=(32, 64),
            logit_tile_tokens=64, pad_id=0, host_optimizer=host_optimizer,
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


class LinearDotTest(absltest.TestCase):
    def test_weight_gradient_is_a_float32_sum_across_devices(self):
        mesh = sub_mesh(jax.devices()[:2], 1, 2, 1)
        rng = np.random.default_rng(0)
        x = np.asarray(rng.normal(size=(4, 64, 128)), jnp.bfloat16)
        w = rng.normal(size=(128, 32)).astype(np.float32) / 10
        r = np.asarray(rng.normal(size=(4, 64, 32)), jnp.bfloat16)
        dims = (((2,), (0,)), ((), ()))
        with jax.set_mesh(mesh):
            xs = jax.device_put(x, NamedSharding(mesh, P("fsdp", None, None)))
            ws = jax.device_put(w, NamedSharding(mesh, P(None, None)))

            def grads(dot):
                return jax.grad(lambda x, w: jnp.sum(dot(x, w).astype(jnp.float32) * r), argnums=(0, 1))(xs, ws)

            dx, dw = grads(lambda x, w: _linear_dot(x, w, dims, None))
            _, dw_bf16 = grads(lambda x, w: jax.lax.dot_general(x, w.astype(x.dtype), dims))
        x64, r64 = np.asarray(x, np.float64), np.asarray(r, np.float64)
        dw_ref = np.einsum("btd,btf->df", x64, r64)
        err = lambda a: np.abs(np.asarray(a, np.float64) - dw_ref).max() / np.abs(dw_ref).max()
        self.assertEqual(dw.dtype, jnp.float32)
        self.assertLess(err(dw), 1e-6)
        self.assertGreater(err(dw_bf16), 1e-4)
        dx_ref = np.einsum("btf,df->btd", r64, np.asarray(np.asarray(w, jnp.bfloat16), np.float64))
        np.testing.assert_allclose(np.asarray(dx, np.float64), dx_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    absltest.main()
