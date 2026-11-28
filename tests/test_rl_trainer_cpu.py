"""Trainer on an FSDP-2 mesh of forced CPU host devices: no compiles after warmup, the host
optimizer placement takes the same steps as the device placement, a power-of-two loss scale is
undone exactly, Linear weight gradients are float32 sums across devices, and packed rows train like
one sample per row."""

import dataclasses
import logging
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx
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
    def _run(self, host_optimizer: bool, loss_scale: float = 1.0):
        with jax.set_mesh(sub_mesh(jax.devices()[:2], 1, 2, 1)):
            model, cfg = text_api.init_model("qwen3.5-smoke-dense", jax.random.key(0), tp_size=1, fsdp_size=2, dp_size=1)
        set_attn_backend(model, "xla")
        trainer = Trainer(model, cfg, TrainConfig(
            lr=1e-3, weight_decay=0.1, adam_b1=0.9, adam_b2=0.98, adam_eps=1e-8, max_grad_norm=1.0,
            eps_low=0.2, eps_high=0.28, tokens_per_microbatch=256, row_tokens=128,
            logit_tile_tokens=64, pad_id=0, host_optimizer=host_optimizer, loss_scale=loss_scale, objective="ppo", topk=1, drift_every=1,
        ))
        rng = np.random.default_rng(0)
        samples = [
            Sample(0, rng.integers(1, 100, L).tolist(), 4, [0] * 4 + [1] * (L - 4), [-1.0] * L, [0] * L,
                   heads=[(np.zeros((L - 4, 1), np.int32), np.zeros((L - 4, 1), np.float32))])
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


class PackingTest(absltest.TestCase):
    def test_packed_rows_match_one_sample_per_row(self):
        # float32 activations: bfloat16 reduction-order noise would be ~1 %, a leak is ~10 %.
        cfg = dataclasses.replace(text_api.resolve_config("qwen3.5-smoke-dense"), dtype=jnp.float32)
        with jax.set_mesh(sub_mesh(jax.devices()[:1], 1, 1, 1)):
            model, cfg = text_api.init_model(cfg, jax.random.key(0), tp_size=1, fsdp_size=1, dp_size=1)
        set_attn_backend(model, "xla")
        trainer = Trainer(model, cfg, TrainConfig(
            lr=1e-3, weight_decay=0.1, adam_b1=0.9, adam_b2=0.98, adam_eps=1e-8, max_grad_norm=1.0,
            eps_low=0.2, eps_high=0.28, tokens_per_microbatch=512, row_tokens=512,
            logit_tile_tokens=128, pad_id=0, host_optimizer=False, loss_scale=1.0, objective="ppo", topk=1, drift_every=0,
        ))
        rng = np.random.default_rng(0)
        # 64 and 128 fill their chunks, so the next segment in the row starts right after a real token.
        tokens = [rng.integers(1, 1000, L).tolist() for L in (64, 150, 37, 128, 90, 64)]
        # Behaviour logprobs = the trainer's own: every ratio is 1, so no token's gradient is clipped away.
        samples = [
            Sample(0, t, 8, [0] * 8 + [1] * (len(t) - 8), list(trainer.logprobs([t])[0]), [0] * len(t),
                   heads=[(np.zeros((len(t) - 8, 1), np.int32), np.zeros((len(t) - 8, 1), np.float32))])
            for t in tokens
        ]
        adv = [1.0, -0.5, 0.8, -1.2, 0.6, -0.9]
        packed = trainer.microbatches(samples, adv)
        alone = [trainer.microbatches([s], [a])[0] for s, a in zip(samples, adv)]
        self.assertEqual(len(packed), 2)

        def grads(mbs):
            with jax.set_mesh(trainer.mesh):
                acc = trainer._zeros(trainer.params)
                for mb in mbs:
                    acc, _ = trainer._accumulate(trainer.params, acc, trainer._put(mb), np.float32(1 / 1000))
            return [np.asarray(g) for g in jax.tree.leaves(acc)]

        def max_rel_err(a, b):
            return max(float(np.abs(x - y).max() / np.abs(y).max()) for x, y in zip(a, b))

        reference = grads(alone)
        self.assertLess(max_rel_err(grads(packed), reference), 1e-5)
        # The check sees leaks: one segment per row (positions, attention, conv and recurrence span it) differs.
        leaky = [dict(mb, segment_ids=(mb["segment_ids"] > 0).astype(np.int32)) for mb in packed]
        self.assertGreater(max_rel_err(grads(leaky), reference), 1e-2)

        for s, logp in zip(samples, trainer.logprobs([s.tokens for s in samples])):
            np.testing.assert_allclose(logp, trainer.logprobs([s.tokens])[0], rtol=1e-5, atol=1e-5)

        # RoPE is relative, so the checks above barely see positions; the engine counts each sequence from 0.
        seg = packed[0]["segment_ids"]
        restarted = np.zeros_like(seg)
        for row, ids in zip(restarted, seg):
            for t in range(1, len(ids)):
                row[t] = 0 if ids[t] != ids[t - 1] else row[t - 1] + 1
        hidden = jax.jit(lambda p, *a: nnx.merge(trainer.graphdef, p).text(*a)[0])
        with jax.set_mesh(trainer.mesh):
            default = hidden(trainer.params, packed[0]["tokens"], None, seg)
            explicit = hidden(trainer.params, packed[0]["tokens"], None, seg, restarted)
        np.testing.assert_array_equal(np.asarray(default)[seg > 0], np.asarray(explicit)[seg > 0])


if __name__ == "__main__":
    absltest.main()
