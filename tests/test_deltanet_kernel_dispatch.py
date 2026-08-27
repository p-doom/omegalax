"""Backend selection in the gated-delta-rule dispatcher.

The implicit default probes ``jax.devices()`` to decide pallas-vs-xla. That
probe used to be wrapped in ``except Exception: pass``, so a failing probe on
a GPU run silently ran the XLA reference and the run's throughput was then
attributed to the accelerated backend. These tests pin that the probe raises,
and that the legitimate XLA paths (explicit request, and CPU-host default)
still resolve and still execute.
"""

from __future__ import annotations

import os
import types
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from omegalax.models.qwen3_5 import kernels
from omegalax.models.qwen3_5.kernels.xla_reference import chunk_gated_delta_rule_xla


def _fake_devices(*platforms: str) -> list[types.SimpleNamespace]:
    return [types.SimpleNamespace(platform=p) for p in platforms]


def _inputs(B=1, T=128, H=1, A=16, U=16):
    rng = np.random.RandomState(0)
    q = jnp.asarray(rng.randn(B, T, H, A).astype(np.float32) * 0.1)
    k = jnp.asarray(rng.randn(B, T, H, A).astype(np.float32) * 0.1)
    v = jnp.asarray(rng.randn(B, T, H, U).astype(np.float32) * 0.1)
    a = jnp.asarray(rng.randn(B, T, H).astype(np.float32) * 0.5)
    g = -jnp.exp(a) * jax.nn.softplus(a)
    beta = jax.nn.sigmoid(jnp.asarray(rng.randn(B, T, H).astype(np.float32) * 0.5))
    return q, k, v, g, beta


class DispatcherTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        # test_qwen3_5_deltanet_xla_hf_smoke.py sets this env var at import time,
        # so a shared pytest process leaks it in; every case here sets its own.
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("OMEGALAX_DELTANET_KERNEL", None)

    def test_probe_failure_raises_instead_of_substituting_xla(self):
        with (
            mock.patch.object(jax, "devices", side_effect=RuntimeError("cuda init failed")),
            self.assertRaisesRegex(RuntimeError, "cuda init failed"),
        ):
            kernels.resolve_backend()

    def test_no_preference_on_gpu_selects_pallas(self):
        with mock.patch.object(jax, "devices", return_value=_fake_devices("cuda", "cuda")):
            self.assertEqual(kernels.resolve_backend(), "pallas")

    def test_no_preference_on_cpu_selects_xla(self):
        with mock.patch.object(jax, "devices", return_value=_fake_devices("cpu")):
            self.assertEqual(kernels.resolve_backend(), "xla")

    def test_explicit_request_skips_the_device_probe(self):
        os.environ["OMEGALAX_DELTANET_KERNEL"] = "pallas"
        with mock.patch.object(jax, "devices", side_effect=AssertionError("probed")):
            self.assertEqual(kernels.resolve_backend(), "pallas")

    def test_unknown_backend_raises(self):
        os.environ["OMEGALAX_DELTANET_KERNEL"] = "flash"
        with self.assertRaisesRegex(ValueError, "flash"):
            kernels.resolve_backend()

    def test_explicit_xla_executes_the_reference(self):
        os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"
        args = _inputs()
        out = np.asarray(kernels.chunk_gated_delta_rule(*args))
        np.testing.assert_array_equal(out, np.asarray(chunk_gated_delta_rule_xla(*args)))

    def test_no_preference_on_cpu_executes_the_reference(self):
        args = _inputs()
        with mock.patch.object(jax, "devices", return_value=_fake_devices("cpu")):
            out = np.asarray(kernels.chunk_gated_delta_rule(*args))
        np.testing.assert_array_equal(out, np.asarray(chunk_gated_delta_rule_xla(*args)))


if __name__ == "__main__":
    absltest.main()
