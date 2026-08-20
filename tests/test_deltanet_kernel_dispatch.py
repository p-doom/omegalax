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

from omegalax.models.qwen3 import registry as qwen3_registry
from omegalax.models.qwen3_5 import kernels
from omegalax.models.qwen3_5.config import make_config as make_qwen3_5_config
from omegalax.models.qwen3_5.kernels.xla_reference import chunk_gated_delta_rule_xla
from omegalax.trainers.perf import record_deltanet_kernel


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
        with mock.patch.object(jax, "devices", side_effect=RuntimeError("cuda init failed")):
            with self.assertRaisesRegex(RuntimeError, "cuda init failed"):
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
            kernels.chunk_gated_delta_rule(*_inputs())

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


class KernelRecordTest(absltest.TestCase):
    """``record_deltanet_kernel`` is what ties a run's reported MFU to a named kernel."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

    def test_records_the_resolved_kernel_for_a_text_run(self):
        cfg = make_qwen3_5_config("qwen3.5-smoke").text_config
        wandb_run = mock.Mock()
        self.assertEqual(record_deltanet_kernel(cfg, wandb_run), "xla")
        wandb_run.config.update.assert_called_once_with(
            {"deltanet_kernel": "xla"}, allow_val_change=True
        )

    def test_records_the_resolved_kernel_for_a_vlm_run(self):
        cfg = make_qwen3_5_config("qwen3.5-smoke")
        wandb_run = mock.Mock()
        self.assertEqual(record_deltanet_kernel(cfg, wandb_run), "xla")
        wandb_run.config.update.assert_called_once_with(
            {"deltanet_kernel": "xla"}, allow_val_change=True
        )

    def test_resolves_without_wandb(self):
        # wandb is opt-in (--wandb_project), so no-wandb is the common path.
        cfg = make_qwen3_5_config("qwen3.5-smoke").text_config
        self.assertEqual(record_deltanet_kernel(cfg, None), "xla")

    def test_records_nothing_for_an_architecture_without_deltanet(self):
        cfg = qwen3_registry.build_config("Qwen/Qwen3-0.6B")
        wandb_run = mock.Mock()
        self.assertIsNone(record_deltanet_kernel(cfg, wandb_run))
        wandb_run.config.update.assert_not_called()

    def test_a_failing_probe_fails_the_run_before_training(self):
        os.environ.pop("OMEGALAX_DELTANET_KERNEL")
        cfg = make_qwen3_5_config("qwen3.5-smoke").text_config
        with mock.patch.object(jax, "devices", side_effect=RuntimeError("cuda init failed")):
            with self.assertRaisesRegex(RuntimeError, "cuda init failed"):
                record_deltanet_kernel(cfg, mock.Mock())


if __name__ == "__main__":
    absltest.main()
