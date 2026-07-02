"""Unit tests for the GPU XLA perf-flags helper (CPU-only, no jax backend).

Covers: append/no-clobber of a pre-existing XLA_FLAGS, user-flag precedence,
idempotency, the opt-out env var and ``enable=False``, GPU-only (CPU no-op) gating,
and that the produced string contains the intended flags. These tests do not import
jax and do not create any XLA backend.
"""

import os
import unittest
from unittest import mock

from omegalax.distributed.xla_flags import (
    DISABLE_ENV_VAR,
    build_gpu_xla_flags,
    configure_gpu_xla_flags,
    _default_flags,
)

# The flags we always expect to install (spot-checked; full set is _default_flags()).
_CORE_FLAGS = (
    "--xla_gpu_enable_latency_hiding_scheduler",
    "--xla_gpu_all_reduce_combine_threshold_bytes",
    "--xla_gpu_reduce_scatter_combine_threshold_bytes",
    "--xla_gpu_all_gather_combine_threshold_bytes",
    "--xla_gpu_enable_pipelined_all_gather",
    "--xla_gpu_enable_pipelined_reduce_scatter",
    "--xla_gpu_enable_pipelined_all_reduce",
    "--xla_gpu_enable_while_loop_double_buffering",
)


def _flag_names(s: str) -> list[str]:
    return [tok.split("=", 1)[0] for tok in s.split() if tok.startswith("--")]


class BuildGpuXlaFlagsTest(unittest.TestCase):
    def test_contains_intended_flags(self):
        s = build_gpu_xla_flags(existing=None)
        for f in _CORE_FLAGS:
            self.assertIn(f, s, f"expected {f} in built flags")
        # Every default flag name should appear exactly once.
        names = _flag_names(s)
        for name in _default_flags():
            self.assertEqual(names.count(name), 1, f"{name} should appear once")

    def test_no_duplicate_flag_names_from_defaults(self):
        names = _flag_names(build_gpu_xla_flags(existing=None))
        self.assertEqual(len(names), len(set(names)), "no duplicate default flags")

    def test_appends_and_preserves_unrelated_user_flags(self):
        existing = "--xla_dump_to=/tmp/hlo --xla_force_host_platform_device_count=4"
        s = build_gpu_xla_flags(existing=existing)
        # user's unrelated flags preserved verbatim
        self.assertIn("--xla_dump_to=/tmp/hlo", s)
        self.assertIn("--xla_force_host_platform_device_count=4", s)
        # our defaults still added
        self.assertIn("--xla_gpu_enable_latency_hiding_scheduler", s)

    def test_user_flag_takes_precedence_no_clobber(self):
        # User overrides one of our defaults with a different value.
        existing = "--xla_gpu_all_reduce_combine_threshold_bytes=999"
        s = build_gpu_xla_flags(existing=existing)
        # We must NOT add our own default for a key the user set.
        names = _flag_names(s)
        self.assertEqual(
            names.count("--xla_gpu_all_reduce_combine_threshold_bytes"),
            1,
            "user-set flag must not be duplicated by our default",
        )
        # The single occurrence must be the user's value.
        self.assertIn("--xla_gpu_all_reduce_combine_threshold_bytes=999", s)
        self.assertNotIn("--xla_gpu_all_reduce_combine_threshold_bytes=33554432", s)

    def test_user_flags_appended_last(self):
        # Even if we didn't detect it as a key overlap, XLA honors the last occurrence;
        # user string must come after ours.
        existing = "--xla_dump_to=/tmp/x"
        s = build_gpu_xla_flags(existing=existing)
        self.assertTrue(s.rstrip().endswith("--xla_dump_to=/tmp/x"))


class ConfigureGpuXlaFlagsTest(unittest.TestCase):
    def setUp(self):
        # Isolate os.environ mutations per test.
        self._patcher = mock.patch.dict(os.environ, {}, clear=False)
        self._patcher.start()
        os.environ.pop("XLA_FLAGS", None)
        os.environ.pop(DISABLE_ENV_VAR, None)
        os.environ.pop("JAX_PLATFORMS", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def tearDown(self):
        self._patcher.stop()

    def test_gpu_forced_sets_environ(self):
        out = configure_gpu_xla_flags(enable=True, force=True)
        self.assertIsNotNone(out)
        self.assertEqual(os.environ["XLA_FLAGS"], out)
        self.assertIn("--xla_gpu_enable_latency_hiding_scheduler", os.environ["XLA_FLAGS"])

    def test_idempotent(self):
        first = configure_gpu_xla_flags(enable=True, force=True)
        second = configure_gpu_xla_flags(enable=True, force=True)
        self.assertEqual(first, second)
        # Latency-hiding flag appears exactly once after two calls.
        self.assertEqual(
            os.environ["XLA_FLAGS"].count("--xla_gpu_enable_latency_hiding_scheduler"), 1
        )

    def test_no_clobber_existing_env(self):
        os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/pre"
        configure_gpu_xla_flags(enable=True, force=True)
        self.assertIn("--xla_dump_to=/tmp/pre", os.environ["XLA_FLAGS"])
        self.assertIn("--xla_gpu_enable_latency_hiding_scheduler", os.environ["XLA_FLAGS"])

    def test_opt_out_env_var(self):
        os.environ[DISABLE_ENV_VAR] = "1"
        out = configure_gpu_xla_flags(enable=True, force=True)
        self.assertIsNone(out)
        self.assertNotIn("XLA_FLAGS", os.environ)

    def test_opt_out_enable_false(self):
        out = configure_gpu_xla_flags(enable=False, force=True)
        self.assertIsNone(out)
        self.assertNotIn("XLA_FLAGS", os.environ)

    def test_cpu_platform_is_noop(self):
        os.environ["JAX_PLATFORMS"] = "cpu"
        out = configure_gpu_xla_flags(enable=True)  # force=False -> platform-gated
        self.assertIsNone(out)
        self.assertNotIn("XLA_FLAGS", os.environ)

    def test_gpu_platform_detected_via_jax_platforms(self):
        os.environ["JAX_PLATFORMS"] = "cuda"
        out = configure_gpu_xla_flags(enable=True)
        self.assertIsNotNone(out)

    def test_gpu_platform_detected_via_cuda_visible_devices(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        out = configure_gpu_xla_flags(enable=True)
        self.assertIsNotNone(out)

    def test_cuda_visible_devices_empty_is_noop(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        out = configure_gpu_xla_flags(enable=True)
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
