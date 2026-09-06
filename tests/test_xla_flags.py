"""Unit test for the GPU XLA perf-flags helper (CPU-only, no jax backend)."""

import os
import unittest
from unittest import mock

from omegalax.distributed.xla_flags import configure_gpu_xla_flags


class ConfigureGpuXlaFlagsTest(unittest.TestCase):
    def test_sets_environ_and_preserves_user_flags(self):
        with mock.patch.dict(os.environ, {"XLA_FLAGS": "--xla_dump_to=/tmp/pre"}, clear=False):
            configure_gpu_xla_flags()
            self.assertIn("--xla_gpu_enable_latency_hiding_scheduler=true", os.environ["XLA_FLAGS"])
            self.assertTrue(os.environ["XLA_FLAGS"].rstrip().endswith("--xla_dump_to=/tmp/pre"))


if __name__ == "__main__":
    unittest.main()
