"""Run the Qwen3.5 multi-device checks in a fresh process."""

import os
import subprocess
import sys
from pathlib import Path

from absl.testing import absltest


class Qwen3_5MultideviceTest(absltest.TestCase):
    def test_qwen3_5_multidevice_in_fresh_process(self):
        worker = Path(__file__).with_name("qwen3_5_multidevice_worker.py")
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        env["OMEGALAX_DELTANET_KERNEL"] = "xla"
        env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
        root = str(worker.parents[1])
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, str(worker)],
            capture_output=True,
            check=False,
            env=env,
            text=True,
            timeout=600,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    absltest.main()
