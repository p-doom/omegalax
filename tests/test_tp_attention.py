"""Run the distributed TP attention check in a fresh process."""

import subprocess
import sys
from pathlib import Path

from absl.testing import absltest


class TpAttentionTest(absltest.TestCase):
    def test_tp_attention_in_fresh_process(self):
        worker = Path(__file__).with_name("tp_attention_worker.py")
        result = subprocess.run(
            [sys.executable, str(worker)],
            capture_output=True,
            check=False,
            text=True,
            timeout=600,
        )
        output = result.stdout + result.stderr
        if result.returncode == 77:
            self.skipTest(output.strip())
        self.assertEqual(result.returncode, 0, output)


if __name__ == "__main__":
    absltest.main()
