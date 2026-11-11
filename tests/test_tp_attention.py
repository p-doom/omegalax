"""Run the distributed TP attention check before JAX can initialize in pytest."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_tp_attention_in_fresh_process():
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
        pytest.skip(output.strip())
    assert result.returncode == 0, output
