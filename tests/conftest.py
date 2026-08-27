"""Test-session environment."""

import os

from absl import flags

os.environ.setdefault("HF_HOME", "/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface")
flags.FLAGS(["pytest"], known_only=True)
