"""Test-session setup.

Points ``HF_HOME`` at the shared p-doom cache so tokenizer / processor loads in
CI and on a dev node resolve OFFLINE. Measured coverage of that cache: 12 of the
22 renderer conftest models and 9 of the 13 multimodal (tokenizer, processor)
pairs are already materialized there, including ``Qwen/Qwen3-VL-2B-Instruct``,
``Qwen/Qwen3-VL-4B-Instruct`` and ``Qwen/Qwen3-0.6B`` used by the collator tests.

Must run before ``transformers`` / ``huggingface_hub`` import, which read
``HF_HOME`` at module import time — pytest imports conftest before any test
module, so setting it here is early enough. ``setdefault`` so an explicit
``HF_HOME`` in the environment still wins.
"""

import os

os.environ.setdefault("HF_HOME", "/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface")
