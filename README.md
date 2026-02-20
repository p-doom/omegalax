<div align="center">
  <img src="https://github.com/p-doom/crowd-code/blob/main/img/pdoom-logo.png?raw=true" width="60%" alt="p(doom)" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.pdoom.org/"><img alt="Homepage"
    src="https://img.shields.io/badge/Homepage-p%28doom%29-white?logo=home&logoColor=black"/></a>
  <a href="https://huggingface.co/p-doom"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-p--doom-ffc107?color=ffc107&logoColor=white"/></a>
  <br>
  <a href="https://discord.gg/G4JNuPX2VR"><img alt="Discord"
    src="https://img.shields.io/badge/Discord-p%28doom%29-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="https://github.com/p-doom"><img alt="GitHub"
    src="https://img.shields.io/badge/GitHub-p--doom-24292e?logo=github&logoColor=white"/></a>
  <a href="https://twitter.com/prob_doom"><img alt="Twitter Follow"
    src="https://img.shields.io/badge/Twitter-prob__doom-white?logo=x&logoColor=white"/></a>
  <br>
  <a href="LICENSE.md" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <br>
</div>

# `omegalax`:  A JAX-based training codebase for LLMs/VLMs.

## Overview
- Qwen3 dense and MoE (`omegalax/models/qwen3`) with cache-aware decode in `omegalax/text/api.py`.
- Qwen3.5 MoE and Qwen3-VL (`omegalax/models/qwen3_5`, `omegalax/models/qwen3_vl`, `omegalax/vlm/api.py`).
- HuggingFace safetensor loaders for all architectures: `create_qwen3_from_safe_tensors`, `create_qwen3_5_from_safe_tensors`, and `create_qwen3_vl_from_safe_tensors`.
- Supported models:
  - Qwen3 dense: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3-32B`.
  - Qwen3 MoE: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
  - Qwen3.5: `Qwen/Qwen3.5-397B-A17B`.
  - Qwen3-VL: `Qwen/Qwen3-VL-2B-Instruct`.

## Install
Use Python 3.11+ with a JAX build that matches your accelerator (e.g., `jax[cuda12]` for CUDA 12):
```bash
uv sync
```

## Quickstart (language-only)
Create a Qwen3 text model and run a forward+decode step:
```python
import jax
import jax.numpy as jnp
from omegalax.text import api

rng = jax.random.key(0)
model, cfg = api.init_model("Qwen/Qwen3-0.6B", rng)
tokens = jax.random.randint(rng, (2, 32), 0, cfg.vocab_size, jnp.int32)
logits, aux_loss = api.forward(model, tokens, pad_id=0, cfg=cfg)
cache = api.make_cache(cfg, batch_size=2, token_len=32, generate_steps=8)
next_logits, cache, aux_loss = api.decode(model, cache, tokens, pad_id=0, cfg=cfg)
```

Run the synthetic training loop:
```bash
uv run scripts/train_smoke.py --model-id Qwen/Qwen3-0.6B --num-steps 5
```

## Quickstart (vision-language)
Initialize a VLM (Qwen3.5 or Qwen3-VL) and run a multimodal forward pass:
```python
import jax
import jax.numpy as jnp
from omegalax import vlm

rng = jax.random.key(0)
model, cfg = vlm.api.init_model("qwen3.5-smoke", rng)
tokens = jnp.ones((1, 16), dtype=jnp.int32)
pixel_values = jnp.zeros((1, 3, 2, 14, 14), dtype=jnp.float32)  # B, C, T, H, W
image_grid_thw = jnp.array([[1, 1, 1]], dtype=jnp.int32)
logits, aux_loss = vlm.api.forward(
    model, tokens, pad_id=0, cfg=cfg, pixel_values=pixel_values, image_grid_thw=image_grid_thw
)
```

## Loading HuggingFace checkpoints
All loaders expect a directory containing safetensors and `config.json`:
```python
from huggingface_hub import snapshot_download
from omegalax.models.qwen3.params import create_qwen3_from_safe_tensors

ckpt_dir = snapshot_download("Qwen/Qwen3-8B")
model = create_qwen3_from_safe_tensors(ckpt_dir, "Qwen/Qwen3-8B")
```
For Qwen3.5 and Qwen3-VL, use `create_qwen3_5_from_safe_tensors` or `create_qwen3_vl_from_safe_tensors` respectively. When starting from a raw HF config, `omegalax.models.qwen3_vl.make_vl_config_from_hf()` will build a matching JAX config.

## Tests
Tests use `absltest`:
- Smoke/tiny-model checks (CPU-friendly, no HF downloads):
```bash
uv run --extra=torch-tests -- python -m unittest discover -s tests -p "test_*smoke.py"
```
- Registry/config + module-level:
```bash
uv run --extra=torch-tests -- python -m unittest discover -s tests -p "test_qwen3_*modules.py"
uv run --extra=torch-tests -- python -m unittest discover -s tests -p "test_qwen3_configs.py"
```
- HuggingFace parity (downloads checkpoints; slow):
```bash
uv run --extra=torch-tests -- python -m unittest discover -s tests -p "test_qwen3_*"
```
Run a single suite via `uv run --extra=torch-tests -- python -m unittest tests.test_qwen3_0_6b`.
