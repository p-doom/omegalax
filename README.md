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
- HuggingFace safetensor loaders for all architectures: `create_qwen3_from_safetensors`, `create_qwen3_5_from_safetensors`, and `create_qwen3_vl_from_safetensors`.
- Supported models:
  - Qwen3 dense: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3-32B`.
  - Qwen3 MoE: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
  - Qwen3.5: `Qwen/Qwen3.5-397B-A17B`.
  - Qwen3-VL: `Qwen/Qwen3-VL-2B-Instruct`.

Tiny CPU-sized aliases exist for every architecture (`qwen3-smoke`,
`qwen3-smoke-moe`, `qwen3.5-smoke`, `qwen3.5-smoke-dense`, `qwen3-vl-smoke`,
`qwen3-vl-smoke-moe`). They have no HF repo behind them, so anything needing a
tokenizer or image processor must be given one explicitly (`--tokenizer` /
`--processor`).

## Tensor naming convention
All tensor variables use [Shazeer's shape-suffix notation](https://medium.com/@noamshazeer/shape-suffixes-good-coding-style-f836e72e24fd).
The full dimension key lives in the `omegalax.models` package docstring (`omegalax/models/__init__.py`).

## Install
Use Python 3.11+ with a JAX build that matches your accelerator (e.g., `jax[cuda12]` for CUDA 12):
```bash
uv sync
```

## Two things that bite before anything runs
Every script here uses `absl.flags`, so flag names are spelled with
**underscores**. `--model-id` is a hard `UnrecognizedFlagError`, not an alias.

The trainers and the exporter call `jax.distributed.initialize()`, so they run
under `srun`, not on a login node. They also take no defaults: every flag in the
script's own `_REQUIRED` list must be passed, and `_validate_flags()` lists every
missing one at startup rather than falling back.

## Quickstart (language-only)
Create a Qwen3 text model and run a forward+decode step. `api.forward` returns
hidden states before `lm_head`, not logits:
```python
import jax
import jax.numpy as jnp
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api

rng = jax.random.key(0)
model, cfg = api.init_model("qwen3-smoke", rng, tp_size=1, fsdp_size=1, dp_size=1)
set_attn_backend(model, text_backend="xla")  # the "mosaic_gpu" default has no CPU lowering
tokens = jax.random.randint(rng, (2, 32), 0, cfg.vocab_size, jnp.int32)
hidden, aux_loss = api.forward(model, tokens, pad_id=0, cfg=cfg)
cache = api.make_cache(cfg, batch_size=2, token_len=32, generate_steps=8)
next_hidden, cache, aux_loss = api.decode(model, cache, tokens, pad_id=0, cfg=cfg)
```
`init_model` needs all three of `tp_size`, `fsdp_size` and `dp_size`, or there is
no mesh to place the parameters on.

## Quickstart (vision-language)
Initialize a VLM (Qwen3.5 or Qwen3-VL) and run a multimodal forward pass.
`pixel_values` is `B, C, T, H, W`, where `H`/`W` are the config's `patch_size`
(16) and `T` its `temporal_patch_size` (2):
```python
import jax
import jax.numpy as jnp
from omegalax import vlm

rng = jax.random.key(0)
model, cfg = vlm.api.init_model("qwen3.5-smoke", rng, tp_size=1, fsdp_size=1, dp_size=1)
tokens = jnp.ones((1, 16), dtype=jnp.int32)
pixel_values = jnp.zeros((1, 3, 2, 16, 16), dtype=jnp.float32)
image_grid_thw = jnp.array([[1, 1, 1]], dtype=jnp.int32)
hidden, aux_loss = vlm.api.forward(
    model, tokens, pad_id=0, cfg=cfg, pixel_values=pixel_values, image_grid_thw=image_grid_thw
)
```
This one needs a GPU: `set_attn_backend` only reaches the text decoder, and the
vision tower's cuDNN attention has no CPU lowering.

## Training
Three steps: a raw JSONL of sessions, binned into ArrayRecord shards whose
records ARE the training examples, then trained from.

Example raw JSONL row:
```json
{"session_id":"demo-0","messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi there"}]}
```

Build the records:
```bash
uv run scripts/build_sft_records_from_chat.py \
  --data_path /path/to/chat.jsonl \
  --out_dir /path/to/train_records \
  --model_id qwen3-smoke \
  --tokenizer Qwen/Qwen3-0.6B \
  --max_length 512 \
  --overwrite
```
Pass `--processor` (and optionally `--preprocessor_config`) when the dataset
contains images. `--message_lengths_path` caches per-message token lengths, so
re-binning at a different `--max_length` or `--overflow_mode` never re-tokenizes.

Run text SFT — the full required flag set, nothing optional:
```bash
uv run scripts/train_text_sft.py \
  --model_id qwen3-smoke --tokenizer Qwen/Qwen3-0.6B \
  --data_path /path/to/train_records --max_length 512 \
  --num_steps 2 --batch_size 2 --learning_rate 1e-4 --weight_decay 0.0 \
  --warmup_steps 1 --lr_schedule wsd --lr_stable_fraction 0.9 --lr_end_factor 0.0 \
  --max_grad_norm 1.0 --grad_accum_steps 1 --gc_period 0 --seed 0 \
  --tp_size 1 --fsdp_size 1 --dp_size 1 \
  --save_dir /path/to/ckpt --jax_cache_dir /path/to/jaxcache \
  --save_every 100 --log_every 1 --resume never --pad_id 0 --peak_tflops 1 \
  --grain_read_threads 1 --grain_read_buffer_size 1 \
  --grain_workers 0 --grain_worker_buffer_size 1 --text_attn_backend xla
```
`--resume` is an enum, not a boolean: `never`, `if_present` (the right mode for
Slurm time-limit resubmits) or `required`. Checkpoints persist the Grain iterator
state alongside the weights.

`scripts/train_vlm_sft.py` takes the same shape plus `--processor`, the vision
budgets (`--max_vision_patches_per_sample`, `--max_vision_images_per_sample`),
`--num_loss_tiles`, `--keep_period`, `--keep_latest`, `--log_memory`,
`--enable_lora` and `--freeze_vision_tower`. `--enable_lora` and
`--freeze_vision_tower` are mutually exclusive; `--enable_lora` additionally
requires `--lora_rank` and `--lora_alpha`.

Export any supported model (Qwen3 dense/MoE, Qwen3.5, Qwen3-VL) to HuggingFace safetensors:
```bash
uv run scripts/export_to_hf.py --model_id qwen3-smoke \
  --out_dir /tmp/qwen3-smoke-export --tp_size 1 --fsdp_size 1 --dp_size 1
```

## Loading HuggingFace checkpoints
All loaders expect a directory containing safetensors and `config.json`:
```python
from huggingface_hub import snapshot_download
from omegalax.models.qwen3.params import create_qwen3_from_safetensors

ckpt_dir = snapshot_download("Qwen/Qwen3-8B")
model = create_qwen3_from_safetensors(ckpt_dir, "Qwen/Qwen3-8B", tp_size=1, fsdp_size=1, dp_size=1)
```
For Qwen3.5 and Qwen3-VL, use `create_qwen3_5_from_safetensors(...)` or
`create_qwen3_vl_from_safetensors(...)` with the same three mesh sizes. When
starting from a raw HF config, `omegalax.models.qwen3_vl.make_vl_config_from_hf()`
builds a matching JAX config.

## Tests
Tests are `absltest` cases, run under pytest. tokamax reads its own flags off
`sys.argv`, so **any dash-flag on the command line** makes every test that
reaches tokamax attention die with `UnrecognizedFlagError`. Put pytest options in
`PYTEST_ADDOPTS`, never in argv:

```bash
PYTEST_ADDOPTS="-q -p no:cacheprovider" JAX_PLATFORMS=cpu \
  uv run --extra=torch-tests -- python -m pytest tests
```

`JAX_PLATFORMS=cpu` keeps a GPU node from being claimed by a suite that does not
need one. Add `OMEGALAX_RUN_REAL_WEIGHTS_TESTS=1` for the real-weight parity
suites (downloads checkpoints; slow), and name a file to run one suite:
`... -m pytest tests/test_qwen3_0_6b.py`.

The `tests/test_*_smoke.py` logits-vs-HF cases need a GPU despite the tiny
models — 6 of 26 fail on numerical agreement with HF when run on CPU.
