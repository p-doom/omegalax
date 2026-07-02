# fp8 training — deferred Hopper (sm_90) numerics verification recipe

The CPU-developable parts (config plumbing, Hopper gating, qwix rules + wrap
injection, strict-no-op guarantees, wrapping-traces composition) are DONE and
CPU-verified (`tests/test_fp8_gating.py`). The actual fp8 **numerics** cannot be
validated on CPU/A100 — there are no fp8 tensor cores, so qwix falls back to
emulated math with no speedup. Everything below MUST run on a Hopper node
(H100/H200, `compute_capability >= 9.0`). Do NOT attempt on A100/CPU.

## 0. Preconditions on the Hopper node

```bash
cd <this worktree>            # omegalax-wt-fp8, branch feat/fp8
source ~/.bashrc              # sets UV_CACHE_DIR / LD_LIBRARY_PATH (cuDNN)
# Confirm the gate flips true on the real GPU:
uv run --no-sync python -c "from omegalax.quant.detect import is_hopper; print('is_hopper=', is_hopper())"
# Expect: is_hopper= True   (NO OMEGALAX_FORCE_FP8 — the real device reports sm_90)
```

Do NOT set `OMEGALAX_FORCE_FP8` on Hopper — the real device satisfies the gate.
The force flag is a CPU-only development escape hatch.

## 1. Per-matmul relative-error bounds (fp8 vs bf16)

Goal: confirm each quantized GEMM (q/k/v/o, gate/up/down, expert einsums +
grouped ragged_dot, lm_head) stays within an acceptable relative error vs the
bf16 reference on real inputs.

```python
# On the Hopper node, JAX_PLATFORMS unset (use GPU).
import jax, jax.numpy as jnp, numpy as np, dataclasses
from flax import nnx
from omegalax.distributed.mesh import make_mesh, mesh_rules
from omegalax.models.shard_config import axis_rules_for_mesh
from omegalax.models.qwen3.config import make_config
from omegalax.models.qwen3.model import Qwen3
from omegalax.models.sharding_runtime import init_model_sharded, set_attn_backend

cfg = make_config("Qwen/Qwen3-30B-A3B-Instruct-2507")   # or a smaller real id
mesh = make_mesh(tp_size=..., fsdp_size=..., dp_size=...)
rng = jax.random.key(0)

cfg_bf16 = dataclasses.replace(cfg, fp8=False)
cfg_fp8  = dataclasses.replace(cfg, fp8=True, fp8_recipe="e4m3_dynamic")
m_bf16 = init_model_sharded(Qwen3, cfg_bf16, rng, mesh, axis_rules_for_mesh(mesh))
m_fp8  = init_model_sharded(Qwen3, cfg_fp8,  rng, mesh, axis_rules_for_mesh(mesh))
# NB: init the fp8 model from the SAME weights as bf16 (copy state) so the only
# difference is quantization, not init RNG.
set_attn_backend(m_bf16, "mosaic_gpu"); set_attn_backend(m_fp8, "mosaic_gpu")

# Forward both on identical inputs; compare logits and, ideally, per-layer
# activations (add nnx.Intermediate taps or compare hidden states):
#   rel_err = ||fp8 - bf16|| / ||bf16||
# Accept guideline: end-to-end logits rel-err ~< 2e-2 for e4m3 dynamic; larger
# per-matmul errors are fine as long as they don't compound into the loss.
```

Suggested bound (tune to the model): per-tensor dynamic e4m3 typically gives
1–2% relative error per GEMM; the softmax/router/norms are excluded so routing
decisions are unchanged. If a specific matmul blows up, exclude it via an
extra `QtRule(module_path=..., )` (no `weight_qtype`) in `rules.py`.

## 2. Loss-curve parity over a short run

Run the real trainer for a few hundred steps with fp8 on vs off (same seed,
same data order) and overlay the loss curves.

```bash
# fp8 OFF baseline
uv run --no-sync python -m omegalax.trainers.text  <args...>   # fp8=False (default)
# fp8 ON  (set fp8=true in the config / config source, recipe e4m3_dynamic)
uv run --no-sync python -m omegalax.trainers.text  <args... with fp8 enabled>
```

Accept: fp8 loss tracks bf16 within noise (no divergence, no NaN). The
grouped-MoE path already forces fp32 accumulation (`OMEGALAX_MOE_F32_ACC=1`,
default) — fp8 quantizes the ragged_dot *inputs* and the accumulation stays
fp32, so the two fixes compose. Watch the first ~50 steps for early NaN
(the classic fp8 failure mode); if seen, try `blockwise_128` or widen the
excluded set.

## 3. MFU vs fp8 peak — and the XLA fusion-gap risk (⚠ top perf risk)

fp8 quantization does NOT guarantee a speedup: XLA may fail to fuse the
amax/scale computation into the fp8 matmul, leaving the fp8 cast + separate
bf16 matmul (SLOWER than plain bf16). This is the known XLA gap
(openxla/xla#22313). Verify in the HLO before trusting any MFU number.

```bash
# Dump + inspect the compiled HLO for fp8 matmuls actually lowering to the
# fp8 cublasLt / __cublas$lt custom-calls with fused scaling:
XLA_FLAGS="--xla_dump_to=/tmp/fp8_hlo --xla_dump_hlo_as_text" \
  uv run --no-sync python -m omegalax.trainers.text <args... fp8 on, 3 steps>
grep -riE "f8e4m3|f8e5m2|cublas.*lt|custom-call" /tmp/fp8_hlo/*after_optimizations* | head
# Confirm: fp8 operands feed a single fused matmul custom-call, NOT a
# quantize+dequantize sandwich around a bf16 dot.
```

Then measure MFU with the **fp8** peak preset so the denominator is right:

```python
from omegalax.trainers.perf import resolve_peak_tflops
peak = resolve_peak_tflops("h100_sxm_fp8")   # 1979.0 TFLOPS (vs 989 bf16)
# feed `peak` as peak_tflops into maybe_log_step_metrics / step_metrics.
```

Accept: fp8 step time strictly < bf16 step time AND `train/mfu` computed against
`h100_sxm_fp8`. If step time is >= bf16, the fusion gap bit you — the fp8 is
correct but not accelerated; do NOT ship the fp8 flag as a perf win until the
HLO shows the fused fp8 custom-call.

## 4. blockwise_128 (397B flagship only)

`fp8_recipe="blockwise_128"` adds 1x128 / 128x128 subchannel tiling
(`tile_size=128`). This needs contraction axes >= 128 (true for the flagship;
NOT for smoke configs — hence CPU only tests rule construction). Repeat steps
1–3 with `blockwise_128` on the 397B config; blockwise is more accurate than
per-tensor at similar cost and is the recommended recipe at that scale.

## Notes / honest caveats carried from CPU dev
- The qwix `e4m3_dynamic` recipe is fully DYNAMIC: it creates NO persistent
  `quant_stats` state and adds NO trainable params (CPU-verified). So the
  optimizer/`wrt=nnx.Param` set is provably untouched. A future static-scale
  recipe (`act_static_scale=True`) WOULD populate the `quant_stats` collection
  — that collection is still not an `nnx.Param`, but re-verify the optimizer
  filter if you add one.
- VLM (`Qwen3_5ForConditionalGeneration`, `Qwen3VL`) fp8 is NOT wired: the
  `maybe_quantize_fp8` dummy trace uses the text `__call__` signature. The VLM
  configs carry the fp8 fields, but wrapping the VLM forward (image inputs)
  is a follow-up. `fp8_active` returns False for the VLM wrapper unless
  `text_config.fp8` is set; even then the text-only dummy trace would need the
  VLM signature. Extend `omegalax/quant/apply.py:_dummy_text_inputs` for VLM.
- tokamax's own `ragged_dot` (the grouped-MoE default `primitive="tokamax"`)
  is a Pallas kernel and is NOT `jax.lax.ragged_dot`, so qwix does not
  intercept it. To get fp8 on the grouped MoE expert GEMMs on Hopper, run the
  grouped path with `primitive="jax"` (which qwix's `ragged_dot` rule
  intercepts) OR quantize inside the tokamax kernel — decide on Hopper after
  measuring. The dense-path expert einsums ARE intercepted today.
```
