# fp8 mixed precision — deferred

## Decision

fp8 mixed-precision training was **scratched for now** (removed 2026-07). The
scaffolding (the `omegalax/quant/` qwix package, the `_maybe_wrap_fp8` wiring at
`init_model_sharded`, the `fp8` / `fp8_recipe` config fields on all three model
families, the H100 fp8 MFU peak presets, and `tests/test_fp8_gating.py`) has been
excised from the `integration` trunk.

It is **recoverable**: the full implementation lives on the `feat/fp8` branch
(commit `feee5c4`, "feat: fp8 training via qwix (Hopper-gated; strict no-op on
A100/CPU)") and in git history, so this is a clean "scratch for now," not a
permanent loss.

## Why

- **Premature for the pre-multi-node stage.** The win is scale-gated: fp8 was a
  net *slowdown* at smoke scale (quantize/scale overhead not amortized), and the
  payoff only shows up when compute-bound at large scale.
- **It does not touch the compute-dominant MoE.** The grouped-MoE expert GEMMs
  run through the tokamax `ragged_dot` **Pallas** kernel, which is not
  `jax.lax.ragged_dot`, so qwix never intercepts it. There is **no fp8×fp8
  grouped-GEMM kernel in tokamax** — getting fp8 onto the expert GEMMs would be a
  bespoke kernel project, not a config flag.
- **The real near-term precision win is elsewhere.** Fast bf16 MoE comes from the
  tokamax **VJP grad-dtype fix** (decouple tokamax's backward grad-storage dtype
  from the operand dtype), which lets us drop the fp32 operand-upcast workaround
  (`_ragged_dot` in `omegalax/models/moe_grouped.py`, `lhs/rhs.astype(f32)`). That
  fp32-upcast workaround is **unrelated to qwix fp8** and stays until the tokamax
  PR merges.
- **Adds unused complexity** for no current benefit.

## Reference recipe when revisiting

Best-documented fp8 training recipe to date is **MAI-Thinking-1**
(`https://microsoft.ai/pdf/mai-thinking-1.pdf` §2.6.3):

- **FP8 E4M3 for fprop, E5M2 for dgrad, bf16 for wgrad**; FP32 gradient
  accumulation.
- **Per-tensor delayed scaling** with a **1024-step amax history**.
- **Stochastic rounding on gradient downcasts.**
- Keep **FP32** for the sensitive ops: attention pre-softmax scores, MoE router
  logits, output logits, MoE combine, the residual stream, embeddings, norms, and
  optimizer state.
- **RL reverts to bf16.**

Finer-grained alternative template: **DeepSeek-V3**'s fine-grained tile/block
scaling (1×128 / 128×128, applied across all three GEMMs) paired with **DeepGEMM**.

## Revisit criteria

Reintroduce fp8 when:

- We are **compute-bound at multi-node GH200 scale** and the throughput win is
  **measurable** (fp8 step time strictly below bf16, MFU computed against the fp8
  peak — watch the XLA amax/scale fusion-gap risk documented on `feat/fp8`).
- An **fp8×fp8 grouped-GEMM kernel** exists for the MoE experts: either a port of
  DeepGEMM via FFI, or a native Pallas/Mosaic kernel.

Tracking: **PDOOM-1069**.
