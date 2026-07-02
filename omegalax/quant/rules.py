"""qwix quantization rules for fp8 training.

Builds a :class:`qwix.QtProvider` of :class:`qwix.QtRule` s keyed by module
path (a regex matched with ``re.fullmatch`` against the ``/``-joined NNX module
path). Rules are matched in order; the FIRST matching rule wins, and only a
rule whose ``weight_qtype`` is set actually quantizes -- a matching rule with
``weight_qtype=None`` is an explicit *exclusion* (runs the original bf16 op).

What we quantize (the compute-bound GEMMs):
  * attention q/k/v/o projections (``nnx.Linear`` -> ``jax.lax.dot_general``)
  * dense MLP gate/up/down (``nnx.Linear`` -> ``jax.lax.dot_general``)
  * MoE expert projections: the dense-path einsums ``BTD,EDF->BTEF`` etc.
    (``jax.numpy.einsum``) and the grouped path (``jax.lax.ragged_dot``)
  * lm_head (``nnx.Linear`` -> ``jax.lax.dot_general``)

What we DO NOT touch (correctness / stability sensitive):
  * RMSNorm / layernorms (no GEMM to intercept anyway)
  * the MoE ``router`` and Qwen3.5 ``shared_expert_gate`` (tiny, precision
    sensitive, cheap -- explicitly excluded by a leading no-quant rule)
  * the softmax / top-k routing math (not a GEMM)
  * the tokamax attention core (a Pallas kernel; qwix disables interception
    inside ``pallas_call`` and tokamax's own ``ragged_dot`` is NOT
    ``jax.lax.ragged_dot`` so it is never intercepted)
  * LoRA adapters ``lora_A`` / ``lora_B`` and the ``LoRALinear`` low-rank
    delta matmul (explicitly excluded by a leading no-quant rule so only the
    BASE projection GEMM is quantized)
  * the optimizer / optimizer state (qwix scales live in the ``quant_stats``
    Flax collection, never as ``nnx.Param``, so ``wrt=nnx.Param`` grads and
    ``MixedPrecisionOptimizer`` never see them)

Phase-A recipe (``e4m3_dynamic``): per-tensor DYNAMIC quantization, e4m3 for
the forward operands and e5m2 for the backward gradient (via ``bwd_qtype``),
which is the standard fp8 training split. ``blockwise_128`` adds DeepSeek-style
1x128 / 128x128 subchannel tiling (``tile_size=128``) for the 397B flagship;
otherwise identical.

fp8 composes with fp32 accumulation: qwix quantizes the GEMM *inputs* to fp8
and the matmul still accumulates in a wider type, so this stacks with the
grouped-MoE fp32-accumulation NaN fix in ``moe_grouped.py`` (fp8 lhs/rhs, fp32
accumulate).
"""

from __future__ import annotations

import re
from typing import Sequence

import jax.numpy as jnp
import qwix

# Recipe names accepted by the config ``fp8_recipe`` field.
RECIPE_OFF = "off"
RECIPE_E4M3_DYNAMIC = "e4m3_dynamic"
RECIPE_BLOCKWISE_128 = "blockwise_128"

SUPPORTED_RECIPES: tuple[str, ...] = (
    RECIPE_OFF,
    RECIPE_E4M3_DYNAMIC,
    RECIPE_BLOCKWISE_128,
)

# fp8 element types. e4m3 (more mantissa) for forward operands; e5m2 (more
# range) for backward gradients -- the canonical fp8 training split.
_FWD_QTYPE = jnp.float8_e4m3fn
_BWD_QTYPE = jnp.float8_e5m2

# Base projection attribute names to quantize. These are matched at the end of
# the module path. The attention/MLP/lm_head names cover every ``nnx.Linear``
# GEMM we want in fp8; ``router`` / ``shared_expert_gate`` are deliberately
# absent (excluded below).
_PROJECTION_LEAVES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
)

# Module paths that must NEVER be quantized (matched FIRST, weight_qtype=None).
#   * router / shared_expert_gate: precision-sensitive routing GEMMs.
#   * lora_A / lora_B and the LoRALinear delta scope: low-rank adapter matmuls.
#     A LoRA-wrapped ``q_proj`` becomes ``.../q_proj`` (a LoRALinear whose
#     __call__ runs the delta ``jnp.matmul`` -> dot_general at that path) with
#     the frozen base at ``.../q_proj/base``. Excluding the ``.../<leaf>``
#     (non-``/base``) delta scope while still quantizing ``.../<leaf>/base``
#     keeps fp8 on the base weight only. When LoRA is not injected the plain
#     ``nnx.Linear`` lives directly at ``.../<leaf>`` and is quantized.
_EXCLUDE_PATTERNS: tuple[str, ...] = (
    r".*router",
    r".*shared_expert_gate",
    r".*lora_A",
    r".*lora_B",
    r".*_lora",  # LoRAMoEExperts adapter slots (gate_proj_lora, ...)
)


def lora_delta_paths(model) -> tuple[str, ...]:
    """Return the ``/``-joined module paths of every ``LoRALinear`` in ``model``.

    Used to exclude the low-rank delta matmuls from fp8 quantization while
    still quantizing the frozen base at ``.../<leaf>/base`` (see
    :func:`build_provider`). Returns ``()`` when the model has no LoRA adapters
    (import of the LoRA module is lazy so non-fp8 / non-LoRA paths are untouched).
    """
    from flax import nnx

    try:
        from omegalax.trainers.lora import LoRALinear
    except Exception:
        return ()
    paths: list[str] = []
    for path, module in nnx.iter_modules(model):
        if isinstance(module, LoRALinear):
            paths.append("/".join(map(str, path)))
    return tuple(paths)


def _quant_rule(module_path: str, recipe: str) -> qwix.QtRule:
    """A fp8 quantization QtRule for ``module_path`` under ``recipe``."""
    tile_size = 128 if recipe == RECIPE_BLOCKWISE_128 else None
    return qwix.QtRule(
        module_path=module_path,
        weight_qtype=_FWD_QTYPE,
        act_qtype=_FWD_QTYPE,
        bwd_qtype=_BWD_QTYPE,
        tile_size=tile_size,
    )


def _exclude_rule(module_path: str) -> qwix.QtRule:
    """A no-quant QtRule (weight_qtype=None) that excludes ``module_path``."""
    return qwix.QtRule(module_path=module_path)


def build_provider(
    recipe: str = RECIPE_E4M3_DYNAMIC,
    *,
    lora_delta_paths: Sequence[str] = (),
) -> qwix.QtProvider:
    """Build the fp8 :class:`qwix.QtProvider` for the given recipe.

    Rules are ordered: exclusions first (router, LoRA adapters, and any
    ``lora_delta_paths``), then the quantized base projections (attention, MLP,
    expert einsums/ragged_dot, lm_head). First match wins, so the exclusions
    shadow the projection rules for LoRA/router paths.

    ``lora_delta_paths`` are the exact ``/``-joined module paths of any
    ``LoRALinear`` wrappers in the model (e.g. ``"layers/0/attn/q_proj"``).
    When LoRA is injected, a wrapped ``q_proj`` becomes a ``LoRALinear`` at
    ``.../q_proj`` whose ``__call__`` runs the low-rank delta matmuls at THAT
    path, with the frozen base linear at ``.../q_proj/base``. Passing the
    LoRALinear paths here adds an exact-path exclusion for each so the delta
    matmuls stay in bf16 while the base GEMM (``.../q_proj/base``, which does
    NOT fullmatch the delta path) is still quantized. When LoRA is absent this
    is empty and the plain ``nnx.Linear`` at ``.../q_proj`` is quantized.
    """
    if recipe not in SUPPORTED_RECIPES:
        raise ValueError(f"Unsupported fp8_recipe {recipe!r}. Supported: {SUPPORTED_RECIPES}.")
    if recipe == RECIPE_OFF:
        raise ValueError("build_provider called with recipe 'off'; gate on fp8_active first.")

    rules: list[qwix.QtRule] = [_exclude_rule(p) for p in _EXCLUDE_PATTERNS]
    # Exclude each LoRALinear delta scope by its exact path (escaped so regex
    # metacharacters in a path can't misfire). ``.../q_proj/base`` is a longer
    # path and does not fullmatch ``.../q_proj``, so the base stays quantized.
    rules.extend(_exclude_rule(re.escape(p)) for p in lora_delta_paths)

    # Quantize each base projection leaf, both as a plain nnx.Linear at
    # ``.../<leaf>`` and (when LoRA-wrapped) the frozen base at
    # ``.../<leaf>/base``. The dense-path MoE expert einsums live on the
    # MoEFeedForward module itself, whose gate_proj/up_proj/down_proj are
    # rank-3 Params (not submodules), so the einsum op is attributed to the
    # ``.../mlp`` module path -- we match those via the expert-einsum rule
    # below (keyed on the ``einsum``/``ragged_dot`` op at the mlp path).
    leaf_alt = "|".join(_PROJECTION_LEAVES)
    # ``.../q_proj`` (plain Linear) and ``.../q_proj/base`` (LoRA base).
    rules.append(_quant_rule(rf".*({leaf_alt})", recipe))
    rules.append(_quant_rule(rf".*({leaf_alt})/base", recipe))

    # MoE expert compute lives directly on the feed-forward module (the
    # gate/up/down are stacked Params applied via jnp.einsum / jax.lax.ragged_dot
    # inside MoEFeedForward.__call__ or moe_grouped). Quantize the einsum and
    # ragged_dot ops on any ``.../mlp`` path. The router (also on ``.../mlp``)
    # is a dot_general already excluded above, and softmax/top-k are not GEMMs.
    rules.append(
        qwix.QtRule(
            module_path=r".*mlp",
            op_names=("einsum", "ragged_dot"),
            weight_qtype=_FWD_QTYPE,
            act_qtype=_FWD_QTYPE,
            bwd_qtype=_BWD_QTYPE,
            tile_size=128 if recipe == RECIPE_BLOCKWISE_128 else None,
        )
    )

    return qwix.QtProvider(rules)
