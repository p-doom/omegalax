"""Apply fp8 quantization to an NNX model via qwix (Hopper-gated no-op).

``maybe_quantize_fp8`` is the single entry point called from the model-build
path (``init_model`` / ``init_model_sharded``). It:

  1. Checks :func:`omegalax.quant.detect.fp8_active` -- returns the model
     UNCHANGED when fp8 is off, the recipe is ``off``, or the host is not
     Hopper (A100/CPU). This is the strict no-op guarantee.
  2. Otherwise builds the qwix ``QtProvider`` for the recipe and calls
     ``qwix.quantize_model(model, provider, *dummy_inputs)`` UNDER THE MESH so
     the traced ``__call__`` (which qwix runs once to convert weights and
     create the ``quant_stats`` collection) composes with the pervasive
     ``out_sharding=`` usage.

Dummy inputs are small abstract token batches -- ``quantize_model`` needs to
trace the model once but the actual values do not matter (qwix docstring).
"""

from __future__ import annotations

import inspect
from typing import Any

import jax
import jax.numpy as jnp
import qwix
from flax import nnx

from omegalax.quant import detect, rules as rules_mod

# Tiny (batch, seq) for the dummy trace. Values are irrelevant (qwix docstring);
# quantize_model runs the forward once only to convert weights / create scales.
_DUMMY_B, _DUMMY_T = 1, 8


def _dummy_inputs_for(model: nnx.Module):
    """Positional dummy inputs matching ``model.__call__`` (text-decoder path).

    Handles the two positional signatures used by the causal-LM entry points:

      * text / qwen3.5-VLM:
        ``__call__(token_ids_BT, segment_ids_BT, cache, num_right_pads, ...)``
        -- ``cache=None`` selects the training forward (the path we quantize);
        for the VLM the trailing image args default to ``None`` so the
        text-only branch is traced.
      * qwen3-VL (``Qwen3VL``):
        ``__call__(token_ids_BT, attention_mask_BT, position_ids_ZBT=None,
        pixel_values=None, ...)`` -- two required leading tensors, image args
        default to ``None``.

    Dispatches on the 2nd positional parameter name (``segment_ids`` vs
    ``attention_mask``) so both compose without a hardcoded per-class table.
    """
    token_ids = jnp.ones((_DUMMY_B, _DUMMY_T), dtype=jnp.int32)
    mask_like = jnp.ones((_DUMMY_B, _DUMMY_T), dtype=jnp.int32)

    params = list(inspect.signature(type(model).__call__).parameters.values())[1:]  # drop self
    names = [p.name for p in params]
    second = names[1] if len(names) > 1 else ""

    if second.startswith("attention_mask"):
        # Qwen3VL: (token_ids, attention_mask); image args default to None.
        return (token_ids, mask_like)
    # Text / qwen3.5-VLM: (token_ids, segment_ids, cache=None, num_right_pads).
    return (token_ids, mask_like, None, jnp.array(0, dtype=jnp.int32))


def maybe_quantize_fp8(model: nnx.Module, cfg: Any, mesh: Any = None) -> nnx.Module:
    """Quantize ``model`` to fp8 when active; otherwise return it unchanged.

    Args:
      model: the freshly-built (sharded) NNX model.
      cfg: the model config (must expose ``fp8`` / ``fp8_recipe``).
      mesh: the mesh the model was built under. When provided, the qwix trace
        runs inside ``jax.set_mesh(mesh)`` so ``out_sharding=`` in the model's
        forward resolves against Explicit-typed mesh axes.

    Returns:
      The (possibly quantized) model. Byte-identical to the input when fp8 is
      not active (strict no-op).
    """
    if not detect.fp8_active(cfg):
        return model

    recipe = getattr(cfg, "fp8_recipe", rules_mod.RECIPE_E4M3_DYNAMIC)
    # Exclude any LoRA low-rank delta matmuls (quantize only the frozen base).
    lora_paths = rules_mod.lora_delta_paths(model)
    provider = rules_mod.build_provider(recipe, lora_delta_paths=lora_paths)
    dummy_inputs = _dummy_inputs_for(model)

    # qwix runs the model's ``__call__`` once to convert weights / create the
    # quant_stats collection. On a CPU host (only reached under the
    # OMEGALAX_FORCE_FP8 validation override -- real fp8 is Hopper-only) the
    # default tokamax ``mosaic_gpu`` attention kernel is unsupported, so force
    # the CPU-correct ``xla`` attention backend for the trace. On Hopper the
    # backend is left untouched (``mosaic_gpu``) so the real kernel is traced.
    if jax.default_backend() == "cpu":
        from omegalax.models.sharding_runtime import set_attn_backend

        set_attn_backend(model, text_backend="xla")

    def _wrap() -> nnx.Module:
        return qwix.quantize_model(model, provider, *dummy_inputs)

    if mesh is not None:
        with jax.set_mesh(mesh):
            return _wrap()
    return _wrap()
