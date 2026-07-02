"""fp8 training support (Hopper-only; strict no-op on A100/CPU).

See ``omegalax/quant/detect.py`` (Hopper gating), ``rules.py`` (qwix QtRules)
and ``apply.py`` (model-build wrap injection).
"""

from omegalax.quant.apply import maybe_quantize_fp8
from omegalax.quant.detect import fp8_active, is_hopper
from omegalax.quant.rules import (
    RECIPE_BLOCKWISE_128,
    RECIPE_E4M3_DYNAMIC,
    RECIPE_OFF,
    SUPPORTED_RECIPES,
    build_provider,
)

__all__ = [
    "maybe_quantize_fp8",
    "fp8_active",
    "is_hopper",
    "build_provider",
    "RECIPE_OFF",
    "RECIPE_E4M3_DYNAMIC",
    "RECIPE_BLOCKWISE_128",
    "SUPPORTED_RECIPES",
]
