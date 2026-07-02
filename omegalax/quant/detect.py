"""Hardware gating for fp8 training (Hopper-only; no-op elsewhere).

fp8 tensor cores exist on NVIDIA Hopper (sm_90) and later. On A100 (sm_80),
Ada (sm_89), and CPU there is no fp8 matmul acceleration, so quantizing the
GEMMs would be pure overhead (and on CPU/A100 the fp8 math is emulated, not
accelerated). We therefore gate the entire fp8 code path behind
:func:`is_hopper`: when it returns ``False`` the model runs the UNCHANGED bf16
path and fp8 is a strict no-op regardless of ``cfg.fp8``.

The compute-capability probe mirrors the idiom in
``jax._src.cudnn.fused_attention_stablehlo.check_compute_capability`` (see also
``omegalax/compat/cudnn_ampere_packed.py``): only devices that report a CUDA
compute capability >= 9.0 count as Hopper. CPU-only processes (no non-CPU
devices) always return ``False``.

Escape hatch (CPU development only): ``OMEGALAX_FORCE_FP8=1`` forces
``is_hopper`` (hence ``fp8_active``) to ``True`` so the qwix wrapping/tracing
composition can be validated on a CPU login node. This does NOT give real fp8
numerics (CPU has no fp8 tensor cores; the math falls back), it only exercises
that ``quantize_model`` composes with the pervasive ``out_sharding=`` usage.
Never set this in a real run.
"""

from __future__ import annotations

import os

import jax

# fp8 requires NVIDIA Hopper (sm_90) or newer.
_HOPPER_MIN_CAPABILITY = (9, 0)

_FORCE_FP8_ENV = "OMEGALAX_FORCE_FP8"


def _force_fp8() -> bool:
    """CPU-development override: bypass the hardware gate (see module docstring)."""
    return os.environ.get(_FORCE_FP8_ENV, "0") not in ("0", "", "false", "False")


def _parse_capability(cap: str | None) -> tuple[int, ...] | None:
    """Parse a ``"9.0"``-style compute-capability string into a tuple, or None."""
    if not cap:
        return None
    try:
        return tuple(int(x) for x in str(cap).split("."))
    except ValueError:
        return None


def is_hopper() -> bool:
    """True only when a visible non-CPU device reports compute capability >= 9.0.

    Returns ``False`` on CPU-only processes and on pre-Hopper GPUs (A100 sm_80,
    Ada sm_89), so the fp8 path stays a no-op there. Honors the
    ``OMEGALAX_FORCE_FP8=1`` CPU-development override (see module docstring).
    """
    if _force_fp8():
        return True
    try:
        devices = jax.devices()
    except Exception:
        return False
    for d in devices:
        if getattr(d, "platform", None) == "cpu":
            continue
        cap = _parse_capability(getattr(d, "compute_capability", None))
        if cap is not None and cap >= _HOPPER_MIN_CAPABILITY:
            return True
    return False


def fp8_active(cfg) -> bool:
    """Whether fp8 quantization should actually be applied for this config+host.

    ``True`` iff the config requests fp8 (``cfg.fp8`` and a non-``off`` recipe)
    AND the host is Hopper (or the force override is set). On A100/CPU this is
    ``False`` even when ``cfg.fp8=True`` -> the model runs the unchanged bf16
    path (strict no-op).
    """
    requested = bool(getattr(cfg, "fp8", False))
    recipe = getattr(cfg, "fp8_recipe", "off")
    if not requested or recipe == "off":
        return False
    return is_hopper()
