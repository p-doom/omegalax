"""Hopper (sm_90+) capability probe for fp8 training.

fp8 tensor cores exist only on NVIDIA Hopper and later. Requesting fp8 on a
non-Hopper device is a misconfiguration: :func:`omegalax.quant.apply.maybe_quantize_fp8`
asserts :func:`is_hopper` rather than silently falling back to bf16. Default
configs have fp8 off, so nothing asserts unless fp8 is explicitly requested.

The compute-capability probe mirrors ``check_compute_capability`` in
``jax._src.cudnn.fused_attention_stablehlo``: only devices reporting a CUDA
compute capability >= 9.0 count as Hopper (CPU-only processes return ``False``).
"""

from __future__ import annotations

import jax

# fp8 requires NVIDIA Hopper (sm_90) or newer.
_HOPPER_MIN_CAPABILITY = (9, 0)


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
    Ada sm_89), so the fp8 path stays a no-op there.
    """
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
    """Whether the config requests fp8 quantization (``cfg.fp8`` and recipe != ``off``)."""
    requested = bool(getattr(cfg, "fp8", False))
    recipe = getattr(cfg, "fp8_recipe", "off")
    return requested and recipe != "off"
