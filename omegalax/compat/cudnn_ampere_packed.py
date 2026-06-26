"""Relax JAX's Hopper-only gate on cuDNN packed/THD attention to include Ampere/Ada.

JAX (as of 0.9.x) hard-gates cuDNN's packed/THD ("ragged offsets") flash-attention
layout to Hopper+ in ``jax._src.cudnn.fused_attention_stablehlo.check_is_flash_attention``::

    if is_packed and not check_compute_capability("9.0"):
        raise NotImplementedError(
            "Packed layout requires a GPU with at least Hopper architecture.")

cuDNN added THD/ragged packed support for Ampere/Ada (sm80/sm89) in cuDNN 9.18.1, so
this gate is overly conservative on those GPUs. The omegalax vision encoder
(``omegalax/models/qwen3_vl/vision.py``) always calls the packed cuDNN kernel via
``cu_seqlens``/``q_offsets``, so on A100 (sm80) it trips this gate. This shim wraps
``check_is_flash_attention`` so the packed layout is allowed on sm80/sm89 when the
*loaded* cuDNN runtime is >= 9.18.1.

This is a local workaround pending an upstream JAX fix (the upstream patch would replace
the gate with ``not (check_compute_capability("9.0") or cudnn_version >= 91801)``). The
wrap-and-swallow approach here is version-robust: it only suppresses the specific
Hopper-only rejection and re-raises every other ``NotImplementedError`` unchanged.

Requirements / caveats:
- A cuDNN runtime >= 9.18.1 must actually be loaded (set ``LD_LIBRARY_PATH`` so the
  newer cuDNN sub-libs win over any system/module cuDNN; see the recipe ``[env]``).
- cuDNN 9.18.1 release notes describe Ampere THD *backward* for the "F16" datatype;
  bf16 coverage on sm80 is not independently confirmed here -- validate on hardware.
- sm80 does not support the deterministic algorithm with ragged tensors.

Apply by importing this module once before any cuDNN attention is traced; it applies
itself on import (idempotent) and also exposes ``apply()``.
"""

from __future__ import annotations

import jax._src.cudnn.fused_attention_stablehlo as _fa

# cuDNN version int (MAJOR*10000 + MINOR*100 + PATCH) that introduced THD/ragged
# packed attention support on Ampere/Ada (sm80/sm89).
CUDNN_AMPERE_PACKED_MIN_VERSION = 91801

_PATCH_FLAG = "_omegalax_ampere_packed_patched"


def apply() -> None:
    """Idempotently relax the Hopper-only packed-layout gate to include sm80/sm89."""
    if getattr(_fa.check_is_flash_attention, _PATCH_FLAG, False):
        return

    _orig = _fa.check_is_flash_attention

    def check_is_flash_attention(
        query,
        key,
        value,
        layout,
        cudnn_version,
        has_bias,
        is_training,
        is_packed=False,
        is_paged_attention=False,
        is_fp8=False,
    ):
        try:
            return _orig(
                query,
                key,
                value,
                layout,
                cudnn_version,
                has_bias,
                is_training,
                is_packed,
                is_paged_attention,
                is_fp8,
            )
        except NotImplementedError as exc:
            # Suppress only the Hopper-only packed-layout rejection, and only when
            # cuDNN is new enough to support packed/THD on Ampere/Ada. Hopper itself
            # never reaches here (the original allows it), so this strictly widens
            # support to sm80/sm89 + cuDNN >= 9.18.1.
            if (
                is_packed
                and "Packed layout" in str(exc)
                and cudnn_version >= CUDNN_AMPERE_PACKED_MIN_VERSION
                and _fa.check_compute_capability("8.0")
            ):
                return None
            raise

    check_is_flash_attention._omegalax_ampere_packed_patched = True
    _fa.check_is_flash_attention = check_is_flash_attention


apply()
