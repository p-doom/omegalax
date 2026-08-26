from __future__ import annotations

import functools
import inspect

import jax
import jax._src.cudnn.fused_attention_stablehlo as _fused_attention

_JAX_VERSION = "0.9.2"
_CUDNN_VERSION = 91801
_PACKED_HOPPER_ERROR = "Packed layout requires a GPU with at least Hopper architecture."
_PATCH_ATTRIBUTE = "_omegalax_ampere_packed_patch"
_POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD
_EMPTY = inspect.Parameter.empty
_EXPECTED_PARAMETERS = (
    ("query", _POSITIONAL, _EMPTY),
    ("key", _POSITIONAL, _EMPTY),
    ("value", _POSITIONAL, _EMPTY),
    ("layout", _POSITIONAL, _EMPTY),
    ("cudnn_version", _POSITIONAL, _EMPTY),
    ("has_bias", _POSITIONAL, _EMPTY),
    ("is_training", _POSITIONAL, _EMPTY),
    ("is_packed", _POSITIONAL, False),
    ("is_paged_attention", _POSITIONAL, False),
    ("is_fp8", _POSITIONAL, False),
)
_EXPECTED_CAPABILITY_PARAMETERS = (("capability", _POSITIONAL, _EMPTY),)


def _parameters(function):
    return tuple(
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(function, follow_wrapped=False).parameters.values()
    )


def enable_ampere_packed_attention() -> None:
    capability_check = _fused_attention.is_cuda_compute_capability_equal
    if _parameters(capability_check) != _EXPECTED_CAPABILITY_PARAMETERS:
        raise RuntimeError(
            "unsupported JAX is_cuda_compute_capability_equal signature: "
            f"{inspect.signature(capability_check, follow_wrapped=False)}"
        )
    if not capability_check("8.0"):
        return
    if jax.__version__ != _JAX_VERSION:
        raise RuntimeError(
            f"unsupported JAX version for Ampere packed attention: {jax.__version__}"
        )

    current = _fused_attention.check_is_flash_attention
    if getattr(current, _PATCH_ATTRIBUTE, False):
        return
    if _parameters(current) != _EXPECTED_PARAMETERS:
        raise RuntimeError(
            "unsupported JAX check_is_flash_attention signature: "
            f"{inspect.signature(current, follow_wrapped=False)}"
        )

    @functools.wraps(current)
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
            return current(
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
        except NotImplementedError as error:
            if is_packed and cudnn_version == _CUDNN_VERSION and str(error) == _PACKED_HOPPER_ERROR:
                return None
            raise

    setattr(check_is_flash_attention, _PATCH_ATTRIBUTE, True)
    _fused_attention.check_is_flash_attention = check_is_flash_attention
