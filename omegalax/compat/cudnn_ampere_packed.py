from __future__ import annotations

import jax._src.cudnn.fused_attention_stablehlo as _fused_attention


def enable_ampere_packed_attention() -> None:
    if not _fused_attention.check_compute_capability("8.0"):
        raise RuntimeError("VLM training requires compute capability 8.0 or newer")

    check = _fused_attention.check_is_flash_attention

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
        del is_packed
        return check(
            query,
            key,
            value,
            layout,
            cudnn_version,
            has_bias,
            is_training,
            False,
            is_paged_attention,
            is_fp8,
        )

    _fused_attention.check_is_flash_attention = check_is_flash_attention
