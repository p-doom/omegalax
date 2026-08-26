from __future__ import annotations

import pytest

from omegalax.compat import cudnn_ampere_packed


def test_packed_attention_disables_the_packed_validator_gate(monkeypatch):
    received = None

    def check(
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
        nonlocal received
        received = (
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

    monkeypatch.setattr(cudnn_ampere_packed._fused_attention, "check_is_flash_attention", check)
    monkeypatch.setattr(
        cudnn_ampere_packed._fused_attention,
        "check_compute_capability",
        lambda capability: capability == "8.0",
    )
    cudnn_ampere_packed.enable_ampere_packed_attention()

    cudnn_ampere_packed._fused_attention.check_is_flash_attention(
        "query", "key", "value", 0, 0, False, True, True, True, False
    )

    assert received == ("query", "key", "value", 0, 0, False, True, False, True, False)


def test_unsupported_gpu_fails_before_patch(monkeypatch):
    check = cudnn_ampere_packed._fused_attention.check_is_flash_attention
    monkeypatch.setattr(
        cudnn_ampere_packed._fused_attention,
        "check_compute_capability",
        lambda capability: False,
    )

    with pytest.raises(RuntimeError, match="compute capability 8.0"):
        cudnn_ampere_packed.enable_ampere_packed_attention()

    assert cudnn_ampere_packed._fused_attention.check_is_flash_attention is check
