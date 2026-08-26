from __future__ import annotations

import pytest

from omegalax.compat import cudnn_ampere_packed

_ERROR = "Packed layout requires a GPU with at least Hopper architecture."


def _check(
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
    del query, key, value, layout, cudnn_version, has_bias, is_training
    del is_packed, is_paged_attention, is_fp8
    raise NotImplementedError(_ERROR)


def _invoke(function, *, cudnn_version=91801, is_packed=True):
    return function(None, None, None, 0, cudnn_version, False, True, is_packed)


def _set_capability(monkeypatch, expected):
    monkeypatch.setattr(
        cudnn_ampere_packed._fused_attention,
        "is_cuda_compute_capability_equal",
        lambda capability: capability == expected,
    )


def test_ampere_suppresses_exact_packed_hopper_gate(monkeypatch):
    monkeypatch.setattr(cudnn_ampere_packed._fused_attention, "check_is_flash_attention", _check)
    _set_capability(monkeypatch, "8.0")

    cudnn_ampere_packed.enable_ampere_packed_attention()

    assert _invoke(cudnn_ampere_packed._fused_attention.check_is_flash_attention) is None


@pytest.mark.parametrize("capability", ["8.9", "9.0"])
def test_non_ampere_device_is_not_patched(monkeypatch, capability):
    monkeypatch.setattr(cudnn_ampere_packed._fused_attention, "check_is_flash_attention", _check)
    _set_capability(monkeypatch, capability)

    cudnn_ampere_packed.enable_ampere_packed_attention()

    assert cudnn_ampere_packed._fused_attention.check_is_flash_attention is _check


def test_cudnn_version_drift_keeps_the_hopper_gate(monkeypatch):
    monkeypatch.setattr(cudnn_ampere_packed._fused_attention, "check_is_flash_attention", _check)
    _set_capability(monkeypatch, "8.0")
    cudnn_ampere_packed.enable_ampere_packed_attention()

    with pytest.raises(NotImplementedError, match="at least Hopper"):
        _invoke(
            cudnn_ampere_packed._fused_attention.check_is_flash_attention,
            cudnn_version=91802,
        )


def test_jax_version_drift_fails_before_patch(monkeypatch):
    monkeypatch.setattr(cudnn_ampere_packed._fused_attention, "check_is_flash_attention", _check)
    monkeypatch.setattr(cudnn_ampere_packed.jax, "__version__", "0.9.3")
    _set_capability(monkeypatch, "8.0")

    with pytest.raises(RuntimeError, match="unsupported JAX version"):
        cudnn_ampere_packed.enable_ampere_packed_attention()
    assert cudnn_ampere_packed._fused_attention.check_is_flash_attention is _check


def test_signature_drift_fails_before_patch(monkeypatch):
    def changed(query):
        del query

    monkeypatch.setattr(cudnn_ampere_packed._fused_attention, "check_is_flash_attention", changed)
    _set_capability(monkeypatch, "8.0")

    with pytest.raises(RuntimeError, match="unsupported JAX check_is_flash_attention signature"):
        cudnn_ampere_packed.enable_ampere_packed_attention()
    assert cudnn_ampere_packed._fused_attention.check_is_flash_attention is changed


def test_capability_signature_drift_fails_before_patch(monkeypatch):
    def changed(capability, *, backend):
        del capability, backend

    monkeypatch.setattr(cudnn_ampere_packed._fused_attention, "check_is_flash_attention", _check)
    monkeypatch.setattr(
        cudnn_ampere_packed._fused_attention,
        "is_cuda_compute_capability_equal",
        changed,
    )

    with pytest.raises(
        RuntimeError,
        match="unsupported JAX is_cuda_compute_capability_equal signature",
    ):
        cudnn_ampere_packed.enable_ampere_packed_attention()
    assert cudnn_ampere_packed._fused_attention.check_is_flash_attention is _check


def test_unrelated_error_is_rethrown(monkeypatch):
    error = NotImplementedError("unsupported head dimension")

    def reject(
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
        del query, key, value, layout, cudnn_version, has_bias, is_training
        del is_packed, is_paged_attention, is_fp8
        raise error

    monkeypatch.setattr(cudnn_ampere_packed._fused_attention, "check_is_flash_attention", reject)
    _set_capability(monkeypatch, "8.0")
    cudnn_ampere_packed.enable_ampere_packed_attention()

    with pytest.raises(NotImplementedError) as raised:
        _invoke(cudnn_ampere_packed._fused_attention.check_is_flash_attention)
    assert raised.value is error
