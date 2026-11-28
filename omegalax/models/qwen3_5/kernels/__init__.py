"""Kernels for Qwen3.5 gated DeltaNet.

Two backends, selected at call time via the ``OMEGALAX_DELTANET_KERNEL`` env var:

- ``pallas`` (default on GPU): a Pallas Triton-lowered state-pass kernel
  that walks the J chunks sequentially per ``(batch, head)`` with state
  resident in registers, fused with parallel JAX einsums for the per-chunk
  output. Hopper-only.

- ``xla`` (default on CPU, oracle elsewhere): the pure-JAX/XLA chunked WY/UT
  implementation. Slow on Hopper but correct everywhere; the Pallas backend
  uses it as a numerical oracle and as the CPU fallback.

The dispatcher picks ``pallas`` when any non-CPU device is visible to JAX,
otherwise ``xla``. Override with ``OMEGALAX_DELTANET_KERNEL=xla|pallas``.

Both share the same signature so callers don't care which path runs:

    chunk_gated_delta_rule(q_BTHA, k_BTHA, v_BTHU, g_BTH, beta_BTH, init_NHAU, init_idx_BJ,
                           final_idx_BW, chunk_size=64) -> (out_BTHU, final_state_BWHAU)
"""

import os

import jax

from .xla_reference import chunk_gated_delta_rule_xla, gated_delta_decode_xla

__all__ = ["chunk_gated_delta_rule_xla", "gated_delta_decode_xla"]


def _resolve_backend():
    explicit = os.environ.get("OMEGALAX_DELTANET_KERNEL")
    if explicit is not None:
        return explicit.lower()
    # Implicit default: pallas if a GPU is reachable, else xla. We check
    # ``jax.devices()`` lazily so import order doesn't force a backend choice.
    try:
        if any(d.platform != "cpu" for d in jax.devices()):
            return "pallas"
    except Exception:
        pass
    return "xla"


def chunk_gated_delta_rule(
    q_BTHA,
    k_BTHA,
    v_BTHU,
    g_BTH,
    beta_BTH,
    init_NHAU,
    init_idx_BJ,
    final_idx_BW,
    chunk_size: int = 64,
):
    """Dispatcher. Late-binds the backend so env-var changes take effect per process."""
    args = (
        q_BTHA,
        k_BTHA,
        v_BTHU,
        g_BTH,
        beta_BTH,
        init_NHAU,
        init_idx_BJ,
        final_idx_BW,
        chunk_size,
    )
    backend = _resolve_backend()
    if backend == "xla":
        return chunk_gated_delta_rule_xla(*args)
    if backend == "pallas":
        from .pallas_triton import chunk_gated_delta_rule_pallas

        return chunk_gated_delta_rule_pallas(*args)
    raise ValueError(f"Unknown OMEGALAX_DELTANET_KERNEL={backend!r}. Use 'xla' or 'pallas'.")


def gated_delta_decode(*args, eps):
    """Dispatcher for one decode token per row; same backends as ``chunk_gated_delta_rule``.
    Arguments as :func:`gated_delta_decode_xla`."""
    backend = _resolve_backend()
    if backend == "xla":
        return gated_delta_decode_xla(*args, eps=eps)
    if backend == "pallas":
        from .pallas_triton import gated_delta_decode_pallas

        return gated_delta_decode_pallas(*args, eps=eps)
    raise ValueError(f"Unknown OMEGALAX_DELTANET_KERNEL={backend!r}. Use 'xla' or 'pallas'.")
