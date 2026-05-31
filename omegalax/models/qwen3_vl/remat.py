"""Activation-rematerialization policies for Qwen3-VL transformer blocks.

The active policy is selected at run time via the ``--remat_policy`` flag on
``scripts/train_vlm_sft.py`` and applied with ``set_remat_policy`` in
``sharding_runtime``. ``TextDecoderLayer`` / ``VisionBlock.__call__`` dispatch
through ``remat_wrap`` which caches the wrapped callable per (fn, policy).

Policies (most-recompute → most-save):
    ``nothing``        Default ``jax.remat`` behavior: recompute every
                       intermediate (Q/K/V/O proj outputs, MLP matmul outputs,
                       softmax, activations). Lowest HBM, highest FLOPs.
    ``dots_no_batch``  Save dot_general outputs whose contracting axes have no
                       batch dim — i.e. the weight matmuls (Q/K/V/O proj,
                       gate/up/down). Cheapest meaningful upgrade from default.
    ``dots``           Save ALL dot_general outputs, including ``QK^T`` and
                       ``attn·V``. Highest HBM, lowest recompute.
    ``offload``        Like ``dots_no_batch`` but offloads to pinned host RAM.
    ``none``           Skip ``jax.remat`` entirely (full activations kept).
"""

from __future__ import annotations

import functools
from typing import Callable

import jax


_NO_REMAT = object()  # sentinel for "skip remat entirely"

_POLICIES: dict[str, object] = {
    "nothing": None,
    "dots_no_batch": jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    "dots": jax.checkpoint_policies.checkpoint_dots,
    "offload": jax.checkpoint_policies.offload_dot_with_no_batch_dims(
        offload_src="device", offload_dst="pinned_host"
    ),
    "none": _NO_REMAT,
}


def remat_policy_choices() -> tuple[str, ...]:
    return tuple(_POLICIES)


@functools.cache
def remat_wrap(fn: Callable, policy_name: str) -> Callable:
    if policy_name not in _POLICIES:
        raise ValueError(
            f"unknown remat_policy {policy_name!r}; must be one of {sorted(_POLICIES)}"
        )
    policy = _POLICIES[policy_name]
    if policy is _NO_REMAT:
        return fn
    if policy is None:
        return jax.remat(fn, static_argnums=0)
    return jax.remat(fn, static_argnums=0, policy=policy)
