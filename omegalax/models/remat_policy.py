"""Resolve remat-policy names to the ``policy=`` argument for ``nnx.remat`` /
``jax.remat``. All policies are numerically transparent (recompute/save/offload
only trade memory for compute or memory kind, never changing results).
"""

from __future__ import annotations

from typing import Callable

import jax
from jax.ad_checkpoint import checkpoint_name

# Selective: save dot_general (matmul) outputs, recompute cheap ops.
DEFAULT_REMAT_POLICY = "dots_saveable"

# Offload policies stage saved activations to host instead of recomputing them.
_OFFLOAD_SRC = "device"
_OFFLOAD_DST = "pinned_host"

# ``checkpoint_name`` handle the named-offload policy matches on the per-layer residual.
OFFLOAD_RESIDUAL_NAME = "offload_residual"

# Policy names needing the residual tagged with OFFLOAD_RESIDUAL_NAME.
_NAMED_OFFLOAD_POLICIES = frozenset({"offload_named"})


def _offload_dot_policy():
    return jax.checkpoint_policies.offload_dot_with_no_batch_dims(
        _OFFLOAD_SRC, _OFFLOAD_DST
    )


def _offload_named_policy():
    # Host-offload analogue of "checkpoint only the residual".
    return jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=(),
        names_which_can_be_offloaded=(OFFLOAD_RESIDUAL_NAME,),
        offload_src=_OFFLOAD_SRC,
        offload_dst=_OFFLOAD_DST,
    )


# name -> zero-arg factory returning a jax remat policy (``None`` == full remat,
# recompute everything).
_POLICY_FACTORIES: dict[str, Callable[[], object | None]] = {
    "full": lambda: None,
    "dots_saveable": lambda: jax.checkpoint_policies.dots_saveable,
    "dots_with_no_batch_dims_saveable": (
        lambda: jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    ),
    "everything_saveable": lambda: jax.checkpoint_policies.everything_saveable,
    # offload_named needs the residual tagged via tag_offload_residual.
    "offload_dot": _offload_dot_policy,
    "offload_named": _offload_named_policy,
}


def policy_uses_named_offload(name: str | None) -> bool:
    """True iff ``name`` is a name-based offload policy needing tagged residuals."""
    return name in _NAMED_OFFLOAD_POLICIES


def tag_offload_residual(x, policy_name: str | None):
    """Tag ``x`` with ``OFFLOAD_RESIDUAL_NAME`` iff a named-offload policy is set;
    otherwise a no-op, leaving the jaxpr unchanged from trunk."""
    if policy_uses_named_offload(policy_name):
        return checkpoint_name(x, OFFLOAD_RESIDUAL_NAME)
    return x


def available_remat_policies() -> tuple[str, ...]:
    """Return the sorted names accepted by :func:`resolve_remat_policy`."""
    return tuple(sorted(_POLICY_FACTORIES))


def resolve_remat_policy(name: str | None):
    """Resolve a policy name to a ``jax`` remat policy object (``None`` for
    ``"full"``). Raises ``ValueError`` for unknown names."""
    if name is None:
        name = "full"
    if name not in _POLICY_FACTORIES:
        raise ValueError(
            f"Unknown remat_policy '{name}'. Available: {available_remat_policies()}"
        )
    return _POLICY_FACTORIES[name]()
