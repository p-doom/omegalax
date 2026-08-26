"""Resolve remat-policy names to the ``policy=`` argument for ``nnx.remat`` /
``jax.remat``. All policies are numerically transparent (recompute/save only
trade memory for compute, never changing results).
"""

from __future__ import annotations

from typing import Callable

import jax

# Full remat: selective policies spike saved-matmul HBM past 80GB at 8B/16k (acts not fsdp-sharded).
DEFAULT_REMAT_POLICY = "full"


# name -> zero-arg factory returning a jax remat policy (``None`` == full remat,
# recompute everything).
_POLICY_FACTORIES: dict[str, Callable[[], object | None]] = {
    "full": lambda: None,
    "dots_saveable": lambda: jax.checkpoint_policies.dots_saveable,
    "dots_with_no_batch_dims_saveable": (
        lambda: jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    ),
    "everything_saveable": lambda: jax.checkpoint_policies.everything_saveable,
}


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
