"""Resolver mapping remat-policy names to ``jax.checkpoint_policies`` policies.

Activation checkpointing (rematerialization) trades memory for recompute in the
backward pass. Full remat (``"full"`` / ``"nothing_saveable"``) saves nothing and
recomputes the entire layer -- minimal memory, maximal recompute FLOPs. A
*selective* policy instead saves expensive intermediates (matmul / ``dot_general``
outputs) and recomputes only the cheap ops, usually a net throughput win.

Policies are numerically transparent: recompute vs. save yields identical math,
so choosing a policy never changes forward/backward results.

Use :func:`resolve_remat_policy` to turn a config-provided name into the policy
object passed as the ``policy=`` argument of ``nnx.remat`` (for nnx.Module
layers) or ``jax.remat`` (for pure functions).
"""

from __future__ import annotations

from typing import Callable

import jax

# Default policy for decoder / vision layers: save dot_general (matmul) outputs
# and recompute the cheap elementwise/norm ops. Selective, good throughput.
DEFAULT_REMAT_POLICY = "dots_saveable"

# name -> zero-arg factory returning a jax remat policy (or ``None`` for
# "recompute everything", i.e. classic full remat with nothing saved).
_POLICY_FACTORIES: dict[str, Callable[[], object | None]] = {
    # Full remat: recompute the entire layer, save nothing. Minimal memory.
    "full": lambda: None,
    "nothing_saveable": lambda: jax.checkpoint_policies.nothing_saveable,
    # Selective: save matmul / dot_general outputs, recompute cheap ops.
    "dots_saveable": lambda: jax.checkpoint_policies.dots_saveable,
    "checkpoint_dots": lambda: jax.checkpoint_policies.dots_saveable,
    # Selective, more aggressive: only save dots whose operands have no batch
    # dims (contraction-heavy matmuls), recompute batched dots too.
    "dots_with_no_batch_dims_saveable": (
        lambda: jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    ),
    "checkpoint_dots_with_no_batch_dims": (
        lambda: jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    ),
    # No remat: save every intermediate. Maximal memory, zero recompute.
    "everything_saveable": lambda: jax.checkpoint_policies.everything_saveable,
    # Offload no-batch-dim dot outputs to host memory instead of recomputing.
    "offload": lambda: jax.checkpoint_policies.offload_dot_with_no_batch_dims(
        "device", "pinned_host"
    ),
    "offload_dots": lambda: jax.checkpoint_policies.offload_dot_with_no_batch_dims(
        "device", "pinned_host"
    ),
}


def available_remat_policies() -> tuple[str, ...]:
    """Return the sorted names accepted by :func:`resolve_remat_policy`."""
    return tuple(sorted(_POLICY_FACTORIES))


def resolve_remat_policy(name: str | None):
    """Resolve a policy name to a ``jax`` remat policy object.

    Returns ``None`` for ``"full"`` / ``"nothing_saveable"`` semantics of
    recomputing everything (``jax.remat`` with ``policy=None`` is classic full
    remat). Any other name maps to a selective / offload policy callable.

    Raises ``ValueError`` for unknown names.
    """
    if name is None:
        name = "full"
    if name not in _POLICY_FACTORIES:
        raise ValueError(
            f"Unknown remat_policy '{name}'. Available: {available_remat_policies()}"
        )
    return _POLICY_FACTORIES[name]()
