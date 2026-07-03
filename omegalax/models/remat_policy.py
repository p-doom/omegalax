"""Resolve remat-policy names to ``jax.checkpoint_policies`` objects.

Activation checkpointing trades memory for backward-pass recompute: ``"full"`` saves
nothing (max recompute); selective ``dots_saveable`` saves matmul outputs; ``offload_*``
stages saved activations to host memory (GH200; see :mod:`omegalax.trainers.offload`).
All are numerically transparent -- the policy never changes forward/backward results.
:func:`resolve_remat_policy` returns the ``policy=`` argument for ``nnx.remat`` /
``jax.remat``.
"""

from __future__ import annotations

from typing import Callable

import jax
from jax.ad_checkpoint import checkpoint_name

# Default policy for decoder / vision layers: save dot_general (matmul) outputs
# and recompute the cheap elementwise/norm ops. Selective, good throughput.
DEFAULT_REMAT_POLICY = "dots_saveable"

# --- Host/activation offload -------------------------------------------------
# Instead of *recomputing* a saved activation in the backward pass (classic
# remat) or keeping it resident in HBM (save), an *offload* policy stages the
# activation to host memory ("pinned_host") for the forward-to-backward gap and
# stages it back to "device" when the backward pass needs it. On a coherent-host
# platform (GH200 Grace + NVLink-C2C) this trades cheap host memory for HBM with
# little time cost and XLA overlaps the H2D/D2H copies with compute; on PCIe
# A100/H100 the same policy works but is transfer-bound. Offload is numerically
# transparent exactly like save/recompute — it only moves bytes between memory
# kinds, never changing the math. See omegalax.trainers.offload for the platform
# gating and the string memory kinds used here.
_OFFLOAD_SRC = "device"
_OFFLOAD_DST = "pinned_host"

# ``checkpoint_name`` tag applied to the per-layer residual stream so the
# name-based offload policy (``save_and_offload_only_these_names``) has a stable
# handle to offload. Tagging is only active under a name-based offload policy;
# see :func:`tag_offload_residual` / :func:`policy_uses_named_offload`.
OFFLOAD_RESIDUAL_NAME = "offload_residual"

# Policy names that drive the ``save_and_offload_only_these_names`` machinery and
# therefore need the residual stream tagged with ``OFFLOAD_RESIDUAL_NAME``.
_NAMED_OFFLOAD_POLICIES = frozenset({"offload_named"})


def _offload_dot_policy():
    return jax.checkpoint_policies.offload_dot_with_no_batch_dims(
        _OFFLOAD_SRC, _OFFLOAD_DST
    )


def _offload_named_policy():
    # Save nothing to HBM by name; offload the tagged residual to host. Cheap
    # dots are still recomputed (they are not named), so this is the
    # host-offload analogue of "checkpoint only the residual".
    return jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=(),
        names_which_can_be_offloaded=(OFFLOAD_RESIDUAL_NAME,),
        offload_src=_OFFLOAD_SRC,
        offload_dst=_OFFLOAD_DST,
    )


# name -> zero-arg factory returning a jax remat policy (or ``None`` for
# "recompute everything", i.e. classic full remat with nothing saved).
_POLICY_FACTORIES: dict[str, Callable[[], object | None]] = {
    # Full remat: recompute the entire layer, save nothing. Minimal memory.
    "full": lambda: None,
    # Selective: save matmul / dot_general outputs, recompute cheap ops.
    "dots_saveable": lambda: jax.checkpoint_policies.dots_saveable,
    # Selective, more aggressive: only save dots whose operands have no batch
    # dims (contraction-heavy matmuls), recompute batched dots too.
    "dots_with_no_batch_dims_saveable": (
        lambda: jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    ),
    # No remat: save every intermediate. Maximal memory, zero recompute.
    "everything_saveable": lambda: jax.checkpoint_policies.everything_saveable,
    # --- Offload policies (activations -> host memory) -----------------------
    # Offload no-batch-dim dot outputs to host instead of recomputing them.
    "offload_dot": _offload_dot_policy,
    # Offload the ``checkpoint_name``-tagged residual stream to host. Requires
    # the layer to tag its residual via :func:`tag_offload_residual`.
    "offload_named": _offload_named_policy,
}


def policy_uses_named_offload(name: str | None) -> bool:
    """True iff ``name`` is a name-based offload policy needing tagged residuals.

    A layer consults this to decide whether to wrap its residual stream in
    ``checkpoint_name(x, OFFLOAD_RESIDUAL_NAME)``. For every other policy the
    tag is unnecessary (and would be inert), so it is skipped to keep the traced
    graph byte-identical to trunk when offload is off.
    """
    return name in _NAMED_OFFLOAD_POLICIES


def tag_offload_residual(x, policy_name: str | None):
    """Tag ``x`` with ``OFFLOAD_RESIDUAL_NAME`` iff a named-offload policy is set.

    ``checkpoint_name`` is an identity op on the value (it only attaches a name
    that the remat policy can match), so this is a strict no-op — same value,
    same dtype/shape — for every non-named-offload policy, and is skipped
    entirely in that case so the jaxpr is unchanged from trunk.
    """
    if policy_uses_named_offload(policy_name):
        return checkpoint_name(x, OFFLOAD_RESIDUAL_NAME)
    return x


def available_remat_policies() -> tuple[str, ...]:
    """Return the sorted names accepted by :func:`resolve_remat_policy`."""
    return tuple(sorted(_POLICY_FACTORIES))


def resolve_remat_policy(name: str | None):
    """Resolve a policy name to a ``jax`` remat policy object.

    Returns ``None`` for ``"full"`` (``jax.remat`` with ``policy=None`` is classic
    full remat). Any other name maps to a selective / offload policy callable.

    Raises ``ValueError`` for unknown names.
    """
    if name is None:
        name = "full"
    if name not in _POLICY_FACTORIES:
        raise ValueError(
            f"Unknown remat_policy '{name}'. Available: {available_remat_policies()}"
        )
    return _POLICY_FACTORIES[name]()
