"""Host/CPU offload of optimizer state (fp32 Adam moments) to ``pinned_host``.

Stages moments to host memory between steps, freeing HBM; the headline target is
GH200 (Grace + NVLink-C2C, where the staging is cheap). A plain on/off config bool:
correctness-equivalent everywhere (only the buffer's memory kind changes, never
shapes/dtypes/arithmetic -- so the update is bit-identical on vs off), just
PCIe-transfer-bound off-GH200. Default off is a strict no-op. The ``pinned_host``
placement resolves on CPU too, so the plumbing is exercisable on a login node;
peak-memory reduction / C2C overlap are GH200-only.
"""

from __future__ import annotations

import jax

# String memory kinds understood by ``NamedSharding.with_memory_kind`` /
# ``jax.device_put`` on this JAX (0.9.2). ``jax.TransferToMemoryKind`` does NOT
# exist here, so we drive everything off these strings.
DEVICE_MEMORY_KIND = "device"
HOST_MEMORY_KIND = "pinned_host"

def resolve_offload_enabled(setting: bool) -> bool:
    """Resolve the config offload setting (a plain on/off bool)."""
    if isinstance(setting, bool):
        return setting
    raise ValueError(f"offload_optimizer must be a bool, got {setting!r}.")


def _to_memory_kind(x, memory_kind: str):
    """Place a single concrete array on ``memory_kind`` (device or pinned_host).

    Rewrites the array's ``NamedSharding`` memory kind in place via
    ``jax.device_put``; the spec (FSDP/TP partitioning) is preserved, only the
    memory kind changes. A no-op if the array already has that memory kind or
    carries no ``NamedSharding``.
    """
    sharding = getattr(x, "sharding", None)
    if sharding is None or not hasattr(sharding, "with_memory_kind"):
        return x
    if getattr(sharding, "memory_kind", None) == memory_kind:
        return x
    return jax.device_put(x, sharding.with_memory_kind(memory_kind))


def place_tree_on_memory_kind(tree, memory_kind: str):
    """Place every array leaf of ``tree`` on ``memory_kind``.

    Used to move the optimizer's moment buffers (fp32 Adam ``mu``/``nu``) to
    ``pinned_host`` at build time (before checkpoint restore, so restored
    shardings match). Non-array leaves and leaves without a ``NamedSharding``
    are left untouched.
    """
    return jax.tree.map(lambda x: _to_memory_kind(x, memory_kind), tree)


def sharding_on_memory_kind(sharding, memory_kind: str):
    """Return ``sharding`` rewritten to ``memory_kind`` (or unchanged).

    Used to build the ``out_shardings`` / ``ShapeDtypeStruct`` shardings that
    carry the memory kind through a jitted step so XLA can overlap the H2D/D2H
    staging with compute. A no-op for shardings that don't support memory kinds.
    """
    if sharding is None or not hasattr(sharding, "with_memory_kind"):
        return sharding
    if getattr(sharding, "memory_kind", None) == memory_kind:
        return sharding
    return sharding.with_memory_kind(memory_kind)
