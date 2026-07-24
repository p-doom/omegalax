"""Host/CPU offload of optimizer state (fp32 Adam moments) to ``pinned_host``.

Stages moments to host between steps to free HBM. Memory-kind-only movement, so
the update is bit-identical on vs off; default off is a strict no-op.
"""

from __future__ import annotations

import jax

# String memory kinds (``jax.TransferToMemoryKind`` does not exist on JAX 0.9.2).
DEVICE_MEMORY_KIND = "device"
HOST_MEMORY_KIND = "pinned_host"

def resolve_offload_enabled(setting: bool) -> bool:
    """Resolve the config offload setting (a plain on/off bool)."""
    if isinstance(setting, bool):
        return setting
    raise ValueError(f"offload_optimizer must be a bool, got {setting!r}.")


def _to_memory_kind(x, memory_kind: str):
    """Place a concrete array on ``memory_kind``, preserving its partition spec (no-op if already there)."""
    sharding = getattr(x, "sharding", None)
    if sharding is None or not hasattr(sharding, "with_memory_kind"):
        return x
    if getattr(sharding, "memory_kind", None) == memory_kind:
        return x
    return jax.device_put(x, sharding.with_memory_kind(memory_kind))


def place_tree_on_memory_kind(tree, memory_kind: str):
    """Place every array leaf of ``tree`` on ``memory_kind``."""
    return jax.tree.map(lambda x: _to_memory_kind(x, memory_kind), tree)


def sharding_on_memory_kind(sharding, memory_kind: str):
    """Return ``sharding`` rewritten to ``memory_kind`` (no-op if unsupported)."""
    if sharding is None or not hasattr(sharding, "with_memory_kind"):
        return sharding
    if getattr(sharding, "memory_kind", None) == memory_kind:
        return sharding
    return sharding.with_memory_kind(memory_kind)
