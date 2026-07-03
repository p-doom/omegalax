"""Host/CPU offload of optimizer state (and activation-offload gating).

This module implements the CPU/host side of *coherent-memory offload*, whose
headline target is the NVIDIA GH200 (Grace + Hopper joined by NVLink-C2C, a
cache-coherent interconnect). On such a platform the fp32 Adam moments and/or
saved activations can live in Grace (host) memory and be staged to the GPU only
for the moment they are needed, freeing HBM for larger models / longer
sequences. On A100/H100 (PCIe) the same mechanism works but every stage is a
PCIe copy, so it is transfer-bound and typically *slower* — it demonstrates the
mechanism, not the payoff. On CPU / TPU there is a single memory kind, so
offload is a semantic no-op.

Design invariants (mirrors the ``ensure_mesh`` discipline):

* **Default OFF is a strict no-op.** With ``offload_optimizer=False`` (the
  default) nothing in this module runs and the optimizer/build path is
  byte-identical to trunk.
* **Plain on/off.** Offload is correctness-equivalent everywhere (only the
  memory kind of a buffer changes), so it is a simple config bool -- no platform
  auto-detection. Off-GH200 it still works but is PCIe transfer-bound.
* **Arithmetic is untouched.** Offload only changes the *memory kind* of a
  buffer (``"device"`` vs ``"pinned_host"``); it never changes shapes, dtypes,
  layouts, or the sequence of arithmetic ops. The optimizer update is therefore
  bit-identical with offload on vs off.

The actual memory-kind *placement* (``jax.device_put(x, sharding
.with_memory_kind("pinned_host"))``) works even on the CPU backend in this JAX
(0.9.2) — it just resolves to the single available memory kind — so the
plumbing is exercisable on a login node. Peak-memory reduction and C2C overlap
are GH200-only and are verified there (see ``tests/test_offload.py`` deferred
recipe).
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
