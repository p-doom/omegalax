"""Zig-zag (load-balanced) sequence sharding for context parallelism (Stage 1b).

Contiguous CP sharding leaves the causal triangle unbalanced (rank 0 attends to
almost nothing, the last rank to everything). Zig-zag splits the sequence into
``2*cp`` chunks and gives rank ``r`` the pair ``{r, 2*cp-1-r}``, realized as a
PERMUTATION of the T axis applied once to every per-token array, then sharded
contiguously. CP attention builds its mask from GLOBAL positions/segment ids (not
``arange`` of the local slice), so the permutation is numerically invisible and
the loss stays permutation-invariant (all per-token arrays share the same map, so
no un-permute is needed). ``cp_size == 1`` (or a sequence not divisible by
``2*cp``) is a strict no-op (identity permutation).
"""

from __future__ import annotations

import numpy as np


def zigzag_permutation(seq_len: int, cp_size: int) -> np.ndarray:
    """Return the zig-zag permutation of ``range(seq_len)`` for ``cp_size`` ranks.

    ``perm[i]`` is the ORIGINAL index landing at permuted slot ``i``: per rank the
    order is chunk ``r`` then chunk ``2*cp-1-r``, so a contiguous cp-shard of the
    permuted layout gives rank ``r`` exactly the pair ``{r, 2*cp-1-r}``. Identity
    (strict no-op) when ``cp_size <= 1`` or ``seq_len % (2*cp_size) != 0``.
    """
    if cp_size <= 1:
        return np.arange(seq_len, dtype=np.int64)
    n_chunks = 2 * cp_size
    if seq_len % n_chunks != 0:
        return np.arange(seq_len, dtype=np.int64)
    cs = seq_len // n_chunks
    chunk_order = []
    for r in range(cp_size):
        chunk_order.append(r)
        chunk_order.append(n_chunks - 1 - r)
    perm = np.concatenate(
        [np.arange(c * cs, (c + 1) * cs, dtype=np.int64) for c in chunk_order]
    )
    return perm


def is_identity(perm: np.ndarray) -> bool:
    """True if ``perm`` is the identity (the no-op / disabled case)."""
    return bool(np.array_equal(perm, np.arange(perm.shape[0])))


def apply_zigzag_to_batch(batch: dict, perm: np.ndarray, seq_keys: frozenset) -> dict:
    """Permute the T axis of the per-token arrays in ``batch`` by ``perm``.

    ``(B, T)`` arrays in ``seq_keys`` are permuted on axis 1 (this includes an
    already-present ``position_ids_BT`` -- e.g. sequence packing's per-document
    reset positions, which must be permuted WITH the tokens, not overwritten);
    ``position_ids_ZBT`` ((3, B, T)) on axis 2. When neither a ``position_ids_ZBT``
    nor a ``position_ids_BT`` is present, a ``position_ids_BT`` (== ``perm``: the
    original index of each permuted slot) is ADDED so downstream RoPE / the mask use
    true positions. No-op for the identity perm; operates on host numpy arrays
    (called pre-shard in
    :func:`omegalax.models.sharding_runtime.shard_batch_dict`).
    """
    if is_identity(perm):
        return batch
    out = dict(batch)
    for key, arr in batch.items():
        a = np.asarray(arr)
        if key == "position_ids_ZBT" and a.ndim == 3:
            out[key] = a[:, :, perm]
        elif key in seq_keys and a.ndim >= 2:
            out[key] = a[:, perm]
    if "position_ids_ZBT" not in batch and "position_ids_BT" not in batch:
        B = int(np.asarray(next(iter(batch.values()))).shape[0])
        out["position_ids_BT"] = np.broadcast_to(perm[None, :], (B, perm.shape[0])).astype(
            np.int32
        )
    return out
