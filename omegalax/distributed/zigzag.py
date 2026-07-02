"""Zig-zag (load-balanced) sequence sharding for context parallelism (Stage 1b).

Contiguous CP sharding leaves the causal-attention triangle badly unbalanced:
rank 0 (earliest tokens) attends to almost nothing, the last rank (latest tokens)
attends to the whole sequence. Zig-zag sharding fixes this: split the sequence
into ``2 * cp`` equal chunks and give rank ``r`` the pair ``{r, 2*cp - 1 - r}``
(one early chunk + one late chunk), so every rank does ~equal attention work.

We realize this as a PERMUTATION of the T axis, applied once when sharding the
batch and consistently to every per-token array (tokens, labels, loss mask,
segment ids, and the position ids that carry each token's ORIGINAL index). The
permuted array is then sharded CONTIGUOUSLY (rank ``r`` owns the contiguous slice
``[2r*cs, (2r+2)*cs)`` which is exactly its zig-zag chunk pair). Because the CP
attention builds its causal / document mask from GLOBAL position and segment ids
(not from ``arange`` of the local slice — see
:func:`omegalax.attention.context_parallel_attention`), the permutation is
numerically invisible: the mask is computed over original positions regardless of
layout. The loss reduces per position and is permutation-invariant as long as
(hidden, target, mask) stay consistently permuted (they do — all per-token arrays
are permuted by the same map), so no un-permute is needed before the loss.

``cp_size == 1`` (or a sequence not divisible by ``2*cp``) is a strict no-op:
:func:`zigzag_permutation` returns the identity, so behavior is unchanged.
"""

from __future__ import annotations

import numpy as np


def zigzag_permutation(seq_len: int, cp_size: int) -> np.ndarray:
    """Return the zig-zag permutation of ``range(seq_len)`` for ``cp_size`` ranks.

    ``perm[i]`` is the ORIGINAL sequence index that lands at permuted slot ``i``.
    The permuted order places, per rank ``r`` in ``0..cp-1``, first chunk ``r``
    then chunk ``2*cp - 1 - r``, so rank ``r``'s two chunks are contiguous in the
    permuted layout and a plain contiguous cp-shard gives it exactly the zig-zag
    pair ``{r, 2*cp - 1 - r}``.

    Returns the identity permutation (a strict no-op) when ``cp_size <= 1`` or
    when ``seq_len`` is not divisible by ``2 * cp_size``.
    """
    if cp_size <= 1:
        return np.arange(seq_len, dtype=np.int64)
    n_chunks = 2 * cp_size
    if seq_len % n_chunks != 0:
        # Can't tile evenly into 2*cp chunks -> fall back to contiguous (identity).
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

    * ``(B, T)`` arrays in ``seq_keys`` are permuted on axis 1.
    * ``position_ids_ZBT`` (shape ``(3, B, T)``) is permuted on axis 2.
    * A ``position_ids_BT`` key (original index of each permuted slot) is ADDED so
      downstream RoPE / mask uses true positions; if a ``position_ids_ZBT`` is
      present it is permuted in place instead (its values already encode true
      positions and are moved to the permuted slots).

    No-op when ``perm`` is the identity. Operates on host numpy arrays (called
    pre-shard in :func:`omegalax.models.sharding_runtime.shard_batch_dict`).
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
    # Provide the original position of each permuted slot for RoPE + the causal
    # mask (broadcast over batch). This is `perm` itself (slot i -> original i).
    if "position_ids_ZBT" not in batch:
        B = int(np.asarray(next(iter(batch.values()))).shape[0])
        out["position_ids_BT"] = np.broadcast_to(perm[None, :], (B, perm.shape[0])).astype(
            np.int32
        )
    return out
