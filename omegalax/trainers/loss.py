"""Memory-efficient cross-entropy loss with vocabulary tiling.

Adapted from MaxText's vocabulary tiling approach. Instead of materializing the
full ``(B*T, V)`` logit tensor, tiles over the batch-sequence axis and computes
logits + cross-entropy per chunk using ``jax.lax.scan``.

With ``num_tiles=1`` this is mathematically equivalent to the naive approach.
With ``num_tiles>1`` peak memory drops from ``O(B*T*V)`` to ``O(B*T*V / num_tiles)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


def _cross_entropy_with_logits(
    logits_NV: jax.Array,
    targets_N: jax.Array,
    mask_N: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Numerically stable cross-entropy for a chunk. Returns (masked_loss_sum, mask_sum)."""
    logits_NV = logits_NV.astype(jnp.float32)
    # Use one-hot + dot to extract target logits. This avoids take_along_axis
    # which fails on TP-sharded vocab, and avoids reshard which breaks VJP
    # sharding under FSDP.
    one_hot = jax.nn.one_hot(targets_N, logits_NV.shape[-1], dtype=logits_NV.dtype)
    target_logits_N = jnp.sum(logits_NV * one_hot, axis=-1)
    max_logits_N = jnp.max(logits_NV, axis=-1)
    stable_logits_NV = logits_NV - max_logits_N[..., None]
    logsumexp_N = max_logits_N + jnp.log(jnp.sum(jnp.exp(stable_logits_NV), axis=-1))
    nll_N = logsumexp_N - target_logits_N
    mask_f = mask_N.astype(jnp.float32)
    return jnp.sum(nll_N * mask_f), jnp.sum(mask_f)


def _cross_entropy_nll(logits_NV: jax.Array, targets_N: jax.Array) -> jax.Array:
    logits_NV = logits_NV.astype(jnp.float32)
    one_hot = jax.nn.one_hot(targets_N, logits_NV.shape[-1], dtype=logits_NV.dtype)
    target_logits_N = jnp.sum(logits_NV * one_hot, axis=-1)
    max_logits_N = jnp.max(logits_NV, axis=-1)
    stable_logits_NV = logits_NV - max_logits_N[..., None]
    logsumexp_N = max_logits_N + jnp.log(jnp.sum(jnp.exp(stable_logits_NV), axis=-1))
    return logsumexp_N - target_logits_N


def chunked_cross_entropy_stats(
    hidden_BTD: jax.Array,
    lm_head_kernel_DV: jax.Array,
    targets_BT: jax.Array,
    mask_BT: jax.Array,
    num_tiles: int = 8,
    logits_out_sharding: P | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return masked ``(nll_sum, token_count)`` for next-token prediction."""
    B, T, D = hidden_BTD.shape

    hidden_BTD = hidden_BTD[:, :-1, :]
    targets_BT = targets_BT[:, 1:]
    mask_BT = mask_BT[:, 1:]
    T1 = T - 1

    if num_tiles <= 1 or T1 < num_tiles:
        logits_BTV = jnp.einsum(
            "BTD,DV->BTV", hidden_BTD, lm_head_kernel_DV, out_sharding=logits_out_sharding
        )
        return _cross_entropy_with_logits(logits_BTV, targets_BT, mask_BT)

    chunk_size = -(-T1 // num_tiles)
    pad_t = chunk_size * num_tiles - T1
    if pad_t > 0:
        hidden_BTD = jnp.pad(hidden_BTD, ((0, 0), (0, pad_t), (0, 0)))
        targets_BT = jnp.pad(targets_BT, ((0, 0), (0, pad_t)))
        mask_BT = jnp.pad(mask_BT, ((0, 0), (0, pad_t)))

    hidden_BCSD = hidden_BTD.reshape(B, num_tiles, chunk_size, D)
    targets_BCS = targets_BT.reshape(B, num_tiles, chunk_size)
    mask_BCS = mask_BT.reshape(B, num_tiles, chunk_size)

    @jax.remat
    def _remat_chunk(h_BSD, tgt_BS, msk_BS):
        logits_BSV = jnp.einsum(
            "BSD,DV->BSV", h_BSD, lm_head_kernel_DV, out_sharding=logits_out_sharding
        )
        return _cross_entropy_with_logits(logits_BSV, tgt_BS, msk_BS)

    def _scan_body(acc, chunk_data):
        loss_acc, mask_acc = acc
        h_BSD, tgt_BS, msk_BS = chunk_data
        chunk_loss, chunk_mask = _remat_chunk(h_BSD, tgt_BS, msk_BS)
        return (loss_acc + chunk_loss, mask_acc + chunk_mask), None

    (total_loss, total_mask), _ = jax.lax.scan(
        _scan_body,
        (jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)),
        (
            jnp.moveaxis(hidden_BCSD, 1, 0),
            jnp.moveaxis(targets_BCS, 1, 0),
            jnp.moveaxis(mask_BCS, 1, 0),
        ),
        unroll=1,
    )
    return total_loss, total_mask


def chunked_cross_entropy_multi_stats(
    hidden_BTD: jax.Array,
    lm_head_kernel_DV: jax.Array,
    targets_BT: jax.Array,
    masks_NBT: jax.Array,
    num_tiles: int = 8,
    logits_out_sharding: P | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return per-mask ``(nll_sum_N, token_count_N)`` with one logits pass."""
    B, T, D = hidden_BTD.shape

    hidden_BTD = hidden_BTD[:, :-1, :]
    targets_BT = targets_BT[:, 1:]
    masks_NBT = masks_NBT[:, :, 1:]
    T1 = T - 1

    if num_tiles <= 1 or T1 < num_tiles:
        logits_BTV = jnp.einsum(
            "BTD,DV->BTV", hidden_BTD, lm_head_kernel_DV, out_sharding=logits_out_sharding
        )
        nll_BT = _cross_entropy_nll(logits_BTV, targets_BT)
        mask_f_NBT = masks_NBT.astype(jnp.float32)
        return jnp.sum(mask_f_NBT * nll_BT[None, :, :], axis=(1, 2)), jnp.sum(
            mask_f_NBT, axis=(1, 2)
        )

    chunk_size = -(-T1 // num_tiles)
    pad_t = chunk_size * num_tiles - T1
    if pad_t > 0:
        hidden_BTD = jnp.pad(hidden_BTD, ((0, 0), (0, pad_t), (0, 0)))
        targets_BT = jnp.pad(targets_BT, ((0, 0), (0, pad_t)))
        masks_NBT = jnp.pad(masks_NBT, ((0, 0), (0, 0), (0, pad_t)))

    hidden_BCSD = hidden_BTD.reshape(B, num_tiles, chunk_size, D)
    targets_BCS = targets_BT.reshape(B, num_tiles, chunk_size)
    masks_NBCS = masks_NBT.reshape(masks_NBT.shape[0], B, num_tiles, chunk_size)

    @jax.remat
    def _remat_chunk(h_BSD, tgt_BS, msk_NBS):
        logits_BSV = jnp.einsum(
            "BSD,DV->BSV", h_BSD, lm_head_kernel_DV, out_sharding=logits_out_sharding
        )
        nll_BS = _cross_entropy_nll(logits_BSV, tgt_BS)
        mask_f_NBS = msk_NBS.astype(jnp.float32)
        return jnp.sum(mask_f_NBS * nll_BS[None, :, :], axis=(1, 2)), jnp.sum(
            mask_f_NBS, axis=(1, 2)
        )

    def _scan_body(acc, chunk_data):
        loss_acc, mask_acc = acc
        h_BSD, tgt_BS, msk_NBS = chunk_data
        chunk_loss, chunk_mask = _remat_chunk(h_BSD, tgt_BS, msk_NBS)
        return (loss_acc + chunk_loss, mask_acc + chunk_mask), None

    init = (
        jnp.zeros((masks_NBT.shape[0],), dtype=jnp.float32),
        jnp.zeros((masks_NBT.shape[0],), dtype=jnp.float32),
    )
    (total_loss, total_mask), _ = jax.lax.scan(
        _scan_body,
        init,
        (
            jnp.moveaxis(hidden_BCSD, 1, 0),
            jnp.moveaxis(targets_BCS, 1, 0),
            jnp.moveaxis(masks_NBCS, 2, 0),
        ),
        unroll=1,
    )
    return total_loss, total_mask


def chunked_cross_entropy_loss(
    hidden_BTD: jax.Array,
    lm_head_kernel_DV: jax.Array,
    targets_BT: jax.Array,
    mask_BT: jax.Array,
    num_tiles: int = 8,
    logits_out_sharding: P | None = None,
) -> jax.Array:
    """Memory-efficient cross-entropy that never materializes the full logit tensor.

    Tiles over the batch-sequence axis. Each tile computes ``hidden_chunk @ lm_head_kernel``
    to get a ``(chunk_size, V)`` logit slice, then immediately computes the cross-entropy
    for that slice and discards the logits.

    Args:
        hidden_BTD: Hidden states after final norm, shape ``(B, T, D)``, any dtype.
        lm_head_kernel_DV: LM head weight matrix, shape ``(D, V)``, any dtype.
        targets_BT: Target token ids, shape ``(B, T)``, int32.
        mask_BT: Loss mask, shape ``(B, T)``, int32/float32.
        num_tiles: Number of tiles to split B*T into. Higher = less memory.
            Must evenly divide ``B * T``.

    Returns:
        Scalar masked mean cross-entropy loss.
    """
    total_loss, total_mask = chunked_cross_entropy_stats(
        hidden_BTD,
        lm_head_kernel_DV,
        targets_BT,
        mask_BT,
        num_tiles=num_tiles,
        logits_out_sharding=logits_out_sharding,
    )
    return total_loss / jnp.maximum(total_mask, 1.0)
