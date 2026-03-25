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


def _cross_entropy_with_logits(
    logits_NV: jax.Array,
    targets_N: jax.Array,
    mask_N: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Numerically stable cross-entropy for a chunk. Returns (masked_loss_sum, mask_sum)."""
    logits_NV = logits_NV.astype(jnp.float32)
    target_logits_N = jnp.take_along_axis(logits_NV, targets_N[..., None], axis=-1)[..., 0]
    max_logits_N = jnp.max(logits_NV, axis=-1)
    stable_logits_NV = logits_NV - max_logits_N[..., None]
    logsumexp_N = max_logits_N + jnp.log(jnp.sum(jnp.exp(stable_logits_NV), axis=-1))
    nll_N = logsumexp_N - target_logits_N
    mask_f = mask_N.astype(jnp.float32)
    return jnp.sum(nll_N * mask_f), jnp.sum(mask_f)


def chunked_cross_entropy_loss(
    hidden_BTD: jax.Array,
    lm_head_kernel_DV: jax.Array,
    targets_BT: jax.Array,
    mask_BT: jax.Array,
    num_tiles: int = 8,
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
    B, T, D = hidden_BTD.shape
    N = B * T  # total tokens

    # For next-token prediction: predict position t from hidden at t-1
    hidden_ND = hidden_BTD[:, :-1, :].reshape(-1, D)
    targets_N = targets_BT[:, 1:].reshape(-1)
    mask_N = mask_BT[:, 1:].reshape(-1)
    N = hidden_ND.shape[0]  # B * (T - 1)

    if num_tiles <= 1 or N < num_tiles:
        # No tiling
        logits_NV = hidden_ND @ lm_head_kernel_DV
        loss_sum, mask_sum = _cross_entropy_with_logits(logits_NV, targets_N, mask_N)
        return loss_sum / jnp.maximum(mask_sum, 1.0)

    # Pad N to be divisible by num_tiles
    chunk_size = -(-N // num_tiles)  # ceil division
    pad_n = chunk_size * num_tiles - N
    if pad_n > 0:
        hidden_ND = jnp.pad(hidden_ND, ((0, pad_n), (0, 0)))
        targets_N = jnp.pad(targets_N, (0, pad_n))
        mask_N = jnp.pad(mask_N, (0, pad_n))  # padded positions have mask=0

    # Reshape into tiles: (num_tiles, chunk_size, ...)
    hidden_tiled = hidden_ND.reshape(num_tiles, chunk_size, D)
    targets_tiled = targets_N.reshape(num_tiles, chunk_size)
    mask_tiled = mask_N.reshape(num_tiles, chunk_size)

    def _scan_body(acc, chunk_data):
        loss_acc, mask_acc = acc
        hidden_chunk, target_chunk, mask_chunk = chunk_data
        logits_chunk = hidden_chunk @ lm_head_kernel_DV
        chunk_loss, chunk_mask = _cross_entropy_with_logits(logits_chunk, target_chunk, mask_chunk)
        return (loss_acc + chunk_loss, mask_acc + chunk_mask), None

    (total_loss, total_mask), _ = jax.lax.scan(
        _scan_body,
        (jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)),
        (hidden_tiled, targets_tiled, mask_tiled),
    )
    return total_loss / jnp.maximum(total_mask, 1.0)
