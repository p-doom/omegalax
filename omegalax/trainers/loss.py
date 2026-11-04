"""Memory-efficient cross-entropy with vocabulary tiling (MaxText-style).

Tiles over the batch-sequence axis instead of materializing the full ``(B*T, V)``
logits: ``num_tiles=1`` is the naive path, ``num_tiles>1`` drops peak memory to
``O(B*T*V / num_tiles)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


def shift_for_next_token(
    token_ids_BT: jax.Array,
    loss_mask_BT: jax.Array,
    pad_id: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Shift-BEFORE-shard next-token alignment for context parallelism.

    The next-token +1 shift crosses shard boundaries once the T axis is cp-sharded,
    so under CP it must be applied on the FULL sequence before sharding. Returns
    same-length ``(B, T)`` arrays (T stays cp-divisible): ``inputs`` == tokens,
    ``targets`` == tokens rolled left one (last position masked out), ``loss_mask``
    rolled left one with the final position forced to 0. The result is
    position-aligned, so the CP loss consumes it with no further shift.
    """
    del pad_id  # masking is driven by loss_mask, not pad_id, matching the trainer.
    targets_BT = jnp.roll(token_ids_BT, shift=-1, axis=1)
    mask_shifted_BT = jnp.roll(loss_mask_BT, shift=-1, axis=1)
    # Final position has no next token: force its mask to 0.
    mask_shifted_BT = mask_shifted_BT.at[:, -1].set(0)
    return token_ids_BT, targets_BT, mask_shifted_BT


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


def chunked_cross_entropy_loss(
    hidden_BTD: jax.Array,
    lm_head_kernel_DV: jax.Array,
    targets_BT: jax.Array,
    mask_BT: jax.Array,
    num_tiles: int = 8,
    logits_out_sharding: P | None = None,
    *,
    shift: bool = True,
    cp_axis: str | None = None,
) -> jax.Array:
    """Memory-efficient masked-mean cross-entropy that never materializes the full
    logit tensor: each tile computes a ``(chunk, V)`` logit slice and its CE, then
    discards the logits. ``hidden_BTD`` (B, T, D), ``lm_head_kernel_DV`` (D, V),
    ``targets_BT``/``mask_BT`` (B, T); ``num_tiles`` must divide ``B*T``.

    ``shift``: True (default, non-CP) applies the next-token shift INSIDE the loss
    (``hidden[:, :-1]`` vs ``targets[:, 1:]``). False expects already-aligned arrays
    (CP uses this -- the +1 shift is applied shift-before-shard, see
    :func:`shift_for_next_token`).

    ``cp_axis``: when set, the T axis is cp-sharded; the ``(loss_sum, mask_sum)`` sum
    over that axis is GLOBAL because this runs under an Explicit-sharding mesh where
    ``jnp.sum`` over a cp axis already all-reduces (a manual psum would double-count),
    and vocab tiling is disabled (its reshape would resplit the sharded T).
    """
    B, T, D = hidden_BTD.shape

    if shift:
        # Next-token shift; valid only when T is NOT cp-sharded (CP passes shift=False).
        hidden_BTD = hidden_BTD[:, :-1, :]
        targets_BT = targets_BT[:, 1:]
        mask_BT = mask_BT[:, 1:]
        T1 = T - 1
    else:
        T1 = T

    # Under an Explicit-sharding mesh, jnp.sum over the cp-sharded T axis already
    # all-reduces, so loss_sum / mask_sum is the correct global mean (no manual
    # psum). Vocab tiling is skipped under CP (its reshape would resplit the T axis).
    if cp_axis is not None or num_tiles <= 1 or T1 < num_tiles:
        logits_BTV = jnp.einsum(
            "BTD,DV->BTV", hidden_BTD, lm_head_kernel_DV, out_sharding=logits_out_sharding
        )
        loss_sum, mask_sum = _cross_entropy_with_logits(logits_BTV, targets_BT, mask_BT)
        return loss_sum / jnp.maximum(mask_sum, 1.0)

    # Tile within each sequence (B intact, T chunked).
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
        """Compute logits + CE for one chunk; remat discards logits in backward."""
        logits_BSV = jnp.einsum(
            "BSD,DV->BSV", h_BSD, lm_head_kernel_DV, out_sharding=logits_out_sharding
        )
        return _cross_entropy_with_logits(logits_BSV, tgt_BS, msk_BS)

    def _scan_body(acc, chunk_data):
        loss_acc, mask_acc = acc
        h_BSD, tgt_BS, msk_BS = chunk_data
        chunk_loss, chunk_mask = _remat_chunk(h_BSD, tgt_BS, msk_BS)
        return (loss_acc + chunk_loss, mask_acc + chunk_mask), None

    # Scan over chunks (B intact); the body's jax.remat recomputes logits in the
    # backward instead of storing all tiles' O(num_tiles * B * chunk * V).
    (total_loss, total_mask), _ = jax.lax.scan(
        _scan_body,
        (jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)),
        (
            jnp.moveaxis(hidden_BCSD, 1, 0),  # (num_tiles, B, chunk_size, D)
            jnp.moveaxis(targets_BCS, 1, 0),  # (num_tiles, B, chunk_size)
            jnp.moveaxis(mask_BCS, 1, 0),  # (num_tiles, B, chunk_size)
        ),
        unroll=1,
    )
    return total_loss / jnp.maximum(total_mask, 1.0)
