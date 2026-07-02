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


def shift_for_next_token(
    token_ids_BT: jax.Array,
    loss_mask_BT: jax.Array,
    pad_id: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Shift-BEFORE-shard next-token alignment for context parallelism.

    The next-token loss pairs hidden state at position ``t`` with target token at
    position ``t+1``. That shift crosses shard boundaries once the token (T) axis
    is sequence-sharded on ``cp``, so under CP it must be applied on the FULL
    sequence *before* sharding (see :mod:`omegalax.distributed.mesh`), producing
    per-position-aligned arrays where each rank then holds a self-contained
    ``(inputs_local, targets_local, mask_local)`` slice.

    Returns arrays of the SAME ``(B, T)`` length as the input (no length change,
    so the T axis stays evenly cp-divisible):

      * ``inputs_BT``  = ``token_ids_BT`` (fed to the model as-is),
      * ``targets_BT`` = ``token_ids`` rolled left by one (``target[t] =
        token[t+1]``); the last position wraps and is masked out,
      * ``loss_mask_BT`` = ``loss_mask`` rolled left by one, with the final
        position forced to 0 (no valid next token). This reproduces the
        ``mask[:, 1:]`` supervised-token count of the non-CP internal shift.

    The paired ``(inputs[t], targets[t], mask[t])`` are position-aligned, so the
    CP loss consumes them WITHOUT any further (cross-shard) shift.
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
        shift: If True (default, non-CP path), apply the next-token shift INSIDE
            the loss (``hidden[:, :-1]`` vs ``targets[:, 1:]``) -- the historical
            behavior, byte-for-byte unchanged. If False, ``hidden``/``targets``/
            ``mask`` are already position-aligned (``target[t]`` predicted from
            ``hidden[t]``); no internal shift is applied. Context parallelism uses
            ``shift=False`` because the +1 shift crosses cp shard boundaries and
            must be applied shift-BEFORE-shard (see :func:`shift_for_next_token`).
        cp_axis: If set, the token (T) axis is sequence-sharded on this mesh axis
            (context parallelism). Under CP the ``(loss_sum, mask_sum)`` reduction
            is over the sequence axis, which is GLOBAL across ``cp_axis``: this
            loss runs in the outer JIT under an Explicit-sharding mesh, where
            ``jnp.sum`` over a cp-sharded axis already inserts the all-reduce (a
            manual ``psum`` would double-count), so the resulting mean is over the
            whole sequence and the padding/normalization is global. Passing
            ``cp_axis`` also disables the within-sequence vocab tiling (the tiling
            reshape would resplit the cp-sharded T axis), which is a pure
            memory/perf knob orthogonal to CP correctness. ``None`` (default)
            leaves everything unchanged.

    Returns:
        Scalar masked mean cross-entropy loss.
    """
    B, T, D = hidden_BTD.shape

    if shift:
        # For next-token prediction: predict position t from hidden at t-1.
        # Keep B separate (may be FSDP-sharded). Tile only within each sequence.
        # NOTE: this cross-shard slice is only valid when T is NOT cp-sharded;
        # under CP the caller passes shift=False with shift-before-shard arrays.
        hidden_BTD = hidden_BTD[:, :-1, :]
        targets_BT = targets_BT[:, 1:]
        mask_BT = mask_BT[:, 1:]
        T1 = T - 1
    else:
        # Position-aligned: hidden[t] predicts targets[t]. No length change, so
        # the T axis stays evenly cp-divisible.
        T1 = T

    # Under CP, ``loss_sum``/``mask_sum`` sum over the cp-sharded sequence axis;
    # in Explicit-sharding mode that sum is ALREADY a global all-reduce across cp,
    # so ``loss_sum / mask_sum`` is the correct global mean with NO manual psum.
    # We also skip vocab tiling under CP (its reshape would resplit the sharded T).
    if cp_axis is not None or num_tiles <= 1 or T1 < num_tiles:
        logits_BTV = jnp.einsum(
            "BTD,DV->BTV", hidden_BTD, lm_head_kernel_DV, out_sharding=logits_out_sharding
        )
        loss_sum, mask_sum = _cross_entropy_with_logits(logits_BTV, targets_BT, mask_BT)
        return loss_sum / jnp.maximum(mask_sum, 1.0)

    # Tile within each sequence (B stays intact, T gets chunked)
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

    # Scan over chunks (axis 1), keeping B (axis 0, possibly FSDP-sharded) intact.
    # jax.remat on the body ensures logits are recomputed during backward instead
    # of stored for all tiles (which would require O(num_tiles * B * chunk * V)).
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
