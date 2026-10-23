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


def _entropy_with_logits(logits_NV: jax.Array) -> jax.Array:
    logits_NV = logits_NV.astype(jnp.float32)
    log_probs_NV = jax.nn.log_softmax(logits_NV, axis=-1)
    return -jnp.sum(jnp.exp(log_probs_NV) * log_probs_NV, axis=-1)


def chunked_token_entropies(
    hidden_BTD: jax.Array,
    lm_head_kernel_DV: jax.Array,
    num_tiles: int = 8,
    logits_out_sharding: P | None = None,
) -> jax.Array:
    """Return predictive entropy for every next-token position, shape ``(B, T - 1)``."""
    B, T, D = hidden_BTD.shape
    hidden_BTD = hidden_BTD[:, :-1, :]
    T1 = T - 1

    if num_tiles <= 1 or T1 < num_tiles:
        logits_BTV = jnp.einsum(
            "BTD,DV->BTV", hidden_BTD, lm_head_kernel_DV, out_sharding=logits_out_sharding
        )
        return _entropy_with_logits(logits_BTV)

    chunk_size = -(-T1 // num_tiles)
    pad_t = chunk_size * num_tiles - T1
    if pad_t > 0:
        hidden_BTD = jnp.pad(hidden_BTD, ((0, 0), (0, pad_t), (0, 0)))
    hidden_BCSD = hidden_BTD.reshape(B, num_tiles, chunk_size, D)

    @jax.remat
    def _remat_chunk(h_BSD):
        logits_BSV = jnp.einsum(
            "BSD,DV->BSV", h_BSD, lm_head_kernel_DV, out_sharding=logits_out_sharding
        )
        return _entropy_with_logits(logits_BSV)

    entropy_CBS = jax.lax.map(_remat_chunk, jnp.moveaxis(hidden_BCSD, 1, 0))
    entropy_BT = jnp.moveaxis(entropy_CBS, 0, 1).reshape(B, num_tiles * chunk_size)
    return entropy_BT[:, :T1]


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


# Fixed binding of the ``rollout_teacher_v1`` targets (see its ``manifest.json``).
_TEACHER_VOCAB_SIZE = 248_320
_TEACHER_TOP_K = 512
_TEACHER_SPECIAL_IDS = tuple(range(248_044, 248_077))
_TEACHER_TARGET_KEYS = (
    "teacher_positions_BR",
    "teacher_mask_BR",
    "teacher_top512_ids_BRK",
    "teacher_top512_logits_BRK",
    "teacher_special_logits_BRS",
    "teacher_logz_all_BR",
    "teacher_logz_tail_BR",
)

_CLASS_SCATTER_DIMS = jax.lax.ScatterDimensionNumbers(
    update_window_dims=(),
    inserted_window_dims=(2,),
    scatter_dims_to_operand_dims=(2,),
    operand_batching_dims=(0, 1),
    scatter_indices_batching_dims=(0, 1),
)


def chunked_teacher_kl_stats(
    hidden_BTD: jax.Array,
    lm_head_kernel_DV: jax.Array,
    targets: dict[str, jax.Array],
    *,
    block_size: int = 256,
    logits_out_sharding: P | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(kl_sum, target_count)`` of grouped forward KL(teacher || student).

    Per valid target row the classes are its unique Top512 IDs, the special IDs not
    already in that Top512 (counted once), and one tail class holding the rest of the
    vocabulary, all at temperature 1. Teacher log-probabilities are the stored logits and
    ``logz_tail`` minus ``logz_all``. The student uses its full normalizer, and its tail
    is a logsumexp over the complement, so a tiny tail is never a difference of
    near-equal probabilities. This is not the exact full-vocabulary KL.

    ``hidden_BTD`` holds all concatenated segments; row r of batch element b is predicted
    by ``hidden_BTD[b, teacher_positions_BR[b, r]]``. Only those rows are projected, at most
    ``block_size`` per batch element at a time with rematerialization, so the
    ``(B, T, V)`` logits never exist. Masked rows read position and ID 0 and contribute
    exactly zero value and gradient. Both results are float32 sums over the batch.
    """
    if set(targets) != set(_TEACHER_TARGET_KEYS):
        raise ValueError(f"targets must have exactly the keys {_TEACHER_TARGET_KEYS}")
    B, _, D = hidden_BTD.shape
    if lm_head_kernel_DV.shape != (D, _TEACHER_VOCAB_SIZE):
        raise ValueError(
            f"lm_head_kernel_DV must have shape ({D}, {_TEACHER_VOCAB_SIZE}) to match the "
            f"teacher vocabulary, got {lm_head_kernel_DV.shape}"
        )
    R = targets["teacher_positions_BR"].shape[-1]
    expected_shapes = {
        "teacher_positions_BR": (B, R),
        "teacher_mask_BR": (B, R),
        "teacher_top512_ids_BRK": (B, R, _TEACHER_TOP_K),
        "teacher_top512_logits_BRK": (B, R, _TEACHER_TOP_K),
        "teacher_special_logits_BRS": (B, R, len(_TEACHER_SPECIAL_IDS)),
        "teacher_logz_all_BR": (B, R),
        "teacher_logz_tail_BR": (B, R),
    }
    for name, shape in expected_shapes.items():
        if targets[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {targets[name].shape}")
    if targets["teacher_mask_BR"].dtype != jnp.bool_:
        raise ValueError("teacher_mask_BR must be bool")
    if R < 1 or block_size < 1:
        raise ValueError(f"Need R >= 1 and block_size >= 1, got R={R}, block_size={block_size}")

    mask_BR = targets["teacher_mask_BR"]
    positions_BR = jnp.where(mask_BR, targets["teacher_positions_BR"], 0)
    hidden_BRD = jnp.take_along_axis(hidden_BTD, positions_BR[:, :, None], axis=1)
    row_arrays = [
        hidden_BRD,
        mask_BR,
        targets["teacher_top512_ids_BRK"],
        targets["teacher_top512_logits_BRK"],
        targets["teacher_special_logits_BRS"],
        targets["teacher_logz_all_BR"],
        targets["teacher_logz_tail_BR"],
    ]

    block = min(block_size, R)
    num_blocks = -(-R // block)
    pad_r = num_blocks * block - R
    if pad_r > 0:
        row_arrays = [
            jnp.pad(x, ((0, 0), (0, pad_r)) + ((0, 0),) * (x.ndim - 2)) for x in row_arrays
        ]
    blocks = [jnp.moveaxis(x.reshape(B, num_blocks, block, *x.shape[2:]), 1, 0) for x in row_arrays]
    special_ids = jnp.asarray(_TEACHER_SPECIAL_IDS, dtype=jnp.int32)

    # Batched gather/scatter keep (B, S) local to each data shard. Explicit sharding cannot
    # type them along a TP-sharded vocab axis, so that axis is left to the partitioner,
    # which reads locally and reduces only the gathered (B, S, G) values over TP.
    def take_classes(logits_BSV, class_ids_BSG):
        return jnp.take_along_axis(logits_BSV, class_ids_BSG, axis=-1)

    def mark_classes(logits_BSV, class_ids_BSG):
        return jax.lax.scatter(
            jnp.zeros_like(logits_BSV, dtype=jnp.bool_),
            class_ids_BSG[..., None],
            jnp.ones_like(class_ids_BSG, dtype=jnp.bool_),
            _CLASS_SCATTER_DIMS,
            mode="drop",
        )

    vocab_axis = None if logits_out_sharding is None else logits_out_sharding[2]
    if vocab_axis is not None:
        take_classes = jax.sharding.auto_axes(
            take_classes, axes=vocab_axis, out_sharding=P(logits_out_sharding[0], None, None)
        )
        mark_classes = jax.sharding.auto_axes(
            mark_classes, axes=vocab_axis, out_sharding=logits_out_sharding
        )

    @jax.remat
    def _remat_block(h_BSD, msk_BS, ids_BSK, top_BSK, special_BSS, logz_all_BS, logz_tail_BS):
        logits_BSV = jnp.einsum(
            "BSD,DV->BSV", h_BSD, lm_head_kernel_DV, out_sharding=logits_out_sharding
        ).astype(jnp.float32)
        ids_BSK = jnp.where(msk_BS[..., None], ids_BSK, 0)
        class_ids_BSG = jnp.concatenate(
            [ids_BSK, jnp.zeros_like(special_BSS, dtype=ids_BSK.dtype) + special_ids], axis=-1
        )
        class_logits_BSG = take_classes(logits_BSV, class_ids_BSG)
        # Per-block membership mask for the tail complement; rematerialized, never stored.
        in_class_BSV = mark_classes(logits_BSV, class_ids_BSG)
        log_z_BS = jax.nn.logsumexp(logits_BSV, axis=-1)
        log_tail_BS = jax.nn.logsumexp(jnp.where(in_class_BSV, -jnp.inf, logits_BSV), axis=-1)
        student_BSG = (
            jnp.concatenate([class_logits_BSG, log_tail_BS[..., None]], axis=-1)
            - log_z_BS[..., None]
        )

        special_in_top_BSS = jnp.any(ids_BSK[..., :, None] == special_ids, axis=-2)
        keep_BSG = msk_BS[..., None] & jnp.concatenate(
            [
                jnp.ones_like(ids_BSK, dtype=jnp.bool_),
                ~special_in_top_BSS,
                jnp.ones_like(msk_BS[..., None]),
            ],
            axis=-1,
        )
        teacher_BSG = (
            jnp.concatenate([top_BSK, special_BSS, logz_tail_BS[..., None]], axis=-1)
            - logz_all_BS[..., None]
        )
        teacher_BSG = jnp.where(keep_BSG, teacher_BSG, 0.0)
        teacher_prob_BSG = jnp.where(keep_BSG, jnp.exp(teacher_BSG), 0.0)
        return jnp.sum(teacher_prob_BSG * (teacher_BSG - student_BSG))

    def _scan_body(kl_acc, block_data):
        return kl_acc + _remat_block(*block_data), None

    kl_sum, _ = jax.lax.scan(_scan_body, jnp.array(0.0, dtype=jnp.float32), tuple(blocks), unroll=1)
    return kl_sum, jnp.sum(mask_BR.astype(jnp.float32))
