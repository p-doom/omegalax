"""Clipped policy-gradient loss on trainer-forward token log-probabilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def token_logprobs(
    hidden_BTD: jax.Array,
    lm_head_DV: jax.Array,
    tokens_BT: jax.Array,
    num_tiles: int,
    logits_out_sharding=None,
) -> tuple[jax.Array, jax.Array]:
    """Log-probability and entropy of the next-token distribution, tiled over the sequence.

    ``logp_BT[:, t]`` scores ``tokens_BT[:, t]`` given ``tokens_BT[:, :t]``; column 0 is zero.
    Logits are float32 and exist one tile at a time, in the backward pass too.
    """
    B, T, D = hidden_BTD.shape
    T1 = T - 1
    C = -(-T1 // num_tiles)
    pad = C * num_tiles - T1
    hidden_BTD = jnp.pad(hidden_BTD[:, :-1], ((0, 0), (0, pad), (0, 0)))
    targets_BT = jnp.pad(tokens_BT[:, 1:], ((0, 0), (0, pad)))
    hidden_NBCD = jnp.moveaxis(hidden_BTD.reshape(B, num_tiles, C, D), 1, 0)
    targets_NBC = jnp.moveaxis(targets_BT.reshape(B, num_tiles, C), 1, 0)

    @jax.remat
    def tile(h_BCD, tgt_BC):
        logits_BCV = jnp.einsum(
            "BCD,DV->BCV", h_BCD, lm_head_DV, out_sharding=logits_out_sharding
        ).astype(jnp.float32)
        # One-hot dot instead of take_along_axis: the vocab axis may be TP-sharded.
        one_hot_BCV = jax.nn.one_hot(tgt_BC, logits_BCV.shape[-1], dtype=jnp.float32)
        lse_BC = jax.nn.logsumexp(logits_BCV, axis=-1)
        logp_BC = jnp.sum(logits_BCV * one_hot_BCV, axis=-1) - lse_BC
        probs_BCV = jax.lax.stop_gradient(jnp.exp(logits_BCV - lse_BC[..., None]))
        entropy_BC = lse_BC - jnp.sum(probs_BCV * logits_BCV, axis=-1)
        return logp_BC, jax.lax.stop_gradient(entropy_BC)

    _, (logp_NBC, entropy_NBC) = jax.lax.scan(
        lambda carry, xs: (carry, tile(*xs)), None, (hidden_NBCD, targets_NBC)
    )
    shift = lambda x_NBC: jnp.pad(jnp.moveaxis(x_NBC, 0, 1).reshape(B, C * num_tiles)[:, :T1], ((0, 0), (1, 0)))
    return shift(logp_NBC), shift(entropy_NBC)


def policy_loss(
    logp_BT: jax.Array,
    behavior_logp_BT: jax.Array,
    advantage_BT: jax.Array,
    mask_BT: jax.Array,
    eps_low: float,
    eps_high: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """PPO-clip surrogate with the behaviour policy (the sampler's recorded logprobs) as the
    ratio denominator, summed over masked tokens; the caller normalizes. The ratio absorbs both
    policy staleness and trainer/engine numerics.

    Returns ``(loss_sum, log_ratio_BT, clipped_BT)``; ``clipped_BT`` marks tokens whose
    gradient the clip zeroes.
    """
    # The +-20 clamp only keeps exp finite (as Miles does); it never binds at sane mismatch.
    log_ratio_BT = jnp.clip(jnp.where(mask_BT > 0, logp_BT - behavior_logp_BT, 0.0), -20.0, 20.0)
    ratio_BT = jnp.exp(log_ratio_BT)
    clipped_BT = jnp.clip(ratio_BT, 1.0 - eps_low, 1.0 + eps_high)
    unclipped_BT = ratio_BT * advantage_BT
    bounded_BT = clipped_BT * advantage_BT
    pg_BT = -jnp.minimum(unclipped_BT, bounded_BT)
    return jnp.sum(pg_BT * mask_BT), log_ratio_BT, (bounded_BT < unclipped_BT).astype(jnp.float32)
