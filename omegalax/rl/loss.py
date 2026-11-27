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
    cand_BTK: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Log-probability and entropy of the next-token distribution, tiled over the sequence.

    ``logp_BT[:, t]`` scores ``tokens_BT[:, t]`` given ``tokens_BT[:, :t]``; column 0 is zero.
    ``cand_logp_BTK[:, t]`` scores the candidate tokens ``cand_BTK[:, t]`` of the same
    distribution (the vocab axis must be unsharded). Logits are float32 and exist one tile at a
    time, in the backward pass too.
    """
    B, T, D = hidden_BTD.shape
    T1 = T - 1
    C = -(-T1 // num_tiles)
    pad = C * num_tiles - T1
    hidden_BTD = jnp.pad(hidden_BTD[:, :-1], ((0, 0), (0, pad), (0, 0)))
    targets_BT = jnp.pad(tokens_BT[:, 1:], ((0, 0), (0, pad)))
    hidden_NBCD = jnp.moveaxis(hidden_BTD.reshape(B, num_tiles, C, D), 1, 0)
    targets_NBC = jnp.moveaxis(targets_BT.reshape(B, num_tiles, C), 1, 0)
    K = cand_BTK.shape[-1]
    cand_BTK = jnp.pad(cand_BTK[:, 1:], ((0, 0), (0, pad), (0, 0)))
    cand_NBCK = jnp.moveaxis(cand_BTK.reshape(B, num_tiles, C, K), 1, 0)

    @jax.remat
    def tile(h_BCD, tgt_BC, cand_BCK):
        logits_BCV = jnp.einsum(
            "BCD,DV->BCV", h_BCD, lm_head_DV, out_sharding=logits_out_sharding
        ).astype(jnp.float32)
        # One-hot dot instead of take_along_axis: the vocab axis may be TP-sharded.
        one_hot_BCV = jax.nn.one_hot(tgt_BC, logits_BCV.shape[-1], dtype=jnp.float32)
        lse_BC = jax.nn.logsumexp(logits_BCV, axis=-1)
        logp_BC = jnp.sum(logits_BCV * one_hot_BCV, axis=-1) - lse_BC
        probs_BCV = jax.lax.stop_gradient(jnp.exp(logits_BCV - lse_BC[..., None]))
        entropy_BC = lse_BC - jnp.sum(probs_BCV * logits_BCV, axis=-1)
        cand_logp_BCK = jnp.take_along_axis(logits_BCV, cand_BCK, axis=-1) - lse_BC[..., None]
        return logp_BC, jax.lax.stop_gradient(entropy_BC), cand_logp_BCK

    _, (logp_NBC, entropy_NBC, cand_NBCK) = jax.lax.scan(
        lambda carry, xs: (carry, tile(*xs)), None, (hidden_NBCD, targets_NBC, cand_NBCK)
    )
    shift = lambda x_NBC: jnp.pad(jnp.moveaxis(x_NBC, 0, 1).reshape(B, C * num_tiles)[:, :T1], ((0, 0), (1, 0)))
    cand_logp_BTK = jnp.pad(
        jnp.moveaxis(cand_NBCK, 0, 1).reshape(B, C * num_tiles, K)[:, :T1], ((0, 0), (1, 0), (0, 0))
    )
    return shift(logp_NBC), shift(entropy_NBC), cand_logp_BTK


def policy_loss(
    logp_BT: jax.Array,
    behavior_logp_BT: jax.Array,
    advantage_BT: jax.Array,
    mask_BT: jax.Array,
    eps_low: float,
    eps_high: float,
    objective: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """PPO-clip surrogate with the behaviour policy (the sampler's recorded logprobs) as the
    ratio denominator (``ppo``, ``ppo_sc``), or the plain policy gradient ``-A * log p`` without
    importance sampling (``pg_sc``), summed over masked tokens; the caller normalizes. The ratio
    absorbs both policy staleness and trainer/engine numerics.

    Returns ``(loss_sum, log_ratio_BT, clipped_BT)``; ``clipped_BT`` marks tokens whose
    gradient the clip zeroes.
    """
    # The +-20 clamp only keeps exp finite (as Miles does); it never binds at sane mismatch.
    log_ratio_BT = jnp.clip(jnp.where(mask_BT > 0, logp_BT - behavior_logp_BT, 0.0), -20.0, 20.0)
    ratio_BT = jnp.exp(log_ratio_BT)
    clipped_BT = jnp.clip(ratio_BT, 1.0 - eps_low, 1.0 + eps_high)
    unclipped_BT = ratio_BT * advantage_BT
    bounded_BT = clipped_BT * advantage_BT
    if objective == "pg_sc":
        pg_BT = -advantage_BT * jnp.exp(logp_BT - jax.lax.stop_gradient(logp_BT))
        return jnp.sum(pg_BT * mask_BT), log_ratio_BT, jnp.zeros_like(log_ratio_BT)
    pg_BT = -jnp.minimum(unclipped_BT, bounded_BT)
    return jnp.sum(pg_BT * mask_BT), log_ratio_BT, (bounded_BT < unclipped_BT).astype(jnp.float32)


def score_centering(
    cand_logp_BTK: jax.Array,
    cand_logq_BTK: jax.Array,
    advantage_BT: jax.Array,
    mask_BT: jax.Array,
    objective: str,
    eps_low: float,
    eps_high: float,
) -> jax.Array:
    """Summed ``A_t * C_t``: added to the loss it subtracts, at every position, the expectation of
    that position's policy-gradient score under the sampler q (Marek & Ryabinin 2026, top-k form):
    ``C = sum_{v in H} sg(m_v - alpha * p_v) log p_v`` over the sampler's top-k head ``H``, tail
    modelled as ``q ~ rho * p`` with ``rho = (1 - q(H)) / (1 - p(H))``.

    ``pg_sc`` (plain policy gradient): ``m = q``, ``alpha = rho``. ``ppo_sc`` (PPO clip on
    ``p / q``): only unclipped tokens carry gradient, so ``m = p * unclipped_v`` and
    ``alpha = unclipped_tail``, with the tail's ratio ``1 / rho``; with nothing clipped C = 0.
    """
    sg = jax.lax.stop_gradient
    p_BTK, q_BTK = jnp.exp(sg(cand_logp_BTK)), jnp.exp(cand_logq_BTK)
    p_tail_BT = jnp.maximum(1.0 - jnp.sum(p_BTK, -1), 1e-6)
    q_tail_BT = jnp.maximum(1.0 - jnp.sum(q_BTK, -1), 0.0)
    rho_BT = q_tail_BT / p_tail_BT
    if objective == "pg_sc":
        coef_BTK = q_BTK - rho_BT[..., None] * p_BTK
    elif objective == "ppo_sc":
        a_BT = advantage_BT

        def unclipped(ratio):
            a = a_BT[..., None] if ratio.ndim == 3 else a_BT
            return ((a > 0) & (ratio <= 1.0 + eps_high)) | ((a < 0) & (ratio >= 1.0 - eps_low)) | (a == 0)

        u_BTK = unclipped(p_BTK / jnp.maximum(q_BTK, 1e-30)).astype(jnp.float32)
        u_tail_BT = unclipped(p_tail_BT / jnp.maximum(q_tail_BT, 1e-30)).astype(jnp.float32)
        coef_BTK = p_BTK * (u_BTK - u_tail_BT[..., None])
    else:
        raise ValueError(f"Unknown score-centering objective {objective!r}.")
    c_BT = jnp.sum(sg(coef_BTK) * cand_logp_BTK, -1)
    return jnp.sum(advantage_BT * c_BT * mask_BT)
