"""Jitted device programs of the rollout engine: K-step decode, packed extend, scoring."""

import jax
import jax.numpy as jnp
from flax import nnx

from omegalax.models.qwen3.cache import LayerCache, PagedBatch


def _logits_BV(model, hidden_BD: jax.Array) -> jax.Array:
    logits_BTV = jnp.einsum("BTD,DV->BTV", hidden_BD[:, None], model.lm_head.kernel[...])
    return logits_BTV[:, 0].astype(jnp.float32)


def _sample(logits_BV, temp_B, key_B, pos_B):
    """Gumbel-max sample of the token at ``pos_B``; noise is a function of (key, position)."""
    logp_BV = jax.nn.log_softmax(logits_BV / temp_B[:, None], axis=-1)
    V = logits_BV.shape[-1]
    gumbel_BV = jax.vmap(lambda k, p: jax.random.gumbel(jax.random.fold_in(k, p), (V,)))(
        key_B, pos_B
    )
    tok_B = jnp.argmax(logp_BV + gumbel_BV, axis=-1).astype(jnp.int32)
    return tok_B, jnp.take_along_axis(logp_BV, tok_B[:, None], axis=1)[:, 0]


def _forward(model, kv, tokens_BT, batch: PagedBatch):
    cache = [LayerCache(k, v, batch) for k, v in kv]
    hidden_BTD, _ = model(tokens_BT, jnp.ones_like(tokens_BT), cache, None)
    return hidden_BTD, [(c.k_pages_GPSK, c.v_pages_GPSK) for c in cache]


def decode(
    graphdef,
    state,
    kv,
    tok_B,
    pos_B,
    remaining_B,
    temp_B,
    key_data_B2,
    stop_B,
    page_table_BP,
    eos_E,
    *,
    num_steps: int,
    num_work: int,
):
    """Runs ``num_steps`` decode steps for every row with ``remaining_B > 0``.

    ``tok_B`` is each row's last sampled token, not yet in the cache, at position
    ``pos_B``. Returns the updated cache and per-step ``(token, logprob, valid)``.
    """
    model = nnx.merge(graphdef, state)
    key_B = jax.random.wrap_key_data(key_data_B2)
    _, P, S, _ = kv[0][0].shape
    rows_B = jnp.arange(tok_B.shape[0], dtype=jnp.int32)

    def step(carry, _):
        kv, tok_B, pos_B, remaining_B = carry
        active_B = remaining_B > 0
        page_B = jnp.take_along_axis(page_table_BP, (pos_B // S)[:, None], axis=1)[:, 0]
        batch = PagedBatch(
            positions_BT=pos_B[:, None],
            write_src_W=rows_B,
            write_slot_W=jnp.where(active_B, page_B * S + pos_B % S, P * S + rows_B),
            page_table_NP=page_table_BP,
            q_pos_N=pos_B,
            kv_len_N=jnp.where(active_B, pos_B + 1, 0),
            block_q=1,
            num_work=num_work,
        )
        hidden_BTD, kv = _forward(model, kv, tok_B[:, None], batch)
        new_B, logp_B = _sample(_logits_BV(model, hidden_BTD[:, 0]), temp_B, key_B, pos_B + 1)
        stopped_B = stop_B & jnp.isin(new_B, eos_E)
        remaining_B = jnp.where(active_B & ~stopped_B, remaining_B - 1, 0)
        carry = (kv, jnp.where(active_B, new_B, tok_B), pos_B + active_B, remaining_B)
        return carry, (new_B, logp_B, active_B)

    (kv, *_), (tok_KB, logp_KB, valid_KB) = jax.lax.scan(
        step, (kv, tok_B, pos_B, remaining_B), length=num_steps
    )
    return kv, tok_KB, logp_KB, valid_KB


def extend(
    graphdef,
    state,
    kv,
    tokens_T,
    positions_T,
    write_src_W,
    write_slot_W,
    page_table_NP,
    q_pos_N,
    kv_len_N,
    sample_idx_R,
    temp_R,
    key_data_R2,
    *,
    block_q: int,
    num_work: int,
):
    """Writes a packed batch of token chunks to the cache, then samples one token per row.

    Each sequence occupies whole query blocks of ``block_q`` tokens. Row ``r`` samples
    from the hidden state at ``sample_idx_R[r]``.
    """
    model = nnx.merge(graphdef, state)
    batch = PagedBatch(
        positions_BT=positions_T[None],
        write_src_W=write_src_W,
        write_slot_W=write_slot_W,
        page_table_NP=page_table_NP,
        q_pos_N=q_pos_N,
        kv_len_N=kv_len_N,
        block_q=block_q,
        num_work=num_work,
    )
    hidden_BTD, kv = _forward(model, kv, tokens_T[None], batch)
    tok_R, logp_R = _sample(
        _logits_BV(model, hidden_BTD[0, sample_idx_R]),
        temp_R,
        jax.random.wrap_key_data(key_data_R2),
        positions_T[sample_idx_R] + 1,
    )
    return kv, tok_R, logp_R


def score(graphdef, state, tokens_BT, pad_id: int):
    """Trainer-forward log-probability of each next token: ``logp_BT[:, t]`` scores ``t + 1``."""
    model = nnx.merge(graphdef, state)
    hidden_BTD, _ = model(tokens_BT, 1 * (tokens_BT != pad_id), None, None)
    logits_BTV = jnp.einsum("BTD,DV->BTV", hidden_BTD, model.lm_head.kernel[...])
    logp_BTV = jax.nn.log_softmax(logits_BTV.astype(jnp.float32), axis=-1)
    return jnp.take_along_axis(logp_BTV[:, :-1], tokens_BT[:, 1:, None], axis=-1)[..., 0]
