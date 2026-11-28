"""Jitted device programs of the rollout engine: K-step decode, packed extend, scoring.

``pools[i]`` holds layer ``i``'s cache arrays: ``(k_pages, v_pages)`` for attention layers,
``(conv_SKC, state_SHAU)`` for DeltaNet layers. DeltaNet pool slot ``ZERO_SLOT`` stays all
zeros (the state of an empty prefix); decode rows without work step slot ``TRASH_SLOT``.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from omegalax.models.kv_cache import LayerCache, PagedBatch
from omegalax.models.qwen3_5.cache import Cache, LinearCache, LinearDecode, LinearExtend
from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM

ZERO_SLOT, TRASH_SLOT = 0, 1


def _model(graphdef, leaves):
    """``graphdef`` is ``(nnx graphdef, state treedef)``; ``leaves`` the flat state."""
    nnx_graphdef, treedef = graphdef
    return nnx.merge(nnx_graphdef, jax.tree.unflatten(treedef, leaves))


def _logits_BV(model, hidden_BD: jax.Array) -> jax.Array:
    logits_BTV = jnp.einsum("BTD,DV->BTV", hidden_BD[:, None], model.lm_head.kernel[...])
    return logits_BTV[:, 0].astype(jnp.float32)


def _pick(logp_XV, tok_X):
    """``logp_XV[..., tok_X]`` as a reduction, which works on vocab-sharded (TP) logits."""
    hit_XV = jnp.arange(logp_XV.shape[-1], dtype=tok_X.dtype) == tok_X[..., None]
    return jnp.sum(jnp.where(hit_XV, logp_XV, 0.0), axis=-1)


def _sample(logits_BV, temp_B, key_B, pos_B, topk: int):
    """Gumbel-max sample of the token at ``pos_B``; noise is a function of (key, position).
    Also returns the ``topk`` most likely tokens of the sampled distribution and their logprobs."""
    logp_BV = jax.nn.log_softmax(logits_BV / temp_B[:, None], axis=-1)
    V = logits_BV.shape[-1]
    gumbel_BV = jax.vmap(lambda k, p: jax.random.gumbel(jax.random.fold_in(k, p), (V,)))(
        key_B, pos_B
    )
    tok_B = jnp.argmax(logp_BV + gumbel_BV, axis=-1).astype(jnp.int32)
    head_logp_Bk, head_Bk = _topk(logp_BV, topk)
    return tok_B, _pick(logp_BV, tok_B), head_Bk.astype(jnp.int32), head_logp_Bk


def _topk(x_BV, k: int, block: int = 128):
    """Exact top-k (up to the order of ties): the k blocks with the largest maxima contain it.
    ``lax.top_k`` over the whole vocabulary costs ~2.3 ms per decode call at B=128 on an H100; this ~0.1 ms."""
    B, V = x_BV.shape
    nb = -(-V // block)
    x_BNC = jnp.pad(x_BV, ((0, 0), (0, nb * block - V)), constant_values=-jnp.inf).reshape(B, nb, block)
    _, block_Bk = jax.lax.top_k(x_BNC.max(-1), k)
    cand_BX = jnp.take_along_axis(x_BNC, block_Bk[:, :, None], axis=1).reshape(B, k * block)
    vals_Bk, j_Bk = jax.lax.top_k(cand_BX, k)
    return vals_Bk, jnp.take_along_axis(block_Bk, j_Bk // block, axis=1) * block + j_Bk % block


def _forward(model, pools, tokens_BT, paged: PagedBatch, linear):
    layers = [
        LayerCache(a, b, paged) if a.ndim == 4 else LinearCache(a, b, linear) for a, b in pools
    ]
    segment_ids_BT = jnp.ones_like(tokens_BT)
    if isinstance(model, Qwen3_5ForCausalLM):
        cache = Cache(layers, paged.positions_BT)
        hidden_BTD, _ = model(tokens_BT, segment_ids_BT, cache, None)
        layers = cache.layers
    else:
        hidden_BTD, _ = model(tokens_BT, segment_ids_BT, layers, None)
    pools = [
        (c.k_pages_GPSK, c.v_pages_GPSK)
        if isinstance(c, LayerCache)
        else (c.conv_SKC, c.state_SHAU)
        for c in layers
    ]
    return hidden_BTD, pools


def _linear_decode(pools, slot_B, active_B):
    if not any(a.ndim == 3 for a, _ in pools):
        return None
    return LinearDecode(row_slot_N=jnp.where(active_B, slot_B, TRASH_SLOT))


def decode(
    graphdef,
    state,
    pools,
    carry_S3,
    tok_B,
    pos_B,
    remaining_B,
    carried_B,
    temp_B,
    key_data_B2,
    stop_B,
    page_table_BP,
    slot_B,
    eos_E,
    *,
    num_steps: int,
    num_work: int,
    topk: int,
):
    """Runs ``num_steps`` decode steps for every row with ``remaining_B > 0``.

    ``tok_B`` is each row's last sampled token, not yet in the cache, at position
    ``pos_B``. Rows with ``carried_B`` instead continue from ``carry_S3[slot_B]``, the
    (token, position, remaining) a previous call left there, so the host can queue this call
    before reading the previous one's results. Returns the updated pools and carry and the
    per-step ``(token, logprob, valid)``.
    """
    prev_B3 = carry_S3[slot_B]
    tok_B = jnp.where(carried_B, prev_B3[:, 0], tok_B)
    pos_B = jnp.where(carried_B, prev_B3[:, 1], pos_B)
    remaining_B = jnp.where(carried_B, prev_B3[:, 2], remaining_B)
    model = _model(graphdef, state)
    key_B = jax.random.wrap_key_data(key_data_B2)
    _, P, S, _ = next(a for a, _ in pools if a.ndim == 4).shape
    rows_B = jnp.arange(tok_B.shape[0], dtype=jnp.int32)

    def step(carry, _):
        pools, tok_B, pos_B, remaining_B = carry
        active_B = remaining_B > 0
        page_B = jnp.take_along_axis(page_table_BP, (pos_B // S)[:, None], axis=1)[:, 0]
        paged = PagedBatch(
            positions_BT=pos_B[:, None],
            write_src_W=rows_B,
            write_slot_W=jnp.where(active_B, page_B * S + pos_B % S, P * S + rows_B),
            page_table_NP=page_table_BP,
            q_pos_N=pos_B,
            kv_len_N=jnp.where(active_B, pos_B + 1, 0),
            block_q=1,
            num_work=num_work,
        )
        linear = _linear_decode(pools, slot_B, active_B)
        hidden_BTD, pools = _forward(model, pools, tok_B[:, None], paged, linear)
        new_B, logp_B, head_Bk, head_logp_Bk = _sample(
            _logits_BV(model, hidden_BTD[:, 0]), temp_B, key_B, pos_B + 1, topk
        )
        stopped_B = stop_B & jnp.isin(new_B, eos_E)
        remaining_B = jnp.where(active_B & ~stopped_B, remaining_B - 1, 0)
        carry = (pools, jnp.where(active_B, new_B, tok_B), pos_B + active_B, remaining_B)
        return carry, (new_B, logp_B, active_B, head_Bk, head_logp_Bk)

    (pools, tok_B, pos_B, remaining_B), (tok_KB, logp_KB, valid_KB, head_KBk, head_logp_KBk) = jax.lax.scan(
        step, (pools, tok_B, pos_B, remaining_B), length=num_steps
    )
    carry_S3 = carry_S3.at[slot_B].set(jnp.stack([tok_B, pos_B, remaining_B], axis=1))
    return pools, carry_S3, tok_KB, logp_KB, valid_KB, head_KBk, head_logp_KBk


def extend(
    graphdef,
    state,
    pools,
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
    linear: LinearExtend | None,
    *,
    block_q: int,
    topk: int,
):
    """Writes a packed batch of token chunks to the cache, then samples one token per row.

    Each sequence occupies whole query blocks of ``block_q`` tokens. Row ``r`` samples
    from the hidden state at ``sample_idx_R[r]``.
    """
    model = _model(graphdef, state)
    paged = PagedBatch(
        positions_BT=positions_T[None],
        write_src_W=write_src_W,
        write_slot_W=write_slot_W,
        page_table_NP=page_table_NP,
        q_pos_N=q_pos_N,
        kv_len_N=kv_len_N,
        block_q=block_q,
        num_work=0,
    )
    hidden_BTD, pools = _forward(model, pools, tokens_T[None], paged, linear)
    tok_R, logp_R, head_Rk, head_logp_Rk = _sample(
        _logits_BV(model, hidden_BTD[0, sample_idx_R]),
        temp_R,
        jax.random.wrap_key_data(key_data_R2),
        positions_T[sample_idx_R] + 1,
        topk,
    )
    return pools, tok_R, logp_R, head_Rk, head_logp_Rk


def score(graphdef, state, tokens_BT, pad_id: int):
    """Trainer-forward log-probability of each next token: ``logp_BT[:, t]`` scores ``t + 1``."""
    model = _model(graphdef, state)
    hidden_BTD, _ = model(tokens_BT, 1 * (tokens_BT != pad_id), None, None)
    logits_BTV = jnp.einsum("BTD,DV->BTV", hidden_BTD, model.lm_head.kernel[...])
    logp_BTV = jax.nn.log_softmax(logits_BTV.astype(jnp.float32), axis=-1)
    return _pick(logp_BTV[:, :-1], tokens_BT[:, 1:])
