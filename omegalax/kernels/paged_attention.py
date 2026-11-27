"""Causal attention of query blocks over a paged KV cache (Pallas, Triton lowering).

Module-local shape key: N query blocks, Q tokens per query block, W work items,
P pages, S tokens per page, M kernel rows per query block (Q * R padded to 16).

Every query block belongs to one sequence and holds consecutive positions
starting at ``q_pos_N``; it attends to KV positions ``[0, kv_len_N)`` read
through its row of ``page_table_NP``, causally. The KV of the queries
themselves must already be written to the pages. Work is split into
(query block, page chunk) items sized at runtime so that the total
``sum(kv_len)`` is spread over a fixed number of programs; partial results
are merged with their log-sum-exp.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

_MIN_ROWS = 16


def _kernel(
    work_n_ref,
    work_c_ref,
    chunk_pages_ref,
    page_table_ref,
    q_pos_ref,
    kv_len_ref,
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    lse_ref,
    *,
    rep: int,
    scale: float,
):
    w = pl.program_id(0)
    n = work_n_ref[w]
    first = work_c_ref[w] * chunk_pages_ref[0]
    kv_len = kv_len_ref[n]
    max_pages = page_table_ref.shape[1]
    page_size = k_ref.shape[1]
    num_iters = jnp.maximum(jnp.minimum(chunk_pages_ref[0], pl.cdiv(kv_len, page_size) - first), 0)

    q_MK = q_ref[n]
    rows = q_MK.shape[0]
    row_pos_M = q_pos_ref[n] + jnp.arange(rows, dtype=jnp.int32) // rep

    # The next page id rides in the loop carry so its load is off the K/V address path,
    # which lets Triton pipeline the K/V loads (~1.4x bandwidth at small batch).
    def body(i, carry):
        acc_MK, m_M, l_M, page = carry
        next_page = page_table_ref[n, jnp.minimum(first + i + 1, max_pages - 1)]
        k_SK = k_ref[page]
        v_SK = v_ref[page]
        s_MS = pl.dot(q_MK, k_SK, trans_b=True) * scale
        kv_pos_S = (first + i) * page_size + jnp.arange(page_size, dtype=jnp.int32)
        mask_MS = (kv_pos_S[None, :] <= row_pos_M[:, None]) & (kv_pos_S[None, :] < kv_len)
        s_MS = jnp.where(mask_MS, s_MS, -jnp.inf)
        m_new_M = jnp.maximum(m_M, s_MS.max(axis=1))
        m_safe_M = jnp.where(m_new_M == -jnp.inf, 0.0, m_new_M)
        alpha_M = jnp.exp(m_M - m_safe_M)
        p_MS = jnp.exp(s_MS - m_safe_M[:, None])
        l_M = l_M * alpha_M + p_MS.sum(axis=1)
        acc_MK = acc_MK * alpha_M[:, None] + pl.dot(p_MS.astype(v_SK.dtype), v_SK)
        return acc_MK, m_new_M, l_M, next_page

    init = (
        jnp.zeros(q_MK.shape, jnp.float32),
        jnp.full((rows,), -jnp.inf, jnp.float32),
        jnp.zeros((rows,), jnp.float32),
        page_table_ref[n, jnp.minimum(first, max_pages - 1)],
    )
    acc_MK, m_M, l_M, _ = jax.lax.fori_loop(0, num_iters, body, init)

    @pl.when(num_iters > 0)
    def _():
        nonempty_M = l_M > 0
        l_safe_M = jnp.where(nonempty_M, l_M, 1.0)
        o_ref[...] = acc_MK / l_safe_M[:, None]
        lse_ref[...] = jnp.where(nonempty_M, m_M + jnp.log(l_safe_M), -jnp.inf)


def _combine_kernel(start_ref, count_ref, o_ref, lse_ref, out_ref):
    """Merges the partial results of one query block's work items by log-sum-exp."""
    n = pl.program_id(0)
    start = start_ref[n]
    rows, head_dim = out_ref.shape

    def body(i, carry):
        acc_MK, m_M, l_M = carry
        lse_M = lse_ref[start + i, pl.ds(0, rows)]
        m_new_M = jnp.maximum(m_M, lse_M)
        m_safe_M = jnp.where(m_new_M == -jnp.inf, 0.0, m_new_M)
        alpha_M = jnp.exp(m_M - m_safe_M)
        beta_M = jnp.exp(lse_M - m_safe_M)
        o_MK = o_ref[start + i, pl.ds(0, rows), :]
        acc_MK = acc_MK * alpha_M[:, None] + o_MK * beta_M[:, None]
        return acc_MK, m_new_M, l_M * alpha_M + beta_M

    init = (
        jnp.zeros((rows, head_dim), jnp.float32),
        jnp.full((rows,), -jnp.inf, jnp.float32),
        jnp.zeros((rows,), jnp.float32),
    )
    acc_MK, _, l_M = jax.lax.fori_loop(0, count_ref[n], body, init)
    out_ref[...] = (acc_MK / jnp.where(l_M == 0, 1.0, l_M)[:, None]).astype(out_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("scale", "num_work", "min_chunk_pages", "num_warps", "num_stages"),
)
def paged_attention(
    q_NQHK: jax.Array,
    k_pages_GPSK: jax.Array,
    v_pages_GPSK: jax.Array,
    page_table_NP: jax.Array,
    q_pos_N: jax.Array,
    kv_len_N: jax.Array,
    *,
    scale: float,
    num_work: int,
    min_chunk_pages: int = 16,
    num_warps: int = 4,
    num_stages: int = 3,
) -> jax.Array:
    """Returns attention outputs shaped like ``q_NQHK``.

    ``num_work`` is the static number of work items per KV head; it must be at
    least ``N`` and bounds how finely long sequences are split.
    """
    N, Q, H, K = q_NQHK.shape
    G, _, S, _ = k_pages_GPSK.shape
    R = H // G
    rows = Q * R
    M = max(rows, _MIN_ROWS)
    max_pages = page_table_NP.shape[1]
    if num_work < N:
        raise ValueError(f"num_work={num_work} must be >= number of query blocks {N}.")

    q_GNMK = jnp.transpose(q_NQHK.reshape(N, Q, G, R, K), (2, 0, 1, 3, 4)).reshape(G, N, rows, K)
    q_GNMK = jnp.pad(q_GNMK, ((0, 0), (0, 0), (0, M - rows), (0, 0)))

    pages_N = (kv_len_N + S - 1) // S
    spare = num_work - N
    chunk_pages = jnp.maximum(min_chunk_pages, -(-jnp.sum(pages_N) // max(spare, 1)))
    chunks_N = (pages_N + chunk_pages - 1) // chunk_pages
    ends_N = jnp.cumsum(chunks_N)
    w_W = jnp.arange(num_work, dtype=jnp.int32)
    work_n_W = jnp.minimum(jnp.sum(w_W[:, None] >= ends_N[None, :], axis=1), N - 1).astype(jnp.int32)
    starts_N = ends_N - chunks_N
    work_c_W = jnp.where(w_W < ends_N[-1], w_W - starts_N[work_n_W], max_pages).astype(jnp.int32)

    kernel = functools.partial(_kernel, rep=R, scale=scale)
    whole = pl.BlockSpec()
    o_WGMK, lse_WGM = pl.pallas_call(
        kernel,
        grid=(num_work, G),
        in_specs=[
            whole,
            whole,
            whole,
            whole,
            whole,
            whole,
            pl.BlockSpec((None, N, M, K), lambda w, g: (g, 0, 0, 0)),
            pl.BlockSpec((None, *k_pages_GPSK.shape[1:]), lambda w, g: (g, 0, 0, 0)),
            pl.BlockSpec((None, *v_pages_GPSK.shape[1:]), lambda w, g: (g, 0, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((None, None, M, K), lambda w, g: (w, g, 0, 0)),
            pl.BlockSpec((None, None, M), lambda w, g: (w, g, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((num_work, G, M, K), jnp.float32),
            jax.ShapeDtypeStruct((num_work, G, M), jnp.float32),
        ],
        compiler_params=plt.CompilerParams(num_warps=num_warps, num_stages=num_stages),
        name="paged_attention",
    )(
        work_n_W,
        work_c_W,
        chunk_pages.reshape(1).astype(jnp.int32),
        page_table_NP,
        q_pos_N,
        kv_len_N,
        q_GNMK,
        k_pages_GPSK,
        v_pages_GPSK,
    )

    out_NGMK = pl.pallas_call(
        _combine_kernel,
        grid=(N, G),
        in_specs=[
            whole,
            whole,
            pl.BlockSpec((num_work, None, M, K), lambda n, g: (0, g, 0, 0)),
            pl.BlockSpec((num_work, None, M), lambda n, g: (0, g, 0)),
        ],
        out_specs=pl.BlockSpec((None, None, rows, K), lambda n, g: (n, g, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((N, G, rows, K), q_NQHK.dtype),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=3),
        name="paged_attention_combine",
    )(starts_N.astype(jnp.int32), chunks_N.astype(jnp.int32), o_WGMK, lse_WGM)
    out_NQHK = jnp.transpose(out_NGMK.reshape(N, G, Q, R, K), (0, 2, 1, 3, 4)).reshape(N, Q, H, K)
    return out_NQHK
