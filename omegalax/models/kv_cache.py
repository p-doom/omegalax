"""Paged KV cache threaded through attention layers by the rollout engine."""

import dataclasses
import functools
from typing import TypeAlias

import jax
import jax.numpy as jnp

from omegalax.kernels.paged_attention import paged_attention


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["positions_BT", "write_src_W", "write_slot_W", "page_table_NP", "q_pos_N", "kv_len_N"],
    meta_fields=["block_q", "num_work"],
)
@dataclasses.dataclass(frozen=True)
class PagedBatch:
    """Token placement shared by all layers; see ``omegalax.kernels.paged_attention``.

    KV write ``w`` stores flattened token ``write_src_W[w]`` at ``write_slot_W[w]`` (``page *
    page_size + offset``). A token may be written to several slots (forked prompts); slots
    are unique, and writes to skip carry distinct out-of-range slots.
    """

    positions_BT: jax.Array
    write_src_W: jax.Array
    write_slot_W: jax.Array
    page_table_NP: jax.Array
    q_pos_N: jax.Array
    kv_len_N: jax.Array
    block_q: int
    num_work: int


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LayerCache:
    k_pages_GPSK: jax.Array
    v_pages_GPSK: jax.Array
    batch: PagedBatch

    def write(self, k_BTGK: jax.Array, v_BTGK: jax.Array) -> "LayerCache":
        G, P, S, K = self.k_pages_GPSK.shape
        src_W, slot_W = self.batch.write_src_W, self.batch.write_slot_W

        def put(pages_GPSK, x_BTGK):
            x_GWK = jnp.swapaxes(x_BTGK.reshape(-1, G, K)[src_W], 0, 1).astype(pages_GPSK.dtype)
            pages_GTK = pages_GPSK.reshape(G, P * S, K).at[:, slot_W].set(
                x_GWK, mode="drop", unique_indices=True
            )
            return pages_GTK.reshape(G, P, S, K)

        return dataclasses.replace(
            self,
            k_pages_GPSK=put(self.k_pages_GPSK, k_BTGK),
            v_pages_GPSK=put(self.v_pages_GPSK, v_BTGK),
        )

    def attend(self, q_BTHK: jax.Array, scale: float) -> jax.Array:
        B, T, H, K = q_BTHK.shape
        Q = self.batch.block_q
        out_NQHK = paged_attention(
            q_BTHK.reshape(B * T // Q, Q, H, K),
            self.k_pages_GPSK,
            self.v_pages_GPSK,
            self.batch.page_table_NP,
            self.batch.q_pos_N,
            self.batch.kv_len_N,
            scale=scale,
            num_work=self.batch.num_work,
            # Tuned on H100: decode is bandwidth-bound, extend (Q x R = 128 rows) compute-bound.
            min_chunk_pages=16 if Q == 1 else 64,
            num_stages=3 if Q == 1 else 2,
        )
        return out_NQHK.reshape(B, T, H, K)


Cache: TypeAlias = list[LayerCache]
