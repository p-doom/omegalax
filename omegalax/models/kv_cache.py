"""Paged KV cache threaded through attention layers by the rollout engine."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from omegalax.kernels.paged_attention import paged_attention
from omegalax.kernels.paged_extend_attention import TILE as EXTEND_TILE
from omegalax.kernels.paged_extend_attention import paged_extend_attention
from omegalax.models.sharding_runtime import per_shard

# Pages hold the KV heads of omegalax's tensor-parallel mesh axis.
_PAGES = P("tp", None, None, None)
_HEADS = P(None, None, "tp", None)


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "positions_BT",
        "write_src_W",
        "write_slot_W",
        "page_table_NP",
        "q_pos_N",
        "kv_len_N",
    ],
    meta_fields=["block_q", "num_work"],
)
@dataclasses.dataclass(frozen=True)
class PagedBatch:
    """Token placement shared by all layers; see ``omegalax.kernels.paged_attention``.

    KV write ``w`` stores flattened token ``write_src_W[w]`` at ``write_slot_W[w]`` (``page *
    page_size + offset``). A token may be written to several slots (forked prompts); slots
    are unique, and writes to skip carry distinct out-of-range slots. ``num_work`` sizes the
    decode kernel's work list; extends do not split work.
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
        def put(pages_GPSK, x_BTGK, src_W, slot_W):
            G, _, _, K = pages_GPSK.shape
            x_GWK = jnp.swapaxes(x_BTGK.reshape(-1, G, K)[src_W], 0, 1).astype(pages_GPSK.dtype)
            pages_GTK = (
                pages_GPSK.reshape(G, -1, K)
                .at[:, slot_W]
                .set(x_GWK, mode="drop", unique_indices=True)
            )
            return pages_GTK.reshape(pages_GPSK.shape)

        def put_kv(k_pages, v_pages, k, v, src_W, slot_W):
            return put(k_pages, k, src_W, slot_W), put(v_pages, v, src_W, slot_W)

        k_pages, v_pages = per_shard(
            put_kv,
            (_PAGES, _PAGES, _HEADS, _HEADS, P(), P()),
            (_PAGES, _PAGES),
            self.k_pages_GPSK,
            self.v_pages_GPSK,
            k_BTGK,
            v_BTGK,
            self.batch.write_src_W,
            self.batch.write_slot_W,
        )
        return dataclasses.replace(self, k_pages_GPSK=k_pages, v_pages_GPSK=v_pages)

    def attend(self, q_BTHK: jax.Array, scale: float) -> jax.Array:
        """Decode steps (one token per query block) use the split-KV Triton kernel; extends
        (``EXTEND_TILE`` tokens per query block) the Mosaic GPU kernel."""
        Q, num_work = self.batch.block_q, self.batch.num_work

        def decode(q_BTHK, k_pages, v_pages, page_table_NP, q_pos_N, kv_len_N):
            B, T, H, K = q_BTHK.shape
            out_NQHK = paged_attention(
                q_BTHK.reshape(B * T, 1, H, K),
                k_pages,
                v_pages,
                page_table_NP,
                q_pos_N,
                kv_len_N,
                scale=scale,
                num_work=num_work,
            )
            return out_NQHK.reshape(B, T, H, K)

        def extend(q_BTHK, k_pages, v_pages, page_table_NP, q_pos_N, kv_len_N):
            B, T, H, K = q_BTHK.shape
            out_THK = paged_extend_attention(
                q_BTHK.reshape(B * T, H, K),
                k_pages,
                v_pages,
                page_table_NP,
                q_pos_N,
                kv_len_N,
                scale=scale,
            )
            return out_THK.reshape(B, T, H, K)

        if Q not in (1, EXTEND_TILE):
            raise ValueError(f"Query blocks must hold 1 or {EXTEND_TILE} tokens, got {Q}.")
        return per_shard(
            decode if Q == 1 else extend,
            (_HEADS, _PAGES, _PAGES, P(), P(), P()),
            _HEADS,
            q_BTHK,
            self.k_pages_GPSK,
            self.v_pages_GPSK,
            self.batch.page_table_NP,
            self.batch.q_pos_N,
            self.batch.kv_len_N,
        )
