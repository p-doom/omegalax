"""KV-cache allocation for Qwen3.5 full-attention layers."""

from __future__ import annotations

from typing import TypeAlias

from flax import nnx
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard

from .config import Qwen3_5TextConfig


class LayerCache(nnx.Module):
    def __init__(
        self,
        cfg: Qwen3_5TextConfig,
        batch_size: int,
        cache_size: int,
        dtype: jnp.dtype,
    ):
        shape = (batch_size, cache_size, cfg.num_key_value_heads, cfg.head_dim)
        self.k_cache = nnx.Cache(reshard(jnp.zeros(shape, dtype=dtype), cfg.shd_cfg.act_btnh))
        self.v_cache = nnx.Cache(reshard(jnp.zeros(shape, dtype=dtype), cfg.shd_cfg.act_btnh))
        self.size = cache_size
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32), sharding=P())


Cache: TypeAlias = list[LayerCache | None]


def init_cache(
    cfg: Qwen3_5TextConfig,
    batch_size: int,
    cache_size: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Cache:
    return [
        LayerCache(cfg, batch_size, cache_size, dtype) if layer_type == "full_attention" else None
        for layer_type in cfg.layer_types
    ]


def reset_cache(cache: Cache) -> None:
    for layer_cache in cache:
        if layer_cache is not None:
            layer_cache.k_cache[...] = jnp.zeros_like(layer_cache.k_cache[...])
            layer_cache.v_cache[...] = jnp.zeros_like(layer_cache.v_cache[...])
            layer_cache.cur_ind[...] = jnp.array(0, dtype=jnp.int32)


def carry_cache(cache: Cache, carry_length: int) -> int:
    carried_tokens = None
    for layer_cache in cache:
        if layer_cache is None:
            continue
        valid_tokens = int(layer_cache.cur_ind[...])
        layer_carried_tokens = min(carry_length, valid_tokens)
        if carried_tokens is None:
            carried_tokens = layer_carried_tokens
        elif layer_carried_tokens != carried_tokens:
            raise ValueError("Full-attention cache lengths diverged")
        start = valid_tokens - layer_carried_tokens
        carried_k = layer_cache.k_cache[:, start:valid_tokens]
        carried_v = layer_cache.v_cache[:, start:valid_tokens]
        layer_cache.k_cache[...] = (
            jnp.zeros_like(layer_cache.k_cache[...]).at[:, :layer_carried_tokens].set(carried_k)
        )
        layer_cache.v_cache[...] = (
            jnp.zeros_like(layer_cache.v_cache[...]).at[:, :layer_carried_tokens].set(carried_v)
        )
        layer_cache.cur_ind[...] = jnp.array(layer_carried_tokens, dtype=jnp.int32)
    return 0 if carried_tokens is None else carried_tokens
