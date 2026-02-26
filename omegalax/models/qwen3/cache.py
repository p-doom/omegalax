import math
from typing import TypeAlias

import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P, reshard


Cache: TypeAlias = list["LayerCache"]


class LayerCache(nnx.Module):
    def __init__(self, cfg, batch_size: int, cache_size: int, dtype: jnp.dtype):
        cache_shape = (batch_size, cache_size, cfg.num_kv_heads, cfg.head_dim)
        self.k_cache = nnx.Cache(reshard(jnp.zeros(cache_shape, dtype=dtype), cfg.shd_cfg.act_btnh))
        self.v_cache = nnx.Cache(reshard(jnp.zeros(cache_shape, dtype=dtype), cfg.shd_cfg.act_btnh))
        self.size = self.k_cache.shape[1]
        batch_sharding = P(cfg.shd_cfg.act_btnh[0]) if cfg.shd_cfg.act_btnh else P(None)
        self.start_ind = nnx.Variable(reshard(-1 * jnp.ones((batch_size,), dtype=jnp.int32), batch_sharding))
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))


def init_cache(cfg, batch_size: int, token_len: int, generate_steps: int, dtype: jnp.dtype = jnp.bfloat16) -> Cache:
    cache_size = 2 ** math.ceil(math.log2(max(token_len + generate_steps, 1)))
    return [LayerCache(cfg, batch_size, cache_size, dtype) for _ in range(cfg.num_layers)]
