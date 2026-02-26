import dataclasses
from typing import Any

import jax.numpy as jnp
from omegalax.models.shard_config import ShardConfig


@dataclasses.dataclass(frozen=True, slots=True)
class Qwen3Config:
    """Shared Qwen3 configuration."""

    variant: str  # "dense" or "moe"
    num_layers: int
    vocab_size: int
    emb_dim: int
    mlp_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    rope_theta: int
    rope_scaling_factor: float | None
    local_rope_theta: float | None
    norm_eps: float
    tie_word_embeddings: bool
    shd_cfg: ShardConfig = dataclasses.field(default_factory=ShardConfig.default)
    dtype: Any = jnp.bfloat16

    @classmethod
    def with_sharding(cls, **kwargs):
        kwargs["shd_cfg"] = ShardConfig.default()
        return cls(**kwargs)
