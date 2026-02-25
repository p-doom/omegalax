import dataclasses
from typing import Any, TypeAlias

import jax.numpy as jnp
from jax.sharding import PartitionSpec

P = PartitionSpec
ShardingSpec: TypeAlias = PartitionSpec


@dataclasses.dataclass(slots=True, frozen=True)
class ShardConfig:
    """Sharding layout for Qwen3. Kept minimal and shared across variants."""

    emb_vd: ShardingSpec
    emb_dv: ShardingSpec
    q_weight_ndh: ShardingSpec
    kv_weight_ndh: ShardingSpec
    o_weight_nhd: ShardingSpec
    ffw_weight_df: ShardingSpec
    ffw_weight_fd: ShardingSpec
    rms_norm: ShardingSpec
    act_btd: ShardingSpec
    act_btf: ShardingSpec
    act_btnh: ShardingSpec

    @staticmethod
    def no_sharding():
        """Configuration with no sharding (all None)."""
        return ShardConfig(
            emb_vd=P(None, None),
            emb_dv=P(None, None),
            q_weight_ndh=P(None, None, None),
            kv_weight_ndh=P(None, None, None),
            o_weight_nhd=P(None, None, None),
            ffw_weight_df=P(None, None),
            ffw_weight_fd=P(None, None),
            rms_norm=P(None),
            act_btd=P(None, None, None),
            act_btf=P(None, None, None),
            act_btnh=P(None, None, None, None),
        )

    @staticmethod
    def default():
        return ShardConfig(
            emb_vd=P("tp", "fsdp"),
            emb_dv=P("fsdp", "tp"),
            q_weight_ndh=P("tp", "fsdp", None),
            kv_weight_ndh=P("tp", "fsdp", None),
            o_weight_nhd=P("tp", None, "fsdp"),
            ffw_weight_df=P("fsdp", "tp"),
            ffw_weight_fd=P("tp", "fsdp"),
            rms_norm=P("tp"),
            act_btd=P("fsdp", None, "tp"),
            act_btf=P("fsdp", None, "tp"),
            act_btnh=P("fsdp", None, "tp", None),
        )


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
    shd_cfg: ShardConfig = dataclasses.field(default_factory=ShardConfig.no_sharding)
    dtype: Any = jnp.bfloat16

    @classmethod
    def with_sharding(cls, use_sharding: bool, **kwargs):
        if use_sharding:
            kwargs["shd_cfg"] = ShardConfig.default()
        return cls(**kwargs)
