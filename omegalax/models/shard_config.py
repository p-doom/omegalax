import dataclasses
from typing import TypeAlias

from jax.sharding import Mesh, PartitionSpec

P = PartitionSpec
ShardingSpec: TypeAlias = PartitionSpec


@dataclasses.dataclass(slots=True, frozen=True)
class ShardConfig:
    """Shared sharding layout used across model families."""

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


def shard_config_for_mesh(shd_cfg: ShardConfig, mesh: Mesh) -> ShardConfig:
    """Drop mesh axes with size 1 from all partition specs in a sharding config."""
    return dataclasses.replace(
        shd_cfg,
        **{
            field.name: P(
                *(
                    axis if axis is not None and int(mesh.shape[axis]) > 1 else None
                    for axis in getattr(shd_cfg, field.name)
                )
            )
            for field in dataclasses.fields(shd_cfg)
        },
    )
