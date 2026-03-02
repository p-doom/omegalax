import dataclasses
from typing import TypeAlias

from jax.sharding import Mesh, PartitionSpec

P = PartitionSpec
ShardingSpec: TypeAlias = PartitionSpec

# Logical axis names for parameter sharding, mapped to device mesh axes.
# Used with nnx.logical_axis_rules() so model code can annotate with semantic names.
# Tuple of (logical_name, device_axis_name); None = replicated.
DEFAULT_AXIS_RULES: tuple[tuple[str, str | None], ...] = (
    ("batch", "fsdp"),
    ("vocab", "tp"),
    ("embed", "fsdp"),
    ("hidden", "tp"),
    ("heads", "tp"),
    ("kv_heads", "tp"),
    ("mlp", "tp"),
    # FIXME (f.srambical)
    ("experts", None),
)


def axis_rules_for_mesh(mesh: Mesh) -> tuple[tuple[str, str | None], ...]:
    """Drop rules for mesh axes with size 1 (replicate instead of shard)."""
    return tuple(
        (logical, axis if axis is not None and int(mesh.shape[axis]) > 1 else None)
        for logical, axis in DEFAULT_AXIS_RULES
    )


@dataclasses.dataclass(slots=True, frozen=True)
class ShardConfig:
    """Activation sharding layout for forward passes (device-axis PartitionSpecs)."""

    act_btd: ShardingSpec
    act_btf: ShardingSpec
    act_btnh: ShardingSpec

    @staticmethod
    def no_sharding():
        """Configuration with no sharding (all None)."""
        return ShardConfig(
            act_btd=P(None, None, None),
            act_btf=P(None, None, None),
            act_btnh=P(None, None, None, None),
        )

    @staticmethod
    def default():
        return ShardConfig(
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
