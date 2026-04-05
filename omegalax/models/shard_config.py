import dataclasses
from typing import TypeAlias

from jax.sharding import Mesh, PartitionSpec

P = PartitionSpec
ShardingSpec: TypeAlias = PartitionSpec

# Logical axis names for parameter sharding, mapped to device mesh axes.
# Used with nnx.logical_axis_rules() so model code can annotate with semantic names.
# Tuple of (logical_name, device_axis_name); None = replicated.
DEFAULT_AXIS_RULES: tuple[tuple[str, str | None], ...] = (
    ("batch", ("dp", "fsdp")),
    ("vocab", "tp"),
    ("embed", "fsdp"),
    ("hidden", None),
    ("heads", "tp"),
    ("kv_heads", "tp"),
    ("mlp", "tp"),
    # Experts replicated; TP on F and FSDP on D within each expert.
    # For expert parallelism, add an "expert" mesh axis and map here.
    ("experts", None),
)


def _filter_axis(axis, mesh: Mesh):
    """Drop mesh axes with size 1 from a single axis spec or tuple of axis specs."""
    if axis is None:
        return None
    axes = (axis,) if isinstance(axis, str) else axis
    kept = tuple(a for a in axes if mesh.shape[a] > 1)
    return kept[0] if len(kept) == 1 else (kept or None)


def axis_rules_for_mesh(mesh: Mesh) -> tuple[tuple[str, str | None], ...]:
    """Drop rules for mesh axes with size 1 (replicate instead of shard)."""
    return tuple(
        (logical, _filter_axis(axis, mesh))
        for logical, axis in DEFAULT_AXIS_RULES
    )


@dataclasses.dataclass(slots=True, frozen=True)
class ShardConfig:
    """Activation sharding layout for forward passes (device-axis PartitionSpecs)."""

    act_btd: ShardingSpec
    act_btf: ShardingSpec
    act_btnh: ShardingSpec

    @property
    def logits_btv(self) -> ShardingSpec:
        """Logits sharding: (batch, seq, vocab), batch from act_btd, vocab from act_btf (TP)."""
        return P(self.act_btd[0], None, self.act_btf[2])

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
            act_btd=P(("dp", "fsdp"), None, None),
            act_btf=P(("dp", "fsdp"), None, "tp"),
            act_btnh=P(("dp", "fsdp"), None, "tp", None),
        )


def shard_config_for_mesh(shd_cfg: ShardConfig, mesh: Mesh) -> ShardConfig:
    """Drop mesh axes with size 1 from all partition specs in a sharding config."""
    return dataclasses.replace(
        shd_cfg,
        **{
            field.name: P(
                *(
                    _filter_axis(axis, mesh)
                    for axis in getattr(shd_cfg, field.name)
                )
            )
            for field in dataclasses.fields(shd_cfg)
        },
    )
