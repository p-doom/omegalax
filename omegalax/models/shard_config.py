import dataclasses
from typing import TypeAlias

from jax.sharding import Mesh, PartitionSpec

P = PartitionSpec
ShardingSpec: TypeAlias = PartitionSpec

# Logical (logical_name, device_axis_name) rules for nnx.logical_axis_rules();
# None = replicated. The physical ICI/DCN placement of these axes is topology-aware
# (see omegalax.distributed.mesh) but the names here are unaffected.
DEFAULT_AXIS_RULES: tuple[tuple[str, str | None], ...] = (
    ("batch", ("dp", "fsdp")),
    ("seq", "cp"),  # context parallelism; dropped by _filter_axis at cp_size == 1
    ("vocab", "tp"),
    ("embed", "fsdp"),
    ("hidden", None),
    ("heads", "tp"),
    ("kv_heads", "tp"),
    ("mlp", "tp"),
    ("experts", None),  # for EP, add an "expert" mesh axis and map here
)


def _filter_axis(axis, mesh: Mesh):
    """Drop mesh axes with size 1 or absent from the mesh from an axis spec.

    An absent axis (e.g. ``"cp"`` on a plain ``('tp','fsdp','dp')`` mesh) is treated
    like a size-1 axis and dropped -- the "strict no-op when cp_size == 1"
    guarantee, and it also avoids a ``KeyError`` from ``mesh.shape[a]``."""
    if axis is None:
        return None
    axes = (axis,) if isinstance(axis, str) else axis
    kept = tuple(a for a in axes if a in mesh.shape and mesh.shape[a] > 1)
    return kept[0] if len(kept) == 1 else (kept or None)


def axis_rules_for_mesh(mesh: Mesh) -> tuple[tuple[str, str | None], ...]:
    """Drop rules for mesh axes with size 1 (replicate instead of shard)."""
    return tuple((logical, _filter_axis(axis, mesh)) for logical, axis in DEFAULT_AXIS_RULES)


@dataclasses.dataclass(slots=True, frozen=True)
class ShardConfig:
    """Activation sharding layout for forward passes (device-axis PartitionSpecs).

    The T (sequence) axis of the ``act_bt*`` specs carries context parallelism:
    :meth:`context_parallel` shards it on ``"cp"``, which :func:`shard_config_for_mesh`
    drops at cp_size == 1 (collapsing to :meth:`default`).
    """

    act_btd: ShardingSpec
    act_btf: ShardingSpec
    act_btnh: ShardingSpec

    @property
    def logits_btv(self) -> ShardingSpec:
        """Logits (batch, seq, vocab) sharding: batch + seq from ``act_btd`` (so CP
        seq sharding carries into the logits einsum), vocab (TP) from ``act_btf``."""
        return P(self.act_btd[0], self.act_btd[1], self.act_btf[2])

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

    @staticmethod
    def context_parallel():
        """Like :meth:`default` but with the T (sequence) axis sharded on ``"cp"``
        (composed with TP head sharding). Strict no-op at cp_size == 1."""
        return ShardConfig(
            act_btd=P(("dp", "fsdp"), "cp", None),
            act_btf=P(("dp", "fsdp"), "cp", "tp"),
            act_btnh=P(("dp", "fsdp"), "cp", "tp", None),
        )


def shard_config_for_mesh(shd_cfg: ShardConfig, mesh: Mesh) -> ShardConfig:
    """Drop mesh axes with size 1 from all partition specs in a sharding config."""
    return dataclasses.replace(
        shd_cfg,
        **{
            field.name: P(*(_filter_axis(axis, mesh) for axis in getattr(shd_cfg, field.name)))
            for field in dataclasses.fields(shd_cfg)
        },
    )
