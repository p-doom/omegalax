import dataclasses
from typing import TypeAlias

from jax.sharding import Mesh, PartitionSpec

P = PartitionSpec
ShardingSpec: TypeAlias = PartitionSpec

# Logical axis names for parameter sharding, mapped to device mesh axes.
# Used with nnx.logical_axis_rules() so model code can annotate with semantic names.
# Tuple of (logical_name, device_axis_name); None = replicated.
#
# Physical placement (see omegalax.distributed.mesh): the mesh axes are laid out
# hierarchically -- "tp" (and part of "fsdp") ride the intra-node NVLink domain
# (ICI), while "dp" (and any "fsdp" spill) ride the inter-node data-center
# network (DCN). The axis *names* below are unchanged; only device placement is
# topology-aware, so these rules and all downstream NamedShardings are unaffected.
DEFAULT_AXIS_RULES: tuple[tuple[str, str | None], ...] = (
    ("batch", ("dp", "fsdp")),
    # Context / sequence parallelism: the token (T) axis is sharded on "cp".
    # When cp_size == 1 this rule is dropped by _filter_axis (strict no-op), so
    # the T axis is replicated exactly as before. See omegalax.distributed.mesh.
    ("seq", "cp"),
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
    return tuple((logical, _filter_axis(axis, mesh)) for logical, axis in DEFAULT_AXIS_RULES)


@dataclasses.dataclass(slots=True, frozen=True)
class ShardConfig:
    """Activation sharding layout for forward passes (device-axis PartitionSpecs).

    The T (token / sequence) axis of the ``act_bt*`` specs is where context
    parallelism (CP) lives: :meth:`context_parallel` shards it on ``"cp"``. When
    ``cp_size == 1`` the ``"cp"`` entry is dropped by :func:`shard_config_for_mesh`
    (via :func:`_filter_axis`), so a CP config collapses to exactly the non-CP
    ``default`` layout -- CP is a strict no-op at cp_size==1.
    """

    act_btd: ShardingSpec
    act_btf: ShardingSpec
    act_btnh: ShardingSpec

    @property
    def logits_btv(self) -> ShardingSpec:
        """Logits sharding: (batch, seq, vocab).

        batch from ``act_btd`` (axis 0), seq (T) from ``act_btd`` (axis 1) so the
        CP sequence sharding carries through to the logits einsum, and vocab from
        ``act_btf`` (TP). Under the non-CP ``default``/``no_sharding`` configs the
        seq axis is ``None`` (unchanged); under a CP config it is ``"cp"``.
        """
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
        """Sharding layout with the T (sequence) axis sharded on ``"cp"``.

        Identical to :meth:`default` except the token axis (axis 1 of the
        ``act_bt*`` specs) carries the ``"cp"`` mesh axis, so activations are
        sequence-sharded for context parallelism. Composed with TP head sharding
        (``"tp"`` on the head axis) as usual. At ``cp_size == 1`` this is a strict
        no-op (the ``"cp"`` axis is filtered out and it becomes :meth:`default`).
        """
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
