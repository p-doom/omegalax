from __future__ import annotations

from typing import Any

from flax import nnx
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from omegalax.models.shard_config import ShardConfig

P = PartitionSpec

_REPLICATED_BATCH_KEYS = frozenset(
    {
        # Qwen3-VL vision tensors are concatenated over images/patches, not
        # shaped as (B, ...), so their leading axis is not data parallel.
        "pixel_values",
        "image_grid_thw",
        "vision_cu_seqlens",
    }
)


def init_model_sharded(
    model_cls: type[nnx.Module],
    cfg: Any,
    rng: jax.Array,
    mesh: Mesh,
    axis_rules: tuple[tuple[str, str | None], ...],
) -> nnx.Module:
    """Create a model with params born sharded. jax.jit is mandatory to avoid
    materializing a full unsharded copy (OOM for large models)."""
    with jax.set_mesh(mesh), nnx.logical_axis_rules(axis_rules):
        model = jax.jit(lambda rng: model_cls(cfg, rngs=nnx.Rngs(rng)))(rng)
    _finalize_q_shardings(model, mesh)
    return model


def _finalize_q_shardings(model: nnx.Module, mesh: Mesh) -> None:
    """Convert ``_q_sharding_spec`` stored during ``__init__`` into ``NamedSharding``.

    Modules set ``_q_sharding_spec`` in ``__init__`` (which runs inside
    ``jax.jit``), but ``NamedSharding`` requires a concrete ``Mesh`` that is
    only available outside ``jax.jit``.  This function bridges the gap.
    """
    for _, module in nnx.iter_modules(model):
        spec = getattr(module, "_q_sharding_spec", None)
        if spec is not None:
            object.__setattr__(module, "_q_sharding", NamedSharding(mesh, spec))


def set_attn_backend(
    model: nnx.Module,
    text_backend: str = "mosaic_gpu",
) -> None:
    """Set ``_attn_backend`` on every text attention sub-module."""

    for _, module in nnx.iter_modules(model):
        if getattr(module, "_attn_kind", None) == "text":
            object.__setattr__(module, "_attn_backend", text_backend)


def batch_partition_spec(shd_cfg: ShardConfig) -> PartitionSpec:
    return P(shd_cfg.act_btd[0], None)


def shard_batch(token_ids_BT: jax.Array, shd_cfg: ShardConfig, mesh: Mesh) -> jax.Array:
    sharding = NamedSharding(mesh, batch_partition_spec(shd_cfg))
    return jax.make_array_from_process_local_data(sharding, token_ids_BT)


def shard_batch_dict(
    batch: dict[str, Any],
    shd_cfg: ShardConfig,
    mesh: Mesh,
) -> dict[str, jax.Array]:
    """Shard every array in a batch dict: batch dim sharded, rest replicated.

    Note: ``position_ids_ZBT`` has shape ``(3, B, T)`` and the batch dim is axis 1,
    """
    batch_axis = shd_cfg.act_btd[0]
    result = {}
    for key, arr in batch.items():
        if key in _REPLICATED_BATCH_KEYS:
            if jax.process_count() > 1 and batch_axis is not None:
                spec = P(batch_axis, *((None,) * (arr.ndim - 1)))
            else:
                spec = P(*((None,) * arr.ndim))
        elif key == "position_ids_ZBT" and arr.ndim == 3:
            spec = P(None, batch_axis, None)
        else:
            spec = P(batch_axis, *((None,) * (arr.ndim - 1)))
        sharding = NamedSharding(mesh, spec)
        result[key] = jax.make_array_from_process_local_data(sharding, arr)
    return result
