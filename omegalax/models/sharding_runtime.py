from __future__ import annotations

from typing import Any

from flax import nnx
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from omegalax.models.shard_config import ShardConfig

P = PartitionSpec


def init_model_sharded(
    model_cls: type[nnx.Module],
    cfg: Any,
    rng: jax.Array,
    mesh: Mesh,
    axis_rules: tuple[tuple[str, str | None], ...],
) -> nnx.Module:
    """Create a model with params born sharded. jax.jit is mandatory to avoid
    materializing a full unsharded copy (OOM for large models).

    When fp8 training is active (``cfg.fp8`` and the host is Hopper -- see
    ``omegalax.quant.detect.fp8_active``) the freshly-built model is wrapped
    with qwix so the compute-bound GEMMs run in fp8. This is a strict no-op
    (returns the model unchanged) when fp8 is off or the host is not Hopper.
    """
    with jax.set_mesh(mesh), nnx.logical_axis_rules(axis_rules):
        model = jax.jit(lambda rng: model_cls(cfg, rngs=nnx.Rngs(rng)))(rng)
    _finalize_q_shardings(model, mesh)
    model = _maybe_wrap_fp8(model, cfg, mesh)
    return model


def _maybe_wrap_fp8(model: nnx.Module, cfg: Any, mesh: Mesh) -> nnx.Module:
    """Apply the Hopper-gated fp8 quantization wrap. No-op unless fp8 is active.

    Imported lazily so the quant package (and qwix) is only touched when a
    model is actually built, keeping import graphs and non-fp8 runs untouched.
    """
    from omegalax.quant.apply import maybe_quantize_fp8

    return maybe_quantize_fp8(model, cfg, mesh=mesh)


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
        if key == "position_ids_ZBT" and arr.ndim == 3:
            spec = P(None, batch_axis, None)
        else:
            spec = P(batch_axis, *((None,) * (arr.ndim - 1)))
        sharding = NamedSharding(mesh, spec)
        result[key] = jax.make_array_from_process_local_data(sharding, arr)
    return result
