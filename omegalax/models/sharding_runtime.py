from __future__ import annotations

from typing import Any

from flax import nnx
import jax
import numpy as np
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


# Per-token batch arrays whose token (T) axis should be sharded on "cp" for
# context parallelism. These are the (B, T) integer arrays consumed per-token by
# the model / loss. The T axis carries the CP sequence shard (see
# ShardConfig.context_parallel). When cp_size == 1, ``seq_axis`` is None (dropped
# by shard_config_for_mesh) and these collapse to plain batch-dim sharding -- a
# strict no-op vs the pre-CP behavior.
_SEQ_SHARDED_KEYS = frozenset(
    {"token_ids_BT", "targets_BT", "segment_ids_BT", "loss_mask_BT", "attention_mask_BT"}
)


def _cp_shift_before_shard(batch: dict[str, Any]) -> dict[str, Any]:
    """Apply the next-token shift on the UNSHARDED host batch (shift-before-shard).

    The next-token +1 shift crosses cp shard boundaries, so under CP it must be
    applied on the full sequence BEFORE it is sequence-sharded (see
    :mod:`omegalax.distributed.mesh` and :func:`omegalax.trainers.loss.\
    shift_for_next_token`). We do it here, in numpy, on the pre-shard arrays,
    adding a ``targets_BT`` key (``target[t] = token[t+1]``) and shifting
    ``loss_mask_BT`` left by one (final position forced to 0 -- no next token).
    ``token_ids_BT`` is left intact as the model input. The loss then consumes
    ``targets_BT`` / ``loss_mask_BT`` position-aligned (``shift=False``) with no
    further cross-shard shift.

    No-op if ``token_ids_BT`` / ``loss_mask_BT`` are absent (e.g. VLM batches or
    already-prepared inputs). Idempotent-safe: only runs when ``targets_BT`` is
    not already present.
    """
    if "token_ids_BT" not in batch or "targets_BT" in batch:
        return batch
    out = dict(batch)
    tokens = np.asarray(batch["token_ids_BT"])
    out["targets_BT"] = np.roll(tokens, shift=-1, axis=1)
    if "loss_mask_BT" in batch:
        mask = np.asarray(batch["loss_mask_BT"])
        mask = np.roll(mask, shift=-1, axis=1)
        mask[:, -1] = 0
        out["loss_mask_BT"] = mask
    return out


def shard_batch_dict(
    batch: dict[str, Any],
    shd_cfg: ShardConfig,
    mesh: Mesh,
) -> dict[str, jax.Array]:
    """Shard every array in a batch dict.

    Batch dim sharded on ``act_btd[0]`` (dp/fsdp). For context parallelism the
    token (T) axis of the per-token arrays is additionally sharded on ``"cp"``
    (``act_btd[1]``):

      * ``(B, T)`` per-token arrays (``token_ids_BT``, ``targets_BT``,
        ``segment_ids_BT``, ``loss_mask_BT``, ``attention_mask_BT``) ->
        ``P(batch_axis, seq_axis)``.
      * ``position_ids_ZBT`` has shape ``(3, B, T)`` (batch is axis 1, T axis 2)
        -> ``P(None, batch_axis, seq_axis)``.
      * everything else -> batch dim sharded, rest replicated.

    ``seq_axis`` is ``shd_cfg.act_btd[1]`` which is ``"cp"`` under a CP config and
    ``None`` otherwise, so at cp_size == 1 this is a strict no-op. When CP is
    active, the next-token shift is applied shift-BEFORE-shard here (see
    :func:`_cp_shift_before_shard`), adding a ``targets_BT`` key.
    """
    batch_axis = shd_cfg.act_btd[0]
    seq_axis = shd_cfg.act_btd[1]
    if seq_axis is not None:
        batch = _cp_shift_before_shard(batch)
    result = {}
    for key, arr in batch.items():
        if key == "position_ids_ZBT" and arr.ndim == 3:
            spec = P(None, batch_axis, seq_axis)
        elif key in _SEQ_SHARDED_KEYS and arr.ndim >= 2:
            spec = P(batch_axis, seq_axis, *((None,) * (arr.ndim - 2)))
        else:
            spec = P(batch_axis, *((None,) * (arr.ndim - 1)))
        sharding = NamedSharding(mesh, spec)
        result[key] = jax.make_array_from_process_local_data(sharding, arr)
    return result
