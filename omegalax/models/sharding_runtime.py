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
    materializing a full unsharded copy (OOM for large models).
    """
    with jax.set_mesh(mesh), nnx.logical_axis_rules(axis_rules):
        model = jax.jit(lambda rng: model_cls(cfg, rngs=nnx.Rngs(rng)))(rng)
    _finalize_q_shardings(model, mesh)
    return model


def _finalize_q_shardings(model: nnx.Module, mesh: Mesh) -> None:
    """Convert each module's ``_q_sharding_spec`` (set in ``__init__`` under
    ``jax.jit``) into a ``NamedSharding`` (needs the concrete ``Mesh``, only
    available outside jit)."""
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


def set_cp_document_mask(model: nnx.Module, enabled: bool = True) -> None:
    """Enable/disable the CP block-diagonal document mask on text attention.

    When enabled AND under CP (cp_size > 1), the all-gather-KV attention adds
    ``q_seg == k_seg`` so packed sequences don't attend across a document boundary.
    Default disabled keeps CP causal-only, matching the (also causal-only) non-CP
    tokamax path. Only affects the CP path.
    """
    for _, module in nnx.iter_modules(model):
        if getattr(module, "_attn_kind", None) == "text":
            object.__setattr__(module, "_cp_document_mask", enabled)


def batch_partition_spec(shd_cfg: ShardConfig) -> PartitionSpec:
    return P(shd_cfg.act_btd[0], None)


def shard_batch(token_ids_BT: jax.Array, shd_cfg: ShardConfig, mesh: Mesh) -> jax.Array:
    sharding = NamedSharding(mesh, batch_partition_spec(shd_cfg))
    return jax.make_array_from_process_local_data(sharding, token_ids_BT)


# Per-token (B, T) arrays whose T axis is sharded on "cp" for context parallelism.
# At cp_size == 1 ``seq_axis`` is None, so these collapse to plain batch sharding.
_SEQ_SHARDED_KEYS = frozenset(
    {
        "token_ids_BT",
        "targets_BT",
        "segment_ids_BT",
        "loss_mask_BT",
        "attention_mask_BT",
        "position_ids_BT",
    }
)


def _infer_seq_len(batch: dict[str, Any]) -> int | None:
    """Length of the token (T) axis from the first per-token (B, T) array."""
    for key in ("token_ids_BT", "targets_BT", "segment_ids_BT", "loss_mask_BT"):
        arr = batch.get(key)
        if arr is not None and getattr(arr, "ndim", 0) >= 2:
            return int(np.asarray(arr).shape[1])
    return None


def _cp_shift_before_shard(batch: dict[str, Any]) -> dict[str, Any]:
    """Apply the next-token shift on the UNSHARDED host batch (shift-before-shard).

    The +1 shift crosses cp shard boundaries, so under CP it must be applied on the
    full sequence before sharding (see :func:`omegalax.trainers.loss.\
    shift_for_next_token`). Adds ``targets_BT`` (``target[t] = token[t+1]``) and
    rolls ``loss_mask_BT`` left by one (final position forced to 0); the loss then
    consumes them position-aligned (``shift=False``). No-op / idempotent when
    ``token_ids_BT`` is absent or ``targets_BT`` already present.
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
    cp_load_balance: bool = True,
) -> dict[str, jax.Array]:
    """Shard every array in a batch dict: batch dim on ``act_btd[0]`` (dp/fsdp),
    and per-token (B, T) arrays' T axis additionally on ``seq_axis`` (``act_btd[1]``,
    == "cp" under CP, else None -> strict no-op). ``position_ids_ZBT`` (3, B, T) gets
    ``P(None, batch, seq)``; everything else is batch-sharded only.

    When CP is active the next-token shift is applied shift-before-shard (see
    :func:`_cp_shift_before_shard`) and, if ``cp_load_balance``, the T axis is
    zig-zag permuted before sharding (numerically invisible -- CP attention masks
    over global positions; see :mod:`omegalax.distributed.zigzag`).
    """
    batch_axis = shd_cfg.act_btd[0]
    seq_axis = shd_cfg.act_btd[1]
    if seq_axis is not None:
        batch = _cp_shift_before_shard(batch)
        if cp_load_balance:
            from omegalax.distributed.zigzag import (
                apply_zigzag_to_batch,
                zigzag_permutation,
            )

            cp_size = int(mesh.shape[seq_axis])
            seq_len = _infer_seq_len(batch)
            if seq_len is not None:
                perm = zigzag_permutation(seq_len, cp_size)
                batch = apply_zigzag_to_batch(batch, perm, _SEQ_SHARDED_KEYS)
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
