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

    When the config requests fp8 (``cfg.fp8`` / recipe != ``off``) the freshly-built
    model is wrapped with qwix so the compute-bound GEMMs run in fp8; this asserts
    a Hopper host. No-op (returns the model unchanged) when fp8 is off.
    """
    with jax.set_mesh(mesh), nnx.logical_axis_rules(axis_rules):
        model = jax.jit(lambda rng: model_cls(cfg, rngs=nnx.Rngs(rng)))(rng)
    _finalize_q_shardings(model, mesh)
    model = _maybe_wrap_fp8(model, cfg, mesh)
    return model


def _maybe_wrap_fp8(model: nnx.Module, cfg: Any, mesh: Mesh) -> nnx.Module:
    """Apply the fp8 quantization wrap (asserts Hopper). No-op unless fp8 is requested.

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


def set_cp_document_mask(model: nnx.Module, enabled: bool = True) -> None:
    """Enable/disable the CP block-diagonal document mask on text attention.

    When enabled AND running under context parallelism (cp_size > 1), the
    all-gather-KV attention adds ``q_seg == k_seg`` (over globally-gathered
    segment ids) so packed multi-document sequences never attend across a
    document boundary. Default (disabled) keeps CP causal-only, matching the
    non-CP tokamax path. Only affects the CP path; non-CP attention is
    unchanged either way.
    """
    for _, module in nnx.iter_modules(model):
        if getattr(module, "_attn_kind", None) == "text":
            object.__setattr__(module, "_cp_document_mask", enabled)


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
    cp_load_balance: bool = True,
) -> dict[str, jax.Array]:
    """Shard every array in a batch dict.

    Batch dim sharded on ``act_btd[0]`` (dp/fsdp). For context parallelism the
    token (T) axis of the per-token arrays is additionally sharded on ``"cp"``
    (``act_btd[1]``):

      * ``(B, T)`` per-token arrays (``token_ids_BT``, ``targets_BT``,
        ``segment_ids_BT``, ``loss_mask_BT``, ``attention_mask_BT``,
        ``position_ids_BT``) -> ``P(batch_axis, seq_axis)``.
      * ``position_ids_ZBT`` has shape ``(3, B, T)`` (batch is axis 1, T axis 2)
        -> ``P(None, batch_axis, seq_axis)``.
      * everything else -> batch dim sharded, rest replicated.

    ``seq_axis`` is ``shd_cfg.act_btd[1]`` which is ``"cp"`` under a CP config and
    ``None`` otherwise, so at cp_size == 1 this is a strict no-op. When CP is
    active:
      * the next-token shift is applied shift-BEFORE-shard (see
        :func:`_cp_shift_before_shard`), adding a ``targets_BT`` key; and
      * if ``cp_load_balance`` (default), the T axis is ZIG-ZAG permuted (Stage
        1b) BEFORE sharding so contiguous cp-sharding of the permuted sequence
        gives each rank a balanced chunk pair (see :mod:`omegalax.distributed.\
        zigzag`). The permutation is numerically invisible because CP attention
        masks over global positions; a ``position_ids_BT`` carrying each token's
        original index is added so RoPE / the mask use true positions.
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
