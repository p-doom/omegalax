"""Text pretraining helpers for IID and fixed-window statepassing experiments."""

from __future__ import annotations

import dataclasses
import datetime
import enum
import gc
import math
import numpy as np
import subprocess
import time
from pathlib import Path
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
import optax

from omegalax.data.pretrain_data_set import pop_pretrain_metadata
from omegalax.distributed.mesh import ensure_mesh, mesh_rules, required_batch_multiple
from omegalax.text import api as text_api
from omegalax.trainers import checkpoint_utils
from omegalax.trainers.loss import (
    chunked_cross_entropy_multi_stats,
    chunked_cross_entropy_stats,
)
from omegalax.trainers.lr_schedule import build_lr_schedule
from omegalax.trainers.optim import MixedPrecisionOptimizer
from omegalax.trainers.perf import (
    maybe_log_step_metrics,
    per_device_flops_per_step,
    StepTimer,
)
from omegalax.trainers.text import (
    _make_checkpoint_manager,
    _restore_sft_checkpoint,
    _save_sft_checkpoint,
    _write_checkpoint_config,
    build_optimizer,
    init_model,
    startup_log,
    TrainConfig,
)

P = PartitionSpec
_NUM_LOSS_TILES = 4


class PretrainMode(enum.StrEnum):
    IID_BASELINE = "iid_baseline"
    STATEPASSING_NO_BPTT = "statepassing_no_bptt"
    STATEPASSING_BPTT = "statepassing_bptt"

    @property
    def is_statepassing(self) -> bool:
        return self is not PretrainMode.IID_BASELINE


@dataclasses.dataclass(frozen=True)
class StatepassingTargetMasks:
    total: jax.Array
    iid_comparable: jax.Array
    boundary: jax.Array
    segments: tuple[jax.Array, ...]


def _nll(sum_value: jax.Array, count: jax.Array) -> jax.Array:
    return sum_value / jnp.maximum(count, 1.0)


def _ppl(nll: jax.Array) -> jax.Array:
    return jnp.exp(jnp.minimum(nll, 20.0))


def _metrics_from_nll(
    prefix: str,
    nll_sum: jax.Array,
    token_count: jax.Array,
) -> dict[str, jax.Array]:
    nll = _nll(nll_sum, token_count)
    return {
        f"{prefix}_nll": nll,
        f"{prefix}_ppl": _ppl(nll),
        f"{prefix}_tokens": token_count,
    }


def _resolve_bptt_chunks(
    pretrain_mode: PretrainMode,
    *,
    num_segments: int,
    bptt_chunks: int | None,
) -> int:
    if bptt_chunks is None:
        return 0 if pretrain_mode is PretrainMode.STATEPASSING_NO_BPTT else int(num_segments)
    bptt_chunks = int(bptt_chunks)
    if bptt_chunks < 0 or bptt_chunks > num_segments:
        raise ValueError(f"bptt_chunks must be in [0, {num_segments}], got {bptt_chunks}")
    return bptt_chunks


def _prepare_carried_states_for_segment(
    states: tuple[jax.Array, ...],
    *,
    segment_idx: int,
    bptt_chunks: int,
) -> tuple[jax.Array, ...]:
    if segment_idx == 0:
        return states
    if bptt_chunks <= 1 or segment_idx % bptt_chunks == 0:
        return tuple(jax.lax.stop_gradient(state) for state in states)
    return states


def _select_gdn_states_for_carry(
    states: tuple[jax.Array, ...],
    *,
    pass_gdn_state: bool,
    gdn_layer_limit: int | None,
) -> tuple[jax.Array, ...] | None:
    return _select_layer_states_for_carry(
        states,
        pass_state=pass_gdn_state,
        layer_limit=gdn_layer_limit,
        limit_name="gdn_layer_limit",
    )


def _select_conv_states_for_carry(
    states: tuple[jax.Array, ...],
    *,
    pass_conv_state: bool,
    gdn_layer_limit: int | None,
) -> tuple[jax.Array, ...] | None:
    return _select_layer_states_for_carry(
        states,
        pass_state=pass_conv_state,
        layer_limit=gdn_layer_limit,
        limit_name="gdn_layer_limit",
    )


def _select_layer_states_for_carry(
    states: tuple[jax.Array, ...],
    *,
    pass_state: bool,
    layer_limit: int | None,
    limit_name: str,
) -> tuple[jax.Array, ...] | None:
    if not pass_state:
        return None
    if layer_limit is None:
        return states
    layer_limit = int(layer_limit)
    if layer_limit < 0 or layer_limit > len(states):
        raise ValueError(f"{limit_name} must be in [0, {len(states)}], got {layer_limit}")
    if layer_limit == 0:
        return None
    return tuple(
        state if state_idx < layer_limit else jnp.zeros_like(state)
        for state_idx, state in enumerate(states)
    )


def apply_state_reset(
    states: tuple[jax.Array, ...],
    reset_B: jax.Array,
) -> tuple[jax.Array, ...]:
    reset = reset_B.astype(jnp.bool_)
    return tuple(
        jnp.where(
            reset.reshape((reset.shape[0],) + (1,) * (state.ndim - 1)),
            jnp.zeros_like(state),
            state,
        )
        for state in states
    )


def _position_ids_zbt_from_chunk_idx(chunk_idx_B: Any, seq_len: int) -> jax.Array:
    chunk_idx_B = jnp.asarray(chunk_idx_B, dtype=jnp.int32)
    token_pos_T = jnp.arange(seq_len, dtype=jnp.int32)
    position_ids_BT = chunk_idx_B[:, None] * jnp.asarray(seq_len, dtype=jnp.int32) + token_pos_T
    return jnp.stack([position_ids_BT] * 3, axis=0)


def statepassing_target_masks(
    loss_mask_BCT: jax.Array,
    reset_state_BC: jax.Array | None = None,
    bptt_chunks: int | None = None,
) -> StatepassingTargetMasks:
    B, C, T = loss_mask_BCT.shape
    if bptt_chunks is not None and (bptt_chunks < 0 or bptt_chunks > C):
        raise ValueError(f"bptt_chunks must be in [0, {C}], got {bptt_chunks}")
    total_BT = loss_mask_BCT.reshape(B, C * T)
    boundary_BT = jnp.zeros_like(total_BT)
    iid_comparable_BT = jnp.zeros_like(total_BT)
    segment_masks = []

    for segment_idx in range(C):
        start = segment_idx * T
        segment_BT = jnp.zeros_like(total_BT)
        segment_BT = segment_BT.at[:, start + 1 : start + T].set(loss_mask_BCT[:, segment_idx, 1:])
        segment_masks.append(segment_BT)
        iid_comparable_BT = iid_comparable_BT + segment_BT

    for segment_idx in range(1, C):
        boundary_pos = segment_idx * T
        boundary_mask_B = loss_mask_BCT[:, segment_idx, 0]
        if bptt_chunks is not None and (bptt_chunks <= 1 or segment_idx % bptt_chunks == 0):
            boundary_mask_B = jnp.zeros_like(boundary_mask_B)
        if reset_state_BC is not None:
            boundary_mask_B = boundary_mask_B * (
                1 - reset_state_BC[:, segment_idx].astype(boundary_mask_B.dtype)
            )
            total_BT = total_BT.at[:, boundary_pos].set(boundary_mask_B)
        boundary_BT = boundary_BT.at[:, boundary_pos].set(boundary_mask_B)

    return StatepassingTargetMasks(
        total=total_BT,
        iid_comparable=iid_comparable_BT,
        boundary=boundary_BT,
        segments=tuple(segment_masks),
    )


def prepare_pretrain_batch(
    batch: dict[str, Any],
    pretrain_mode: PretrainMode,
    model_cfg: text_api.TextConfig,
    mesh,
    *,
    pass_rope_positions: bool = False,
) -> tuple[dict[str, jax.Array], dict[str, Any] | None, dict[str, Any]]:
    metadata = pop_pretrain_metadata(batch)
    chunk_idx_B = batch.pop("chunk_idx_B", None)
    chunk_idx_BC = batch.pop("chunk_idx_BC", None)
    debug = {
        "chunk_idx_B": chunk_idx_B,
        "chunk_idx_BC": chunk_idx_BC,
    }

    if pretrain_mode is PretrainMode.IID_BASELINE:
        required = ("token_ids_BT", "attention_mask_BT", "loss_mask_BT")
    else:
        required = ("token_ids_BCT", "attention_mask_BCT", "loss_mask_BCT", "reset_state_BC")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Missing required pretrain batch keys for {pretrain_mode}: {missing}")
    if pretrain_mode.is_statepassing:
        batch_multiple = required_batch_multiple(text_api.batch_partition_spec(model_cfg), mesh)
        window_batch = int(batch["token_ids_BCT"].shape[0])
        if window_batch % batch_multiple != 0:
            raise ValueError(
                f"Statepassing window batch size {window_batch} must be divisible by "
                f"batch sharding multiple {batch_multiple}."
            )

    device_batch = {key: batch[key] for key in required}
    if pass_rope_positions:
        if pretrain_mode is PretrainMode.IID_BASELINE:
            if chunk_idx_B is None:
                raise KeyError("pass_rope_positions=True requires chunk_idx_B in IID batches")
            device_batch["position_ids_ZBT"] = _position_ids_zbt_from_chunk_idx(
                chunk_idx_B,
                int(device_batch["token_ids_BT"].shape[1]),
            )
        else:
            if chunk_idx_BC is None:
                raise KeyError(
                    "pass_rope_positions=True requires chunk_idx_BC in statepassing batches"
                )
            device_batch["chunk_idx_BC"] = chunk_idx_BC
    return text_api.shard_batch_dict(device_batch, model_cfg, mesh), metadata, debug


def _iid_loss_stats(model, batch: dict[str, jax.Array], cfg, pad_id: int):
    token_ids_BT = batch["token_ids_BT"]
    attention_mask_BT = batch["attention_mask_BT"]
    loss_mask_BT = batch["loss_mask_BT"]

    hidden_BTD, aux_loss = text_api.forward(
        model,
        token_ids_BT,
        pad_id,
        cfg,
        attention_mask_BT=attention_mask_BT,
        position_ids_ZBT=batch.get("position_ids_ZBT"),
    )
    nll_sum, token_count = chunked_cross_entropy_stats(
        hidden_BTD,
        model.lm_head.kernel[...],
        token_ids_BT,
        loss_mask_BT,
        num_tiles=_NUM_LOSS_TILES,
        logits_out_sharding=cfg.shd_cfg.logits_btv,
    )
    nll = _nll(nll_sum, token_count)
    loss = nll + aux_loss
    metrics = {
        "loss": loss,
        "nll": nll,
        "ppl": _ppl(nll),
        "aux_loss": aux_loss,
        "nll_sum": nll_sum,
        "supervised_tokens": token_count,
    }
    return loss, metrics


def _statepassing_loss_stats(
    model,
    batch: dict[str, jax.Array],
    cfg,
    pad_id: int,
    pretrain_mode: PretrainMode,
    bptt_chunks: int | None = None,
    pass_gdn_state: bool = True,
    gdn_layer_limit: int | None = None,
    pass_rope_positions: bool = False,
    pass_conv_state: bool = False,
):
    token_ids_BCT = batch["token_ids_BCT"]
    attention_mask_BCT = batch["attention_mask_BCT"]
    loss_mask_BCT = batch["loss_mask_BCT"]
    reset_state_BC = batch["reset_state_BC"]
    C = token_ids_BCT.shape[1]
    chunk_idx_BC = batch.get("chunk_idx_BC")
    if pass_rope_positions and chunk_idx_BC is None:
        raise KeyError("pass_rope_positions=True requires chunk_idx_BC in statepassing batches")
    resolved_bptt_chunks = _resolve_bptt_chunks(
        pretrain_mode, num_segments=C, bptt_chunks=bptt_chunks
    )

    hidden_segments = []
    aux_losses = []
    carried_states = None
    carried_conv_states = None
    for segment_idx in range(C):
        if carried_states is not None:
            carried_states = _prepare_carried_states_for_segment(
                carried_states,
                segment_idx=segment_idx,
                bptt_chunks=resolved_bptt_chunks,
            )
            carried_states = apply_state_reset(carried_states, reset_state_BC[:, segment_idx])
        if carried_conv_states is not None:
            carried_conv_states = _prepare_carried_states_for_segment(
                carried_conv_states,
                segment_idx=segment_idx,
                bptt_chunks=resolved_bptt_chunks,
            )
            carried_conv_states = apply_state_reset(
                carried_conv_states, reset_state_BC[:, segment_idx]
            )
        forward_kwargs = {
            "attention_mask_BT": attention_mask_BCT[:, segment_idx, :],
            "initial_gdn_states": carried_states,
        }
        if pass_conv_state:
            forward_kwargs["initial_conv_states"] = carried_conv_states
            forward_kwargs["return_conv_states"] = True
        if pass_rope_positions:
            forward_kwargs["position_ids_ZBT"] = _position_ids_zbt_from_chunk_idx(
                chunk_idx_BC[:, segment_idx],
                int(token_ids_BCT.shape[2]),
            )
        forward_result = text_api.forward_with_gdn_state(
            model,
            token_ids_BCT[:, segment_idx, :],
            pad_id,
            cfg,
            **forward_kwargs,
        )
        if pass_conv_state:
            hidden_BTD, aux_loss, final_states, final_conv_states = forward_result
        else:
            hidden_BTD, aux_loss, final_states = forward_result
        carried_states = _select_gdn_states_for_carry(
            final_states,
            pass_gdn_state=pass_gdn_state,
            gdn_layer_limit=gdn_layer_limit,
        )
        if pass_conv_state:
            carried_conv_states = _select_conv_states_for_carry(
                final_conv_states,
                pass_conv_state=pass_conv_state,
                gdn_layer_limit=gdn_layer_limit,
            )
        hidden_segments.append(hidden_BTD)
        aux_losses.append(aux_loss)

    hidden_BT_D = jnp.concatenate(hidden_segments, axis=1)
    token_ids_BT = token_ids_BCT.reshape(token_ids_BCT.shape[0], -1)
    masks = statepassing_target_masks(loss_mask_BCT, reset_state_BC, bptt_chunks=bptt_chunks)
    mask_names = ("total", "iid_comparable", "boundary") + tuple(
        f"segment{idx}" for idx in range(C)
    )
    mask_stack = jnp.stack(
        [masks.total, masks.iid_comparable, masks.boundary, *masks.segments],
        axis=0,
    )
    nll_sums, token_counts = chunked_cross_entropy_multi_stats(
        hidden_BT_D,
        model.lm_head.kernel[...],
        token_ids_BT,
        mask_stack,
        num_tiles=_NUM_LOSS_TILES,
        logits_out_sharding=cfg.shd_cfg.logits_btv,
    )

    total_nll = _nll(nll_sums[0], token_counts[0])
    aux_loss = jnp.sum(jnp.stack(aux_losses)) / float(C)
    loss = total_nll + aux_loss
    metrics = {
        "loss": loss,
        "nll": total_nll,
        "ppl": _ppl(total_nll),
        "aux_loss": aux_loss,
        "nll_sum": nll_sums[0],
        "supervised_tokens": token_counts[0],
    }
    for mask_idx, name in enumerate(mask_names[1:], start=1):
        metrics.update(_metrics_from_nll(name, nll_sums[mask_idx], token_counts[mask_idx]))
    return loss, metrics


def make_pretrain_train_step(
    pretrain_mode: PretrainMode,
    cfg,
    pad_id: int = 0,
    bptt_chunks: int | None = None,
    pass_gdn_state: bool = True,
    gdn_layer_limit: int | None = None,
    pass_rope_positions: bool = False,
    pass_conv_state: bool = False,
):
    @nnx.jit(donate_argnums=0)
    def pretrain_train_step(optimizer: MixedPrecisionOptimizer, batch: dict[str, jax.Array]):
        def loss_fn(model):
            if pretrain_mode is PretrainMode.IID_BASELINE:
                return _iid_loss_stats(model, batch, cfg, pad_id)
            return _statepassing_loss_stats(
                model,
                batch,
                cfg,
                pad_id,
                pretrain_mode,
                bptt_chunks=bptt_chunks,
                pass_gdn_state=pass_gdn_state,
                gdn_layer_limit=gdn_layer_limit,
                pass_rope_positions=pass_rope_positions,
                pass_conv_state=pass_conv_state,
            )

        (loss, aux_metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(optimizer.model)
        optimizer.update(grads)
        metrics = dict(aux_metrics)
        metrics["grad_norm"] = optax.tree.norm(grads)
        return loss, metrics

    return pretrain_train_step


def make_pretrain_eval_step(
    pretrain_mode: PretrainMode,
    cfg,
    pad_id: int = 0,
    bptt_chunks: int | None = None,
    pass_gdn_state: bool = True,
    gdn_layer_limit: int | None = None,
    pass_rope_positions: bool = False,
    pass_conv_state: bool = False,
):
    @nnx.jit
    def pretrain_eval_step(model: nnx.Module, batch: dict[str, jax.Array]):
        if pretrain_mode is PretrainMode.IID_BASELINE:
            _, metrics = _iid_loss_stats(model, batch, cfg, pad_id)
        else:
            _, metrics = _statepassing_loss_stats(
                model,
                batch,
                cfg,
                pad_id,
                pretrain_mode,
                bptt_chunks=bptt_chunks,
                pass_gdn_state=pass_gdn_state,
                gdn_layer_limit=gdn_layer_limit,
                pass_rope_positions=pass_rope_positions,
                pass_conv_state=pass_conv_state,
            )
        return metrics

    return pretrain_eval_step


def _gpu_util_snapshot() -> dict[str, float]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        values = [
            float(pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(i)).gpu)
            for i in range(pynvml.nvmlDeviceGetCount())
        ]
    except Exception:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                check=False,
                capture_output=True,
                text=True,
            )
            values = [float(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
        except Exception:
            values = []

    if not values:
        return {}
    return {
        "gpu_util_avg": sum(values) / len(values),
        "gpu_util_min": min(values),
        "gpu_util_max": max(values),
    }


def _host_float(value) -> float:
    if isinstance(value, jax.Array):
        local_value = value.addressable_data(0)
        local_value.block_until_ready()
        return float(np.asarray(local_value))
    return float(value)


def _accumulate_metric_sums(acc: dict[str, float], metrics: dict[str, Any]) -> None:
    acc["nll_sum"] = acc.get("nll_sum", 0.0) + _host_float(metrics["nll_sum"])
    acc["supervised_tokens"] = acc.get("supervised_tokens", 0.0) + _host_float(
        metrics["supervised_tokens"]
    )
    acc["aux_loss"] = acc.get("aux_loss", 0.0) + _host_float(metrics.get("aux_loss", 0.0))
    acc["steps"] = acc.get("steps", 0.0) + 1.0
    for tok_key in sorted(key for key in metrics if key.endswith("_tokens")):
        if tok_key == "supervised_tokens":
            continue
        prefix = tok_key[: -len("_tokens")]
        nll_key = f"{prefix}_nll"
        if nll_key not in metrics:
            continue
        tokens = _host_float(metrics[tok_key])
        acc[f"{prefix}_nll_sum"] = acc.get(f"{prefix}_nll_sum", 0.0) + (
            _host_float(metrics[nll_key]) * tokens
        )
        acc[tok_key] = acc.get(tok_key, 0.0) + tokens


def _finalize_accumulated_metrics(acc: dict[str, float]) -> dict[str, float]:
    tokens = acc.get("supervised_tokens", 0.0)
    nll = acc.get("nll_sum", 0.0) / max(tokens, 1.0)
    aux_loss = acc.get("aux_loss", 0.0) / max(acc.get("steps", 0.0), 1.0)
    out = {
        "nll": nll,
        "ppl": math.exp(min(nll, 20.0)),
        "loss": nll + aux_loss,
        "aux_loss": aux_loss,
        "supervised_tokens": tokens,
        "nll_sum": acc.get("nll_sum", 0.0),
    }
    for tokens_key in sorted(key for key in acc if key.endswith("_tokens")):
        if tokens_key == "supervised_tokens":
            continue
        prefix = tokens_key[: -len("_tokens")]
        tokens_value = acc[tokens_key]
        prefix_nll = acc.get(f"{prefix}_nll_sum", 0.0) / max(tokens_value, 1.0)
        out[f"{prefix}_nll"] = prefix_nll
        out[f"{prefix}_ppl"] = math.exp(min(prefix_nll, 20.0))
        out[tokens_key] = tokens_value
    return out


def run_pretrain(
    model_id_or_cfg,
    train_cfg: TrainConfig,
    data_iter: checkpoint_utils.GrainIterator,
    *,
    pretrain_mode: PretrainMode | str,
    save_dir: str | Path | None = None,
    save_every: int = 0,
    log_every: int = 1,
    resume: checkpoint_utils.ResumeMode = checkpoint_utils.ResumeMode.NEVER,
    pad_id: int = 0,
    peak_tflops: float | None = None,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    dp_size: int | None = None,
    wandb_run=None,
    val_data_iter: checkpoint_utils.GrainIterator | None = None,
    val_every: int | None = None,
    val_steps: int = 10,
    text_attn_backend: str = "mosaic_gpu",
    gc_period: int = 0,
    bptt_chunks: int | None = None,
    pass_gdn_state: bool = True,
    gdn_layer_limit: int | None = None,
    pass_rope_positions: bool = False,
    pass_conv_state: bool = False,
) -> tuple[MixedPrecisionOptimizer, dict[str, float]]:
    pretrain_mode = PretrainMode(pretrain_mode)
    save_path = Path(save_dir).expanduser().resolve() if save_dir is not None else None

    checkpoint_manager = None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        checkpoint_manager = _make_checkpoint_manager(save_path, save_interval=save_every or None)

    latest_step = checkpoint_manager.latest_step() if checkpoint_manager is not None else None
    if resume == checkpoint_utils.ResumeMode.REQUIRED and latest_step is None:
        raise ValueError(f"resume='required' but no checkpoint found at {save_path}")
    will_resume = (
        resume in (checkpoint_utils.ResumeMode.IF_PRESENT, checkpoint_utils.ResumeMode.REQUIRED)
        and latest_step is not None
    )

    model_cfg = (
        text_api.resolve_config(str(save_path))
        if will_resume
        else text_api.resolve_config(model_id_or_cfg)
    )
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    model_cfg = text_api.align_config_to_mesh(model_cfg, mesh)
    batch_multiple = required_batch_multiple(text_api.batch_partition_spec(model_cfg), mesh)
    if train_cfg.batch_size % batch_multiple != 0:
        raise ValueError(
            f"Global batch size {train_cfg.batch_size} must be divisible by {batch_multiple}."
        )
    replicated_rng_sharding = NamedSharding(mesh, P())
    root_rng = jax.device_put(jax.random.key(train_cfg.seed), replicated_rng_sharding)
    init_rng, rng = jax.random.split(root_rng)
    init_rng = jax.device_put(init_rng, replicated_rng_sharding)
    rng = jax.device_put(rng, replicated_rng_sharding)
    is_primary_process = jax.process_index() == 0

    model, model_cfg = init_model(
        model_cfg, init_rng, tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size
    )
    from omegalax.models.sharding_runtime import set_attn_backend

    set_attn_backend(model, text_backend=text_attn_backend)
    with mesh_rules(mesh):
        optimizer = build_optimizer(model, train_cfg)

    lr_schedule_fn = build_lr_schedule(
        peak_lr=train_cfg.learning_rate,
        num_steps=train_cfg.num_steps,
        warmup_steps=train_cfg.warmup_steps,
        schedule=train_cfg.lr_schedule,
        end_factor=train_cfg.lr_end_factor,
        stable_fraction=train_cfg.lr_stable_fraction,
    )
    train_step = make_pretrain_train_step(
        pretrain_mode,
        model_cfg,
        pad_id=pad_id,
        bptt_chunks=bptt_chunks,
        pass_gdn_state=pass_gdn_state,
        gdn_layer_limit=gdn_layer_limit,
        pass_rope_positions=pass_rope_positions,
        pass_conv_state=pass_conv_state,
    )
    eval_step = make_pretrain_eval_step(
        pretrain_mode,
        model_cfg,
        pad_id=pad_id,
        bptt_chunks=bptt_chunks,
        pass_gdn_state=pass_gdn_state,
        gdn_layer_limit=gdn_layer_limit,
        pass_rope_positions=pass_rope_positions,
        pass_conv_state=pass_conv_state,
    )

    if checkpoint_manager is not None and not will_resume:
        _write_checkpoint_config(save_path, model_cfg)

    start_step = 0
    if will_resume:
        optimizer, start_step, rng, data_iter = _restore_sft_checkpoint(
            checkpoint_manager, optimizer, rng, data_iter
        )
        rng = jax.device_put(rng, replicated_rng_sharding)

    accum_steps = train_cfg.grad_accum_steps
    timer = StepTimer(warmup=2 * accum_steps)
    global_tokens_per_step = train_cfg.seq_len * train_cfg.batch_size * accum_steps
    last_metrics: dict[str, float] = {}
    prev_metrics: tuple[int, dict[str, Any], datetime.timedelta, float] | None = None

    def _log_prev_metrics(force: bool = False) -> None:
        nonlocal last_metrics
        if prev_metrics is None:
            return
        step_to_log, metrics_to_log, step_delta, step_per_device_flops = prev_metrics
        result = maybe_log_step_metrics(
            step_to_log,
            metrics_to_log,
            step_delta,
            is_primary_process=is_primary_process,
            log_every=log_every,
            force=force,
            per_device_flops=step_per_device_flops,
            global_tokens_per_step=global_tokens_per_step,
            peak_tflops=peak_tflops,
            wandb_run=wandb_run,
            batch_size=train_cfg.batch_size * accum_steps,
        )
        if result is not None:
            last_metrics = result

    startup_log(f"entering pretrain loop mode={pretrain_mode}")
    for step_idx in range(start_step, train_cfg.num_steps):
        step = step_idx + 1
        accum = {}
        accum_grad_norm = 0.0
        accum_flops = 0.0
        accum_time = datetime.timedelta(0)
        accum_data_wait_s = 0.0

        for _micro in range(accum_steps):
            wait_start = time.perf_counter()
            raw_batch = next(data_iter)
            accum_data_wait_s += time.perf_counter() - wait_start
            batch, _, _ = prepare_pretrain_batch(
                raw_batch,
                pretrain_mode,
                model_cfg,
                mesh,
                pass_rope_positions=pass_rope_positions,
            )
            micro_flops = per_device_flops_per_step(
                model_cfg,
                train_cfg.seq_len,
                train_cfg.batch_size,
            )
            _, metrics = train_step(optimizer, batch)
            micro_delta = timer.step()

            _accumulate_metric_sums(accum, metrics)
            accum_grad_norm = accum_grad_norm + metrics["grad_norm"]
            accum_flops += micro_flops
            accum_time += micro_delta

        with jax.default_device("cpu"):
            current_lr = (
                float(lr_schedule_fn(step_idx))
                if callable(lr_schedule_fn)
                else float(lr_schedule_fn)
            )
        window_metrics = _finalize_accumulated_metrics(accum)
        window_metrics["grad_norm"] = accum_grad_norm / accum_steps
        window_metrics["lr"] = current_lr
        window_metrics["total_tokens"] = step * global_tokens_per_step
        window_metrics["data_wait_s"] = accum_data_wait_s
        step_time_s = accum_time.total_seconds()
        window_metrics["data_wait_frac"] = (
            accum_data_wait_s / step_time_s if step_time_s > 0 else 0.0
        )
        if is_primary_process:
            window_metrics.update(_gpu_util_snapshot())

        _log_prev_metrics()
        prev_metrics = (step, window_metrics, accum_time, accum_flops)

        if checkpoint_manager is not None and save_every and step % save_every == 0:
            _save_sft_checkpoint(checkpoint_manager, optimizer, rng, step, data_iter)

        if gc_period and step % gc_period == 0:
            gc.collect()

        if val_data_iter is not None and val_every and step % val_every == 0:
            val_acc = {}
            for _ in range(val_steps):
                raw_val_batch = next(val_data_iter)
                val_batch, _, _ = prepare_pretrain_batch(
                    raw_val_batch,
                    pretrain_mode,
                    model_cfg,
                    mesh,
                    pass_rope_positions=pass_rope_positions,
                )
                val_metrics = eval_step(optimizer.model, val_batch)
                _accumulate_metric_sums(val_acc, val_metrics)
            val_host = _finalize_accumulated_metrics(val_acc)
            if wandb_run is not None and is_primary_process:
                wandb_run.log({f"val/{key}": value for key, value in val_host.items()}, step=step)

    _log_prev_metrics(force=True)

    if checkpoint_manager is not None:
        if last_metrics and (not save_every or last_metrics["step"] % save_every != 0):
            _save_sft_checkpoint(
                checkpoint_manager, optimizer, rng, int(last_metrics["step"]), data_iter, force=True
            )
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()

    return optimizer, last_metrics
