"""Text pretraining helpers for IID and 2-segment statepassing experiments."""

from __future__ import annotations

import dataclasses
import datetime
import enum
import gc
import math
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
    segment0: jax.Array
    boundary: jax.Array
    segment1: jax.Array


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


def prepare_carried_states(
    states: tuple[jax.Array, ...],
    pretrain_mode: PretrainMode,
) -> tuple[jax.Array, ...]:
    if pretrain_mode is PretrainMode.STATEPASSING_NO_BPTT:
        return tuple(jax.lax.stop_gradient(state) for state in states)
    return states


def apply_state_reset(
    states: tuple[jax.Array, ...],
    reset_B: jax.Array,
) -> tuple[jax.Array, ...]:
    reset = reset_B.astype(jnp.bool_)
    return tuple(
        jnp.where(reset[:, None, None, None], jnp.zeros_like(state), state) for state in states
    )


def statepassing_target_masks(
    loss_mask_BCT: jax.Array,
    reset_state_BC: jax.Array | None = None,
) -> StatepassingTargetMasks:
    if loss_mask_BCT.shape[1] != 2:
        raise ValueError(f"Statepassing pretraining requires C=2, got {loss_mask_BCT.shape[1]}")

    B, C, T = loss_mask_BCT.shape
    total_BT = loss_mask_BCT.reshape(B, C * T)
    segment0_BT = jnp.zeros_like(total_BT)
    boundary_BT = jnp.zeros_like(total_BT)
    segment1_BT = jnp.zeros_like(total_BT)

    segment0_BT = segment0_BT.at[:, 1:T].set(loss_mask_BCT[:, 0, 1:])
    boundary_mask_B = loss_mask_BCT[:, 1, 0]
    if reset_state_BC is not None:
        boundary_mask_B = boundary_mask_B * (1 - reset_state_BC[:, 1].astype(boundary_mask_B.dtype))
        total_BT = total_BT.at[:, T].set(boundary_mask_B)
    boundary_BT = boundary_BT.at[:, T].set(boundary_mask_B)
    segment1_BT = segment1_BT.at[:, T + 1 :].set(loss_mask_BCT[:, 1, 1:])
    return StatepassingTargetMasks(
        total=total_BT,
        segment0=segment0_BT,
        boundary=boundary_BT,
        segment1=segment1_BT,
    )


def prepare_pretrain_batch(
    batch: dict[str, Any],
    pretrain_mode: PretrainMode,
    model_cfg: text_api.TextConfig,
    mesh,
) -> tuple[dict[str, jax.Array], dict[str, Any] | None, dict[str, Any]]:
    metadata = pop_pretrain_metadata(batch)
    debug = {
        "chunk_idx_B": batch.pop("chunk_idx_B", None),
        "chunk_idx_BC": batch.pop("chunk_idx_BC", None),
        "is_last_chunk_BC": batch.pop("is_last_chunk_BC", None),
    }

    if pretrain_mode is PretrainMode.IID_BASELINE:
        required = ("token_ids_BT", "attention_mask_BT", "loss_mask_BT")
    else:
        required = ("token_ids_BCT", "attention_mask_BCT", "loss_mask_BCT", "reset_state_BC")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Missing required pretrain batch keys for {pretrain_mode}: {missing}")

    device_batch = {key: batch[key] for key in required}
    return text_api.shard_batch_dict(device_batch, model_cfg, mesh), metadata, debug


def _iid_loss_stats(model, batch: dict[str, jax.Array], cfg, pad_id: int):
    token_ids_BT = batch["token_ids_BT"]
    attention_mask_BT = batch["attention_mask_BT"]
    loss_mask_BT = batch["loss_mask_BT"]

    hidden_BTD, aux_loss = text_api.forward(
        model, token_ids_BT, pad_id, cfg, attention_mask_BT=attention_mask_BT
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
):
    token_ids_BCT = batch["token_ids_BCT"]
    attention_mask_BCT = batch["attention_mask_BCT"]
    loss_mask_BCT = batch["loss_mask_BCT"]
    reset_state_BC = batch["reset_state_BC"]
    if token_ids_BCT.shape[1] != 2:
        raise ValueError(f"Statepassing pretraining requires C=2, got {token_ids_BCT.shape[1]}")

    hidden0_BTD, aux0, state0 = text_api.forward_with_gdn_state(
        model,
        token_ids_BCT[:, 0, :],
        pad_id,
        cfg,
        attention_mask_BT=attention_mask_BCT[:, 0, :],
        initial_gdn_states=None,
    )
    state0 = prepare_carried_states(state0, pretrain_mode)
    state0 = apply_state_reset(state0, reset_state_BC[:, 1])
    hidden1_BTD, aux1, _ = text_api.forward_with_gdn_state(
        model,
        token_ids_BCT[:, 1, :],
        pad_id,
        cfg,
        attention_mask_BT=attention_mask_BCT[:, 1, :],
        initial_gdn_states=state0,
    )

    hidden_BT_D = jnp.concatenate([hidden0_BTD, hidden1_BTD], axis=1)
    token_ids_BT = token_ids_BCT.reshape(token_ids_BCT.shape[0], -1)
    masks = statepassing_target_masks(loss_mask_BCT, reset_state_BC)
    mask_stack = jnp.stack([masks.total, masks.segment0, masks.boundary, masks.segment1], axis=0)
    nll_sums, token_counts = chunked_cross_entropy_multi_stats(
        hidden_BT_D,
        model.lm_head.kernel[...],
        token_ids_BT,
        mask_stack,
        num_tiles=_NUM_LOSS_TILES,
        logits_out_sharding=cfg.shd_cfg.logits_btv,
    )

    total_nll = _nll(nll_sums[0], token_counts[0])
    aux_loss = 0.5 * (aux0 + aux1)
    loss = total_nll + aux_loss
    metrics = {
        "loss": loss,
        "nll": total_nll,
        "ppl": _ppl(total_nll),
        "aux_loss": aux_loss,
        "nll_sum": nll_sums[0],
        "supervised_tokens": token_counts[0],
        **_metrics_from_nll("segment0", nll_sums[1], token_counts[1]),
        **_metrics_from_nll("boundary", nll_sums[2], token_counts[2]),
        **_metrics_from_nll("segment1", nll_sums[3], token_counts[3]),
    }
    return loss, metrics


def make_pretrain_train_step(pretrain_mode: PretrainMode, cfg, pad_id: int = 0):
    @nnx.jit(donate_argnums=0)
    def pretrain_train_step(optimizer: MixedPrecisionOptimizer, batch: dict[str, jax.Array]):
        def loss_fn(model):
            if pretrain_mode is PretrainMode.IID_BASELINE:
                return _iid_loss_stats(model, batch, cfg, pad_id)
            return _statepassing_loss_stats(model, batch, cfg, pad_id, pretrain_mode)

        (loss, aux_metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(optimizer.model)
        optimizer.update(grads)
        metrics = dict(aux_metrics)
        metrics["grad_norm"] = optax.tree.norm(grads)
        return loss, metrics

    return pretrain_train_step


def make_pretrain_eval_step(pretrain_mode: PretrainMode, cfg, pad_id: int = 0):
    @nnx.jit
    def pretrain_eval_step(model: nnx.Module, batch: dict[str, jax.Array]):
        if pretrain_mode is PretrainMode.IID_BASELINE:
            _, metrics = _iid_loss_stats(model, batch, cfg, pad_id)
        else:
            _, metrics = _statepassing_loss_stats(model, batch, cfg, pad_id, pretrain_mode)
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
    return float(value)


def _accumulate_metric_sums(acc: dict[str, float], metrics: dict[str, Any]) -> None:
    acc["nll_sum"] = acc.get("nll_sum", 0.0) + _host_float(metrics["nll_sum"])
    acc["supervised_tokens"] = acc.get("supervised_tokens", 0.0) + _host_float(
        metrics["supervised_tokens"]
    )
    acc["aux_loss"] = acc.get("aux_loss", 0.0) + _host_float(metrics.get("aux_loss", 0.0))
    acc["steps"] = acc.get("steps", 0.0) + 1.0
    for prefix in ("segment0", "boundary", "segment1"):
        sum_key = f"{prefix}_nll"
        tok_key = f"{prefix}_tokens"
        if sum_key in metrics and tok_key in metrics:
            tokens = _host_float(metrics[tok_key])
            acc[f"{prefix}_nll_sum"] = acc.get(f"{prefix}_nll_sum", 0.0) + (
                _host_float(metrics[sum_key]) * tokens
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
    for prefix in ("segment0", "boundary", "segment1"):
        tokens_key = f"{prefix}_tokens"
        tokens_value = acc.get(tokens_key)
        if tokens_value is None:
            continue
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
    if pretrain_mode.is_statepassing and train_cfg.batch_size % (2 * batch_multiple) != 0:
        raise ValueError(
            f"Statepassing global batch size {train_cfg.batch_size} must leave an even "
            f"per-shard segment batch after batch sharding multiple {batch_multiple}."
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
    train_step = make_pretrain_train_step(pretrain_mode, model_cfg, pad_id=pad_id)
    eval_step = make_pretrain_eval_step(pretrain_mode, model_cfg, pad_id=pad_id)

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
            batch, _, _ = prepare_pretrain_batch(raw_batch, pretrain_mode, model_cfg, mesh)
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
                    raw_val_batch, pretrain_mode, model_cfg, mesh
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
                checkpoint_manager, optimizer, rng, int(last_metrics["step"]), data_iter
            )
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()

    return optimizer, last_metrics
