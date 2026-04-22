"""Training helpers for vision-language models (text-only or multimodal batches)."""

from __future__ import annotations

import dataclasses
import datetime
from pathlib import Path
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
import optax
import orbax.checkpoint as ocp

from omegalax import export as export_lib
from omegalax.distributed.mesh import ensure_mesh, mesh_rules, required_batch_multiple
from omegalax.models.params_utils import save_hf_config
from omegalax.trainers import checkpoint_utils
from omegalax.trainers.loss import chunked_cross_entropy_loss
from omegalax.trainers.lr_schedule import build_lr_schedule
from omegalax.trainers.perf import (
    maybe_log_step_metrics,
    per_device_flops_per_step,
    StepTimer,
)
from omegalax.trainers.optim import MixedPrecisionOptimizer
from omegalax.trainers.text import startup_log
from omegalax.vlm import api as vlm_api

P = PartitionSpec


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    batch_size: int = 8
    seq_len: int = 64
    num_steps: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    lr_schedule: str = "linear"
    lr_end_factor: float = 0.0 
    lr_stable_fraction: float = 0.8 
    max_grad_norm: float = 0.0
    grad_accum_steps: int = 1
    print_every: int = 1


def init_model(
    cfg_or_model_id,
    rng: jax.Array,
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    dp_size: int | None = None,
) -> nnx.Module:
    model, _ = vlm_api.init_model(
        cfg_or_model_id,
        rng,
        tp_size=tp_size,
        fsdp_size=fsdp_size,
        dp_size=dp_size,
    )
    return model


def build_optimizer(model: nnx.Module, lr_schedule_fn: optax.Schedule | float, train_cfg: TrainConfig) -> MixedPrecisionOptimizer:
    chain = []
    if train_cfg.max_grad_norm > 0:
        chain.append(optax.clip_by_global_norm(train_cfg.max_grad_norm))
    chain.append(optax.adamw(lr_schedule_fn, weight_decay=train_cfg.weight_decay))
    tx = optax.chain(*chain)
    if train_cfg.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=train_cfg.grad_accum_steps)
    opt = MixedPrecisionOptimizer(model, tx)
    return opt


_NUM_LOSS_TILES = 4


def _train_state(optimizer: MixedPrecisionOptimizer, rng: jax.Array) -> dict[str, object]:
    return {"optimizer": nnx.state(optimizer), "rng": rng}


def _abstract_train_state(optimizer: MixedPrecisionOptimizer, rng: jax.Array) -> dict[str, object]:
    return {
        "optimizer": jax.tree.map(
            lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding),
            nnx.state(optimizer),
        ),
        "rng": jax.ShapeDtypeStruct(rng.shape, rng.dtype, sharding=rng.sharding),
    }


def _make_checkpoint_manager(save_dir: Path, save_interval: int | None) -> ocp.CheckpointManager:
    """Orbax requires an absolute checkpoint path."""
    save_dir = Path(save_dir).expanduser().resolve()
    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add("train_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)
    handler_registry.add("train_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    checkpoint_utils.register_grain_iterator_handler(handler_registry)
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=save_interval,
        max_to_keep=2,
        step_format_fixed_length=6,
        cleanup_tmp_directories=True,
    )
    return ocp.CheckpointManager(save_dir, options=options, handler_registry=handler_registry)


def _write_checkpoint_config(save_dir: Path, cfg) -> None:
    save_hf_config(export_lib.model_config_to_hf_dict(cfg), save_dir)


def _save_sft_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    optimizer: MixedPrecisionOptimizer,
    rng: jax.Array,
    step: int,
    input_iter: checkpoint_utils.GrainIterator,
) -> None:
    train_state = _train_state(optimizer, rng)
    save_args = checkpoint_utils.make_grain_save_args(train_state, input_iter)
    checkpoint_manager.save(step, args=save_args)


def _restore_sft_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    optimizer: MixedPrecisionOptimizer,
    rng: jax.Array,
    input_iter: checkpoint_utils.GrainIterator,
) -> tuple[MixedPrecisionOptimizer, int, jax.Array, checkpoint_utils.GrainIterator]:
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        raise ValueError("No checkpoint found to restore.")

    abstract_state = _abstract_train_state(optimizer, rng)
    restore_args = checkpoint_utils.make_grain_restore_args(abstract_state, input_iter)
    restored = checkpoint_manager.restore(latest_step, args=restore_args)
    train_state = restored["train_state"]
    nnx.update(optimizer, train_state["optimizer"])
    return optimizer, int(latest_step), train_state["rng"], checkpoint_utils.restored_input_iter(restored)


def make_sft_train_step(cfg, pad_id: int = 0):
    """Build a JIT-compiled VLM SFT train step that consumes a batch dict.

    The batch dict must contain ``token_ids_BT``, ``attention_mask_BT``, and
    ``loss_mask_BT``.  It may also contain ``pixel_values`` and
    ``image_grid_thw`` for multimodal batches.
    """

    @nnx.jit(donate_argnums=0)
    def sft_train_step(optimizer: MixedPrecisionOptimizer, batch: dict[str, jax.Array]):
        token_ids_BT = batch["token_ids_BT"]
        attention_mask_BT = batch["attention_mask_BT"]
        loss_mask_BT = batch["loss_mask_BT"]
        pixel_values = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")
        vision_cu_seqlens = batch.get("vision_cu_seqlens")
        position_ids_ZBT = batch.get("position_ids_ZBT")

        def loss_fn(model):
            hidden_BTD, aux_loss = vlm_api.forward(
                model,
                token_ids_BT,
                pad_id,
                cfg,
                attention_mask_BT=attention_mask_BT,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                vision_cu_seqlens=vision_cu_seqlens,
                position_ids_ZBT=position_ids_ZBT,
            )
            lm_weight = model.lm_head.kernel[...]
            loss = chunked_cross_entropy_loss(
                hidden_BTD, lm_weight, token_ids_BT, loss_mask_BT,
                num_tiles=_NUM_LOSS_TILES,
                logits_out_sharding=cfg.shd_cfg.logits_btv,
            ) + aux_loss
            supervised_tokens = jnp.sum(loss_mask_BT[:, 1:].astype(jnp.float32))
            return loss, supervised_tokens

        (loss, supervised_tokens), grads = nnx.value_and_grad(loss_fn, has_aux=True)(optimizer.model)
        optimizer.update(grads)
        metrics = {
            "loss": loss,
            "grad_norm": optax.tree.norm(grads),
            "supervised_tokens": supervised_tokens,
        }
        return loss, metrics

    return sft_train_step


def make_sft_eval_step(cfg, pad_id: int = 0):
    """Build a JIT-compiled VLM SFT eval step (forward only, no gradients)."""

    @nnx.jit
    def sft_eval_step(model: nnx.Module, batch: dict[str, jax.Array]):
        token_ids_BT = batch["token_ids_BT"]
        attention_mask_BT = batch["attention_mask_BT"]
        loss_mask_BT = batch["loss_mask_BT"]
        pixel_values = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")
        vision_cu_seqlens = batch.get("vision_cu_seqlens")
        position_ids_ZBT = batch.get("position_ids_ZBT")

        hidden_BTD, aux_loss = vlm_api.forward(
            model,
            token_ids_BT,
            pad_id,
            cfg,
            attention_mask_BT=attention_mask_BT,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            vision_cu_seqlens=vision_cu_seqlens,
            position_ids_ZBT=position_ids_ZBT,
        )
        lm_weight = model.lm_head.kernel[...]
        loss = chunked_cross_entropy_loss(
            hidden_BTD, lm_weight, token_ids_BT, loss_mask_BT,
            num_tiles=_NUM_LOSS_TILES,
            logits_out_sharding=cfg.shd_cfg.logits_btv,
        ) + aux_loss
        supervised_tokens = jnp.sum(loss_mask_BT[:, 1:].astype(jnp.float32))
        return loss, supervised_tokens

    return sft_eval_step


def run_sft(
    model_id_or_cfg,
    train_cfg: TrainConfig,
    data_iter: checkpoint_utils.GrainIterator,
    *,
    save_dir: str | Path | None = None,
    save_every: int = 0,
    log_every: int = 1,
    resume: bool = False,
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
) -> tuple[MixedPrecisionOptimizer, dict[str, float]]:
    """SFT a VLM from a Grain iterator; returns final optimizer + last metrics.

    ``data_iter`` must be a checkpointable Grain iterator yielding dicts with keys ``token_ids_BT``,
    ``attention_mask_BT``, and ``loss_mask_BT`` (all numpy ``(B, T)``).
    Optionally ``pixel_values`` and ``image_grid_thw`` for multimodal batches.

    If ``val_data_iter`` is provided, runs ``val_steps`` forward-only batches
    every ``val_every`` training steps and logs the average validation loss.
    """
    save_path = Path(save_dir).expanduser().resolve() if save_dir is not None else None
    if resume:
        if save_path is None:
            raise ValueError("resume=True requires save_dir to be provided.")
        if not save_path.exists():
            raise ValueError(f"resume=True requires an existing checkpoint directory: {save_path}")
        checkpoint_probe = _make_checkpoint_manager(save_path, save_interval=None)
        latest_step = checkpoint_probe.latest_step()
        checkpoint_probe.close()
        if latest_step is None:
            raise ValueError(f"resume=True but no checkpoints found under: {save_path}")
        model_cfg = vlm_api.resolve_config(str(save_path))
        startup_log(f"resolved model config from checkpoint {save_path!r}")
    else:
        model_cfg = vlm_api.resolve_config(model_id_or_cfg)
        startup_log("resolved model config")
    startup_log(f"model_cfg={model_cfg}")
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    model_cfg = vlm_api.align_config_to_mesh(model_cfg, mesh)
    startup_log("mesh ready (tp/fsdp/dp)")
    batch_multiple = required_batch_multiple(vlm_api.batch_partition_spec(model_cfg), mesh)
    if train_cfg.batch_size % batch_multiple != 0:
        raise ValueError(
            f"Global batch size {train_cfg.batch_size} must be divisible by the mesh batch multiple "
            f"{batch_multiple}."
        )

    replicated_rng_sharding = NamedSharding(mesh, P())
    root_rng = jax.device_put(jax.random.key(train_cfg.seed), replicated_rng_sharding)
    init_rng, rng = jax.random.split(root_rng)
    init_rng = jax.device_put(init_rng, replicated_rng_sharding)
    rng = jax.device_put(rng, replicated_rng_sharding)
    startup_log("placed training rng on device mesh")

    is_primary_process = jax.process_index() == 0

    lr_schedule_fn = build_lr_schedule(
        peak_lr=train_cfg.learning_rate,
        num_steps=train_cfg.num_steps,
        warmup_steps=train_cfg.warmup_steps,
        schedule=train_cfg.lr_schedule,
        end_factor=train_cfg.lr_end_factor,
        stable_fraction=train_cfg.lr_stable_fraction,
    )

    if not resume and isinstance(model_id_or_cfg, str):
        model, model_cfg = vlm_api.load_pretrained(
            model_id_or_cfg,
            tp_size=tp_size,
            fsdp_size=fsdp_size,
            dp_size=dp_size,
        )
        model_cfg = vlm_api.align_config_to_mesh(model_cfg, mesh)
        startup_log("loaded pretrained model")
    else:
        model, model_cfg = vlm_api.init_model(
            model_cfg,
            init_rng,
            tp_size=tp_size,
            fsdp_size=fsdp_size,
            dp_size=dp_size,
        )
        startup_log("initialized model (random init)")
    from omegalax.models.sharding_runtime import set_attn_backend
    set_attn_backend(model, text_backend=text_attn_backend)
    startup_log(f"set attn backend: text={text_attn_backend}")
    with mesh_rules(mesh):
        optimizer = build_optimizer(model, lr_schedule_fn, train_cfg)

    startup_log("built optimizer")
    sft_step = make_sft_train_step(model_cfg, pad_id=pad_id)
    eval_step = make_sft_eval_step(model_cfg, pad_id=pad_id) if val_data_iter is not None else None
    startup_log("built train step (jit)" + (" and eval step (jit)" if eval_step is not None else ""))

    accum_steps = train_cfg.grad_accum_steps
    timer = StepTimer(warmup=2 * accum_steps)
    global_tokens_per_step = train_cfg.seq_len * train_cfg.batch_size * accum_steps

    checkpoint_manager = None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        _write_checkpoint_config(save_path, model_cfg)
        checkpoint_manager = _make_checkpoint_manager(save_path, save_interval=save_every or None)
        startup_log(f"checkpoint manager ready at {save_path!r}")

    start_step = 0
    if resume:
        if checkpoint_manager is None:
            raise ValueError("resume=True requires save_dir to be provided.")
        optimizer, start_step, rng, data_iter = _restore_sft_checkpoint(checkpoint_manager, optimizer, rng, data_iter)
        rng = jax.device_put(rng, replicated_rng_sharding)
        startup_log(f"restored checkpoint at step {start_step}")

    last_metrics: dict[str, float] = {}
    prev_metrics: tuple[int, dict[str, jax.Array], datetime.timedelta, float] | None = None

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
            batch_size=train_cfg.batch_size,
        )
        if result is not None:
            last_metrics = result

    startup_log("entering training loop")
    for step_idx in range(start_step, train_cfg.num_steps):
        step = step_idx + 1

        accum_loss = 0.0
        accum_sup_tokens = 0.0
        accum_grad_norm = 0.0
        accum_flops = 0.0
        accum_time = datetime.timedelta(0)

        for _micro in range(accum_steps):
            batch = next(data_iter)
            micro_flops = per_device_flops_per_step(
                model_cfg,
                train_cfg.seq_len,
                train_cfg.batch_size,
                image_grid_thw=batch.get("image_grid_thw"),
            )
            batch = vlm_api.shard_batch_dict(batch, model_cfg, mesh)
            _, metrics = sft_step(optimizer, batch)
            micro_delta = timer.step()

            accum_loss = accum_loss + metrics["loss"]
            accum_sup_tokens = accum_sup_tokens + metrics["supervised_tokens"]
            accum_grad_norm = accum_grad_norm + metrics["grad_norm"]
            accum_flops += micro_flops
            accum_time += micro_delta

        
        with jax.default_device('cpu'):
            window_metrics = {
                "loss": accum_loss / accum_steps,
                "grad_norm": accum_grad_norm / accum_steps,
                "supervised_tokens": accum_sup_tokens,
                "lr": lr_schedule_fn(step_idx),
            }
            _log_prev_metrics()

            prev_metrics = (step, window_metrics, accum_time, accum_flops)

        if checkpoint_manager is not None and save_every and step % save_every == 0:
            _save_sft_checkpoint(checkpoint_manager, optimizer, rng, step, data_iter)

        if eval_step is not None and val_every and step % val_every == 0:
            total_val_loss = 0.0
            total_val_sup_tokens = 0.0
            for _ in range(val_steps):
                val_batch = next(val_data_iter)
                val_batch = vlm_api.shard_batch_dict(val_batch, model_cfg, mesh)
                val_loss, val_sup_tokens = eval_step(optimizer.model, val_batch)
                total_val_loss += float(val_loss)
                total_val_sup_tokens += float(val_sup_tokens)
            avg_val_loss = total_val_loss / val_steps
            if wandb_run is not None and is_primary_process:
                wandb_run.log(
                    {"val/loss": avg_val_loss, "val/sup_tokens": total_val_sup_tokens},
                    step=step,
                )

    _log_prev_metrics(force=True)

    if checkpoint_manager is not None:
        if last_metrics and (not save_every or last_metrics["step"] % save_every != 0):
            _save_sft_checkpoint(checkpoint_manager, optimizer, rng, int(last_metrics["step"]), data_iter)
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()

    return optimizer, last_metrics
