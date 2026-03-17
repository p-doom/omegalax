"""Training helpers for vision-language models (text-only or multimodal batches)."""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Iterator
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
from omegalax.trainers.perf import (
    maybe_log_step_metrics,
    per_device_flops_per_step,
    StepTimer,
)
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
    print_every: int = 1

    @classmethod
    def smoke(cls):
        return cls()


def init_model(
    cfg_or_model_id,
    rng: jax.Array,
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
) -> nnx.Module:
    model, _ = vlm_api.init_model(
        cfg_or_model_id,
        rng,
        tp_size=tp_size,
        fsdp_size=fsdp_size,
    )
    return model


def build_optimizer(model: nnx.Module, train_cfg: TrainConfig) -> nnx.ModelAndOptimizer:
    tx = optax.adamw(learning_rate=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    return nnx.ModelAndOptimizer(model, tx)


def _local_batch_size(global_batch_size: int, seq_len: int, batch_spec: PartitionSpec, mesh: Mesh) -> int:
    sharding = NamedSharding(mesh, batch_spec)
    global_shape = (global_batch_size, seq_len)
    unique_batch_slices = {
        indices[0].indices(global_batch_size)
        for indices in sharding.addressable_devices_indices_map(global_shape).values()
    }
    return sum((stop - start + step - 1) // step for start, stop, step in unique_batch_slices)


def _masked_next_token_loss(
    logits_BTV: jax.Array,
    targets_BT: jax.Array,
    mask_BT: jax.Array,
) -> jax.Array:
    logits_BTV = logits_BTV.astype(jnp.float32)
    target_logits_BT = jnp.take_along_axis(logits_BTV, targets_BT[..., None], axis=-1)[..., 0]
    max_logits_BT = jnp.max(logits_BTV, axis=-1)
    stable_logits_BTV = logits_BTV - max_logits_BT[..., None]
    logsumexp_BT = max_logits_BT + jnp.log(jnp.sum(jnp.exp(stable_logits_BTV), axis=-1))
    nll_BT = logsumexp_BT - target_logits_BT
    mask_BT = mask_BT.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(mask_BT), 1.0)
    return jnp.sum(nll_BT * mask_BT) / denom


def make_train_step(cfg, pad_id: int = 0):
    @nnx.jit(donate_argnums=0)
    def train_step(
        optimizer: nnx.ModelAndOptimizer,
        token_ids_BT: jax.Array,
        *,
        attention_mask_BT: jax.Array | None = None,
        pixel_values: jax.Array | None = None,
        image_grid_thw: jax.Array | None = None,
        position_ids_ZBT: jax.Array | None = None,
    ):
        def loss_fn(model):
            logits_BTV, aux_loss = vlm_api.forward(
                model,
                token_ids_BT,
                pad_id,
                cfg,
                attention_mask_BT=attention_mask_BT,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids_ZBT=position_ids_ZBT,
            )
            targets = token_ids_BT[:, 1:]
            mask = attention_mask_BT[:, 1:] if attention_mask_BT is not None else (targets != pad_id)
            loss = _masked_next_token_loss(logits_BTV[:, :-1, :], targets, mask) + aux_loss
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
        optimizer.update(grads)
        metrics = {
            "loss": loss,
            "grad_norm": optax.tree.norm(grads),
        }
        return loss, metrics

    return train_step


def make_synthetic_batch(
    rng: jax.Array, batch_size: int, seq_len: int, vocab_size: int, pad_id: int = 0
) -> np.ndarray:
    """Random token batch generator used for smoke training.

    Returns a numpy array so the result is always process-local, which
    is required by ``jax.make_array_from_process_local_data`` in
    multi-process setups.
    """
    return np.asarray(
        jax.random.randint(rng, (batch_size, seq_len), minval=pad_id, maxval=vocab_size, dtype=jnp.int32)
    )


def _train_state(optimizer: nnx.ModelAndOptimizer, rng: jax.Array) -> dict[str, object]:
    return {"optimizer": nnx.state(optimizer), "rng": rng}


def _abstract_train_state(optimizer: nnx.ModelAndOptimizer, rng: jax.Array) -> dict[str, object]:
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
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=save_interval,
        max_to_keep=2,
        step_format_fixed_length=6,
        cleanup_tmp_directories=True,
    )
    return ocp.CheckpointManager(save_dir, options=options, handler_registry=handler_registry)


def _write_checkpoint_config(save_dir: Path, cfg) -> None:
    save_hf_config(export_lib.model_config_to_hf_dict(cfg), save_dir)


def _save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager, optimizer: nnx.ModelAndOptimizer, rng: jax.Array, step: int
) -> None:
    train_state = _train_state(optimizer, rng)
    save_args = ocp.args.Composite(train_state=ocp.args.PyTreeSave(train_state))
    checkpoint_manager.save(step, args=save_args)


def _restore_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    optimizer: nnx.ModelAndOptimizer,
    rng: jax.Array,
) -> tuple[nnx.ModelAndOptimizer, int, jax.Array]:
    """Restore optimizer/model state and RNG key from latest checkpoint."""
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        raise ValueError("No checkpoint found to restore.")

    abstract_state = _abstract_train_state(optimizer, rng)
    restore_args = ocp.args.Composite(train_state=ocp.args.PyTreeRestore(abstract_state))
    restored = checkpoint_manager.restore(latest_step, args=restore_args)
    train_state = restored["train_state"]
    nnx.update(optimizer, train_state["optimizer"])
    return optimizer, int(latest_step), train_state["rng"]


def run_training(
    model_id_or_cfg,
    train_cfg: TrainConfig,
    *,
    save_dir: str | Path | None = None,
    save_every: int = 0,
    log_every: int = 1,
    resume: bool = False,
    pad_id: int = 0,
    peak_tflops: float | None = None,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    tb_writer=None,
) -> tuple[nnx.ModelAndOptimizer, dict[str, float]]:
    """Train a VLM with synthetic data; returns final optimizer + last metrics."""
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
    else:
        model_cfg = vlm_api.resolve_config(model_id_or_cfg)
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size)
    model_cfg = vlm_api.align_config_to_mesh(model_cfg, mesh)
    batch_spec = vlm_api.batch_partition_spec(model_cfg)
    required_multiple = required_batch_multiple(batch_spec, mesh)
    if train_cfg.batch_size % max(1, required_multiple) != 0:
        raise ValueError(
            f"Global batch_size={train_cfg.batch_size} must be divisible by {required_multiple} "
            f"for batch sharding {batch_spec}."
        )
    local_batch_size = _local_batch_size(train_cfg.batch_size, train_cfg.seq_len, batch_spec, mesh)
    if local_batch_size <= 0 or local_batch_size > train_cfg.batch_size:
        raise RuntimeError(
            f"Invalid per-process batch size {local_batch_size} for global batch_size={train_cfg.batch_size} "
            f"and sharding {batch_spec}."
        )
    is_batch_partitioned_across_processes = local_batch_size < train_cfg.batch_size

    replicated_rng_sharding = NamedSharding(mesh, P())
    root_rng = jax.device_put(jax.random.key(train_cfg.seed), replicated_rng_sharding)
    init_rng, rng = jax.random.split(root_rng)
    init_rng = jax.device_put(init_rng, replicated_rng_sharding)
    if is_batch_partitioned_across_processes:
        rng = jax.random.fold_in(rng, jax.process_index())
    rng = jax.device_put(rng, replicated_rng_sharding)

    is_primary_process = jax.process_index() == 0

    model, model_cfg = vlm_api.init_model(
        model_cfg,
        init_rng,
        tp_size=tp_size,
        fsdp_size=fsdp_size,
    )
    with mesh_rules(mesh):
        optimizer = build_optimizer(model, train_cfg)
    train_step = make_train_step(model_cfg, pad_id=pad_id)

    per_device_flops = per_device_flops_per_step(
        model_cfg, train_cfg.seq_len, train_cfg.batch_size
    )
    timer = StepTimer(warmup=2)
    tokens_per_step = train_cfg.seq_len * train_cfg.batch_size

    checkpoint_manager = None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        _write_checkpoint_config(save_path, model_cfg)
        checkpoint_manager = _make_checkpoint_manager(save_path, save_interval=save_every or None)

    start_step = 0
    if resume:
        if checkpoint_manager is None:
            raise ValueError("resume=True requires save_dir to be provided.")
        optimizer, start_step, rng = _restore_checkpoint(checkpoint_manager, optimizer, rng)
        rng = jax.device_put(rng, replicated_rng_sharding)

    last_metrics: dict[str, float] = {}
    prev_metrics: tuple[int, dict[str, jax.Array], datetime.timedelta] | None = None

    def _log_prev_metrics(force: bool = False) -> None:
        nonlocal last_metrics
        if prev_metrics is None:
            return
        step_to_log, metrics_to_log, step_delta = prev_metrics
        result = maybe_log_step_metrics(
            step_to_log,
            metrics_to_log,
            step_delta,
            is_primary_process=is_primary_process,
            log_every=log_every,
            force=force,
            per_device_flops=per_device_flops,
            tokens_per_step=tokens_per_step,
            peak_tflops=peak_tflops,
            tb_writer=tb_writer,
        )
        if result is not None:
            last_metrics = result

    for step in range(start_step, train_cfg.num_steps):
        rng, batch_rng = jax.random.split(rng)
        rng = jax.device_put(rng, replicated_rng_sharding)
        batch = make_synthetic_batch(
            batch_rng,
            local_batch_size,
            train_cfg.seq_len,
            vlm_api.vocab_size(model_cfg),
            pad_id,
        )
        batch = vlm_api.shard_batch(batch, model_cfg, mesh)
        attention_mask = (batch != pad_id).astype(jnp.int32)
        attention_mask = vlm_api.shard_batch(attention_mask, model_cfg, mesh)
        _, metrics = train_step(optimizer, batch, attention_mask_BT=attention_mask)
        step_delta = timer.step()

        _log_prev_metrics()
        prev_metrics = (step + 1, metrics, step_delta)

        if checkpoint_manager is not None and save_every and (step + 1) % save_every == 0:
            _save_checkpoint(checkpoint_manager, optimizer, rng, step + 1)

    _log_prev_metrics(force=True)

    if checkpoint_manager is not None:
        if last_metrics and (not save_every or last_metrics["step"] % save_every != 0):
            _save_checkpoint(checkpoint_manager, optimizer, rng, int(last_metrics["step"]))
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()

    return optimizer, last_metrics


def make_sft_train_step(cfg, pad_id: int = 0):
    """Build a JIT-compiled VLM SFT train step that consumes a batch dict.

    The batch dict must contain ``token_ids_BT``, ``attention_mask_BT``, and
    ``loss_mask_BT``.  It may also contain ``pixel_values`` and
    ``image_grid_thw`` for multimodal batches.
    """

    @nnx.jit(donate_argnums=0)
    def sft_train_step(optimizer: nnx.ModelAndOptimizer, batch: dict[str, jax.Array]):
        token_ids_BT = batch["token_ids_BT"]
        attention_mask_BT = batch["attention_mask_BT"]
        loss_mask_BT = batch["loss_mask_BT"]
        pixel_values = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")
        vision_cu_seqlens = batch.get("vision_cu_seqlens")
        position_ids_ZBT = batch.get("position_ids_ZBT")

        def loss_fn(model):
            logits_BTV, aux_loss = vlm_api.forward(
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
            targets = token_ids_BT[:, 1:]
            mask = loss_mask_BT[:, 1:].astype(jnp.float32)
            loss = _masked_next_token_loss(logits_BTV[:, :-1, :], targets, mask) + aux_loss
            supervised_tokens = jnp.sum(mask)
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

        logits_BTV, aux_loss = vlm_api.forward(
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
        targets = token_ids_BT[:, 1:]
        mask = loss_mask_BT[:, 1:].astype(jnp.float32)
        loss = _masked_next_token_loss(logits_BTV[:, :-1, :], targets, mask) + aux_loss
        supervised_tokens = jnp.sum(mask)
        return loss, supervised_tokens

    return sft_eval_step


def run_sft(
    model_id_or_cfg,
    train_cfg: TrainConfig,
    data_iter: Iterator[dict[str, np.ndarray]],
    *,
    save_dir: str | Path | None = None,
    save_every: int = 0,
    log_every: int = 1,
    resume: bool = False,
    pad_id: int = 0,
    peak_tflops: float | None = None,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    profile_dir: str | Path | None = None,
    profile_steps: tuple[int, int] = (3, 8),
    tb_writer=None,
    val_data_iter: Iterator[dict[str, np.ndarray]] | None = None,
    val_every: int | None = None,
    val_steps: int = 10,
) -> tuple[nnx.ModelAndOptimizer, dict[str, float]]:
    """SFT a VLM from an external data iterator; returns final optimizer + last metrics.

    ``data_iter`` must yield dicts with keys ``token_ids_BT``,
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
    else:
        model_cfg = vlm_api.resolve_config(model_id_or_cfg)
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size)
    model_cfg = vlm_api.align_config_to_mesh(model_cfg, mesh)

    replicated_rng_sharding = NamedSharding(mesh, P())
    root_rng = jax.device_put(jax.random.key(train_cfg.seed), replicated_rng_sharding)
    init_rng, rng = jax.random.split(root_rng)
    init_rng = jax.device_put(init_rng, replicated_rng_sharding)
    rng = jax.device_put(rng, replicated_rng_sharding)

    is_primary_process = jax.process_index() == 0

    model, model_cfg = vlm_api.init_model(
        model_cfg,
        init_rng,
        tp_size=tp_size,
        fsdp_size=fsdp_size,
    )
    with mesh_rules(mesh):
        optimizer = build_optimizer(model, train_cfg)
    sft_step = make_sft_train_step(model_cfg, pad_id=pad_id)
    eval_step = make_sft_eval_step(model_cfg, pad_id=pad_id) if val_data_iter is not None else None

    per_device_flops = per_device_flops_per_step(
        model_cfg, train_cfg.seq_len, train_cfg.batch_size
    )
    timer = StepTimer(warmup=2)
    tokens_per_step = train_cfg.seq_len * train_cfg.batch_size

    checkpoint_manager = None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        _write_checkpoint_config(save_path, model_cfg)
        checkpoint_manager = _make_checkpoint_manager(save_path, save_interval=save_every or None)

    start_step = 0
    if resume:
        if checkpoint_manager is None:
            raise ValueError("resume=True requires save_dir to be provided.")
        optimizer, start_step, rng = _restore_checkpoint(checkpoint_manager, optimizer, rng)
        rng = jax.device_put(rng, replicated_rng_sharding)

    last_metrics: dict[str, float] = {}
    prev_metrics: tuple[int, dict[str, jax.Array], datetime.timedelta] | None = None

    def _log_prev_metrics(force: bool = False) -> None:
        nonlocal last_metrics
        if prev_metrics is None:
            return
        step_to_log, metrics_to_log, step_delta = prev_metrics
        result = maybe_log_step_metrics(
            step_to_log,
            metrics_to_log,
            step_delta,
            is_primary_process=is_primary_process,
            log_every=log_every,
            force=force,
            per_device_flops=per_device_flops,
            tokens_per_step=tokens_per_step,
            peak_tflops=peak_tflops,
            tb_writer=tb_writer,
        )
        if result is not None:
            last_metrics = result

    prof_start, prof_end = profile_steps
    is_profiling_active = False

    for step in range(start_step, train_cfg.num_steps):
        if profile_dir is not None and step == prof_start and not is_profiling_active:
            if is_primary_process:
                print(f"[profiler] starting trace at step {step} -> {profile_dir}")
            jax.profiler.start_trace(str(profile_dir))
            is_profiling_active = True

        batch = next(data_iter)
        batch = vlm_api.shard_batch_dict(batch, model_cfg, mesh)
        _, metrics = sft_step(optimizer, batch)
        step_delta = timer.step()

        if is_profiling_active and step + 1 >= prof_end:
            jax.tree.map(lambda x: x.block_until_ready(), metrics)
            jax.profiler.save_device_memory_profile(f"{profile_dir}/memory.prof")
            jax.profiler.stop_trace()
            is_profiling_active = False
            if is_primary_process:
                print(f"[profiler] stopped trace at step {step + 1}")

        _log_prev_metrics()
        prev_metrics = (step + 1, metrics, step_delta)

        if checkpoint_manager is not None and save_every and (step + 1) % save_every == 0:
            _save_checkpoint(checkpoint_manager, optimizer, rng, step + 1)

        if eval_step is not None and val_every and (step + 1) % val_every == 0:
            total_val_loss = 0.0
            for _ in range(val_steps):
                val_batch = next(val_data_iter)
                val_batch = vlm_api.shard_batch_dict(val_batch, model_cfg, mesh)
                val_loss, val_sup_tokens = eval_step(optimizer.model, val_batch)
                total_val_loss += float(val_loss)
            avg_val_loss = total_val_loss / val_steps
            if tb_writer is not None and is_primary_process:
                tb_writer.add_scalar("val/loss", avg_val_loss, step + 1)
                tb_writer.flush()

    if is_profiling_active:
        jax.tree.map(lambda x: x.block_until_ready(), metrics)
        jax.profiler.save_device_memory_profile(f"{profile_dir}/memory.prof")
        jax.profiler.stop_trace()
        if is_primary_process:
            print("[profiler] stopped trace at end of training")

    _log_prev_metrics(force=True)

    if checkpoint_manager is not None:
        if last_metrics and (not save_every or last_metrics["step"] % save_every != 0):
            _save_checkpoint(checkpoint_manager, optimizer, rng, int(last_metrics["step"]))
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()

    return optimizer, last_metrics
