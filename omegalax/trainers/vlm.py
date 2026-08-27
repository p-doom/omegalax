"""Training helpers for vision-language models (text-only or multimodal batches)."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import gc
import signal
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec
from orbax.checkpoint import checkpoint_managers as ocm

from omegalax import export as export_lib
from omegalax.data.grain_pipeline import pop_source_ids
from omegalax.distributed.mesh import ensure_mesh, mesh_rules, required_batch_multiple
from omegalax.models.params_utils import save_hf_config
from omegalax.models.qwen3_5 import Qwen3_5Config
from omegalax.models.qwen3_5.kernels import resolve_backend as resolve_deltanet_backend
from omegalax.models.qwen3_vl import Qwen3VLConfig
from omegalax.models.qwen3_vl.model import DECODER_LAYER_REMAT
from omegalax.models.qwen3_vl.vision import VISION_BLOCK_REMAT
from omegalax.trainers import checkpoint_utils
from omegalax.trainers import tokamax_cache as tokamax_cache_lib
from omegalax.trainers.lora import LoRAParam, inject_lora
from omegalax.trainers.loss import chunked_cross_entropy_loss_sum
from omegalax.trainers.lr_schedule import build_lr_schedule
from omegalax.trainers.optim import (
    MixedPrecisionOptimizer,
    update_from_gradient_sum,
)
from omegalax.trainers.perf import (
    StepFlops,
    StepTimer,
    log_compiled_memory_analysis,
    log_device_memory,
    log_live_arrays,
    log_pytree_bytes,
    log_top_leaves_with_paths,
    maybe_log_step_metrics,
    per_device_step_flops,
)
from omegalax.trainers.text import startup_log
from omegalax.vlm import api as vlm_api

P = PartitionSpec


def require_zero_router_aux_loss(cfg) -> None:
    if isinstance(cfg, Qwen3_5Config):
        text_cfg = cfg.text_config
        has_nonzero_aux = bool(text_cfg.num_experts) and text_cfg.router_aux_loss_coef != 0.0
    elif isinstance(cfg, Qwen3VLConfig):
        has_nonzero_aux = any(cfg.is_moe_layer(layer_idx) for layer_idx in range(cfg.num_layers))
    else:
        raise TypeError(f"Unsupported VLM config type: {type(cfg)}")
    if has_nonzero_aux:
        raise ValueError("Gradient accumulation requires zero router auxiliary loss")


def _require_healthy_at_boundary(healthy: jax.Array, step: int) -> None:
    if not bool(jax.device_get(healthy)):
        raise FloatingPointError(f"Non-finite optimizer inputs at step {step}")


def _validate_resume_request(
    resume: checkpoint_utils.ResumeMode,
    resume_step: int | None,
    save_path: Path | None,
    num_steps: int,
) -> bool:
    if resume not in (checkpoint_utils.ResumeMode.NEVER, checkpoint_utils.ResumeMode.REQUIRED):
        raise ValueError(f"VLM training does not support resume={resume.value!r}")
    will_resume = resume == checkpoint_utils.ResumeMode.REQUIRED
    if will_resume:
        if save_path is None or resume_step is None:
            raise ValueError("resume='required' requires save_dir and resume_step")
        if resume_step <= 0 or resume_step >= num_steps:
            raise ValueError(f"resume_step must be in [1, {num_steps}), got {resume_step}")
    elif resume_step is not None:
        raise ValueError("resume_step is only valid with resume='required'")
    return will_resume


def _validate_train_config(train_cfg: TrainConfig) -> None:
    if train_cfg.num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {train_cfg.num_steps}")
    if train_cfg.schedule_horizon <= 0:
        raise ValueError(f"schedule_horizon must be > 0, got {train_cfg.schedule_horizon}")
    if train_cfg.num_steps > train_cfg.schedule_horizon:
        raise ValueError(
            f"num_steps={train_cfg.num_steps} exceeds schedule_horizon={train_cfg.schedule_horizon}"
        )


def _trainable_non_vision(path, x):
    """NNX filter predicate: select every ``nnx.Param`` whose state-tree path
    does not pass through ``Qwen3VL.vision`` (or any nested ``vision``
    attribute). Used as the ``wrt`` filter for full-FT-with-frozen-vision
    training. Mirrors ``DEFAULT_SKIP_PATHS = ("vision",)`` in ``lora.py``.
    """
    if not isinstance(x, nnx.Param):
        return False
    for part in path:
        key = getattr(part, "key", None) or getattr(part, "name", None) or str(part)
        if key == "vision":
            return False
    return True


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    batch_size: int = 8
    seq_len: int = 64
    num_steps: int = 20
    schedule_horizon: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    lr_schedule: str = "linear"
    lr_end_factor: float = 0.0
    lr_stable_fraction: float = 0.8
    max_grad_norm: float = 0.0
    grad_accum_steps: int = 1
    print_every: int = 1
    enable_lora: bool = False
    lora_rank: int = 32
    lora_alpha: float = 32.0
    freeze_vision_tower: bool = False
    num_loss_tiles: int = 4


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


def build_optimizer(
    model: nnx.Module,
    lr_schedule_fn: optax.Schedule | float,
    train_cfg: TrainConfig,
    *,
    wrt=nnx.Param,
) -> MixedPrecisionOptimizer:
    chain = []
    if train_cfg.max_grad_norm > 0:
        chain.append(optax.clip_by_global_norm(train_cfg.max_grad_norm))
    wd = 0.0 if wrt is LoRAParam else train_cfg.weight_decay
    chain.append(optax.adamw(lr_schedule_fn, weight_decay=wd))
    tx = optax.chain(*chain)
    opt = MixedPrecisionOptimizer(model, tx, wrt=wrt)
    return opt


def _train_state(
    optimizer: MixedPrecisionOptimizer,
    rng: jax.Array,
    schedule_num_steps: int,
) -> dict[str, object]:
    return {
        "optimizer": nnx.state(optimizer),
        "rng": rng,
        "schedule_num_steps": jnp.asarray(schedule_num_steps, dtype=jnp.int32),
    }


def _abstract_train_state(
    optimizer: MixedPrecisionOptimizer,
    rng: jax.Array,
    schedule_num_steps: int,
) -> dict[str, object]:
    return jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding),
        _train_state(optimizer, rng, schedule_num_steps),
    )


def _make_checkpoint_manager(
    save_dir: Path,
    save_interval: int | None,
    keep_period: int | None = None,
    keep_latest: int | None = None,
) -> ocp.CheckpointManager:
    """Orbax requires an absolute checkpoint path.

    ``keep_period`` permanently retains every checkpoint whose step is a multiple
    of it (e.g. full-epoch boundaries); for it to ever fire it must be a multiple
    of ``save_interval`` since the loop only saves at multiples of ``save_interval``.
    ``keep_latest`` additionally retains the N most-recent checkpoints. When
    ``keep_period`` is unset the manager keeps every checkpoint (prior behavior).
    """
    save_dir = Path(save_dir).expanduser().resolve()
    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    train_state_handler = ocp.handlers.PyTreeCheckpointHandler(
        save_device_host_concurrent_gb=2,
        is_prioritized_key_fn=lambda _: False,
    )
    handler_registry.add("train_state", ocp.args.PyTreeSave, train_state_handler)
    handler_registry.add("train_state", ocp.args.PyTreeRestore, train_state_handler)
    checkpoint_utils.register_grain_iterator_handler(handler_registry)
    preservation_policy = None
    if keep_period:
        policies = [ocm.EveryNSteps(keep_period, exact_interval=True)]
        if keep_latest:
            policies.append(ocm.LatestN(keep_latest))
        preservation_policy = ocm.AnyPreservationPolicy(policies)
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=save_interval,
        step_format_fixed_length=6,
        cleanup_tmp_directories=True,
        preservation_policy=preservation_policy,
        enable_async_checkpointing=False,
    )
    return ocp.CheckpointManager(save_dir, options=options, handler_registry=handler_registry)


def _write_checkpoint_config(save_dir: Path, cfg) -> None:
    save_hf_config(export_lib.model_config_to_hf_dict(cfg), save_dir)


def _write_lora_metadata(save_dir: Path, train_cfg: TrainConfig) -> None:
    """Persist LoRA settings alongside the orbax tree."""
    import json

    meta = {
        "enable_lora": bool(train_cfg.enable_lora),
        "lora_rank": int(train_cfg.lora_rank) if train_cfg.enable_lora else None,
        "lora_alpha": float(train_cfg.lora_alpha) if train_cfg.enable_lora else None,
    }
    (Path(save_dir) / "lora_metadata.json").write_text(json.dumps(meta, indent=2))


def _save_sft_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    optimizer: MixedPrecisionOptimizer,
    rng: jax.Array,
    step: int,
    input_iter: checkpoint_utils.GrainIterator,
    schedule_num_steps: int,
) -> None:
    _validate_optimizer_generation(nnx.state(optimizer), step)
    train_state = _train_state(optimizer, rng, schedule_num_steps)
    save_args = checkpoint_utils.make_grain_save_args(train_state, input_iter)
    if not checkpoint_manager.save(step, args=save_args, force=True):
        raise RuntimeError(f"Checkpoint {step} was not accepted")
    checkpoint_manager.wait_until_finished()


def _validate_optimizer_generation(optimizer_state: nnx.State, generation: int) -> None:
    step = optimizer_state["step"][...]
    counts = [
        leaf
        for leaf in jax.tree.leaves(nnx.pure(optimizer_state["opt_state"]))
        if leaf.shape == () and jnp.issubdtype(leaf.dtype, jnp.integer)
    ]
    if not counts:
        raise ValueError("Optimizer state has no integer update count")
    host_step, host_counts = jax.device_get((step, counts))
    values = [int(host_step), *(int(count) for count in host_counts)]
    if any(value != generation for value in values):
        raise ValueError(
            f"Checkpoint generation {generation} does not match optimizer counters {values}"
        )


def _restore_sft_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    optimizer: MixedPrecisionOptimizer,
    rng: jax.Array,
    input_iter: checkpoint_utils.GrainIterator,
    step: int,
    schedule_num_steps: int,
) -> tuple[MixedPrecisionOptimizer, int, jax.Array, checkpoint_utils.GrainIterator]:
    if step not in checkpoint_manager.all_steps():
        raise ValueError(f"Checkpoint {step} does not exist")

    abstract_state = _abstract_train_state(optimizer, rng, schedule_num_steps)
    restore_args = checkpoint_utils.make_grain_restore_args(abstract_state, input_iter)
    restored = checkpoint_manager.restore(step, args=restore_args)
    train_state = restored["train_state"]
    restored_schedule_num_steps = int(jax.device_get(train_state["schedule_num_steps"]))
    if restored_schedule_num_steps != schedule_num_steps:
        raise ValueError(
            f"Checkpoint schedule horizon is {restored_schedule_num_steps}, "
            f"requested {schedule_num_steps}"
        )
    _validate_optimizer_generation(train_state["optimizer"], step)
    nnx.update(optimizer, train_state["optimizer"])
    return (
        optimizer,
        step,
        train_state["rng"],
        checkpoint_utils.restored_input_iter(restored),
    )


def make_sft_train_step(cfg, pad_id: int = 0, *, wrt=nnx.Param, num_loss_tiles: int = 4):
    """Build a JIT-compiled optimizer step over one accumulation window.

    The batch dict must contain ``token_ids_BT``, ``attention_mask_BT``,
    ``loss_mask_BT``, and ``vision_patch_valid``. It may also contain
    ``pixel_values`` and ``image_grid_thw`` for multimodal batches.

    ``wrt`` selects which model variables receive gradients. Defaults to
    ``nnx.Param`` (full FT). Pass ``LoRAParam`` for adapter-only training
    — every other ``nnx.Param`` then sees zero gradient and contributes
    no optimizer state.
    """

    diff_state = nnx.DiffState(0, wrt)

    @nnx.jit(donate_argnums=0)
    def sft_train_step(
        optimizer: MixedPrecisionOptimizer,
        batches: tuple[dict[str, jax.Array], ...],
    ):
        gradient_sum = None
        loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
        supervised_tokens = jnp.asarray(0.0, dtype=jnp.float32)
        total_tokens = jnp.asarray(0.0, dtype=jnp.float32)

        def loss_fn(model, batch):
            token_ids_BT = batch["token_ids_BT"]
            attention_mask_BT = batch["attention_mask_BT"]
            loss_mask_BT = batch["loss_mask_BT"]
            hidden_BTD, aux_loss = vlm_api.forward(
                model,
                token_ids_BT,
                pad_id,
                cfg,
                attention_mask_BT=attention_mask_BT,
                pixel_values=batch.get("pixel_values"),
                vision_patch_valid=batch["vision_patch_valid"],
                image_grid_thw=batch.get("image_grid_thw"),
                vision_cu_seqlens=batch.get("vision_cu_seqlens"),
                position_ids_ZBT=batch.get("position_ids_ZBT"),
            )
            lm_weight = model.output_weight()
            batch_loss_sum, batch_supervised_tokens = chunked_cross_entropy_loss_sum(
                hidden_BTD,
                lm_weight,
                token_ids_BT,
                loss_mask_BT,
                num_tiles=num_loss_tiles,
                logits_out_sharding=cfg.shd_cfg.logits_btv,
            )
            batch_total_tokens = jnp.sum(attention_mask_BT.astype(jnp.float32))
            return batch_loss_sum, (
                batch_supervised_tokens,
                batch_total_tokens,
                aux_loss,
            )

        gradient_fn = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)
        for batch in batches:
            (
                (
                    batch_loss_sum,
                    (batch_supervised_tokens, batch_total_tokens, _),
                ),
                gradients,
            ) = gradient_fn(optimizer.model, batch)
            gradient_sum = (
                jax.tree.map(lambda gradient: gradient.astype(jnp.float32), gradients)
                if gradient_sum is None
                else jax.tree.map(
                    lambda total, gradient: total + gradient.astype(jnp.float32),
                    gradient_sum,
                    gradients,
                )
            )
            loss_sum = loss_sum + batch_loss_sum
            supervised_tokens = supervised_tokens + batch_supervised_tokens
            total_tokens = total_tokens + batch_total_tokens

        if gradient_sum is None:
            raise ValueError("At least one microbatch is required")
        grad_norm, loss, healthy = update_from_gradient_sum(
            optimizer,
            gradient_sum,
            supervised_tokens,
            loss_sum,
        )
        return loss, {
            "loss": loss,
            "grad_norm": grad_norm,
            "optimizer_healthy": healthy,
            "supervised_tokens": supervised_tokens,
            "total_tokens": total_tokens,
        }

    return sft_train_step


def make_sft_eval_step(cfg, pad_id: int = 0, *, num_loss_tiles: int = 4):
    """Build a JIT-compiled VLM SFT eval step (forward only, no gradients)."""

    @nnx.jit
    def sft_eval_step(model: nnx.Module, batch: dict[str, jax.Array]):
        token_ids_BT = batch["token_ids_BT"]
        attention_mask_BT = batch["attention_mask_BT"]
        loss_mask_BT = batch["loss_mask_BT"]
        pixel_values = batch.get("pixel_values")
        vision_patch_valid = batch["vision_patch_valid"]
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
            vision_patch_valid=vision_patch_valid,
            image_grid_thw=image_grid_thw,
            vision_cu_seqlens=vision_cu_seqlens,
            position_ids_ZBT=position_ids_ZBT,
        )
        lm_weight = model.output_weight()
        ce_loss_sum, supervised_tokens = chunked_cross_entropy_loss_sum(
            hidden_BTD,
            lm_weight,
            token_ids_BT,
            loss_mask_BT,
            num_tiles=num_loss_tiles,
            logits_out_sharding=cfg.shd_cfg.logits_btv,
        )
        return ce_loss_sum, supervised_tokens, aux_loss

    return sft_eval_step


@dataclasses.dataclass
class _TrainingCleanup:
    data_iter: object
    val_data_iter: object | None
    checkpoint_manager: ocp.CheckpointManager | None = None
    autotune_context: object | None = None
    signal_handlers: dict[int, object] = dataclasses.field(default_factory=dict)

    def close(self, active_error: BaseException | None) -> None:
        errors: list[BaseException] = []
        if self.autotune_context is not None:
            try:
                self.autotune_context.__exit__(None, None, None)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
        if self.checkpoint_manager is not None:
            try:
                self.checkpoint_manager.wait_until_finished()
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
            try:
                self.checkpoint_manager.close()
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
        for signum, handler in self.signal_handlers.items():
            try:
                signal.signal(signum, handler)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
        closed: set[int] = set()
        for iterator in (self.data_iter, self.val_data_iter):
            close = getattr(iterator, "close", None)
            if close is None or id(iterator) in closed:
                continue
            closed.add(id(iterator))
            try:
                close()
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
        if not errors:
            return
        if active_error is not None:
            for error in errors:
                active_error.add_note(f"Training cleanup also failed: {error!r}")
            return
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup("Training cleanup failed", errors)


def _run_sft(
    model_id_or_cfg,
    train_cfg: TrainConfig,
    data_iter: checkpoint_utils.GrainIterator,
    *,
    save_dir: str | Path | None = None,
    save_every: int = 0,
    keep_period: int = 0,
    keep_latest: int = 1,
    log_every: int = 1,
    resume: checkpoint_utils.ResumeMode = checkpoint_utils.ResumeMode.NEVER,
    resume_step: int | None = None,
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
    log_memory: bool = False,
    tokamax_cache_dir: str | Path | None = None,
    _cleanup: _TrainingCleanup,
) -> tuple[MixedPrecisionOptimizer, dict[str, float]]:
    """SFT a VLM from a Grain iterator; returns final optimizer + last metrics.

    ``data_iter`` must be a checkpointable Grain iterator yielding dicts with keys ``token_ids_BT``,
    ``attention_mask_BT``, ``loss_mask_BT`` (all numpy ``(B, T)``), and
    ``vision_patch_valid``.
    Optionally ``pixel_values`` and ``image_grid_thw`` for multimodal batches.

    If ``val_data_iter`` is provided, runs ``val_steps`` forward-only batches
    every ``val_every`` training steps and logs the average validation loss.

    Required resumes name one exact checkpoint generation with ``resume_step``.
    """
    save_path = Path(save_dir).expanduser().resolve() if save_dir is not None else None
    will_resume = _validate_resume_request(resume, resume_step, save_path, train_cfg.num_steps)

    checkpoint_manager: ocp.CheckpointManager | None = None
    if save_path is not None:
        if will_resume:
            if not save_path.is_dir():
                raise ValueError(f"Checkpoint directory does not exist: {save_path}")
        else:
            if save_path.exists() and any(save_path.iterdir()):
                raise ValueError(f"Fresh checkpoint directory is not empty: {save_path}")
            save_path.mkdir(parents=True, exist_ok=True)
        checkpoint_manager = _make_checkpoint_manager(
            save_path,
            save_interval=save_every or None,
            keep_period=keep_period or None,
            keep_latest=keep_latest or None,
        )
        _cleanup.checkpoint_manager = checkpoint_manager

    if will_resume:
        model_cfg = vlm_api.resolve_config(str(save_path))
        startup_log(f"resolved model config from checkpoint {save_path!r}")
    else:
        model_cfg = vlm_api.resolve_config(model_id_or_cfg)
        startup_log("resolved model config")
    require_zero_router_aux_loss(model_cfg)
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
        num_steps=train_cfg.schedule_horizon,
        warmup_steps=train_cfg.warmup_steps,
        schedule=train_cfg.lr_schedule,
        end_factor=train_cfg.lr_end_factor,
        stable_fraction=train_cfg.lr_stable_fraction,
    )

    if not will_resume and isinstance(model_id_or_cfg, str):
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
    if wandb_run is not None and is_primary_process:
        wandb_run.config.update(
            {"model_cfg": export_lib.model_config_to_hf_dict(model_cfg)},
            allow_val_change=True,
        )
    from omegalax.models.sharding_runtime import set_attn_backend

    set_attn_backend(model, text_backend=text_attn_backend)
    startup_log(f"set attn backend: text={text_attn_backend}")
    if isinstance(model_cfg, Qwen3_5Config):
        deltanet_kernel = resolve_deltanet_backend()
        if wandb_run is not None and is_primary_process:
            wandb_run.config.update({"deltanet_kernel": deltanet_kernel}, allow_val_change=True)
        startup_log(f"deltanet kernel: {deltanet_kernel}")
    if train_cfg.enable_lora and train_cfg.freeze_vision_tower:
        raise ValueError(
            "--enable_lora already freezes the vision tower; "
            "--freeze_vision_tower is redundant. Pass at most one."
        )
    if train_cfg.enable_lora:
        with mesh_rules(mesh):
            n_wrapped = inject_lora(
                model,
                r=train_cfg.lora_rank,
                alpha=train_cfg.lora_alpha,
                rngs=nnx.Rngs(train_cfg.seed),
            )
        startup_log(
            f"LoRA enabled: r={train_cfg.lora_rank} alpha={train_cfg.lora_alpha} "
            f"wrapped {n_wrapped} text-decoder Linear projections; vision frozen"
        )
        wrt_filter = LoRAParam
    elif train_cfg.freeze_vision_tower:
        wrt_filter = _trainable_non_vision
        startup_log(
            "vision tower frozen; full FT on text decoder + embedder + lm_head + layernorms"
        )
    else:
        wrt_filter = nnx.Param

    if log_memory:
        log_pytree_bytes("params (after load)", nnx.state(model, nnx.Param), save_dir=save_path)
        log_device_memory("after model load", save_dir=save_path)

    with mesh_rules(mesh):
        optimizer = build_optimizer(model, lr_schedule_fn, train_cfg, wrt=wrt_filter)

    startup_log("built optimizer")
    if log_memory:
        log_pytree_bytes("optimizer.opt_state", nnx.state(optimizer.opt_state), save_dir=save_path)
        log_pytree_bytes("optimizer (params + state)", nnx.state(optimizer), save_dir=save_path)
        log_top_leaves_with_paths(
            "optimizer (params + state) by path", nnx.state(optimizer), save_dir=save_path
        )
        log_device_memory("after optimizer build", save_dir=save_path)

    sft_train_step = make_sft_train_step(
        model_cfg,
        pad_id=pad_id,
        wrt=wrt_filter,
        num_loss_tiles=train_cfg.num_loss_tiles,
    )
    eval_step = (
        make_sft_eval_step(model_cfg, pad_id=pad_id, num_loss_tiles=train_cfg.num_loss_tiles)
        if val_data_iter is not None
        else None
    )
    startup_log(
        "built train step (jit)" + (" and eval step (jit)" if eval_step is not None else "")
    )

    accum_steps = train_cfg.grad_accum_steps
    timer = StepTimer(warmup=2)
    global_tokens_per_step = train_cfg.seq_len * train_cfg.batch_size * accum_steps

    if checkpoint_manager is not None and not will_resume:
        # Write the HF config alongside the orbax tree only on a fresh start;
        # on resume the file was written by the original run and matches by
        # construction (we just resolved model_cfg from it).
        _write_checkpoint_config(save_path, model_cfg)
        _write_lora_metadata(save_path, train_cfg)
    if checkpoint_manager is not None:
        startup_log(f"checkpoint manager ready at {save_path!r}")

    start_step = 0
    if will_resume:
        optimizer, start_step, rng, data_iter = _restore_sft_checkpoint(
            checkpoint_manager,
            optimizer,
            rng,
            data_iter,
            resume_step,
            train_cfg.schedule_horizon,
        )
        rng = jax.device_put(rng, replicated_rng_sharding)
        startup_log(f"restored checkpoint at step {start_step}")

    last_metrics: dict[str, float] = {}
    prev_metrics: tuple[int, dict[str, jax.Array], datetime.timedelta, float] | None = None

    def _log_prev_metrics(force: bool = False) -> bool:
        nonlocal last_metrics
        if prev_metrics is None:
            return False
        step_to_log, metrics_to_log, step_delta, step_flops = prev_metrics
        result = maybe_log_step_metrics(
            step_to_log,
            metrics_to_log,
            step_delta,
            is_primary_process=is_primary_process,
            log_every=log_every,
            force=force,
            step_flops=step_flops,
            global_tokens_per_step=global_tokens_per_step,
            peak_tflops=peak_tflops,
            wandb_run=wandb_run,
            batch_size=train_cfg.batch_size * accum_steps,
        )
        if result is not None:
            if not result.pop("optimizer_healthy"):
                raise FloatingPointError(f"Non-finite optimizer inputs at step {step_to_log}")
            last_metrics = result
            return True
        return False

    # Flag-only handler: doing an orbax save or JAX collective from inside
    # the signal handler deadlocks (handlers run on arbitrary threads and
    # re-enter the runtime). The flag is read at a safe per-step point.
    stop_requested = False

    def _request_stop(signum, _frame):
        nonlocal stop_requested
        if not stop_requested:
            startup_log(f"[signal] received {signum}; will stop after current step")
        stop_requested = True

    for signum in (signal.SIGUSR1, signal.SIGTERM):
        _cleanup.signal_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _request_stop)

    autotune_result = None
    pending_batches = None
    if tokamax_cache_dir is not None:
        autotune_result = tokamax_cache_lib.try_load(tokamax_cache_dir)
        if autotune_result is None:
            startup_log("priming tokamax autotuning with first training step")
            pending_batches = tuple(next(data_iter) for _ in range(accum_steps))
            autotune_batches = []
            for pending_batch in pending_batches:
                autotune_batch = dict(pending_batch)
                pop_source_ids(autotune_batch)
                autotune_batches.append(vlm_api.shard_batch_dict(autotune_batch, model_cfg, mesh))
            autotune_result = tokamax_cache_lib.autotune_and_save(
                tokamax_cache_dir,
                sft_train_step,
                optimizer,
                tuple(autotune_batches),
            )

    _autotune_ctx = autotune_result if autotune_result is not None else contextlib.nullcontext()
    _autotune_ctx.__enter__()
    _cleanup.autotune_context = _autotune_ctx

    startup_log("entering training loop")
    if log_memory:
        log_device_memory("before first step", save_dir=save_path)
    _mem_logged_after_first_step = not log_memory
    _mem_logged_steady_state = not log_memory
    last_saved_step = start_step if will_resume else None
    optimizer_healthy_since_boundary = None

    for step_idx in range(start_step, train_cfg.num_steps):
        step = step_idx + 1

        accum_model_flops = 0.0
        accum_hardware_flops = 0.0
        source_counts: dict[int, int] = {}
        step_batches = []

        for micro_idx in range(accum_steps):
            if pending_batches is not None:
                batch = pending_batches[micro_idx]
            else:
                batch = next(data_iter)
            sids = pop_source_ids(batch)
            if sids is not None:
                for sid in sids.tolist():
                    source_counts[sid] = source_counts.get(sid, 0) + 1
            grid_thw = batch.get("image_grid_thw")
            # LoRA freezes base weights (no weight-grad); remat flags from the model modules.
            micro_flops = per_device_step_flops(
                model_cfg,
                train_cfg.seq_len,
                train_cfg.batch_size,
                image_grid_thw=grid_thw,
                base_weights_trainable=not train_cfg.enable_lora,
                vision_trainable=not (train_cfg.freeze_vision_tower or train_cfg.enable_lora),
                decoder_remat=DECODER_LAYER_REMAT,
                vision_remat=VISION_BLOCK_REMAT,
            )
            batch = vlm_api.shard_batch_dict(batch, model_cfg, mesh)
            step_batches.append(batch)
            accum_model_flops += micro_flops.model
            accum_hardware_flops += micro_flops.hardware
        pending_batches = None

        _, metrics = sft_train_step(
            optimizer,
            tuple(step_batches),
        )
        accum_time = timer.step()

        if not _mem_logged_after_first_step:
            jax.block_until_ready(metrics["grad_norm"])
            log_device_memory("after first step (compile done)", save_dir=save_path)
            log_live_arrays("after first step (compile done)", save_dir=save_path)
            log_compiled_memory_analysis(
                "sft_train_step", sft_train_step, save_path, optimizer, tuple(step_batches)
            )
            _mem_logged_after_first_step = True
        elif not _mem_logged_steady_state and step_idx >= 4:
            jax.block_until_ready(metrics["grad_norm"])
            log_device_memory("after step 5 (steady state)", save_dir=save_path)
            _mem_logged_steady_state = True

        logged = _log_prev_metrics()
        if logged or optimizer_healthy_since_boundary is None:
            optimizer_healthy_since_boundary = metrics["optimizer_healthy"]
        else:
            optimizer_healthy_since_boundary = (
                optimizer_healthy_since_boundary & metrics["optimizer_healthy"]
            )

        with jax.default_device("cpu"):
            window_metrics = {
                "loss": metrics["loss"],
                "grad_norm": metrics["grad_norm"],
                "supervised_tokens": metrics["supervised_tokens"],
                "total_tokens": metrics["total_tokens"],
                "optimizer_healthy": optimizer_healthy_since_boundary,
                "lr": (
                    float(lr_schedule_fn(step_idx))
                    if callable(lr_schedule_fn)
                    else float(lr_schedule_fn)
                ),
            }
            if len(source_counts) > 1:
                total = float(sum(source_counts.values()))
                for sid, cnt in source_counts.items():
                    window_metrics[f"data_source_{sid}_frac"] = cnt / total
            prev_metrics = (
                step,
                window_metrics,
                accum_time,
                StepFlops(model=accum_model_flops, hardware=accum_hardware_flops),
            )

        if checkpoint_manager is not None and save_every and step % save_every == 0:
            _require_healthy_at_boundary(optimizer_healthy_since_boundary, step)
            _save_sft_checkpoint(
                checkpoint_manager,
                optimizer,
                rng,
                step,
                data_iter,
                train_cfg.schedule_horizon,
            )
            last_saved_step = step

        if gc_period and step % gc_period == 0:
            gc.collect()

        if eval_step is not None and val_every and step % val_every == 0:
            total_val_ce_loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
            total_val_sup_tokens = jnp.asarray(0.0, dtype=jnp.float32)
            total_val_aux_loss = jnp.asarray(0.0, dtype=jnp.float32)
            for _ in range(val_steps):
                val_batch = next(val_data_iter)
                pop_source_ids(val_batch)
                val_batch = vlm_api.shard_batch_dict(val_batch, model_cfg, mesh)
                val_ce_loss_sum, val_sup_tokens, val_aux_loss = eval_step(
                    optimizer.model, val_batch
                )
                total_val_ce_loss_sum += val_ce_loss_sum
                total_val_sup_tokens += val_sup_tokens
                total_val_aux_loss += val_aux_loss
            total_val_ce_loss_sum, total_val_sup_tokens, total_val_aux_loss = jax.device_get(
                (total_val_ce_loss_sum, total_val_sup_tokens, total_val_aux_loss)
            )
            if total_val_sup_tokens <= 0:
                raise ValueError("Validation window has no supervised tokens")
            avg_val_ce_loss = float(total_val_ce_loss_sum / total_val_sup_tokens)
            avg_val_aux_loss = float(total_val_aux_loss / val_steps)
            avg_val_loss = avg_val_ce_loss + avg_val_aux_loss
            if wandb_run is not None and is_primary_process:
                wandb_run.log(
                    {
                        "val/loss": avg_val_loss,
                        "val/ce_loss": avg_val_ce_loss,
                        "val/aux_loss": avg_val_aux_loss,
                        "val/sup_tokens": float(total_val_sup_tokens),
                    },
                    step=step,
                )

        if stop_requested:
            startup_log(f"[signal] saving checkpoint at step={step} and stopping")
            if checkpoint_manager is not None and last_saved_step != step:
                _require_healthy_at_boundary(optimizer_healthy_since_boundary, step)
                _save_sft_checkpoint(
                    checkpoint_manager,
                    optimizer,
                    rng,
                    step,
                    data_iter,
                    train_cfg.schedule_horizon,
                )
            return optimizer, last_metrics

    _log_prev_metrics(force=True)

    if (
        checkpoint_manager is not None
        and last_metrics
        and last_saved_step != int(last_metrics["step"])
    ):
        _save_sft_checkpoint(
            checkpoint_manager,
            optimizer,
            rng,
            int(last_metrics["step"]),
            data_iter,
            train_cfg.schedule_horizon,
        )
    return optimizer, last_metrics


def run_sft(
    model_id_or_cfg,
    train_cfg: TrainConfig,
    data_iter: checkpoint_utils.GrainIterator,
    *,
    save_dir: str | Path | None = None,
    save_every: int = 0,
    keep_period: int = 0,
    keep_latest: int = 1,
    log_every: int = 1,
    resume: checkpoint_utils.ResumeMode = checkpoint_utils.ResumeMode.NEVER,
    resume_step: int | None = None,
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
    log_memory: bool = False,
    tokamax_cache_dir: str | Path | None = None,
) -> tuple[MixedPrecisionOptimizer, dict[str, float]]:
    cleanup = _TrainingCleanup(data_iter, val_data_iter)
    active_error: BaseException | None = None
    try:
        return _run_sft(
            model_id_or_cfg,
            train_cfg,
            data_iter,
            save_dir=save_dir,
            save_every=save_every,
            keep_period=keep_period,
            keep_latest=keep_latest,
            log_every=log_every,
            resume=resume,
            resume_step=resume_step,
            pad_id=pad_id,
            peak_tflops=peak_tflops,
            tp_size=tp_size,
            fsdp_size=fsdp_size,
            dp_size=dp_size,
            wandb_run=wandb_run,
            val_data_iter=val_data_iter,
            val_every=val_every,
            val_steps=val_steps,
            text_attn_backend=text_attn_backend,
            gc_period=gc_period,
            log_memory=log_memory,
            tokamax_cache_dir=tokamax_cache_dir,
            _cleanup=cleanup,
        )
    except BaseException as error:
        active_error = error
        raise
    finally:
        cleanup.close(active_error)
