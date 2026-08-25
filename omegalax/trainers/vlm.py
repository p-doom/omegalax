"""Training helpers for vision-language models (text-only or multimodal batches)."""

from __future__ import annotations

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
from omegalax.models.qwen3_vl.model import DECODER_LAYER_REMAT
from omegalax.models.qwen3_vl.vision import VISION_BLOCK_REMAT
from omegalax.trainers import checkpoint_utils
from omegalax.trainers.lora import LoRAParam, inject_lora
from omegalax.trainers.loss import chunked_cross_entropy_loss_sum
from omegalax.trainers.lr_schedule import build_lr_schedule
from omegalax.trainers.optim import (
    MixedPrecisionOptimizer,
    accumulate_gradient_sum,
    apply_normalized_gradient_sum,
    initialize_gradient_sum,
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
    record_deltanet_kernel,
)
from omegalax.trainers.text import startup_log
from omegalax.vlm import api as vlm_api
from omegalax.vlm.local_snapshot import LocalVLMSnapshot

P = PartitionSpec


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
    schedule_horizon: int
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
    enable_lora: bool = False
    lora_rank: int = 32
    lora_alpha: float = 32.0
    freeze_vision_tower: bool = False
    num_loss_tiles: int = 4


def init_model(
    config,
    rng: jax.Array,
    *,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    dp_size: int | None = None,
) -> nnx.Module:
    model, _ = vlm_api.init_model(
        config,
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
    schedule_horizon: int,
) -> dict[str, object]:
    return {
        "opt_state": nnx.state(optimizer.opt_state),
        "step": optimizer.step[...],
        "rng": rng,
        "schedule_horizon": jnp.asarray(schedule_horizon, dtype=jnp.int32),
    }


def _abstract_train_state(
    optimizer: MixedPrecisionOptimizer,
    rng: jax.Array,
    schedule_horizon: int,
) -> dict[str, object]:
    return jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding),
        _train_state(optimizer, rng, schedule_horizon),
    )


def _model_item(optimizer: MixedPrecisionOptimizer, model_identity: jax.Array):
    return {"state": nnx.state(optimizer.model), "identity": model_identity}


def _abstract_model_item(optimizer: MixedPrecisionOptimizer, model_identity: jax.Array):
    return jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding),
        _model_item(optimizer, model_identity),
    )


def _model_identity(model_source) -> jax.Array:
    if isinstance(model_source, LocalVLMSnapshot):
        digest = bytes.fromhex(model_source.sha256)
    else:
        digest = bytes(32)
    return jnp.asarray(list(digest), dtype=jnp.uint8)


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
    handler_registry.add("model", ocp.args.PyTreeSave, train_state_handler)
    handler_registry.add("model", ocp.args.PyTreeRestore, train_state_handler)
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
    schedule_horizon: int,
    model_identity: jax.Array,
    healthy: jax.Array,
) -> None:
    _require_healthy_at_boundary(healthy, step)
    train_state = _train_state(optimizer, rng, schedule_horizon)
    _validate_optimizer_generation(train_state, step)
    save_args = checkpoint_utils.make_grain_save_args(
        train_state,
        input_iter,
        model=_model_item(optimizer, model_identity),
    )
    if not checkpoint_manager.save(step, args=save_args, force=True):
        raise RuntimeError(f"Checkpoint {step} was not accepted")
    checkpoint_manager.wait_until_finished()


def _validate_optimizer_generation(optimizer_state, generation: int) -> None:
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
    schedule_horizon: int,
    model_identity: jax.Array,
) -> tuple[MixedPrecisionOptimizer, int, jax.Array, checkpoint_utils.GrainIterator]:
    abstract_state = _abstract_train_state(optimizer, rng, schedule_horizon)
    abstract_model = _abstract_model_item(optimizer, model_identity)
    restore_args = checkpoint_utils.make_grain_restore_args(
        abstract_state,
        input_iter,
        model=abstract_model,
    )
    restored = checkpoint_manager.restore(step, args=restore_args)
    train_state = restored["train_state"]
    model_item = restored["model"]
    restored_horizon, restored_identity = jax.device_get(
        (train_state["schedule_horizon"], model_item["identity"])
    )
    restored_horizon = int(restored_horizon)
    if restored_horizon != schedule_horizon:
        raise ValueError(
            f"Checkpoint schedule horizon is {restored_horizon}, requested {schedule_horizon}"
        )
    if bytes(restored_identity) != bytes(jax.device_get(model_identity)):
        raise ValueError("Checkpoint model snapshot does not match the requested snapshot")
    _validate_optimizer_generation(train_state, step)
    nnx.update(optimizer.model, model_item["state"])
    nnx.update(optimizer.opt_state, train_state["opt_state"])
    optimizer.step[...] = train_state["step"]
    return (
        optimizer,
        step,
        train_state["rng"],
        checkpoint_utils.restored_input_iter(restored),
    )


def make_sft_gradient_step(cfg, pad_id: int = 0, *, wrt=nnx.Param, num_loss_tiles: int = 4):
    """Build a JIT step returning one batch's masked-loss gradient sum.

    The batch dict must contain ``token_ids_BT``, ``attention_mask_BT``, and
    ``loss_mask_BT``.  It may also contain ``pixel_values`` and
    ``image_grid_thw`` for multimodal batches.

    ``wrt`` selects which model variables receive gradients. Defaults to
    ``nnx.Param`` (full FT). Pass ``LoRAParam`` for adapter-only training
    — every other ``nnx.Param`` then sees zero gradient and contributes
    no optimizer state.
    """

    diff_state = nnx.DiffState(0, wrt)

    @nnx.jit
    def sft_gradient_step(model: nnx.Module, batch: dict[str, jax.Array]):
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
            lm_weight = model.output_weight()
            loss_sum, supervised_tokens = chunked_cross_entropy_loss_sum(
                hidden_BTD,
                lm_weight,
                token_ids_BT,
                loss_mask_BT,
                num_tiles=num_loss_tiles,
                logits_out_sharding=cfg.shd_cfg.logits_btv,
            )
            loss_sum = loss_sum + aux_loss * supervised_tokens
            total_tokens = jnp.sum(attention_mask_BT.astype(jnp.float32))
            return loss_sum, (supervised_tokens, total_tokens)

        (loss_sum, (supervised_tokens, total_tokens)), grads = nnx.value_and_grad(
            loss_fn,
            argnums=diff_state,
            has_aux=True,
        )(model)
        metrics = {
            "loss_sum": loss_sum,
            "supervised_tokens": supervised_tokens,
            "total_tokens": total_tokens,
        }
        return grads, metrics

    return sft_gradient_step


def make_sft_eval_step(cfg, pad_id: int = 0, *, num_loss_tiles: int = 4):
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
        lm_weight = model.output_weight()
        loss_sum, supervised_tokens = chunked_cross_entropy_loss_sum(
            hidden_BTD,
            lm_weight,
            token_ids_BT,
            loss_mask_BT,
            num_tiles=num_loss_tiles,
            logits_out_sharding=cfg.shd_cfg.logits_btv,
        )
        return loss_sum + aux_loss * supervised_tokens, supervised_tokens

    return sft_eval_step


def _evaluate_validation_panel(eval_step, model, val_data_iter, val_steps, model_cfg, mesh):
    total_loss_sum = 0.0
    total_supervised_tokens = 0.0
    initial_state = val_data_iter.get_state()
    validation_error = None
    try:
        for _ in range(val_steps):
            batch = next(val_data_iter)
            pop_source_ids(batch)
            batch = vlm_api.shard_batch_dict(batch, model_cfg, mesh)
            loss_sum, supervised_tokens = eval_step(model, batch)
            total_loss_sum = total_loss_sum + loss_sum
            total_supervised_tokens = total_supervised_tokens + supervised_tokens
    except BaseException as error:
        validation_error = error
        raise
    finally:
        try:
            val_data_iter.set_state(initial_state)
        except BaseException as reset_error:
            if validation_error is None:
                raise
            validation_error.add_note(f"Validation iterator reset also failed: {reset_error!r}")
    healthy = (
        jnp.isfinite(total_loss_sum)
        & jnp.isfinite(total_supervised_tokens)
        & (total_supervised_tokens > 0)
    )
    return (
        total_loss_sum / jnp.maximum(total_supervised_tokens, 1.0),
        (total_supervised_tokens),
        healthy,
    )


def _validate_training_request(
    train_cfg: TrainConfig,
    resume: checkpoint_utils.ResumeMode,
    resume_step: int | None,
    save_path: Path | None,
) -> bool:
    if train_cfg.num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {train_cfg.num_steps}")
    if train_cfg.schedule_horizon <= 0:
        raise ValueError(f"schedule_horizon must be > 0, got {train_cfg.schedule_horizon}")
    if train_cfg.num_steps > train_cfg.schedule_horizon:
        raise ValueError(
            f"num_steps={train_cfg.num_steps} exceeds schedule_horizon={train_cfg.schedule_horizon}"
        )
    if resume == checkpoint_utils.ResumeMode.REQUIRED:
        if save_path is None or resume_step is None:
            raise ValueError("resume='required' requires save_dir and resume_step")
        if resume_step <= 0 or resume_step >= train_cfg.num_steps:
            raise ValueError(
                f"resume_step must be in [1, {train_cfg.num_steps}), got {resume_step}"
            )
        return True
    if resume != checkpoint_utils.ResumeMode.NEVER:
        raise ValueError(f"Unsupported resume mode: {resume!r}")
    if resume_step is not None:
        raise ValueError("resume_step is only valid with resume='required'")
    return False


def _require_healthy_at_boundary(healthy: jax.Array, step: int) -> None:
    if not bool(jax.device_get(healthy)):
        raise FloatingPointError(f"Non-finite optimizer inputs at step {step}")


@dataclasses.dataclass
class _TrainingCleanup:
    iterators: list[object] = dataclasses.field(default_factory=list)
    checkpoint_manager: ocp.CheckpointManager | None = None
    signal_handlers: dict[int, object] = dataclasses.field(default_factory=dict)

    def own_iterator(self, iterator: object | None) -> None:
        if iterator is not None and all(iterator is not owned for owned in self.iterators):
            self.iterators.append(iterator)

    def close(self, active_error: BaseException | None) -> None:
        errors: list[BaseException] = []

        def attempt(operation) -> None:
            try:
                operation()
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        if self.checkpoint_manager is not None:
            attempt(self.checkpoint_manager.wait_until_finished)
            attempt(self.checkpoint_manager.close)
        for signum, handler in self.signal_handlers.items():
            attempt(lambda signum=signum, handler=handler: signal.signal(signum, handler))
        for iterator in reversed(self.iterators):
            close = getattr(iterator, "close", None)
            if close is not None:
                attempt(close)

        if active_error is not None:
            for error in errors:
                active_error.add_note(f"Training cleanup also failed: {error!r}")
        elif len(errors) == 1:
            raise errors[0]
        elif errors:
            raise BaseExceptionGroup("Training cleanup failed", errors)


def _run_sft(
    model_source,
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
    _cleanup: _TrainingCleanup,
) -> tuple[MixedPrecisionOptimizer, dict[str, float]]:
    """SFT a VLM from a Grain iterator; returns final optimizer + last metrics.

    ``data_iter`` must be a checkpointable Grain iterator yielding dicts with keys ``token_ids_BT``,
    ``attention_mask_BT``, and ``loss_mask_BT`` (all numpy ``(B, T)``).
    Optionally ``pixel_values`` and ``image_grid_thw`` for multimodal batches.

    If ``val_data_iter`` is provided, runs ``val_steps`` forward-only batches
    every ``val_every`` training steps and logs the average validation loss.

    Required resumes name one exact checkpoint generation with ``resume_step``.
    """
    save_path = Path(save_dir).expanduser().resolve() if save_dir is not None else None
    will_resume = resume == checkpoint_utils.ResumeMode.REQUIRED

    if isinstance(model_source, str):
        raise TypeError("VLM training requires a LocalVLMSnapshot or an explicit test config")
    model_cfg = vlm_api.resolve_config(model_source)
    model_identity = _model_identity(model_source)
    startup_log("resolved model config")
    startup_log(f"model_cfg={model_cfg}")
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    model_identity = jax.device_put(model_identity, NamedSharding(mesh, P()))
    model_cfg = vlm_api.align_config_to_mesh(model_cfg, mesh)
    startup_log("mesh ready (tp/fsdp/dp)")
    batch_multiple = required_batch_multiple(vlm_api.batch_partition_spec(model_cfg), mesh)
    if train_cfg.batch_size % batch_multiple != 0:
        raise ValueError(
            f"Global batch size {train_cfg.batch_size} must be divisible by the mesh batch multiple "
            f"{batch_multiple}."
        )

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
        if will_resume and resume_step not in checkpoint_manager.all_steps():
            raise ValueError(f"Checkpoint {resume_step} does not exist")

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

    if not will_resume and isinstance(model_source, LocalVLMSnapshot):
        model, model_cfg = vlm_api.load_pretrained(
            model_source,
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
    deltanet_kernel = record_deltanet_kernel(model_cfg, wandb_run)
    if deltanet_kernel is not None:
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

    sft_gradient_step = make_sft_gradient_step(
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
        "built gradient step (jit)" + (" and eval step (jit)" if eval_step is not None else "")
    )

    accum_steps = train_cfg.grad_accum_steps
    timer = StepTimer(warmup=2 * accum_steps)
    global_tokens_per_step = train_cfg.seq_len * train_cfg.batch_size * accum_steps

    if checkpoint_manager is not None and not will_resume:
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
            model_identity,
        )
        _cleanup.own_iterator(data_iter)
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

    stop_requested = False

    def _request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    for signum in (signal.SIGUSR1, signal.SIGTERM):
        _cleanup.signal_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _request_stop)

    startup_log("entering training loop")
    if log_memory:
        log_device_memory("before first step", save_dir=save_path)
    _mem_logged_after_first_step = not log_memory
    _mem_logged_steady_state = not log_memory
    last_saved_step = start_step if will_resume else None
    optimizer_healthy_since_boundary = None

    for step_idx in range(start_step, train_cfg.num_steps):
        step = step_idx + 1

        gradient_sum = None
        accum_loss_sum = 0.0
        accum_sup_tokens = 0.0
        accum_total_tokens = 0.0
        accum_model_flops = 0.0
        accum_hardware_flops = 0.0
        accum_time = datetime.timedelta(0)
        source_counts: dict[int, int] = {}

        for _micro in range(accum_steps):
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
            gradients, metrics = sft_gradient_step(optimizer.model, batch)
            gradient_sum = (
                initialize_gradient_sum(gradients)
                if gradient_sum is None
                else accumulate_gradient_sum(gradient_sum, gradients)
            )
            micro_delta = timer.step()

            accum_loss_sum = accum_loss_sum + metrics["loss_sum"]
            accum_sup_tokens = accum_sup_tokens + metrics["supervised_tokens"]
            accum_total_tokens = accum_total_tokens + metrics["total_tokens"]
            accum_model_flops += micro_flops.model
            accum_hardware_flops += micro_flops.hardware
            accum_time += micro_delta

        if gradient_sum is None:
            raise RuntimeError("Gradient accumulation produced no microbatches")
        grad_norm, optimizer_healthy = apply_normalized_gradient_sum(
            optimizer,
            gradient_sum,
            accum_sup_tokens,
            accum_loss_sum,
        )
        accum_time += timer.step()

        if not _mem_logged_after_first_step:
            jax.block_until_ready(grad_norm)
            log_device_memory("after first step (compile done)", save_dir=save_path)
            log_live_arrays("after first step (compile done)", save_dir=save_path)
            log_compiled_memory_analysis(
                "sft_gradient_step", sft_gradient_step, save_path, optimizer.model, batch
            )
            _mem_logged_after_first_step = True
        elif not _mem_logged_steady_state and step_idx >= 4:
            jax.block_until_ready(grad_norm)
            log_device_memory("after step 5 (steady state)", save_dir=save_path)
            _mem_logged_steady_state = True

        logged = _log_prev_metrics()
        if logged or optimizer_healthy_since_boundary is None:
            optimizer_healthy_since_boundary = optimizer_healthy
        else:
            optimizer_healthy_since_boundary = optimizer_healthy_since_boundary & optimizer_healthy

        window_metrics = {
            "loss": accum_loss_sum / jnp.maximum(accum_sup_tokens, 1.0),
            "grad_norm": grad_norm,
            "supervised_tokens": accum_sup_tokens,
            "total_tokens": accum_total_tokens,
            "optimizer_healthy": optimizer_healthy_since_boundary,
            "lr": lr_schedule_fn(step_idx) if callable(lr_schedule_fn) else lr_schedule_fn,
        }
        if len(source_counts) > 1:
            total = float(sum(source_counts.values()))
            for sid, count in source_counts.items():
                window_metrics[f"data_source_{sid}_frac"] = count / total
        prev_metrics = (
            step,
            window_metrics,
            accum_time,
            StepFlops(model=accum_model_flops, hardware=accum_hardware_flops),
        )

        if stop_requested:
            _require_healthy_at_boundary(optimizer_healthy_since_boundary, step)
            startup_log(f"[signal] stopping after step={step} without checkpointing")
            return optimizer, last_metrics

        if gc_period and step % gc_period == 0:
            gc.collect()

        if eval_step is not None and val_every and step % val_every == 0:
            _require_healthy_at_boundary(optimizer_healthy_since_boundary, step)
            val_loss, total_val_sup_tokens, val_healthy = _evaluate_validation_panel(
                eval_step,
                optimizer.model,
                val_data_iter,
                val_steps,
                model_cfg,
                mesh,
            )
            _require_healthy_at_boundary(val_healthy, step)
            if wandb_run is not None and is_primary_process:
                host_val_loss, host_val_tokens = jax.device_get((val_loss, total_val_sup_tokens))
                wandb_run.log(
                    {
                        "val/loss": float(host_val_loss),
                        "val/sup_tokens": float(host_val_tokens),
                    },
                    step=step,
                )

        if stop_requested:
            _require_healthy_at_boundary(optimizer_healthy_since_boundary, step)
            startup_log(f"[signal] stopping after step={step} without checkpointing")
            return optimizer, last_metrics

        if checkpoint_manager is not None and save_every and step % save_every == 0:
            _save_sft_checkpoint(
                checkpoint_manager,
                optimizer,
                rng,
                step,
                data_iter,
                train_cfg.schedule_horizon,
                model_identity,
                optimizer_healthy_since_boundary,
            )
            last_saved_step = step

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
            model_identity,
            optimizer_healthy_since_boundary,
        )
    return optimizer, last_metrics


def run_sft(
    model_source,
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
) -> tuple[MixedPrecisionOptimizer, dict[str, float]]:
    save_path = Path(save_dir).expanduser().resolve() if save_dir is not None else None
    cleanup = _TrainingCleanup()
    cleanup.own_iterator(data_iter)
    cleanup.own_iterator(val_data_iter)
    active_error: BaseException | None = None
    try:
        _validate_training_request(train_cfg, resume, resume_step, save_path)
        if jax.process_count() != 1:
            raise ValueError("VLM training requires one JAX process")
        ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
        return _run_sft(
            model_source,
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
            _cleanup=cleanup,
        )
    except BaseException as error:
        active_error = error
        raise
    finally:
        cleanup.close(active_error)
