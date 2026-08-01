"""Additive fixed-C tail training from model-only checkpoints."""

from __future__ import annotations

import dataclasses
import datetime
import json
import time
from pathlib import Path
from typing import Any

from flax import nnx
import jax
from jax.sharding import NamedSharding, PartitionSpec
import optax

from omegalax.distributed.mesh import ensure_mesh, mesh_rules, required_batch_multiple
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api
from omegalax.text.checkpoint import ResolvedCheckpoint, resolve_checkpoint, restore_model_params
from omegalax.trainers import checkpoint_utils
from omegalax.trainers.lora import LoRAParam, inject_lora
from omegalax.trainers.optim import MixedPrecisionOptimizer
from omegalax.trainers.perf import maybe_log_step_metrics, per_device_flops_per_step, StepTimer
from omegalax.trainers.lr_schedule import build_lr_schedule
from omegalax.trainers.pretrain import (
    _accumulate_metric_sums,
    _finalize_accumulated_metrics,
    _gpu_util_snapshot,
    _iid_loss_stats,
    _install_requeue_signal_handler,
    _request_slurm_requeue_if_primary,
    _statepassing_loss_stats,
    GdnStateRouting,
    prepare_pretrain_batch,
    PretrainMode,
)
from omegalax.trainers.text import (
    _make_checkpoint_manager,
    _restore_sft_checkpoint,
    _save_sft_checkpoint,
    _write_checkpoint_config,
    init_model,
    startup_log,
)

P = PartitionSpec

LORA_PROFILE: tuple[dict[str, object], ...] = (
    {
        "r": 32,
        "alpha": 32.0,
        "target_modules": (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
        ),
    },
    {
        "r": 6,
        "alpha": 6.0,
        "target_modules": ("in_proj_a", "in_proj_b"),
    },
)


@dataclasses.dataclass(frozen=True)
class TailTrainConfig:
    seed: int = 0
    batch_size: int = 192
    seq_len: int = 2048
    num_steps: int = 4000
    global_step_offset: int = 40000
    start_lr: float = 4.19804819103e-5
    end_lr: float = 3e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_accum_steps: int = 5
    lr_schedule: str = "cosine_bridge"
    learning_rate: float | None = None
    warmup_steps: int = 0
    lr_end_factor: float = 0.0
    lr_stable_fraction: float = 0.8

    @property
    def global_stop_step(self) -> int:
        return int(self.global_step_offset) + int(self.num_steps)


def build_cosine_bridge(config: TailTrainConfig) -> optax.Schedule:
    if config.num_steps <= 1:
        raise ValueError("cosine bridge requires num_steps > 1")
    if config.start_lr <= 0 or config.end_lr < 0:
        raise ValueError("cosine bridge requires start_lr > 0 and end_lr >= 0")
    return optax.cosine_decay_schedule(
        init_value=config.start_lr,
        decay_steps=config.num_steps - 1,
        alpha=config.end_lr / config.start_lr,
    )


def build_tail_schedule(config: TailTrainConfig) -> optax.Schedule | float:
    if config.lr_schedule == "cosine_bridge":
        return build_cosine_bridge(config)
    if config.learning_rate is None:
        raise ValueError(f"learning_rate is required when lr_schedule={config.lr_schedule!r}")
    return build_lr_schedule(
        peak_lr=config.learning_rate,
        num_steps=config.num_steps,
        warmup_steps=config.warmup_steps,
        schedule=config.lr_schedule,
        end_factor=config.lr_end_factor,
        stable_fraction=config.lr_stable_fraction,
    )


def lr_contract(config: TailTrainConfig) -> dict[str, object]:
    if config.lr_schedule != "cosine_bridge":
        if config.learning_rate is None:
            raise ValueError(f"learning_rate is required when lr_schedule={config.lr_schedule!r}")
        return {
            "learning_rate": float(config.learning_rate),
            "warmup_steps": int(config.warmup_steps),
            "lr_schedule": str(config.lr_schedule),
            "lr_end_factor": float(config.lr_end_factor),
            "lr_stable_fraction": float(config.lr_stable_fraction),
            "lr_schedule_steps": int(config.num_steps),
        }
    return {
        "lr_schedule": "cosine_bridge",
        "lr_bridge_global_start_step": int(config.global_step_offset),
        "lr_bridge_total_steps": int(config.num_steps),
        "lr_bridge_start_lr": float(config.start_lr),
        "lr_bridge_end_lr": float(config.end_lr),
    }


def inject_tail_lora(model: nnx.Module, *, adapter_seed: int) -> tuple[int, ...]:
    rngs = nnx.Rngs(int(adapter_seed))
    return tuple(
        inject_lora(
            model,
            r=int(entry["r"]),
            alpha=float(entry["alpha"]),
            rngs=rngs,
            target_modules=entry["target_modules"],
        )
        for entry in LORA_PROFILE
    )


def expected_lora_target_counts(model_cfg) -> tuple[int, int]:
    layer_types = tuple(model_cfg.layer_types)
    full_layers = sum(layer_type == "full_attention" for layer_type in layer_types)
    linear_layers = sum(layer_type == "linear_attention" for layer_type in layer_types)
    return (
        3 * len(layer_types) + 4 * full_layers + 3 * linear_layers,
        2 * linear_layers,
    )


def build_tail_optimizer(
    model: nnx.Module,
    schedule: optax.Schedule,
    config: TailTrainConfig,
    *,
    wrt=nnx.Param,
) -> MixedPrecisionOptimizer:
    chain = []
    if config.max_grad_norm > 0:
        chain.append(optax.clip_by_global_norm(config.max_grad_norm))
    chain.append(
        optax.adamw(
            schedule,
            b1=config.adam_beta1,
            b2=config.adam_beta2,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
    )
    tx = optax.chain(*chain)
    if config.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=config.grad_accum_steps)
    return MixedPrecisionOptimizer(model, tx, wrt=wrt)


def make_tail_train_step(
    pretrain_mode: PretrainMode,
    model_cfg,
    *,
    pad_id: int,
    bptt_chunks: int | None,
    pass_gdn_state: bool,
    gdn_layer_limit: int | None,
    gdn_state_routing: GdnStateRouting | str,
    pass_rope_positions: bool,
    pass_conv_state: bool,
    wrt=nnx.Param,
):
    diff_state = nnx.DiffState(0, wrt)

    @nnx.jit(donate_argnums=0)
    def train_step(optimizer: MixedPrecisionOptimizer, batch: dict[str, jax.Array]):
        def loss_fn(model):
            if pretrain_mode is PretrainMode.IID_BASELINE:
                return _iid_loss_stats(model, batch, model_cfg, pad_id)
            return _statepassing_loss_stats(
                model,
                batch,
                model_cfg,
                pad_id,
                pretrain_mode,
                bptt_chunks=bptt_chunks,
                pass_gdn_state=pass_gdn_state,
                gdn_layer_limit=gdn_layer_limit,
                gdn_state_routing=gdn_state_routing,
                pass_rope_positions=pass_rope_positions,
                pass_conv_state=pass_conv_state,
            )

        (loss, metrics), grads = nnx.value_and_grad(
            loss_fn,
            argnums=diff_state,
            has_aux=True,
        )(optimizer.model)
        optimizer.update(grads)
        metrics = dict(metrics)
        metrics["grad_norm"] = optax.tree.norm(grads)
        return loss, metrics

    return train_step


def _ensure_json(path: Path, requested: dict[str, object], *, will_resume: bool) -> None:
    if path.is_file():
        if json.loads(path.read_text()) != requested:
            raise ValueError(f"Run metadata conflicts with the requested run: {path}")
        return
    if will_resume:
        raise ValueError(f"Resume requires run metadata at {path}")
    path.write_text(json.dumps(requested, indent=2, sort_keys=True) + "\n")


def _initialization_metadata(
    checkpoint: ResolvedCheckpoint | None, init_hf_model_path: str | Path | None
) -> dict[str, object]:
    if init_hf_model_path is not None:
        return {
            "source_hf_model_path": str(Path(init_hf_model_path).expanduser().resolve()),
            "restore": "huggingface_safetensors",
        }
    if checkpoint is None:
        raise ValueError("Tail initialization requires a source checkpoint or HuggingFace path")
    return {
        "source_checkpoint_root": str(checkpoint.root),
        "source_checkpoint_step": int(checkpoint.step),
        "restore": "model_parameters_only",
    }


def _lora_metadata(
    *, adapter_seed: int, wrapped_counts: tuple[int, ...], config: TailTrainConfig
) -> dict[str, object]:
    return {
        "enable_lora": True,
        "adapter_seed": int(adapter_seed),
        "frozen_backbone": True,
        "dropout": 0.0,
        "wrapped_counts": list(wrapped_counts),
        "profile": [
            {
                "r": int(entry["r"]),
                "alpha": float(entry["alpha"]),
                "target_modules": list(entry["target_modules"]),
            }
            for entry in LORA_PROFILE
        ],
        "lr_contract": lr_contract(config),
    }


def run_tail_pretrain(
    init_model_checkpoint: str | Path | None,
    config: TailTrainConfig,
    data_iter: checkpoint_utils.GrainIterator,
    *,
    pretrain_mode: PretrainMode | str,
    save_dir: str | Path,
    enable_lora: bool = False,
    lora_adapter_seed: int = 0,
    save_every: int = 500,
    keep_latest: int | None = 10,
    log_every: int = 10,
    resume: checkpoint_utils.ResumeMode = checkpoint_utils.ResumeMode.IF_PRESENT,
    pad_id: int = 0,
    peak_tflops: float | None = None,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
    dp_size: int | None = None,
    wandb_run=None,
    text_attn_backend: str = "mosaic_gpu",
    bptt_chunks: int | None = None,
    pass_gdn_state: bool = True,
    gdn_layer_limit: int | None = None,
    gdn_state_routing: GdnStateRouting | str = GdnStateRouting.SAME_LAYER,
    pass_rope_positions: bool = False,
    pass_conv_state: bool = False,
    stop_after_local_step: int | None = None,
    init_hf_model_path: str | Path | None = None,
    restore_input_iter: bool = True,
    resume_expected_step: int | None = None,
) -> tuple[MixedPrecisionOptimizer, dict[str, float]]:
    pretrain_mode = PretrainMode(pretrain_mode)
    save_path = Path(save_dir).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = _make_checkpoint_manager(
        save_path, save_interval=save_every or None, max_to_keep=keep_latest
    )
    latest_step = checkpoint_manager.latest_step()
    if resume == checkpoint_utils.ResumeMode.REQUIRED and latest_step is None:
        raise ValueError(f"resume='required' but no checkpoint found at {save_path}")
    will_resume = (
        resume in (checkpoint_utils.ResumeMode.IF_PRESENT, checkpoint_utils.ResumeMode.REQUIRED)
        and latest_step is not None
    )
    if resume_expected_step is not None and latest_step != int(resume_expected_step):
        raise ValueError(
            f"Expected resume checkpoint {resume_expected_step}, found {latest_step} at {save_path}."
        )
    if (init_model_checkpoint is None) == (init_hf_model_path is None):
        raise ValueError("Provide exactly one tail initialization source")
    source_checkpoint = (
        resolve_checkpoint(init_model_checkpoint) if init_model_checkpoint is not None else None
    )
    requested_lr_contract = lr_contract(config)
    if will_resume:
        stored_contract = dict(
            checkpoint_utils.restore_lr_contract(checkpoint_manager, int(latest_step))
        )
        if stored_contract != requested_lr_contract:
            raise ValueError(
                f"LR contract mismatch on tail resume: checkpoint={stored_contract}, "
                f"requested={requested_lr_contract}"
            )
        if not config.global_step_offset <= int(latest_step) < config.global_stop_step:
            raise ValueError(
                f"Checkpoint step {latest_step} is outside tail interval "
                f"({config.global_step_offset}, {config.global_stop_step}]"
            )

    model_cfg = text_api.resolve_config(
        str(save_path)
        if will_resume
        else str(
            init_hf_model_path if init_hf_model_path is not None else source_checkpoint.config_path
        )
    )
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
    model_cfg = text_api.align_config_to_mesh(model_cfg, mesh)
    batch_multiple = required_batch_multiple(text_api.batch_partition_spec(model_cfg), mesh)
    if config.batch_size % batch_multiple:
        raise ValueError(
            f"Global batch size {config.batch_size} must be divisible by {batch_multiple}."
        )

    replicated = NamedSharding(mesh, P())
    root_rng = jax.device_put(jax.random.key(config.seed), replicated)
    init_rng, rng = jax.random.split(root_rng)
    if not will_resume and init_hf_model_path is not None:
        from omegalax.models.qwen3_5.params import create_qwen3_5_from_safetensors

        model, _ = create_qwen3_5_from_safetensors(
            str(init_hf_model_path),
            tp_size=tp_size,
            fsdp_size=fsdp_size,
            dp_size=dp_size,
            text_only=True,
        )
        startup_log(f"loaded text model parameters from {Path(init_hf_model_path).resolve()}")
    else:
        model, model_cfg = init_model(
            model_cfg,
            jax.device_put(init_rng, replicated),
            tp_size=tp_size,
            fsdp_size=fsdp_size,
            dp_size=dp_size,
        )
        if not will_resume:
            model = restore_model_params(model, source_checkpoint)
            startup_log(
                f"restored model parameters only from {source_checkpoint.step_path}; "
                "optimizer, RNG, and iterator start fresh"
            )

    wrt_filter = nnx.Param
    wrapped_counts: tuple[int, ...] = ()
    if enable_lora:
        with mesh_rules(mesh):
            wrapped_counts = inject_tail_lora(model, adapter_seed=lora_adapter_seed)
        expected_counts = expected_lora_target_counts(model_cfg)
        if wrapped_counts != expected_counts:
            raise ValueError(
                f"Unexpected LoRA target counts: expected {expected_counts}, got {wrapped_counts}"
            )
        wrt_filter = LoRAParam

    _ensure_json(
        save_path / "initialization.json",
        _initialization_metadata(source_checkpoint, init_hf_model_path),
        will_resume=will_resume,
    )
    if enable_lora:
        _ensure_json(
            save_path / "lora_metadata.json",
            _lora_metadata(
                adapter_seed=lora_adapter_seed,
                wrapped_counts=wrapped_counts,
                config=config,
            ),
            will_resume=will_resume,
        )

    set_attn_backend(model, text_backend=text_attn_backend)
    schedule = build_tail_schedule(config)
    with mesh_rules(mesh):
        optimizer = build_tail_optimizer(model, schedule, config, wrt=wrt_filter)
    train_step = make_tail_train_step(
        pretrain_mode,
        model_cfg,
        pad_id=pad_id,
        bptt_chunks=bptt_chunks,
        pass_gdn_state=pass_gdn_state,
        gdn_layer_limit=gdn_layer_limit,
        gdn_state_routing=gdn_state_routing,
        pass_rope_positions=pass_rope_positions,
        pass_conv_state=pass_conv_state,
        wrt=wrt_filter,
    )
    if not will_resume:
        _write_checkpoint_config(save_path, model_cfg)

    start_local_step = 0
    if will_resume:
        startup_log(
            f"restoring checkpoint step={latest_step} restore_input_iter={restore_input_iter}"
        )
        optimizer, restored_step, rng, data_iter = _restore_sft_checkpoint(
            checkpoint_manager,
            optimizer,
            rng,
            data_iter,
            restore_input_iter=restore_input_iter,
        )
        start_local_step = int(restored_step) - int(config.global_step_offset)
        rng = jax.device_put(rng, replicated)

    local_stop_step = (
        config.num_steps if stop_after_local_step is None else int(stop_after_local_step)
    )
    if not start_local_step < local_stop_step <= config.num_steps:
        raise ValueError(
            f"Local stop step must be in ({start_local_step}, {config.num_steps}], "
            f"got {local_stop_step}"
        )

    timer = StepTimer(warmup=2 * config.grad_accum_steps)
    global_tokens_per_step = config.seq_len * config.batch_size * config.grad_accum_steps
    is_primary = jax.process_index() == 0
    last_metrics: dict[str, float] = {}
    prev_metrics: tuple[int, dict[str, Any], datetime.timedelta, float] | None = None

    def log_previous(force: bool = False) -> None:
        nonlocal last_metrics
        if prev_metrics is None:
            return
        step, metrics, elapsed, flops = prev_metrics
        result = maybe_log_step_metrics(
            step,
            metrics,
            elapsed,
            is_primary_process=is_primary,
            log_every=log_every,
            force=force,
            per_device_flops=flops,
            global_tokens_per_step=global_tokens_per_step,
            peak_tflops=peak_tflops,
            wandb_run=wandb_run,
            batch_size=config.batch_size * config.grad_accum_steps,
        )
        if result is not None:
            last_metrics = result

    requeue_requested = _install_requeue_signal_handler()
    startup_log(
        f"entering additive tail loop mode={pretrain_mode} "
        f"global_steps={config.global_step_offset + 1}..{config.global_stop_step}"
    )
    for local_step_idx in range(start_local_step, local_stop_step):
        local_step = local_step_idx + 1
        global_step = config.global_step_offset + local_step
        accum: dict[str, float] = {}
        accum_grad_norm = 0.0
        accum_flops = 0.0
        accum_time = datetime.timedelta(0)
        accum_data_wait_s = 0.0
        for _ in range(config.grad_accum_steps):
            wait_start = time.perf_counter()
            raw_batch = next(data_iter)
            accum_data_wait_s += time.perf_counter() - wait_start
            batch, _, _ = prepare_pretrain_batch(
                raw_batch,
                pretrain_mode,
                model_cfg,
                mesh,
                global_batch_size=config.batch_size,
                pass_rope_positions=pass_rope_positions,
            )
            micro_flops = per_device_flops_per_step(model_cfg, config.seq_len, config.batch_size)
            _, metrics = train_step(optimizer, batch)
            micro_delta = timer.step()
            _accumulate_metric_sums(accum, metrics)
            accum_grad_norm += metrics["grad_norm"]
            accum_flops += micro_flops
            accum_time += micro_delta

        with jax.default_device("cpu"):
            current_lr = float(schedule(local_step_idx))
        metrics = _finalize_accumulated_metrics(accum)
        metrics["grad_norm"] = accum_grad_norm / config.grad_accum_steps
        metrics["lr"] = current_lr
        metrics["total_tokens"] = global_step * global_tokens_per_step
        metrics["tail_step"] = local_step
        metrics["tail_tokens"] = local_step * global_tokens_per_step
        metrics["data_wait_s"] = accum_data_wait_s
        step_time_s = accum_time.total_seconds()
        metrics["data_wait_frac"] = accum_data_wait_s / step_time_s if step_time_s > 0 else 0.0
        if is_primary:
            metrics.update(_gpu_util_snapshot())
        log_previous()
        prev_metrics = (global_step, metrics, accum_time, accum_flops)

        if save_every and local_step % save_every == 0:
            _save_sft_checkpoint(
                checkpoint_manager,
                optimizer,
                rng,
                global_step,
                data_iter,
                lr_contract=requested_lr_contract,
            )
        if requeue_requested():
            _save_sft_checkpoint(
                checkpoint_manager,
                optimizer,
                rng,
                global_step,
                data_iter,
                lr_contract=requested_lr_contract,
                force=True,
            )
            checkpoint_manager.wait_until_finished()
            checkpoint_manager.close()
            _request_slurm_requeue_if_primary(is_primary)
            return optimizer, last_metrics

    log_previous(force=True)
    if last_metrics and (
        not save_every or (int(last_metrics["step"]) - config.global_step_offset) % save_every != 0
    ):
        _save_sft_checkpoint(
            checkpoint_manager,
            optimizer,
            rng,
            int(last_metrics["step"]),
            data_iter,
            lr_contract=requested_lr_contract,
            force=True,
        )
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    return optimizer, last_metrics
