"""Training helpers for vision-language models (text-only or multimodal batches)."""

from __future__ import annotations

import dataclasses
import datetime
import enum
import gc
import json
import os
import stat
from pathlib import Path

import grain
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec
from orbax.checkpoint import checkpoint_managers as ocm

from omegalax import export as export_lib
from omegalax.data.grain_pipeline import pop_source_ids
from omegalax.distributed.mesh import ensure_mesh, mesh_rules, required_batch_multiple
from omegalax.models.params_utils import save_hf_config
from omegalax.models.qwen3_5 import Qwen3_5Config
from omegalax.models.qwen3_vl import Qwen3VLConfig
from omegalax.models.qwen3_vl.model import DECODER_LAYER_REMAT
from omegalax.models.qwen3_vl.vision import VISION_BLOCK_REMAT
from omegalax.trainers import checkpoint_utils
from omegalax.trainers.lora import LoRAParam, inject_lora
from omegalax.trainers.loss import chunked_cross_entropy_loss, chunked_cross_entropy_loss_sum
from omegalax.trainers.lr_schedule import build_lr_schedule
from omegalax.trainers.optim import (
    MixedPrecisionOptimizer,
    OptimizerFatalStatus,
    OptimizerStatusBoundary,
    accumulate_gradient_sum,
    apply_normalized_gradient_sum,
    generation_adamw,
    initialize_gradient_sum,
    require_healthy_optimizer_status,
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

P = PartitionSpec
_MAX_GRAIN_CHECKPOINT_BYTES = 16 * 1024 * 1024


def require_zero_router_aux_loss(cfg) -> None:
    """Reject models whose nonlinear router objective cannot be micro-accumulated exactly."""
    if isinstance(cfg, Qwen3_5Config):
        text_cfg = cfg.text_config
        has_nonzero_aux = bool(text_cfg.num_experts) and text_cfg.router_aux_loss_coef != 0.0
    elif isinstance(cfg, Qwen3VLConfig):
        has_nonzero_aux = any(cfg.is_moe_layer(layer_idx) for layer_idx in range(cfg.num_layers))
    else:
        raise TypeError(f"Unsupported VLM config type: {type(cfg)}")
    if has_nonzero_aux:
        raise ValueError(
            "Gradient accumulation does not support a nonzero router auxiliary loss: "
            "router statistics are nonlinear across microbatches. Use a dense model until "
            "the trainer accumulates exact router sufficient statistics."
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
    schedule_horizon: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    lr_schedule: str = "linear"
    lr_end_factor: float = 0.0
    lr_stable_fraction: float = 0.8
    max_grad_norm: float = 1.0
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
    train_cfg: TrainConfig,
    *,
    wrt=nnx.Param,
) -> MixedPrecisionOptimizer:
    if not np.isfinite(train_cfg.max_grad_norm) or train_cfg.max_grad_norm <= 0:
        raise ValueError("VLM training requires a positive finite max_grad_norm.")
    wd = 0.0 if wrt is LoRAParam else train_cfg.weight_decay
    tx = generation_adamw(weight_decay=wd)
    opt = MixedPrecisionOptimizer(model, tx, wrt=wrt)
    return opt


def _train_state(optimizer: MixedPrecisionOptimizer) -> dict[str, object]:
    return {"optimizer": nnx.state(optimizer)}


def _abstract_train_state(optimizer: MixedPrecisionOptimizer) -> dict[str, object]:
    return {
        "optimizer": jax.tree.map(
            lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding),
            nnx.state(optimizer),
        ),
    }


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
    schema_handler = ocp.handlers.JsonCheckpointHandler()
    handler_registry.add("schema", ocp.args.JsonSave, schema_handler)
    handler_registry.add("schema", ocp.args.JsonRestore, schema_handler)
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


class _CheckpointCommitMode(enum.Enum):
    PERIODIC = enum.auto()
    FORCED = enum.auto()
    REUSE = enum.auto()


@dataclasses.dataclass(frozen=True)
class _SFTCheckpointCommit:
    step: int
    checkpoint_manager: ocp.CheckpointManager
    optimizer: MixedPrecisionOptimizer
    input_iter: checkpoint_utils.GrainIterator
    schedule_horizon: int
    invocation_end_step: int


def _commit_sft_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    optimizer: MixedPrecisionOptimizer,
    step: int,
    input_iter: checkpoint_utils.GrainIterator,
    schedule_horizon: int,
    invocation_end_step: int,
    optimizer_status: jax.Array,
    boundary: OptimizerStatusBoundary,
    mode: _CheckpointCommitMode,
    prior_commit: _SFTCheckpointCommit | None,
) -> _SFTCheckpointCommit:
    require_healthy_optimizer_status(optimizer_status, boundary)
    _validate_optimizer_generation(nnx.state(optimizer), step)
    if mode is _CheckpointCommitMode.REUSE:
        if (
            prior_commit is None
            or prior_commit.step != step
            or prior_commit.checkpoint_manager is not checkpoint_manager
            or prior_commit.optimizer is not optimizer
            or prior_commit.input_iter is not input_iter
            or prior_commit.schedule_horizon != schedule_horizon
            or prior_commit.invocation_end_step != invocation_end_step
        ):
            raise ValueError(
                "Checkpoint reuse requires a commit from the identical step, checkpoint manager, "
                "optimizer, train iterator, schedule horizon, and invocation boundary."
            )
        commit = prior_commit
    elif mode in (_CheckpointCommitMode.PERIODIC, _CheckpointCommitMode.FORCED):
        if prior_commit is not None:
            raise ValueError("A new checkpoint save cannot reuse a prior commit.")
        save_args = ocp.args.Composite(
            train_state=ocp.args.PyTreeSave(_train_state(optimizer)),
            input_iter=grain.checkpoint.CheckpointSave(input_iter),
            schema=ocp.args.JsonSave(
                _sft_checkpoint_schema(
                    optimizer,
                    input_iter,
                    schedule_horizon,
                    invocation_end_step,
                )
            ),
        )
        saved = checkpoint_manager.save(
            step,
            args=save_args,
            force=mode is _CheckpointCommitMode.FORCED,
        )
        if saved is not True:
            raise RuntimeError(f"Checkpoint manager did not save step {step}.")
        commit = _SFTCheckpointCommit(
            step,
            checkpoint_manager,
            optimizer,
            input_iter,
            schedule_horizon,
            invocation_end_step,
        )
    else:
        raise ValueError(f"Unsupported checkpoint commit mode: {mode!r}.")

    checkpoint_manager.wait_until_finished()
    latest_step = checkpoint_manager.latest_step()
    if latest_step != step:
        raise RuntimeError(
            f"Checkpoint commit mismatch: expected latest step {step}, got {latest_step}."
        )
    return commit


def _commit_phase_end(
    checkpoint_manager: ocp.CheckpointManager | None,
    optimizer: MixedPrecisionOptimizer,
    input_iter: checkpoint_utils.GrainIterator,
    schedule_horizon: int,
    invocation_end_step: int,
    save_every: int,
    optimizer_status: jax.Array,
) -> _SFTCheckpointCommit | None:
    require_healthy_optimizer_status(optimizer_status, OptimizerStatusBoundary.FINAL)
    if checkpoint_manager is None or (save_every and invocation_end_step % save_every == 0):
        return None
    return _commit_sft_checkpoint(
        checkpoint_manager,
        optimizer,
        invocation_end_step,
        input_iter,
        schedule_horizon,
        invocation_end_step,
        optimizer_status,
        OptimizerStatusBoundary.FINAL,
        _CheckpointCommitMode.FORCED,
        None,
    )


def _state_path(path: tuple[object, ...]) -> str:
    return ".".join(str(part) for part in path)


def _validate_optimizer_restore(expected: nnx.State, restored: object) -> None:
    if type(restored) is not type(expected):
        raise ValueError(
            f"Checkpoint optimizer must be {type(expected).__name__}, got "
            f"{type(restored).__name__}."
        )
    expected_flat = expected.flat_state()
    restored_flat = restored.flat_state()
    expected_paths = tuple(expected_flat.paths)
    restored_paths = tuple(restored_flat.paths)
    if restored_paths != expected_paths:
        raise ValueError(
            "Checkpoint optimizer paths do not match the initialized optimizer: "
            f"expected {expected_paths}, got {restored_paths}."
        )
    for path, expected_leaf, restored_leaf in zip(
        expected_paths, expected_flat.leaves, restored_flat.leaves, strict=True
    ):
        name = _state_path(path)
        if type(restored_leaf) is not type(expected_leaf):
            raise ValueError(
                f"Checkpoint optimizer variable {name} must be "
                f"{type(expected_leaf).__name__}, got {type(restored_leaf).__name__}."
            )
        if restored_leaf.shape != expected_leaf.shape:
            raise ValueError(
                f"Checkpoint optimizer variable {name} shape must be "
                f"{expected_leaf.shape}, got {restored_leaf.shape}."
            )
        if restored_leaf.dtype != expected_leaf.dtype:
            raise ValueError(
                f"Checkpoint optimizer variable {name} dtype must be "
                f"{expected_leaf.dtype}, got {restored_leaf.dtype}."
            )


def _validate_optimizer_generation(optimizer_state: nnx.State, generation: int) -> None:
    if type(generation) is not int or not (0 < generation <= np.iinfo(np.int32).max):
        raise ValueError("Checkpoint generation must be a positive int32-range integer.")
    step = optimizer_state["step"][...]
    if step.shape != () or step.dtype != jnp.uint32:
        raise ValueError(
            f"Checkpoint NNX step must be scalar uint32, got shape={step.shape} dtype={step.dtype}."
        )
    exact_opt_leaves = [
        value
        for value in jax.tree.leaves(nnx.pure(optimizer_state["opt_state"]))
        if not jnp.issubdtype(value.dtype, jnp.inexact)
    ]
    if (
        len(exact_opt_leaves) != 1
        or exact_opt_leaves[0].shape != ()
        or exact_opt_leaves[0].dtype != jnp.int32
    ):
        raise ValueError(
            "Checkpoint optimizer state must contain exactly one scalar int32 Adam count; "
            f"got {[(value.shape, value.dtype) for value in exact_opt_leaves]}."
        )
    host_step, host_count = jax.device_get((step, exact_opt_leaves[0]))
    if int(host_step) != generation or int(host_count) != generation:
        raise ValueError(
            f"Checkpoint generation {generation} must equal NNX step and Adam count; "
            f"got step={int(host_step)} count={int(host_count)}."
        )


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key {key!r} in Grain iterator checkpoint.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant {value!r} in Grain iterator checkpoint.")


def _parse_iterator_state(raw: str) -> object:
    return json.loads(
        raw,
        object_pairs_hook=_json_object,
        parse_constant=_reject_json_constant,
    )


def _validate_json_schema(expected: object, restored: object, path: str = "input_iter") -> None:
    if type(restored) is not type(expected):
        raise ValueError(
            f"Checkpoint {path} type must be {type(expected).__name__}, got "
            f"{type(restored).__name__}."
        )
    if isinstance(expected, dict):
        if set(restored) != set(expected):
            raise ValueError(
                f"Checkpoint {path} keys must be {sorted(expected)}, got {sorted(restored)}."
            )
        for key in expected:
            _validate_json_schema(expected[key], restored[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(restored) != len(expected):
            raise ValueError(
                f"Checkpoint {path} length must be {len(expected)}, got {len(restored)}."
            )
        for index, (expected_value, restored_value) in enumerate(zip(expected, restored)):
            _validate_json_schema(expected_value, restored_value, f"{path}[{index}]")


def _type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _iterator_state(input_iter: checkpoint_utils.GrainIterator) -> object:
    state = input_iter.get_state()
    if isinstance(state, bytes):
        return _parse_iterator_state(state.decode("utf-8"))
    return _parse_iterator_state(json.dumps(state))


def _iterator_schema(value: object) -> object:
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": [[key, _iterator_schema(value[key])] for key in sorted(value)],
        }
    if isinstance(value, list):
        return {"type": "list", "items": [_iterator_schema(item) for item in value]}
    return {"type": _type_name(value)}


def _sft_checkpoint_schema(
    optimizer: MixedPrecisionOptimizer,
    input_iter: checkpoint_utils.GrainIterator,
    schedule_horizon: int,
    invocation_end_step: int,
) -> dict[str, object]:
    optimizer_state = nnx.state(optimizer).flat_state()
    optimizer_leaves = []
    for path, leaf in zip(optimizer_state.paths, optimizer_state.leaves, strict=True):
        optimizer_leaves.append(
            {
                "path": list(path),
                "variable_type": _type_name(leaf),
                "shape": list(leaf.shape),
                "dtype": str(leaf.dtype),
            }
        )
    return {
        "version": 2,
        "optimizer": optimizer_leaves,
        "phase": {
            "schedule_horizon": schedule_horizon,
            "invocation_end_step": invocation_end_step,
        },
        "input_iter": _iterator_schema(_iterator_state(input_iter)),
    }


def _validate_checkpoint_phase(
    restored_schema: object,
    expected_schema: dict[str, object],
    step: int,
    invocation_end_step: int,
) -> None:
    if type(restored_schema) is not dict or set(restored_schema) != set(expected_schema):
        raise ValueError("Checkpoint schema does not match the initialized training contract.")
    restored_phase = restored_schema.get("phase")
    if type(restored_phase) is not dict or set(restored_phase) != {
        "schedule_horizon",
        "invocation_end_step",
    }:
        raise ValueError("Checkpoint phase schema is invalid.")
    stored_horizon = restored_phase["schedule_horizon"]
    stored_end = restored_phase["invocation_end_step"]
    if (
        type(stored_horizon) is not int
        or type(stored_end) is not int
        or stored_horizon <= 0
        or stored_end <= 0
        or stored_end > stored_horizon
    ):
        raise ValueError("Checkpoint phase fields must be bounded positive integers.")
    comparable = dict(expected_schema)
    comparable["phase"] = {
        "schedule_horizon": expected_schema["phase"]["schedule_horizon"],
        "invocation_end_step": stored_end,
    }
    if restored_schema != comparable:
        raise ValueError("Checkpoint schema does not match the initialized training contract.")
    if stored_end == invocation_end_step:
        return
    if step == stored_end and invocation_end_step > stored_end:
        raise PermissionError(
            "Extending invocation_end_step requires a registrar-authorized parent/child phase "
            "capability; local checkpoint arithmetic is not phase authority."
        )
    raise ValueError(
        "A phase extension must restore its parent exactly at the parent's invocation_end_step; "
        f"checkpoint step={step}, stored end={stored_end}, requested end={invocation_end_step}."
    )


def _read_iterator_checkpoint(state_path: Path) -> bytes:
    try:
        fd = os.open(state_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError(
            f"Grain iterator checkpoint must be a readable no-follow regular file: {state_path}."
        ) from error
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Grain iterator checkpoint is not a regular file: {state_path}.")
        if before.st_size > _MAX_GRAIN_CHECKPOINT_BYTES:
            raise ValueError(
                f"Grain iterator checkpoint exceeds {_MAX_GRAIN_CHECKPOINT_BYTES} bytes: "
                f"{state_path} has {before.st_size}."
            )
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 64 * 1024))
            if not chunk:
                raise ValueError(f"Grain iterator checkpoint changed while reading: {state_path}.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ValueError(f"Grain iterator checkpoint changed while reading: {state_path}.")
        after = os.fstat(fd)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if after_identity != before_identity:
            raise ValueError(f"Grain iterator checkpoint changed while reading: {state_path}.")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _restore_iterator_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    step: int,
    input_iter: checkpoint_utils.GrainIterator,
) -> checkpoint_utils.GrainIterator:
    expected = _iterator_state(input_iter)
    process_index = jax.process_index()
    process_count = jax.process_count()
    state_path = (
        Path(checkpoint_manager.directory)
        / f"{step:06d}"
        / "input_iter"
        / f"process_{process_index}-of-{process_count}.json"
    )
    raw_state = _read_iterator_checkpoint(state_path)
    restored = _parse_iterator_state(raw_state.decode("utf-8"))
    _validate_json_schema(expected, restored)
    if isinstance(input_iter, grain.DatasetIterator):
        input_iter.set_state(restored)
    elif isinstance(input_iter, grain.DataLoaderIterator):
        input_iter.set_state(raw_state)
    else:
        raise TypeError(f"Unsupported Grain iterator type: {type(input_iter).__name__}.")
    input_iter.start_prefetch()
    return input_iter


def _restore_sft_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    optimizer: MixedPrecisionOptimizer,
    step: int,
    input_iter: checkpoint_utils.GrainIterator,
    schedule_horizon: int,
    invocation_end_step: int,
) -> tuple[MixedPrecisionOptimizer, int, checkpoint_utils.GrainIterator]:
    abstract_state = _abstract_train_state(optimizer)
    expected_state = nnx.state(optimizer)
    metadata = checkpoint_manager.item_metadata(step)
    if set(metadata.keys()) != {"input_iter", "schema", "train_state"}:
        raise ValueError(
            f"Checkpoint {step} items must be exactly input_iter, schema, and train_state, got "
            f"{sorted(metadata.keys())}."
        )

    restored_schema = checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(schema=ocp.args.JsonRestore()),
    )
    if set(restored_schema.keys()) != {"schema"}:
        raise ValueError(
            f"Checkpoint {step} schema restore returned keys {sorted(restored_schema.keys())}."
        )
    schema = restored_schema["schema"]
    if type(schema) is not dict or type(schema.get("version")) is not int or schema["version"] != 2:
        version = schema.get("version") if isinstance(schema, dict) else None
        raise ValueError(
            f"Checkpoint schema version {version!r} is incompatible with fresh-run schema 2; "
            "historical scheduled-optimizer checkpoints require the separate writer-lineage "
            "recovery path."
        )
    expected_schema = _sft_checkpoint_schema(
        optimizer,
        input_iter,
        schedule_horizon,
        invocation_end_step,
    )
    _validate_checkpoint_phase(
        schema,
        expected_schema,
        step,
        invocation_end_step,
    )

    restored = checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(train_state=ocp.args.PyTreeRestore(abstract_state)),
    )
    if set(restored.keys()) != {"train_state"}:
        raise ValueError(
            f"Checkpoint {step} train-state restore returned keys {sorted(restored.keys())}."
        )
    train_state = restored["train_state"]
    if type(train_state) is not dict or set(train_state) != {"optimizer"}:
        keys = sorted(train_state) if isinstance(train_state, dict) else type(train_state).__name__
        raise ValueError(
            f"Checkpoint {step} train_state must contain exactly optimizer, got {keys}."
        )
    _validate_optimizer_restore(expected_state, train_state["optimizer"])
    _validate_optimizer_generation(train_state["optimizer"], step)
    restored_input_iter = _restore_iterator_checkpoint(checkpoint_manager, step, input_iter)

    nnx.update(optimizer, train_state["optimizer"])
    return optimizer, step, restored_input_iter


def make_sft_gradient_step(cfg, pad_id: int = 0, *, wrt=nnx.Param, num_loss_tiles: int = 4):
    """Build a read-only JIT step returning the masked-CE gradient sum for one batch.

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
            ce_loss_sum, supervised_tokens = chunked_cross_entropy_loss_sum(
                hidden_BTD,
                lm_weight,
                token_ids_BT,
                loss_mask_BT,
                num_tiles=num_loss_tiles,
                logits_out_sharding=cfg.shd_cfg.logits_btv,
            )
            total_tokens = jnp.sum(attention_mask_BT.astype(jnp.float32))
            return ce_loss_sum, (supervised_tokens, total_tokens, aux_loss)

        (ce_loss_sum, (supervised_tokens, total_tokens, aux_loss)), grads = nnx.value_and_grad(
            loss_fn,
            argnums=diff_state,
            has_aux=True,
        )(model)
        metrics = {
            "ce_loss_sum": ce_loss_sum,
            "aux_loss": aux_loss,
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
        loss = (
            chunked_cross_entropy_loss(
                hidden_BTD,
                lm_weight,
                token_ids_BT,
                loss_mask_BT,
                num_tiles=num_loss_tiles,
                logits_out_sharding=cfg.shd_cfg.logits_btv,
            )
            + aux_loss
        )
        supervised_tokens = jnp.sum(loss_mask_BT[:, 1:].astype(jnp.float32))
        return loss, supervised_tokens

    return sft_eval_step


def _validate_resume_request(
    resume: checkpoint_utils.ResumeMode,
    resume_step: int | None,
    save_path: Path | None,
    invocation_end_step: int,
) -> bool:
    if type(resume) is not checkpoint_utils.ResumeMode:
        raise TypeError(f"resume must be ResumeMode, got {type(resume).__name__}.")
    if resume is checkpoint_utils.ResumeMode.IF_PRESENT:
        raise ValueError(
            "VLM resume does not support if_present; choose never for a new checkpoint root or "
            "required with an explicit resume_step."
        )
    if resume is checkpoint_utils.ResumeMode.NEVER:
        if resume_step is not None:
            raise ValueError("resume_step must be unset when resume='never'.")
        if save_path is not None and save_path.exists():
            raise ValueError(
                f"resume='never' requires a new checkpoint root, but {save_path} already exists."
            )
        return False
    if resume is not checkpoint_utils.ResumeMode.REQUIRED:
        raise ValueError(f"Unsupported VLM resume mode: {resume!r}.")
    if save_path is None:
        raise ValueError("resume='required' requires save_dir.")
    if not save_path.is_dir():
        raise ValueError(
            f"resume='required' requires an existing checkpoint root, got {save_path}."
        )
    if type(resume_step) is not int or resume_step <= 0:
        raise ValueError("resume='required' requires a positive integer resume_step.")
    if resume_step >= invocation_end_step:
        raise ValueError(
            f"resume_step={resume_step} must be less than "
            f"invocation_end_step={invocation_end_step}."
        )
    return True


def _validate_training_phase(train_cfg: TrainConfig, invocation_end_step: int) -> None:
    int32_max = int(np.iinfo(np.int32).max)
    if type(train_cfg.schedule_horizon) is not int or not (
        0 < train_cfg.schedule_horizon <= int32_max
    ):
        raise ValueError(f"schedule_horizon must be an integer in [1, {int32_max}].")
    if type(invocation_end_step) is not int or not (
        0 < invocation_end_step <= train_cfg.schedule_horizon
    ):
        raise ValueError(
            "invocation_end_step must be a positive integer no greater than "
            f"schedule_horizon={train_cfg.schedule_horizon}."
        )


def _require_single_jax_process() -> None:
    process_count = jax.process_count()
    if type(process_count) is not int or process_count != 1:
        raise RuntimeError(
            "Production VLM training requires exactly one JAX process; "
            f"got process_count={process_count!r}."
        )


def _require_registrar_compiled_executable_capability() -> None:
    raise RuntimeError(
        "Production VLM training requires a registrar-authorized compiled-executable "
        "capability; the authority adapter is not available."
    )


def _require_checkpoint_frontier(
    checkpoint_manager: ocp.CheckpointManager,
    resume_step: int,
    save_path: Path,
) -> None:
    committed_steps = checkpoint_manager.all_steps()
    committed_frontier = max(committed_steps, default=None)
    if committed_frontier != resume_step:
        raise ValueError(
            f"resume_step={resume_step} does not match the committed checkpoint frontier "
            f"{committed_frontier} at {save_path}."
        )


@dataclasses.dataclass
class _TrainingCleanup:
    checkpoint_manager: ocp.CheckpointManager | None = None

    def close(
        self,
        active_error: BaseException | None,
    ) -> None:
        errors: list[BaseException] = []
        if self.checkpoint_manager is not None:
            try:
                self.checkpoint_manager.wait_until_finished()
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
            try:
                self.checkpoint_manager.close()
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
    invocation_end_step: int,
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

    A fresh run uses ``resume=never`` and a new ``save_dir``. A continuation uses
    ``resume=required`` and names the exact committed frontier with ``resume_step``.
    """
    _validate_training_phase(train_cfg, invocation_end_step)
    save_path = Path(save_dir).expanduser().resolve() if save_dir is not None else None
    will_resume = _validate_resume_request(resume, resume_step, save_path, invocation_end_step)

    checkpoint_manager: ocp.CheckpointManager | None = None
    if save_path is not None:
        if not will_resume:
            save_path.mkdir(parents=True)
        checkpoint_manager = _make_checkpoint_manager(
            save_path,
            save_interval=save_every or None,
            keep_period=keep_period or None,
            keep_latest=keep_latest or None,
        )
        _cleanup.checkpoint_manager = checkpoint_manager
        if will_resume:
            _require_checkpoint_frontier(checkpoint_manager, resume_step, save_path)

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

    replicated_sharding = NamedSharding(mesh, P())
    init_rng = jax.device_put(jax.random.key(train_cfg.seed), replicated_sharding)
    startup_log("placed initialization rng on device mesh")

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
        optimizer = build_optimizer(model, train_cfg, wrt=wrt_filter)

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
        # Write the HF config alongside the orbax tree only on a fresh start;
        # on resume the file was written by the original run and matches by
        # construction (we just resolved model_cfg from it).
        _write_checkpoint_config(save_path, model_cfg)
        _write_lora_metadata(save_path, train_cfg)
    if checkpoint_manager is not None:
        startup_log(f"checkpoint manager ready at {save_path!r}")

    start_step = 0
    if will_resume:
        optimizer, start_step, data_iter = _restore_sft_checkpoint(
            checkpoint_manager,
            optimizer,
            resume_step,
            data_iter,
            train_cfg.schedule_horizon,
            invocation_end_step,
        )
        startup_log(f"restored checkpoint at step {start_step}")

    optimizer_status = jax.device_put(
        jnp.asarray(OptimizerFatalStatus.HEALTHY, dtype=jnp.uint8),
        replicated_sharding,
    )
    last_metrics: dict[str, float] = {}
    prev_metrics: (
        tuple[
            int,
            dict[str, jax.Array],
            datetime.timedelta,
            float,
            jax.Array,
        ]
        | None
    ) = None

    def _log_prev_metrics(force: bool = False) -> None:
        nonlocal last_metrics
        if prev_metrics is None:
            return
        step_to_log, metrics_to_log, step_delta, step_flops, status_to_log = prev_metrics
        if force or (log_every and step_to_log % log_every == 0):
            require_healthy_optimizer_status(
                status_to_log,
                OptimizerStatusBoundary.FINAL if force else OptimizerStatusBoundary.LOG,
            )
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
            last_metrics = result

    startup_log("entering training loop")
    if log_memory:
        log_device_memory("before first step", save_dir=save_path)
    _mem_logged_after_first_step = not log_memory
    _mem_logged_steady_state = not log_memory

    for step_idx in range(start_step, invocation_end_step):
        _log_prev_metrics()
        step = step_idx + 1

        gradient_sum = None
        accum_ce_loss_sum = 0.0
        accum_aux_loss = 0.0
        accum_aux_loss_abs = 0.0
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

            accum_ce_loss_sum = accum_ce_loss_sum + metrics["ce_loss_sum"]
            accum_aux_loss = accum_aux_loss + metrics["aux_loss"]
            accum_aux_loss_abs = accum_aux_loss_abs + jnp.abs(metrics["aux_loss"])
            accum_sup_tokens = accum_sup_tokens + metrics["supervised_tokens"]
            accum_total_tokens = accum_total_tokens + metrics["total_tokens"]
            accum_model_flops += micro_flops.model
            accum_hardware_flops += micro_flops.hardware
            accum_time += micro_delta

        if gradient_sum is None:
            raise RuntimeError("Gradient accumulation produced no microbatches.")
        learning_rate = (
            lr_schedule_fn(step_idx)
            if callable(lr_schedule_fn)
            else jnp.asarray(lr_schedule_fn, dtype=jnp.float32)
        )
        optimizer_status, grad_norm = apply_normalized_gradient_sum(
            optimizer,
            gradient_sum,
            accum_ce_loss_sum,
            accum_sup_tokens,
            accum_aux_loss_abs,
            optimizer_status,
            train_cfg.max_grad_norm,
            learning_rate,
            jnp.asarray(step, dtype=jnp.int32),
        )
        accum_time += timer.step()

        if not _mem_logged_after_first_step:
            require_healthy_optimizer_status(optimizer_status, OptimizerStatusBoundary.LOG)
            jax.block_until_ready(grad_norm)
            log_device_memory("after first step (compile done)", save_dir=save_path)
            log_live_arrays("after first step (compile done)", save_dir=save_path)
            log_compiled_memory_analysis(
                "sft_gradient_step", sft_gradient_step, save_path, optimizer.model, batch
            )
            _mem_logged_after_first_step = True
        elif not _mem_logged_steady_state and step_idx >= 4:
            require_healthy_optimizer_status(optimizer_status, OptimizerStatusBoundary.LOG)
            jax.block_until_ready(grad_norm)
            log_device_memory("after step 5 (steady state)", save_dir=save_path)
            _mem_logged_steady_state = True

        with jax.default_device("cpu"):
            window_metrics = {
                "loss": accum_ce_loss_sum / accum_sup_tokens,
                "ce_loss": accum_ce_loss_sum / accum_sup_tokens,
                "aux_loss": accum_aux_loss / accum_steps,
                "grad_norm": grad_norm,
                "supervised_tokens": accum_sup_tokens,
                "total_tokens": accum_total_tokens,
                "lr": learning_rate,
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
                optimizer_status,
            )

        if gc_period and step % gc_period == 0:
            gc.collect()

        if eval_step is not None and val_every and step % val_every == 0:
            require_healthy_optimizer_status(optimizer_status, OptimizerStatusBoundary.VALIDATION)
            total_val_loss = 0.0
            total_val_sup_tokens = 0.0
            for _ in range(val_steps):
                val_batch = next(val_data_iter)
                pop_source_ids(val_batch)
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

        if checkpoint_manager is not None and save_every and step % save_every == 0:
            _commit_sft_checkpoint(
                checkpoint_manager,
                optimizer,
                step,
                data_iter,
                train_cfg.schedule_horizon,
                invocation_end_step,
                optimizer_status,
                OptimizerStatusBoundary.CHECKPOINT,
                _CheckpointCommitMode.PERIODIC,
                None,
            )

    _log_prev_metrics(force=True)

    _commit_phase_end(
        checkpoint_manager,
        optimizer,
        data_iter,
        train_cfg.schedule_horizon,
        invocation_end_step,
        save_every,
        optimizer_status,
    )

    return optimizer, last_metrics


def run_sft(
    model_id_or_cfg,
    train_cfg: TrainConfig,
    data_iter: checkpoint_utils.GrainIterator,
    *,
    invocation_end_step: int,
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
    """Run one explicitly bounded phase with exception-safe resource cleanup."""
    _require_single_jax_process()
    _require_registrar_compiled_executable_capability()
    cleanup = _TrainingCleanup()
    active_error: BaseException | None = None
    try:
        return _run_sft(
            model_id_or_cfg,
            train_cfg,
            data_iter,
            invocation_end_step=invocation_end_step,
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
