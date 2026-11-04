"""Training helpers for vision-language models (text-only or multimodal batches)."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import gc
import hashlib
import json
import os
import signal
import subprocess
import weakref
from pathlib import Path
from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_managers as ocm

from omegalax import export as export_lib
from omegalax.data.grain_pipeline import pop_source_ids
from omegalax.distributed.mesh import ensure_mesh, mesh_rules, required_batch_multiple
from omegalax.models.params_utils import save_hf_config
from omegalax.trainers import checkpoint_utils
from omegalax.trainers import tokamax_cache as tokamax_cache_lib
from omegalax.trainers.loss import chunked_cross_entropy_loss
from omegalax.trainers.lr_schedule import build_lr_schedule
from omegalax.trainers.perf import (
    log_compiled_memory_analysis,
    log_device_memory,
    log_live_arrays,
    log_pytree_bytes,
    log_top_leaves_with_paths,
    maybe_log_step_metrics,
    per_device_step_flops,
    StepFlops,
    StepTimer,
)
from omegalax.trainers.optim import MixedPrecisionOptimizer
from omegalax.trainers.lora import LoRAParam, inject_lora
from omegalax.trainers.text import startup_log
from omegalax.models.qwen3_vl.model import DECODER_LAYER_REMAT
from omegalax.models.qwen3_vl.vision import VISION_BLOCK_REMAT
from omegalax.vlm import api as vlm_api

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
    tx = optax.MultiSteps(tx, every_k_schedule=train_cfg.grad_accum_steps)
    opt = MixedPrecisionOptimizer(model, tx, wrt=wrt)
    return opt


def _train_state(optimizer: MixedPrecisionOptimizer, rng: jax.Array) -> dict[str, object]:
    return {"optimizer": nnx.state(optimizer), "rng": rng}


def _abstract_train_state(optimizer: MixedPrecisionOptimizer, rng: jax.Array) -> dict[str, object]:
    return _abstract_train_state_from_optimizer_state(nnx.state(optimizer), rng)


def _abstract_train_state_from_optimizer_state(
    optimizer_state: Any, rng: jax.Array
) -> dict[str, object]:
    return {
        "optimizer": jax.tree.map(
            lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding),
            optimizer_state,
        ),
        "rng": jax.ShapeDtypeStruct(rng.shape, rng.dtype, sharding=rng.sharding),
    }


@dataclasses.dataclass(frozen=True)
class _MemorySafeRestoreBlueprint:
    """Array-free material needed to restore and reconstruct an optimizer."""

    optimizer_graphdef: Any
    abstract_train_state: dict[str, object]
    initialized_optimizer_ref: weakref.ReferenceType
    initialized_model_ref: weakref.ReferenceType
    initialized_array_refs: tuple[weakref.ReferenceType, ...]
    device_bytes_in_use_before_release: dict[str, int]


def _gpu_bytes_in_use() -> dict[str, int]:
    result = {}
    for device in jax.local_devices():
        if device.platform != "gpu":
            continue
        stats = device.memory_stats()
        if stats is not None and "bytes_in_use" in stats:
            result[f"{device.platform}:{device.id}"] = int(stats["bytes_in_use"])
    return result


def _prepare_memory_safe_restore(
    optimizer: MixedPrecisionOptimizer, rng: jax.Array
) -> _MemorySafeRestoreBlueprint:
    """Split off an array-free graph/restore spec before dropping fresh state."""
    graphdef, initialized_state = nnx.split(optimizer)
    jax.block_until_ready(initialized_state)
    arrays = {}
    for value in jax.tree.leaves(initialized_state):
        if isinstance(value, jax.Array):
            arrays.setdefault(id(value), value)
    array_refs = tuple(weakref.ref(value) for value in arrays.values())
    abstract_state = _abstract_train_state_from_optimizer_state(initialized_state, rng)
    blueprint = _MemorySafeRestoreBlueprint(
        optimizer_graphdef=graphdef,
        abstract_train_state=abstract_state,
        initialized_optimizer_ref=weakref.ref(optimizer),
        initialized_model_ref=weakref.ref(optimizer.model),
        initialized_array_refs=array_refs,
        device_bytes_in_use_before_release=_gpu_bytes_in_use(),
    )
    # The caller still owns ``optimizer`` (and may own another ``model`` alias),
    # but this helper must not leak a concrete-state alias in its return value.
    del initialized_state, arrays
    return blueprint


def _verify_initialized_state_released(
    blueprint: _MemorySafeRestoreBlueprint,
) -> dict[str, object]:
    """Fail unless all fresh model/optimizer objects and arrays were released."""
    jax.clear_caches()
    gc.collect()
    jax.effects_barrier()
    gc.collect()

    optimizer_alive = blueprint.initialized_optimizer_ref() is not None
    model_alive = blueprint.initialized_model_ref() is not None
    live_array_count = sum(ref() is not None for ref in blueprint.initialized_array_refs)
    memory_after = _gpu_bytes_in_use()
    memory_before = blueprint.device_bytes_in_use_before_release
    strict_gpu_telemetry = os.environ.get("OMEGALAX_REQUIRE_MEMORY_SAFE_RESTORE") == "1"
    gpu_drop_verified = all(
        key in memory_after and memory_after[key] < value for key, value in memory_before.items()
    )
    report = {
        "schema_version": 1,
        "artifact_type": "omegalax_vlm_memory_safe_restore_release",
        "status": "release_pass",
        "strict_gpu_telemetry_required": strict_gpu_telemetry,
        "initialized_optimizer_collected": not optimizer_alive,
        "initialized_model_collected": not model_alive,
        "initialized_array_count": len(blueprint.initialized_array_refs),
        "live_initialized_array_count_after_gc": live_array_count,
        "device_bytes_in_use_before_release": memory_before,
        "device_bytes_in_use_after_release": memory_after,
        "device_bytes_released": {
            key: value - memory_after.get(key, value) for key, value in memory_before.items()
        },
        "gpu_memory_drop_verified": gpu_drop_verified if memory_before else None,
    }
    if optimizer_alive or model_alive or live_array_count:
        report["status"] = "fail"
        raise RuntimeError(f"fresh initialized state still has live aliases: {report}")
    if strict_gpu_telemetry and not memory_before:
        report["status"] = "fail"
        raise RuntimeError(f"strict restore requires GPU bytes_in_use telemetry: {report}")
    if memory_before and not gpu_drop_verified:
        report["status"] = "fail"
        raise RuntimeError(f"GPU bytes_in_use did not drop after releasing fresh state: {report}")
    return report


def _write_restore_release_audit(save_dir: Path, report: dict[str, object]) -> None:
    output = save_dir / "restore_memory_release.json"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def _trained_promotion_group(path: tuple[Any, ...]) -> str | None:
    """Return the exact trained-state branch allowed to promote bf16 to fp32."""
    if len(path) > 2 and path[:2] == ("opt_state", "acc_grads"):
        return "acc_grads"
    if len(path) <= 4 or path[:2] != ("opt_state", "inner_opt_state"):
        return None
    markers = [
        (index, entry)
        for index, entry in enumerate(path[2:], start=2)
        if entry in ("mu", "nu")
    ]
    if len(markers) != 1:
        return None
    marker_index, group = markers[0]
    if (
        marker_index == 2
        or marker_index == len(path) - 1
        or not all(isinstance(entry, int) for entry in path[2:marker_index])
    ):
        return None
    return group


def _assert_restored_optimizer_contract(expected: Any, restored: Any) -> dict[str, object]:
    """Validate exact structure while preserving known trained-state promotions.

    Fresh MultiSteps accumulators and Adam moments begin as bf16 because they
    are initialized from bf16 LoRA parameters. Real fp32 gradients promote
    ``acc_grads``, ``mu``, and ``nu`` after training starts. Those checkpoint
    fp32 arrays are semantically correct trained state and must not be rounded
    back to their fresh-zero dtype.
    """
    expected_flat = nnx.to_flat_state(expected)
    restored_flat = nnx.to_flat_state(restored)
    if len(expected_flat) != len(restored_flat):
        raise RuntimeError("restored optimizer tree structure differs from fresh optimizer")
    memory_before = _gpu_bytes_in_use()
    records = []
    groups: dict[str, dict[str, int]] = {}
    for (expected_path, expected_var), (restored_path, restored_var) in zip(
        expected_flat, restored_flat
    ):
        if expected_path != restored_path:
            raise RuntimeError(f"restored optimizer path mismatch: {expected_path} != {restored_path}")
        wanted = expected_var.get_value()
        got = restored_var.get_value()
        if (
            tuple(got.shape) != tuple(wanted.shape)
            or got.sharding != wanted.sharding
        ):
            raise RuntimeError(
                f"restored optimizer contract mismatch at {expected_path}: "
                f"{got.shape}/{got.dtype}/{got.sharding} != "
                f"{wanted.shape}/{wanted.dtype}/{wanted.sharding}"
            )
        if got.dtype == wanted.dtype:
            continue
        group = _trained_promotion_group(expected_path)
        if group is None or got.dtype != jnp.float32 or wanted.dtype != jnp.bfloat16:
            raise RuntimeError(
                f"unpermitted restored optimizer dtype mismatch at {expected_path}: "
                f"{got.dtype} != {wanted.dtype}"
            )
        source_bytes = int(got.size * got.dtype.itemsize)
        fresh_bytes = int(got.size * wanted.dtype.itemsize)
        record = {
            "path": list(expected_path),
            "shape": list(got.shape),
            "group": group,
            "restored_trained_dtype": str(got.dtype),
            "fresh_zero_state_dtype": str(wanted.dtype),
            "restored_source_bytes": source_bytes,
            "fresh_zero_state_bytes": fresh_bytes,
            "preserved_without_cast": True,
        }
        records.append(record)
        summary = groups.setdefault(
            group,
            {
                "leaf_count": 0,
                "numel": 0,
                "restored_source_bytes": 0,
                "fresh_zero_state_bytes": 0,
            },
        )
        summary["leaf_count"] += 1
        summary["numel"] += int(got.size)
        summary["restored_source_bytes"] += source_bytes
        summary["fresh_zero_state_bytes"] += fresh_bytes
    memory_after = _gpu_bytes_in_use()
    return {
        "mode": "preserve_known_trained_fp32_promotions_without_cast",
        "promoted_leaf_count": len(records),
        "promoted_source_bytes": sum(item["restored_source_bytes"] for item in records),
        "fresh_zero_state_bytes": sum(item["fresh_zero_state_bytes"] for item in records),
        "converted_leaf_count": 0,
        "all_shapes_and_shardings_exact": True,
        "all_other_dtypes_exact": True,
        "promoted_arrays_bitwise_untouched": True,
        "groups": groups,
        "promoted_leaf_records": records,
        "device_bytes_in_use_before_contract": memory_before,
        "device_bytes_in_use_after_contract": memory_after,
        "contract_check_allocated_no_array_outputs": True,
    }


def _path_tuple(path: tuple[Any, ...]) -> tuple[str, ...]:
    result = []
    for entry in path:
        value = getattr(entry, "key", getattr(entry, "name", getattr(entry, "idx", entry)))
        result.append(str(value))
    return tuple(result)


_RESTORE_COUNTER_PATHS = {
    ("step", "value"): "optimizer_micro_step",
    ("opt_state", "gradient_step", "value"): "global_gradient_step",
    ("opt_state", "mini_step", "value"): "gradient_accumulation_remainder",
    ("opt_state", "inner_opt_state", "1", "0", "count", "value"): "adam_count_0",
    ("opt_state", "inner_opt_state", "1", "2", "count", "value"): "adam_count_2",
}


def _restored_optimizer_counters(optimizer_state: Any) -> dict[str, int]:
    counters = {}
    for path, value in jax.tree_util.tree_leaves_with_path(optimizer_state):
        name = _RESTORE_COUNTER_PATHS.get(_path_tuple(path))
        if name is None:
            continue
        array = jax.device_get(value)
        if tuple(array.shape) != () or not jnp.issubdtype(array.dtype, jnp.integer):
            raise RuntimeError(f"restored optimizer counter {path} is not an integer scalar")
        counters[name] = int(array)
    return counters


def _write_exact_restore_attestation(
    checkpoint_root: Path,
    step: int,
    optimizer: MixedPrecisionOptimizer,
    rng: jax.Array,
    restored_input_iter: checkpoint_utils.GrainIterator,
    contract_leaf_count: int,
    target_topology: dict[str, int],
    optimizer_contract: dict[str, object],
) -> None:
    """Optionally gate exact resume scalars and sealed iterator bytes pre-update."""
    if os.environ.get("OMEGALAX_REQUIRE_EXACT_RESTORE_ATTESTATION") != "1":
        return
    required = (
        "OMEGALAX_EXPECT_RESUME_STEP",
        "OMEGALAX_EXPECT_OPTIMIZER_COUNTERS_JSON",
        "OMEGALAX_EXPECT_RNG_KEY_DATA_JSON",
        "OMEGALAX_EXPECT_ITERATOR_STATE_JSON",
        "OMEGALAX_EXPECT_ITERATOR_SHA256",
        "OMEGALAX_EXPECT_PROMOTED_OPTIMIZER_STATE_JSON",
    )
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(f"exact restore attestation environment is incomplete: {missing}")
    expected_step = int(os.environ["OMEGALAX_EXPECT_RESUME_STEP"])
    expected_counters = json.loads(os.environ["OMEGALAX_EXPECT_OPTIMIZER_COUNTERS_JSON"])
    expected_rng = json.loads(os.environ["OMEGALAX_EXPECT_RNG_KEY_DATA_JSON"])
    expected_iterator_state = json.loads(os.environ["OMEGALAX_EXPECT_ITERATOR_STATE_JSON"])
    expected_iterator_sha = os.environ["OMEGALAX_EXPECT_ITERATOR_SHA256"]
    expected_promoted_state = json.loads(
        os.environ["OMEGALAX_EXPECT_PROMOTED_OPTIMIZER_STATE_JSON"]
    )
    actual_counters = _restored_optimizer_counters(nnx.state(optimizer))
    actual_rng = [int(value) for value in jax.device_get(jax.random.key_data(rng)).reshape(-1)]
    actual_iterator_state = restored_input_iter.get_state()
    actual_iterator_bytes = json.dumps(actual_iterator_state, indent=4).encode()
    actual_iterator_sha = hashlib.sha256(actual_iterator_bytes).hexdigest()
    if step != expected_step:
        raise RuntimeError(f"restored step mismatch: {step} != {expected_step}")
    if actual_counters != expected_counters:
        raise RuntimeError(
            f"restored optimizer counter mismatch: {actual_counters} != {expected_counters}"
        )
    if actual_rng != expected_rng:
        raise RuntimeError(f"restored RNG key-data mismatch: {actual_rng} != {expected_rng}")
    if actual_iterator_state != expected_iterator_state:
        raise RuntimeError(
            f"live restored iterator state mismatch: {actual_iterator_state} "
            f"!= {expected_iterator_state}"
        )
    if actual_iterator_sha != expected_iterator_sha:
        raise RuntimeError(
            f"restored iterator payload hash mismatch: {actual_iterator_sha} "
            f"!= {expected_iterator_sha}"
        )
    observed_promoted_state = {
        key: optimizer_contract[key]
        for key in ("promoted_leaf_count", "promoted_source_bytes", "fresh_zero_state_bytes")
    }
    if observed_promoted_state != expected_promoted_state:
        raise RuntimeError(
            "restored promoted optimizer-state mismatch: "
            f"{observed_promoted_state} != {expected_promoted_state}"
        )
    result = {
        "schema_version": 1,
        "artifact_type": "omegalax_vlm_exact_restore_attestation",
        "status": "restore_pass",
        "resume_step": step,
        "optimizer_shape_dtype_sharding_contract_pass": True,
        "optimizer_contract_leaf_count": contract_leaf_count,
        "optimizer_counters": actual_counters,
        "rng_key_data": actual_rng,
        "restored_iterator_state": actual_iterator_state,
        "input_iterator_sha256": actual_iterator_sha,
        "target_topology": target_topology,
        "optimizer_contract": optimizer_contract,
        "written_before_first_optimizer_update": True,
    }
    output = checkpoint_root / "restore_exact_state.json"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


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
    handler_registry.add("train_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)
    handler_registry.add(
        "train_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler
    )
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
    )
    return ocp.CheckpointManager(save_dir, options=options, handler_registry=handler_registry)


def _write_checkpoint_config(save_dir: Path, cfg) -> None:
    save_hf_config(export_lib.model_config_to_hf_dict(cfg), save_dir)


def _write_lora_metadata(save_dir: Path, train_cfg: TrainConfig) -> None:
    """Persist LoRA settings alongside the orbax tree.

    The export driver reads this file to reconstruct the same optimizer
    shape at restore time. Absent file ⇒ checkpoint was full-FT.
    """
    import json

    meta = {
        "enable_lora": bool(train_cfg.enable_lora),
        "lora_rank": int(train_cfg.lora_rank),
        "lora_alpha": float(train_cfg.lora_alpha),
    }
    (Path(save_dir) / "lora_metadata.json").write_text(json.dumps(meta, indent=2))


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
    blueprint: _MemorySafeRestoreBlueprint,
    input_iter: checkpoint_utils.GrainIterator,
    target_topology: dict[str, int],
) -> tuple[MixedPrecisionOptimizer, int, jax.Array, checkpoint_utils.GrainIterator]:
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        raise ValueError("No checkpoint found to restore.")

    restore_args = checkpoint_utils.make_grain_restore_args(
        blueprint.abstract_train_state, input_iter
    )
    restored = checkpoint_manager.restore(latest_step, args=restore_args)
    train_state = restored["train_state"]
    # Orbax can preserve a checkpoint's trained fp32 accumulator/moment dtype
    # even when the fresh abstract target is bf16. The contract below permits
    # only those exact promotion branches and never casts restored arrays.
    expected_optimizer = blueprint.abstract_train_state["optimizer"]
    optimizer_contract = _assert_restored_optimizer_contract(
        expected_optimizer, train_state["optimizer"]
    )
    optimizer = nnx.merge(blueprint.optimizer_graphdef, train_state["optimizer"])
    merged_contract = _assert_restored_optimizer_contract(
        expected_optimizer, nnx.state(optimizer)
    )
    for key in (
        "mode",
        "promoted_leaf_count",
        "promoted_source_bytes",
        "fresh_zero_state_bytes",
        "converted_leaf_count",
        "groups",
        "promoted_leaf_records",
    ):
        if merged_contract[key] != optimizer_contract[key]:
            raise RuntimeError(f"nnx.merge changed restored optimizer contract field {key}")
    optimizer_contract["nnx_merge_preserved_promoted_state"] = True
    optimizer_contract["device_bytes_in_use_after_nnx_merge"] = _gpu_bytes_in_use()
    restored_input_iter = checkpoint_utils.restored_input_iter(restored)
    _write_exact_restore_attestation(
        Path(checkpoint_manager.directory),
        int(latest_step),
        optimizer,
        train_state["rng"],
        restored_input_iter,
        len(
            jax.tree.leaves(
                expected_optimizer,
                is_leaf=lambda value: isinstance(value, jax.ShapeDtypeStruct),
            )
        ),
        target_topology,
        optimizer_contract,
    )
    return (
        optimizer,
        int(latest_step),
        train_state["rng"],
        restored_input_iter,
    )


def make_sft_train_step(cfg, pad_id: int = 0, *, wrt=nnx.Param, num_loss_tiles: int = 4):
    """Build a JIT-compiled VLM SFT train step that consumes a batch dict.

    The batch dict must contain ``token_ids_BT``, ``attention_mask_BT``, and
    ``loss_mask_BT``.  It may also contain ``pixel_values`` and
    ``image_grid_thw`` for multimodal batches.

    ``wrt`` selects which model variables receive gradients. Defaults to
    ``nnx.Param`` (full FT). Pass ``LoRAParam`` for adapter-only training
    — every other ``nnx.Param`` then sees zero gradient and contributes
    no optimizer state.
    """

    diff_state = nnx.DiffState(0, wrt)

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
            total_tokens = jnp.sum(attention_mask_BT.astype(jnp.float32))
            return loss, (supervised_tokens, total_tokens)

        (loss, (supervised_tokens, total_tokens)), grads = nnx.value_and_grad(
            loss_fn,
            argnums=diff_state,
            has_aux=True,
        )(optimizer.model)
        optimizer.update(grads)
        metrics = {
            "loss": loss,
            "grad_norm": optax.tree.norm(grads),
            "supervised_tokens": supervised_tokens,
            "total_tokens": total_tokens,
        }
        return loss, metrics

    return sft_train_step


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
    """SFT a VLM from a Grain iterator; returns final optimizer + last metrics.

    ``data_iter`` must be a checkpointable Grain iterator yielding dicts with keys ``token_ids_BT``,
    ``attention_mask_BT``, and ``loss_mask_BT`` (all numpy ``(B, T)``).
    Optionally ``pixel_values`` and ``image_grid_thw`` for multimodal batches.

    If ``val_data_iter`` is provided, runs ``val_steps`` forward-only batches
    every ``val_every`` training steps and logs the average validation loss.

    See :class:`omegalax.trainers.checkpoint_utils.ResumeMode` for the meaning of
    each ``resume`` mode.
    """
    save_path = Path(save_dir).expanduser().resolve() if save_dir is not None else None

    # Build the canonical CheckpointManager up-front so a single ``latest_step()``
    # query drives both the model_cfg-source decision and the eventual restore.
    # No throwaway probes.
    checkpoint_manager: ocp.CheckpointManager | None = None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        checkpoint_manager = _make_checkpoint_manager(
            save_path,
            save_interval=save_every or None,
            keep_period=keep_period or None,
            keep_latest=keep_latest or None,
        )

    latest_step = checkpoint_manager.latest_step() if checkpoint_manager is not None else None

    if (
        latest_step is not None
        and latest_step >= train_cfg.num_steps
        and resume in (checkpoint_utils.ResumeMode.IF_PRESENT, checkpoint_utils.ResumeMode.REQUIRED)
    ):
        startup_log(f"latest_step={latest_step} >= num_steps={train_cfg.num_steps}; exiting")
        if checkpoint_manager is not None:
            checkpoint_manager.close()
        return None, None

    if resume == checkpoint_utils.ResumeMode.REQUIRED and latest_step is None:
        raise ValueError(
            f"resume='required' but no checkpoint found at "
            f"{save_path if save_path is not None else '<no save_dir provided>'}"
        )

    will_resume = (
        resume in (checkpoint_utils.ResumeMode.IF_PRESENT, checkpoint_utils.ResumeMode.REQUIRED)
        and latest_step is not None
    )
    if resume == checkpoint_utils.ResumeMode.IF_PRESENT:
        startup_log(
            f"resume=if_present: existing checkpoint detected at {save_path}; resuming"
            if will_resume
            else f"resume=if_present: no checkpoint at {save_path}; starting fresh"
        )

    if will_resume:
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

    restore_blueprint = None
    if will_resume:
        # NNX graph metadata and ShapeDtypeStructs are sufficient to restore.
        # Drop *both* concrete owners: ``optimizer.model`` and the local
        # ``model`` alias otherwise keep the entire random initialization live
        # while Orbax allocates the restored checkpoint beside it.
        restore_blueprint = _prepare_memory_safe_restore(optimizer, rng)
        del optimizer
        del model
        release_report = _verify_initialized_state_released(restore_blueprint)
        if save_path is not None and is_primary_process:
            _write_restore_release_audit(save_path, release_report)
        startup_log(
            "released fresh model/optimizer before restore: "
            f"arrays={release_report['initialized_array_count']} "
            f"gpu_bytes={release_report['device_bytes_released']}"
        )

    sft_step = make_sft_train_step(
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
        if restore_blueprint is None or checkpoint_manager is None:
            raise AssertionError("resume restore blueprint/manager was not initialized")
        optimizer, start_step, rng, data_iter = _restore_sft_checkpoint(
            checkpoint_manager,
            restore_blueprint,
            data_iter,
            {axis: int(size) for axis, size in mesh.shape.items()},
        )
        del restore_blueprint
        rng = jax.device_put(rng, replicated_rng_sharding)
        startup_log(f"restored checkpoint at step {start_step}")

    last_metrics: dict[str, float] = {}
    prev_metrics: tuple[int, dict[str, jax.Array], datetime.timedelta, float] | None = None

    def _log_prev_metrics(force: bool = False) -> None:
        nonlocal last_metrics
        if prev_metrics is None:
            return
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
            last_metrics = result

    # Flag-only handler: doing an orbax save or JAX collective from inside
    # the signal handler deadlocks (handlers run on arbitrary threads and
    # re-enter the runtime). The flag is read at a safe per-step point.
    requeue_requested = False

    def _request_requeue(signum, _frame):
        nonlocal requeue_requested
        if not requeue_requested:
            startup_log(f"[signal] received {signum}; will requeue after current step")
        requeue_requested = True

    signal.signal(signal.SIGUSR1, _request_requeue)
    signal.signal(signal.SIGTERM, _request_requeue)

    autotune_result = None
    pending_batch = None
    if tokamax_cache_dir is not None:
        autotune_result = tokamax_cache_lib.try_load(tokamax_cache_dir)
        if autotune_result is None:
            startup_log("priming tokamax autotuning with first training batch")
            pending_batch = next(data_iter)
            pending_batch_sharded = vlm_api.shard_batch_dict(pending_batch, model_cfg, mesh)
            autotune_result = tokamax_cache_lib.autotune_and_save(
                tokamax_cache_dir, sft_step, optimizer, pending_batch_sharded
            )

    # Push the autotuning overlay onto tokamax's lookup stack for the duration
    # of training; this keeps the for-loop indentation unchanged.
    _autotune_ctx = autotune_result if autotune_result is not None else contextlib.nullcontext()
    _autotune_ctx.__enter__()

    startup_log("entering training loop")
    if log_memory:
        log_device_memory("before first step", save_dir=save_path)
    _mem_logged_after_first_step = not log_memory
    _mem_logged_steady_state = not log_memory

    for step_idx in range(start_step, train_cfg.num_steps):
        step = step_idx + 1

        accum_loss = 0.0
        accum_sup_tokens = 0.0
        accum_total_tokens = 0.0
        accum_grad_norm = 0.0
        accum_model_flops = 0.0
        accum_hardware_flops = 0.0
        accum_time = datetime.timedelta(0)
        source_counts: dict[int, int] = {}

        for _micro in range(accum_steps):
            if pending_batch is not None:
                batch = pending_batch
                pending_batch = None
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
            _, metrics = sft_step(optimizer, batch)
            micro_delta = timer.step()

            accum_loss = accum_loss + metrics["loss"]
            accum_sup_tokens = accum_sup_tokens + metrics["supervised_tokens"]
            accum_total_tokens = accum_total_tokens + metrics["total_tokens"]
            accum_grad_norm = accum_grad_norm + metrics["grad_norm"]
            accum_model_flops += micro_flops.model
            accum_hardware_flops += micro_flops.hardware
            accum_time += micro_delta

        # Log memory after first step and after 5 steps
        if not _mem_logged_after_first_step:
            jax.block_until_ready(metrics["loss"])
            log_device_memory("after first step (compile done)", save_dir=save_path)
            log_live_arrays("after first step (compile done)", save_dir=save_path)
            log_compiled_memory_analysis("sft_step", sft_step, save_path, optimizer, batch)
            _mem_logged_after_first_step = True
        elif not _mem_logged_steady_state and step_idx >= 4:
            jax.block_until_ready(metrics["loss"])
            log_device_memory("after step 5 (steady state)", save_dir=save_path)
            _mem_logged_steady_state = True

        with jax.default_device("cpu"):
            window_metrics = {
                "loss": accum_loss / accum_steps,
                "grad_norm": accum_grad_norm / accum_steps,
                "supervised_tokens": accum_sup_tokens,
                "total_tokens": accum_total_tokens,
                "lr": lr_schedule_fn(step_idx),
            }
            if len(source_counts) > 1:
                total = float(sum(source_counts.values()))
                for sid, cnt in source_counts.items():
                    window_metrics[f"data_source_{sid}_frac"] = cnt / total
            _log_prev_metrics()

            prev_metrics = (
                step,
                window_metrics,
                accum_time,
                StepFlops(model=accum_model_flops, hardware=accum_hardware_flops),
            )

        if checkpoint_manager is not None and save_every and step % save_every == 0:
            _save_sft_checkpoint(checkpoint_manager, optimizer, rng, step, data_iter)

        if gc_period and step % gc_period == 0:
            gc.collect()

        if eval_step is not None and val_every and step % val_every == 0:
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

        if requeue_requested:
            startup_log(f"[signal] saving checkpoint at step={step} and requeueing")
            if checkpoint_manager is not None:
                _save_sft_checkpoint(checkpoint_manager, optimizer, rng, step, data_iter)
                checkpoint_manager.wait_until_finished()
                checkpoint_manager.close()
            slurm_job_id = os.environ.get("SLURM_JOB_ID")
            if slurm_job_id and is_primary_process:
                startup_log(f"[signal] scontrol requeue {slurm_job_id}")
                subprocess.run(["scontrol", "requeue", slurm_job_id], check=False)
            _autotune_ctx.__exit__(None, None, None)
            return optimizer, last_metrics

    _log_prev_metrics(force=True)

    if checkpoint_manager is not None:
        if last_metrics and (not save_every or last_metrics["step"] % save_every != 0):
            _save_sft_checkpoint(
                checkpoint_manager, optimizer, rng, int(last_metrics["step"]), data_iter
            )
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()

    _autotune_ctx.__exit__(None, None, None)
    return optimizer, last_metrics
