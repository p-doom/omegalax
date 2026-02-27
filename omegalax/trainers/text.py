"""Training helpers for text-only causal language models."""

from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import optax
import orbax.checkpoint as ocp

from omegalax.distributed.mesh import ensure_mesh
from omegalax.text import api as text_api
from omegalax.trainers.perf import (
    per_device_flops_per_step,
    step_metrics,
    StepTimer,
)

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
) -> tuple[nnx.Module, text_api.TextConfig]:
    return text_api.init_model(cfg_or_model_id, rng, tp_size=tp_size, fsdp_size=fsdp_size)


def build_optimizer(model: nnx.Module, train_cfg: TrainConfig) -> nnx.ModelAndOptimizer:
    tx = optax.adamw(learning_rate=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    return nnx.ModelAndOptimizer(model, tx)


def _replicated_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())


def _batch_sharding(cfg: text_api.TextConfig, mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, text_api.batch_partition_spec(cfg))


def _local_batch_size(global_batch_size: int, seq_len: int, batch_spec: PartitionSpec, mesh: Mesh) -> int:
    sharding = NamedSharding(mesh, batch_spec)
    global_shape = (global_batch_size, seq_len)
    unique_batch_slices = {
        indices[0].indices(global_batch_size)
        for indices in sharding.addressable_devices_indices_map(global_shape).values()
    }
    return sum((stop - start + step - 1) // step for start, stop, step in unique_batch_slices)


def make_train_step(
    cfg: text_api.TextConfig,
    optimizer_graphdef,
    optimizer_state_sharding,
    mesh: Mesh,
    *,
    pad_id: int = 0,
):
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

    def _train_step(optimizer_state, token_ids_BT: jax.Array):
        optimizer = nnx.merge(optimizer_graphdef, optimizer_state)

        def loss_fn(model):
            logits_BTV, aux_loss = text_api.forward(model, token_ids_BT, pad_id, cfg)
            targets = token_ids_BT[:, 1:]
            mask = (targets != pad_id).astype(jnp.float32)
            loss = _masked_next_token_loss(logits_BTV[:, :-1, :], targets, mask) + aux_loss
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
        optimizer.update(grads)
        metrics = {
            "loss": loss,
            "grad_norm": optax.tree.norm(grads),
        }
        return nnx.state(optimizer), metrics

    batch_sharding = _batch_sharding(cfg, mesh)
    replicated = _replicated_sharding(mesh)

    @jax.jit(
        in_shardings=(optimizer_state_sharding, batch_sharding),
        out_shardings=(optimizer_state_sharding, replicated),
        donate_argnums=(0,),
    )
    def train_step(optimizer_state, token_ids_BT: jax.Array):
        return _train_step(optimizer_state, token_ids_BT)

    return train_step


def make_synthetic_batch(
    rng: jax.Array, batch_size: int, seq_len: int, vocab_size: int, pad_id: int = 0
) -> jax.Array:
    """Random token batch generator used for smoke training."""
    return jax.random.randint(rng, (batch_size, seq_len), minval=pad_id, maxval=vocab_size, dtype=jnp.int32)


def _train_state(optimizer_state: nnx.State, rng: jax.Array) -> dict[str, object]:
    return {"optimizer": optimizer_state, "rng": rng}


def _abstract_train_state(optimizer_state: nnx.State, rng: jax.Array) -> dict[str, object]:
    return {
        "optimizer": jax.tree.map(
            lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding),
            optimizer_state,
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


def _save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    optimizer_state: nnx.State,
    rng: jax.Array,
    step: int,
) -> None:
    train_state = _train_state(optimizer_state, rng)
    save_args = ocp.args.Composite(train_state=ocp.args.PyTreeSave(train_state))
    checkpoint_manager.save(step, args=save_args)


def _restore_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    optimizer_state: nnx.State,
    rng: jax.Array,
) -> tuple[nnx.State, int, jax.Array]:
    """Restore optimizer/model state and RNG key if a checkpoint exists."""
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        return optimizer_state, 0, rng

    abstract_state = _abstract_train_state(optimizer_state, rng)
    restore_args = ocp.args.Composite(train_state=ocp.args.PyTreeRestore(abstract_state))
    restored = checkpoint_manager.restore(latest_step, args=restore_args)
    train_state = restored["train_state"]
    return train_state["optimizer"], int(latest_step), train_state["rng"]


def run_training(
    model_id_or_cfg,
    train_cfg: TrainConfig,
    *,
    save_dir: str | Path | None = None,
    save_every: int = 0,
    log_every: int = 1,
    log_jsonl: str | Path | None = None,
    resume: bool = False,
    pad_id: int = 0,
    peak_tflops: float | None = None,
    tp_size: int | None = None,
    fsdp_size: int | None = None,
) -> tuple[nnx.ModelAndOptimizer, dict[str, float]]:
    """Train a text model with synthetic data; returns final optimizer + last metrics."""
    model_cfg = text_api.resolve_config(model_id_or_cfg)
    mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size)
    model_cfg = text_api.align_config_to_mesh(model_cfg, mesh)
    batch_spec = text_api.batch_partition_spec(model_cfg)
    required_multiple = text_api.required_batch_multiple(model_cfg, mesh)
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
    if is_batch_partitioned_across_processes:
        rng = jax.random.fold_in(rng, jax.process_index())
    rng = jax.device_put(rng, replicated_rng_sharding)

    is_primary_process = jax.process_index() == 0

    model, model_cfg = init_model(model_cfg, init_rng, tp_size=tp_size, fsdp_size=fsdp_size)
    optimizer = build_optimizer(model, train_cfg)
    optimizer_graphdef = nnx.graphdef(optimizer)
    optimizer_state = nnx.state(optimizer)
    replicated = NamedSharding(mesh, P())
    optimizer_state_sharding = jax.tree.map(
        lambda leaf: leaf.sharding if isinstance(leaf, jax.Array) else replicated,
        optimizer_state,
    )
    train_step = make_train_step(
        model_cfg,
        optimizer_graphdef,
        optimizer_state_sharding,
        mesh,
        pad_id=pad_id,
    )

    per_device_flops = per_device_flops_per_step(
        model_cfg, train_cfg.seq_len, train_cfg.batch_size
    )
    timer = StepTimer(warmup=2)
    tokens_per_step = train_cfg.seq_len * train_cfg.batch_size

    checkpoint_manager = None
    if save_dir is not None:
        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_manager = _make_checkpoint_manager(save_dir, save_interval=save_every or None)

    log_path = Path(log_jsonl).expanduser() if log_jsonl else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if resume and checkpoint_manager is not None:
        optimizer_state, start_step, rng = _restore_checkpoint(checkpoint_manager, optimizer_state, rng)
        rng = jax.device_put(rng, replicated_rng_sharding)

    last_metrics: dict[str, float] = {}
    prev_metrics: tuple[int, dict[str, jax.Array], datetime.timedelta] | None = None

    def _log_prev_metrics(force: bool = False) -> None:
        """Log metrics for the previous step to avoid per-step device syncs."""
        nonlocal prev_metrics, last_metrics
        if prev_metrics is None:
            return

        step_to_log, metrics_to_log, step_delta = prev_metrics
        should_print = is_primary_process and log_every and step_to_log % log_every == 0
        should_write = is_primary_process and log_path is not None
        if not (should_print or should_write or force):
            return

        host_metrics = {k: float(v) for k, v in metrics_to_log.items()}
        host_metrics["step"] = step_to_log
        perf = step_metrics(
            per_device_flops, step_delta, tokens_per_step, peak_tflops
        )
        host_metrics.update(perf)
        last_metrics = host_metrics

        if should_print:
            acc = host_metrics.get("token_accuracy", 0.0)
            print(
                f"step={host_metrics['step']} loss={host_metrics['loss']:.4f} "
                f"acc={acc:.4f} grad_norm={host_metrics['grad_norm']:.4f} "
                f"step_s={host_metrics['step_time_s']:.3f} tok/s/dev={host_metrics['tokens_per_sec_per_device']:.0f} "
                f"TFLOP/s/dev={host_metrics['tflops_per_device']:.2f} mfu={host_metrics['mfu']:.4f}"
            )

        if should_write:
            with log_path.open("a") as f:
                f.write(json.dumps(host_metrics) + "\n")

    for step in range(start_step, train_cfg.num_steps):
        rng, batch_rng = jax.random.split(rng)
        batch = make_synthetic_batch(batch_rng, local_batch_size, train_cfg.seq_len, model_cfg.vocab_size, pad_id)
        batch = text_api.shard_batch(batch, model_cfg, mesh)
        optimizer_state, metrics = train_step(optimizer_state, batch)
        step_delta = timer.step()

        # Log the previous step while the current step is running to preserve async dispatch.
        _log_prev_metrics()
        prev_metrics = (step + 1, metrics, step_delta)

        if checkpoint_manager is not None and save_every and (step + 1) % save_every == 0:
            _save_checkpoint(checkpoint_manager, optimizer_state, rng, step + 1)

    # Flush the final step's metrics.
    _log_prev_metrics(force=True)

    if checkpoint_manager is not None:
        if last_metrics and (not save_every or last_metrics["step"] % save_every != 0):
            _save_checkpoint(checkpoint_manager, optimizer_state, rng, int(last_metrics["step"]))
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()

    optimizer = nnx.merge(optimizer_graphdef, optimizer_state)
    return optimizer, last_metrics
