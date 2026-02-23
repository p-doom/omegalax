"""Training helpers for vision-language models (text-only or multimodal batches)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from omegalax.vlm import api as vlm_api


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


def init_model(cfg_or_model_id, rng: jax.Array) -> nnx.Module:
    model, _ = vlm_api.init_model(cfg_or_model_id, rng)
    return model


def build_optimizer(model: nnx.Module, train_cfg: TrainConfig) -> nnx.ModelAndOptimizer:
    tx = optax.adamw(learning_rate=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    return nnx.ModelAndOptimizer(model, tx)


def make_train_step(cfg, pad_id: int = 0):
    @nnx.jit(donate_argnums=0)
    def train_step(
        optimizer: nnx.ModelAndOptimizer,
        tokens: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        pixel_values: jax.Array | None = None,
        image_grid_thw: jax.Array | None = None,
        position_ids: jax.Array | None = None,
    ):
        def loss_fn(model):
            logits, aux_loss = vlm_api.forward(
                model,
                tokens,
                pad_id,
                cfg,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
            )
            logits = logits.astype(jnp.float32)
            targets = tokens[:, 1:]
            lm_logits = logits[:, :-1, :]
            mask = attention_mask[:, 1:] if attention_mask is not None else (targets != pad_id)
            mask = mask.astype(jnp.float32)
            ce = optax.softmax_cross_entropy_with_integer_labels(lm_logits, targets)
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            loss = jnp.sum(ce * mask) / denom + aux_loss
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
) -> jax.Array:
    """Random token batch generator used for smoke training."""
    return jax.random.randint(rng, (batch_size, seq_len), minval=pad_id, maxval=vocab_size, dtype=jnp.int32)


def _train_state(optimizer: nnx.ModelAndOptimizer, rng: jax.Array) -> dict[str, object]:
    return {"optimizer": nnx.state(optimizer), "rng": rng}


def _abstract_train_state(optimizer: nnx.ModelAndOptimizer) -> dict[str, object]:
    abstract_optimizer = nnx.eval_shape(lambda: optimizer)
    abstract_rng = jax.eval_shape(lambda: jax.random.key(0))
    return {"optimizer": nnx.state(abstract_optimizer), "rng": abstract_rng}


def _make_checkpoint_manager(save_dir: Path, save_interval: int | None) -> ocp.CheckpointManager:
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
    """Restore optimizer/model state and RNG key if a checkpoint exists."""
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        return optimizer, 0, rng

    abstract_state = _abstract_train_state(optimizer)
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
    log_jsonl: str | Path | None = None,
    resume: bool = False,
    pad_id: int = 0,
) -> tuple[nnx.ModelAndOptimizer, dict[str, float]]:
    """Train a VLM with synthetic data; returns final optimizer + last metrics."""
    rng = jax.random.key(train_cfg.seed)
    rng, init_rng = jax.random.split(rng)

    model, model_cfg = vlm_api.init_model(model_id_or_cfg, init_rng)
    optimizer = build_optimizer(model, train_cfg)
    train_step = make_train_step(model_cfg, pad_id=pad_id)

    checkpoint_manager = None
    if save_dir is not None:
        save_dir = Path(save_dir).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_manager = _make_checkpoint_manager(save_dir, save_interval=save_every or None)

    log_path = Path(log_jsonl).expanduser() if log_jsonl else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if resume and checkpoint_manager is not None:
        optimizer, start_step, rng = _restore_checkpoint(checkpoint_manager, optimizer, rng)

    last_metrics: dict[str, float] = {}
    prev_metrics: tuple[int, dict[str, jax.Array]] | None = None

    def _log_prev_metrics(force: bool = False) -> None:
        """Log metrics for the previous step to avoid per-step device syncs."""
        nonlocal prev_metrics, last_metrics
        if prev_metrics is None:
            return

        step_to_log, metrics_to_log = prev_metrics
        should_print = log_every and step_to_log % log_every == 0
        should_write = log_path is not None
        if not (should_print or should_write or force):
            return

        host_metrics = {k: float(v) for k, v in metrics_to_log.items()}
        host_metrics["step"] = step_to_log
        last_metrics = host_metrics

        if should_print:
            print(
                f"step={host_metrics['step']} loss={host_metrics['loss']:.4f} "
                f"grad_norm={host_metrics['grad_norm']:.4f}"
            )

        if should_write:
            with log_path.open("a") as f:
                f.write(json.dumps(host_metrics) + "\n")

    for step in range(start_step, train_cfg.num_steps):
        rng, batch_rng = jax.random.split(rng)
        batch = make_synthetic_batch(batch_rng, train_cfg.batch_size, train_cfg.seq_len, model_cfg.vocab_size, pad_id)
        _, metrics = train_step(optimizer, batch)

        _log_prev_metrics()
        prev_metrics = (step + 1, metrics)

        if checkpoint_manager is not None and save_every and (step + 1) % save_every == 0:
            _save_checkpoint(checkpoint_manager, optimizer, rng, step + 1)

    _log_prev_metrics(force=True)

    if checkpoint_manager is not None:
        if last_metrics and (not save_every or last_metrics["step"] % save_every != 0):
            _save_checkpoint(checkpoint_manager, optimizer, rng, int(last_metrics["step"]))
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()

    return optimizer, last_metrics
