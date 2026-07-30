"""Run a short Qwen3.5 FSDP optimizer smoke test."""

from __future__ import annotations

import importlib.metadata
import time

from absl import app, flags
from flax import nnx
import jax
from jax.sharding import NamedSharding, PartitionSpec
import numpy as np
import wandb

from omegalax.data.pretrain_statepassing import make_statepassing_iterator
from omegalax.distributed.mesh import ensure_mesh, mesh_rules, process_local_batch_size
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api
from omegalax.trainers.pretrain import (
    make_pretrain_train_step,
    prepare_pretrain_batch,
    PretrainMode,
)
from omegalax.trainers.text import build_optimizer, TrainConfig

P = PartitionSpec
FLAGS = flags.FLAGS

flags.DEFINE_string("model", None, "Qwen3.5 config or local model directory.", required=True)
flags.DEFINE_integer("fsdp_size", None, "FSDP mesh size.", required=True)
flags.DEFINE_integer("batch_size", None, "Global batch size.", required=True)
flags.DEFINE_integer("seq_len", 2048, "Sequence length.")
flags.DEFINE_integer("steps", 5, "Optimizer steps.")
flags.DEFINE_string("train_index_path", None, "Optional C4 statepassing train index.")
flags.DEFINE_string("wandb_group", None, "Optional W&B group for capacity tests.")
flags.DEFINE_string("wandb_name", None, "Optional W&B run name.")


def _tree_bytes_by_device(tree) -> dict[int, int]:
    totals = {device.id: 0 for device in jax.local_devices()}
    for value in jax.tree.leaves(tree):
        for shard in getattr(value, "addressable_shards", ()):
            totals[shard.device.id] += shard.data.nbytes
    return totals


def _print_tree_bytes(category: str, tree) -> dict[int, int]:
    values = _tree_bytes_by_device(tree)
    for device, num_bytes in values.items():
        print(f"tree_memory category={category} device={device} bytes={num_bytes}", flush=True)
    return values


def _print_device_memory(phase: str) -> dict[int, dict]:
    values = {}
    for device in jax.local_devices():
        memory = device.memory_stats() or {}
        values[device.id] = memory
        print(
            f"device_memory phase={phase} device={device.id} "
            f"bytes_in_use={memory.get('bytes_in_use')} "
            f"peak_bytes_in_use={memory.get('peak_bytes_in_use')} "
            f"bytes_limit={memory.get('bytes_limit')}",
            flush=True,
        )
    return values


def main(_) -> None:
    jax.distributed.initialize()
    mesh = ensure_mesh(tp_size=1, fsdp_size=FLAGS.fsdp_size, dp_size=1)
    local_batch_size = process_local_batch_size(FLAGS.batch_size, 1, FLAGS.fsdp_size)
    print(
        f"runtime jax={jax.__version__} nccl={importlib.metadata.version('nvidia-nccl-cu12')} "
        f"devices={jax.devices()} global_batch={FLAGS.batch_size} "
        f"local_batch={local_batch_size} seq_len={FLAGS.seq_len} "
        f"mode={'statepassing_bptt_c4' if FLAGS.train_index_path else 'iid_baseline'}",
        flush=True,
    )
    model_tag = FLAGS.model.rstrip("/").rsplit("/", 1)[-1].lower()
    wandb_run = None
    if FLAGS.wandb_group:
        wandb_run = wandb.init(
            entity="pdoom",
            project="omegalax",
            group=FLAGS.wandb_group,
            name=FLAGS.wandb_name,
            tags=["test", "capacity", "statepassing", "c4", model_tag, f"fsdp{FLAGS.fsdp_size}"],
            config={
                "model": FLAGS.model,
                "fsdp_size": FLAGS.fsdp_size,
                "global_batch_size": FLAGS.batch_size,
                "local_batch_size": local_batch_size,
                "seq_len": FLAGS.seq_len,
                "steps": FLAGS.steps,
                "bptt_chunks": 4,
            },
        )

    replicated = NamedSharding(mesh, P())
    init_rng = jax.device_put(jax.random.key(0), replicated)
    model, cfg = text_api.init_model(
        FLAGS.model,
        init_rng,
        tp_size=1,
        fsdp_size=FLAGS.fsdp_size,
        dp_size=1,
    )
    set_attn_backend(model, text_backend="mosaic_gpu")
    model_params = nnx.state(model, nnx.Param)
    jax.block_until_ready(model_params)
    param_bytes = _print_tree_bytes("model_params", model_params)
    _print_device_memory("after_model_init")
    train_cfg = TrainConfig(
        batch_size=FLAGS.batch_size,
        seq_len=FLAGS.seq_len,
        num_steps=FLAGS.steps,
        learning_rate=1e-4,
        lr_schedule="linear",
        lr_end_factor=1.0,
    )
    with mesh_rules(mesh):
        optimizer = build_optimizer(model, train_cfg)
    optimizer_state = nnx.state(optimizer.opt_state)
    jax.block_until_ready(optimizer_state)
    optimizer_bytes = _print_tree_bytes("optimizer_state", optimizer_state)
    _print_device_memory("after_optimizer_init")

    if FLAGS.train_index_path:
        data_iter = make_statepassing_iterator(
            FLAGS.train_index_path,
            batch_size=FLAGS.batch_size,
            chunk_length=FLAGS.seq_len,
            pad_id=0,
            eos_id=248046,
            shuffle=True,
            seed=0,
            dp_size=1,
            fsdp_size=FLAGS.fsdp_size,
            process_index=jax.process_index(),
            grain_workers=0,
        )
        pretrain_mode = PretrainMode.STATEPASSING_BPTT
        raw_batch = next(data_iter)
        batch, _, _ = prepare_pretrain_batch(
            raw_batch,
            pretrain_mode,
            cfg,
            mesh,
            global_batch_size=FLAGS.batch_size,
        )
        train_step = make_pretrain_train_step(pretrain_mode, cfg, bptt_chunks=4)
    else:
        rng = np.random.default_rng(0)
        tokens = rng.integers(
            1,
            cfg.vocab_size,
            size=(FLAGS.batch_size, FLAGS.seq_len),
            dtype=np.int32,
        )
        host_batch = {
            "token_ids_BT": tokens,
            "attention_mask_BT": np.ones_like(tokens),
            "loss_mask_BT": np.ones_like(tokens),
        }
        batch = text_api.shard_batch_dict(host_batch, cfg, mesh)
        train_step = make_pretrain_train_step(PretrainMode.IID_BASELINE, cfg)
    jax.block_until_ready(batch)
    batch_bytes = _print_tree_bytes("batch", batch)
    before_step_memory = _print_device_memory("before_train_step")
    if wandb_run:
        wandb_run.log(
            {
                "memory/model_params_bytes_per_gpu": max(param_bytes.values()),
                "memory/optimizer_state_bytes_per_gpu": max(optimizer_bytes.values()),
                "memory/batch_bytes_per_gpu": max(batch_bytes.values()),
                "memory/before_step_bytes_in_use_per_gpu": max(
                    memory["bytes_in_use"] for memory in before_step_memory.values()
                ),
            },
            step=0,
        )

    for step in range(1, FLAGS.steps + 1):
        started = time.monotonic()
        loss, metrics = train_step(optimizer, batch)
        jax.block_until_ready((loss, metrics))
        print(
            f"step={step} loss={float(loss):.6f} nll={float(metrics['nll']):.6f} "
            f"grad_norm={float(metrics['grad_norm']):.6f} seconds={time.monotonic() - started:.3f}",
            flush=True,
        )
        step_memory = _print_device_memory(f"after_step_{step}")
        if wandb_run:
            wandb_run.log(
                {
                    "train/loss": float(loss),
                    "train/nll": float(metrics["nll"]),
                    "train/grad_norm": float(metrics["grad_norm"]),
                    "memory/bytes_in_use_per_gpu": max(
                        memory["bytes_in_use"] for memory in step_memory.values()
                    ),
                    "memory/peak_bytes_in_use_per_gpu": max(
                        memory["peak_bytes_in_use"] for memory in step_memory.values()
                    ),
                },
                step=step,
            )
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    app.run(main)
