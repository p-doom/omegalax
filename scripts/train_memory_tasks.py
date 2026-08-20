"""Train controlled symbolic memory tasks through the production pretrain paths."""

from __future__ import annotations

import argparse
import dataclasses
from enum import StrEnum
import json
from pathlib import Path

from absl import flags
import jax
import wandb

from omegalax.data.memory_tasks import (
    MemoryTask,
    MemoryTaskConfig,
    MemoryTaskCue,
    MemoryTaskSplit,
    make_memory_task_iterator,
)
from omegalax.text import api as text_api
from omegalax.text.checkpoint import resolve_checkpoint
from omegalax.trainers import pretrain as pretrain_trainer
from omegalax.trainers import tail_pretrain
from omegalax.trainers.checkpoint_utils import ResumeMode


class TrainingCondition(StrEnum):
    FULL_CONTEXT = "full_context"
    RESET = "reset"
    STATE = "state"


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingLayout:
    batch_size: int
    seq_len: int
    flatten: bool
    pretrain_mode: pretrain_trainer.PretrainMode
    pass_gdn_state: bool
    bptt_chunks: int | None


def training_layout(
    condition: TrainingCondition | str,
    *,
    pairs_per_batch: int,
    num_segments: int,
    segment_length: int,
) -> TrainingLayout:
    condition = TrainingCondition(condition)
    if pairs_per_batch <= 0:
        raise ValueError("pairs_per_batch must be > 0")
    num_windows = 2 * pairs_per_batch
    if condition is TrainingCondition.FULL_CONTEXT:
        return TrainingLayout(
            batch_size=num_windows,
            seq_len=num_segments * segment_length,
            flatten=True,
            pretrain_mode=pretrain_trainer.PretrainMode.IID_BASELINE,
            pass_gdn_state=False,
            bptt_chunks=None,
        )
    return TrainingLayout(
        batch_size=num_windows * num_segments,
        seq_len=segment_length,
        flatten=False,
        pretrain_mode=pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
        pass_gdn_state=condition is TrainingCondition.STATE,
        bptt_chunks=num_segments,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[task.value for task in MemoryTask], required=True)
    parser.add_argument(
        "--condition",
        choices=[condition.value for condition in TrainingCondition],
        required=True,
    )
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--model_id", default="qwen3.5-smoke-dense")
    parser.add_argument("--init_model_checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--segment_length", type=int, default=16)
    parser.add_argument("--num_segments", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--num_rule_examples", type=int, default=5)
    parser.add_argument("--num_exact_bindings", type=int, default=1)
    parser.add_argument(
        "--relevance_cue",
        choices=[cue.value for cue in MemoryTaskCue],
        default=MemoryTaskCue.MARKED.value,
    )
    parser.add_argument("--pairs_per_batch", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_schedule", choices=("linear", "cosine", "wsd"), default="linear")
    parser.add_argument("--lr_end_factor", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--fsdp_size", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--keep_latest", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--resume", choices=[mode.value for mode in ResumeMode], default="never")
    parser.add_argument("--text_attn_backend", default="xla")
    parser.add_argument("--jax_cache_dir", type=Path, required=True)
    parser.add_argument("--wandb_project")
    parser.add_argument("--wandb_entity")
    parser.add_argument("--wandb_group")
    parser.add_argument("--wandb_name")
    parser.add_argument("--wandb_id")
    parser.add_argument("--wandb_resume")
    parser.add_argument("--wandb_tags", nargs="*", default=())
    return parser


def _device_memory_stats() -> list[dict[str, object]]:
    stats = []
    for device in jax.local_devices():
        try:
            memory = device.memory_stats()
        except Exception:
            continue
        if not memory:
            continue
        try:
            stats.append(
                {
                    "device_id": device.id,
                    "platform": device.platform,
                    "bytes_in_use": int(memory["bytes_in_use"]),
                    "peak_bytes_in_use": int(memory["peak_bytes_in_use"]),
                    "bytes_limit": int(memory["bytes_limit"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return stats


def _ensure_contract(save_dir: Path, contract: dict[str, object]) -> None:
    path = save_dir / "memory_task_contract.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != contract:
            raise ValueError(f"Memory-task contract mismatch at {path}")
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["train_memory_tasks.py"])
    jax.config.update("jax_compilation_cache_dir", str(args.jax_cache_dir))
    task_config = MemoryTaskConfig(
        task=args.task,
        seed=args.seed,
        segment_length=args.segment_length,
        num_segments=args.num_segments,
        vocab_size=args.vocab_size,
        num_rule_examples=args.num_rule_examples,
        num_exact_bindings=args.num_exact_bindings,
        relevance_cue=args.relevance_cue,
        split=MemoryTaskSplit.TRAIN,
    )
    layout = training_layout(
        args.condition,
        pairs_per_batch=args.pairs_per_batch,
        num_segments=args.num_segments,
        segment_length=args.segment_length,
    )
    model_source = args.model_id
    source_checkpoint = None
    if args.init_model_checkpoint is not None:
        source_checkpoint = resolve_checkpoint(args.init_model_checkpoint)
        model_source = str(source_checkpoint.config_path)
    model_config = text_api.resolve_config(model_source)
    if task_config.vocab_size > model_config.vocab_size:
        raise ValueError(
            f"Task vocab_size={task_config.vocab_size} exceeds model vocab_size={model_config.vocab_size}"
        )

    iterator = make_memory_task_iterator(
        task_config,
        global_batch_size=layout.batch_size,
        num_batches=args.num_steps * args.grad_accum_steps,
        flatten=layout.flatten,
        dp_size=args.dp_size,
        fsdp_size=args.fsdp_size,
    )
    contract = {
        "schema_version": 1,
        "task": dataclasses.asdict(task_config),
        "condition": str(args.condition),
        "pairs_per_batch": args.pairs_per_batch,
        "layout": {
            **dataclasses.asdict(layout),
            "pretrain_mode": layout.pretrain_mode.value,
        },
        "source": (
            {"model_id": args.model_id}
            if source_checkpoint is None
            else {
                "checkpoint_root": str(source_checkpoint.root),
                "checkpoint_step": source_checkpoint.step,
            }
        ),
        "optimizer": {
            "num_steps": args.num_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "lr_schedule": args.lr_schedule,
            "lr_end_factor": args.lr_end_factor,
            "max_grad_norm": args.max_grad_norm,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_eps": args.adam_eps,
            "grad_accum_steps": args.grad_accum_steps,
        },
        "state": {
            "pass_gdn_state": layout.pass_gdn_state,
            "bptt_chunks": layout.bptt_chunks,
            "pass_rope_positions": False,
            "pass_conv_state": False,
        },
    }
    save_dir = args.save_dir.expanduser().resolve()
    if jax.process_index() == 0:
        _ensure_contract(save_dir, contract)

    wandb_run = None
    if args.wandb_project and jax.process_index() == 0:
        wandb_kwargs = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": args.wandb_group,
            "name": args.wandb_name,
            "tags": list(args.wandb_tags) or None,
            "config": contract,
        }
        if args.wandb_id:
            wandb_kwargs["id"] = args.wandb_id
        if args.wandb_resume:
            wandb_kwargs["resume"] = args.wandb_resume
        wandb_run = wandb.init(**wandb_kwargs)

    try:
        if source_checkpoint is None:
            train_config = pretrain_trainer.TrainConfig(
                seed=args.seed,
                batch_size=layout.batch_size,
                seq_len=layout.seq_len,
                num_steps=args.num_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_steps=args.warmup_steps,
                lr_schedule=args.lr_schedule,
                lr_end_factor=args.lr_end_factor,
                max_grad_norm=args.max_grad_norm,
                adam_beta1=args.adam_beta1,
                adam_beta2=args.adam_beta2,
                adam_eps=args.adam_eps,
                grad_accum_steps=args.grad_accum_steps,
                print_every=args.log_every,
            )
            _, metrics = pretrain_trainer.run_pretrain(
                args.model_id,
                train_config,
                iterator,
                pretrain_mode=layout.pretrain_mode,
                save_dir=save_dir,
                save_every=args.save_every,
                keep_latest=args.keep_latest,
                log_every=args.log_every,
                resume=ResumeMode(args.resume),
                tp_size=args.tp_size,
                fsdp_size=args.fsdp_size,
                dp_size=args.dp_size,
                wandb_run=wandb_run,
                text_attn_backend=args.text_attn_backend,
                bptt_chunks=layout.bptt_chunks,
                pass_gdn_state=layout.pass_gdn_state,
                pass_rope_positions=False,
                pass_conv_state=False,
            )
        else:
            train_config = tail_pretrain.TailTrainConfig(
                seed=args.seed,
                batch_size=layout.batch_size,
                seq_len=layout.seq_len,
                num_steps=args.num_steps,
                global_step_offset=source_checkpoint.step,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_steps=args.warmup_steps,
                lr_schedule=args.lr_schedule,
                lr_end_factor=args.lr_end_factor,
                max_grad_norm=args.max_grad_norm,
                adam_beta1=args.adam_beta1,
                adam_beta2=args.adam_beta2,
                adam_eps=args.adam_eps,
                grad_accum_steps=args.grad_accum_steps,
            )
            _, metrics = tail_pretrain.run_tail_pretrain(
                source_checkpoint.step_path,
                train_config,
                iterator,
                pretrain_mode=layout.pretrain_mode,
                save_dir=save_dir,
                save_every=args.save_every,
                keep_latest=args.keep_latest,
                log_every=args.log_every,
                resume=ResumeMode(args.resume),
                tp_size=args.tp_size,
                fsdp_size=args.fsdp_size,
                dp_size=args.dp_size,
                wandb_run=wandb_run,
                text_attn_backend=args.text_attn_backend,
                bptt_chunks=layout.bptt_chunks,
                pass_gdn_state=layout.pass_gdn_state,
                pass_rope_positions=False,
                pass_conv_state=False,
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    metrics["device_memory"] = _device_memory_stats()
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
