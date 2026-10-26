"""Controlled comparison CLI: common plan, genuine pretrained weights, four local GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time


def main():
    job, node, host = (
        os.environ.get("SLURM_JOB_ID"),
        os.environ.get("SLURMD_NODENAME"),
        socket.gethostname().split(".")[0],
    )
    if not job or not node or node.split(".")[0] != host:
        raise SystemExit(f"Not a Slurm compute node: {job=} {node=} {host=}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--segment-length", type=int, choices=[8192, 2048, 512], required=True)
    parser.add_argument("--init-hf-model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", choices=["never", "if_present", "required"], default="never")
    parser.add_argument("--stop-after-update", type=int)
    parser.add_argument("--wandb-project")
    parser.add_argument("--text-attn-backend", default="triton")
    parser.add_argument("--peak-tflops", default="h100_sxm")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        key: value for key, value in sorted(os.environ.items())
        if key.startswith(("JAX", "XLA", "NCCL", "TF_", "OMEGALAX"))
    }
    (output / f"numerical_environment_{job}_{os.getpid()}.json").write_text(
        json.dumps({"job": job, "host": host, "source": str(Path(__file__).resolve().parents[1]),
                    "text_attn_backend": args.text_attn_backend, "environment": environment}, indent=2)
        + "\n"
    )
    from absl import flags

    # Tokamax reads Abseil defaults lazily; our CLI belongs to argparse.
    flags.FLAGS([str(__file__)])
    import omegalax

    assert Path(omegalax.__file__).resolve().parents[1] == Path(__file__).resolve().parents[1], (
        omegalax.__file__
    )
    import jax
    import numpy as np
    from flax import nnx
    from omegalax.data.rollout_teacher_pipeline import (
        load_rollout_update_plan,
        make_planned_rollout_teacher_iterator,
        PLANNED_MICROBATCH_CHAINS,
    )
    from omegalax.training_contract import ensure_training_contract
    from omegalax.trainers.pretrain import (
        run_pretrain,
        TrainConfig,
        TrainingObjective,
        PretrainMode,
    )
    from omegalax.trainers.checkpoint_utils import ResumeMode
    from omegalax.trainers.perf import resolve_peak_tflops

    if (
        jax.process_count() != 1
        or len(jax.devices()) != 4
        or any(d.platform != "gpu" for d in jax.devices())
    ):
        raise SystemExit(
            "Comparison requires one independent JAX controller with exactly four local GPUs"
        )
    plan = load_rollout_update_plan(args.plan)
    C = -(-plan["max_context_tokens"] // args.segment_length)
    train_cfg = TrainConfig(
        seed=0,
        batch_size=PLANNED_MICROBATCH_CHAINS * C,
        seq_len=args.segment_length,
        num_steps=len(plan["updates"]),
        lr_schedule_steps=len(plan["updates"]),
        learning_rate=2e-5,
        warmup_steps=plan["warmup_end_update"],
        lr_schedule="wsd",
        lr_end_factor=0.0,
        lr_stable_fraction=0.9,
        grad_accum_steps=-(-plan["chains_per_update"] // PLANNED_MICROBATCH_CHAINS),
        max_grad_norm=1.0,
    )
    contract = {
        "schema_version": 1,
        "common_plan_sha256": plan["plan_sha256"],
        "source": plan["source"],
        "lr": plan["lr"],
        "schedule_steps": len(plan["updates"]),
        "checkpoint_updates": plan["checkpoint_updates"],
        "train_config": vars(train_cfg),
        "pretrained_snapshot": str(Path(args.init_hf_model_path).resolve()),
        "mesh": {"dp": 1, "fsdp": 4, "tp": 1},
        "text_attn_backend": args.text_attn_backend,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "objective": "teacher_top512_special_tail_kl",
        "eval_statepassing_config": {
            "pass_gdn_state": True,
            "gdn_layer_limit": None,
            "pass_conv_state": False,
            "pass_rope_positions": False,
            "pad_id": 0,
            "eos_id": 248046,
        },
        "horizon_by_step": [{"start_step": 1, "end_step": None, "c_train": C}],
        "resume": "optimizer update cursor = restored Grain cursor / ceil(chains_per_update / 4); verified before every update",
    }
    ensure_training_contract(output, contract)
    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"rollouts_T{args.segment_length}_fast",
            id=f"{plan['plan_sha256'][:12]}_T{args.segment_length}_fast",
            resume="allow",
            config=contract,
        )
    start = time.perf_counter()
    optimizer, metrics = run_pretrain(
        args.init_hf_model_path,
        train_cfg,
        make_planned_rollout_teacher_iterator(plan, segment_length=args.segment_length),
        pretrain_mode=PretrainMode.STATEPASSING_BPTT,
        training_objective=TrainingObjective.TEACHER_TOP512_SPECIAL_TAIL_KL,
        teacher_segment_buckets=[C],
        common_update_plan=plan,
        init_hf_model_path=args.init_hf_model_path,
        save_dir=output,
        save_every=0,
        keep_latest=None,
        log_every=1,
        resume=ResumeMode(args.resume),
        stop_after_local_step=args.stop_after_update,
        tp_size=1,
        fsdp_size=4,
        dp_size=1,
        bptt_chunks=None,
        pass_gdn_state=True,
        pass_conv_state=False,
        pass_rope_positions=False,
        text_attn_backend=args.text_attn_backend,
        peak_tflops=resolve_peak_tflops(args.peak_tflops),
        wandb_run=wandb_run,
        val_data_iter=make_planned_rollout_teacher_iterator(
            plan, segment_length=args.segment_length, split="val"
        ),
        val_every=500,
        val_steps=-(-len(plan["validation_members"]) // PLANNED_MICROBATCH_CHAINS),
    )
    if not all(np.isfinite(metrics[k]) for k in ("teacher_kl", "grad_norm", "lr")):
        raise RuntimeError(f"Nonfinite comparison result: {metrics}")
    state_digest = None
    if plan.get("gate_parent_plan_sha256"):
        digest = hashlib.sha256()
        for leaf in jax.tree.leaves(nnx.state(optimizer)):
            value = np.asarray(leaf)
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.tobytes())
        state_digest = digest.hexdigest()
    result = {
        "job_id": job,
        "optimizer_state_sha256": state_digest,
        "host": host,
        "devices": [str(d) for d in jax.devices()],
        "plan_sha256": plan["plan_sha256"],
        "segment_length": args.segment_length,
        "elapsed_seconds_including_compile_restore_save": time.perf_counter() - start,
        "device_memory": [d.memory_stats() for d in jax.local_devices()],
        "metrics": metrics,
        "optimizer_microsteps": int(np.asarray(optimizer.step[...])),
    }
    with (output / f"result_step{int(metrics['step'])}.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
