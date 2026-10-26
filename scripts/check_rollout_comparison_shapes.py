"""Abstract four-device full-size production train-step shapes, no Student GPU run."""

import argparse
import json
import os
from pathlib import Path
import socket

assert os.environ.get("SLURM_JOB_ID")
assert os.environ.get("SLURMD_NODENAME") == socket.gethostname().split(".")[0]
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

import omegalax

assert Path(omegalax.__file__).resolve().parents[1] == Path(__file__).resolve().parents[1], (
    omegalax.__file__
)
from flax import nnx
import jax
import jax.numpy as jnp
from omegalax.data.rollout_teacher_pipeline import (
    load_rollout_update_plan,
    _PlannedTeacherBatchBuilder,
    PLANNED_MICROBATCH_CHAINS,
)
from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.models.sharding_runtime import set_attn_backend, _finalize_q_shardings
from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM
from omegalax.text import api as text_api
from omegalax.trainers.pretrain import (
    prepare_pretrain_batch,
    make_pretrain_train_step,
    TrainConfig,
    PretrainMode,
    TrainingObjective,
)
from omegalax.trainers.text import build_optimizer
from omegalax.trainers.lr_schedule import build_common_token_schedule

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
from absl import flags

# Tokamax reads Abseil defaults lazily; our CLI belongs to argparse.
flags.FLAGS([str(__file__)])
assert len(jax.devices()) == 4
plan = load_rollout_update_plan(args.plan)
mesh = ensure_mesh(tp_size=1, fsdp_size=4, dp_size=1)
cfg = text_api.align_config_to_mesh(text_api.resolve_config(args.model), mesh)
results = {}
for T in (8192, 2048, 512):
    print(f"Tracing full 4B production train step at T={T}", flush=True)
    C = -(-plan["max_context_tokens"] // T)
    train_cfg = TrainConfig(
        batch_size=PLANNED_MICROBATCH_CHAINS * C,
        seq_len=T,
        grad_accum_steps=-(-plan["chains_per_update"] // PLANNED_MICROBATCH_CHAINS),
    )

    def initialize():
        model = Qwen3_5ForCausalLM(cfg, rngs=nnx.Rngs(params=0))
        set_attn_backend(model, text_backend="xla")
        return build_optimizer(model, train_cfg, lr_schedule=build_common_token_schedule(plan))

    with mesh_rules(mesh):
        optimizer = nnx.eval_shape(initialize)
        _finalize_q_shardings(optimizer.model, mesh)
        # Real first chain, real Teacher arrays; only model/optimizer execution is abstract.
        raw = _PlannedTeacherBatchBuilder(plan, T, "train", 0)(0)
        batch, _, _ = prepare_pretrain_batch(
            raw,
            PretrainMode.STATEPASSING_BPTT,
            cfg,
            mesh,
            global_batch_size=PLANNED_MICROBATCH_CHAINS * C,
            training_objective=TrainingObjective.TEACHER_TOP512_SPECIAL_TAIL_KL,
        )
        step = make_pretrain_train_step(
            PretrainMode.STATEPASSING_BPTT,
            cfg,
            bptt_chunks=None,
            pass_gdn_state=True,
            pass_conv_state=False,
            pass_rope_positions=False,
            training_objective=TrainingObjective.TEACHER_TOP512_SPECIAL_TAIL_KL,
        )
        loss, metrics = nnx.eval_shape(step, optimizer, batch, jnp.asarray(1.0))
    results[str(T)] = {
        "token_shape": list(batch["token_ids_BCT"].shape),
        "loss_shape": list(loss.shape),
        "metric_shapes": {k: list(v.shape) for k, v in metrics.items()},
    }
    print(f"PASS shapes T={T}", flush=True)
    del optimizer, raw, batch
with Path(args.output).open("x") as stream:
    json.dump(
        {
            "status": "PASS",
            "scope": "CPU abstract tracing only; no GPU execution or pretrained weight load",
            "mesh": {"dp": 1, "fsdp": 4, "tp": 1},
            "results": results,
        },
        stream,
        indent=2,
    )
print(json.dumps(results))
