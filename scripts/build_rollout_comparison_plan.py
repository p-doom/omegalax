"""Build immutable common membership from metadata only, in the shared CPU allocation."""

import argparse
import os
from pathlib import Path
import socket

assert os.environ.get("SLURM_JOB_ID")
assert os.environ.get("SLURMD_NODENAME") == socket.gethostname().split(".")[0]

import omegalax

assert Path(omegalax.__file__).resolve().parents[1] == Path(__file__).resolve().parents[1], (
    omegalax.__file__
)
from omegalax.data.rollout_teacher_pipeline import build_rollout_update_plan

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset-root", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
plan = build_rollout_update_plan(args.dataset_root, args.output)
print(
    {
        k: v
        for k, v in plan.items()
        if k not in ("records", "updates", "validation_members", "source")
    }
)
print(
    "selected chains",
    len(plan["records"]),
    "val",
    len(plan["validation_members"]),
    "updates",
    len(plan["updates"]),
)
