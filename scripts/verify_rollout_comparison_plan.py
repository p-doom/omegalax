"""Bounded genuine-loader equality and resume evidence, no full target corpus scan."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

assert os.environ.get("SLURM_JOB_ID")
assert os.environ.get("SLURMD_NODENAME") == socket.gethostname().split(".")[0]

import omegalax

assert Path(omegalax.__file__).resolve().parents[1] == Path(__file__).resolve().parents[1], (
    omegalax.__file__
)
import numpy as np
from omegalax.data.pretrain_data_set import BATCH_PRETRAIN_METADATA_KEY
from omegalax.data.rollout_teacher_pipeline import (
    TEACHER_BATCH_KEYS,
    load_rollout_update_plan,
    make_planned_rollout_teacher_iterator,
    PLANNED_MICROBATCH_CHAINS,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
plan = load_rollout_update_plan(args.plan)
for epoch in range(2):
    actual = [i for u in plan["updates"] if u["epoch"] == epoch for i in u["members"]]
    expected = [i for i, r in enumerate(plan["records"]) if r["split"] == "train"]
    assert sorted(actual) == expected
    assert len(actual) == len(set(actual))


def digest(batch):
    meta = batch[BATCH_PRETRAIN_METADATA_KEY]
    real = batch["attention_mask_BCT"].astype(bool).reshape(-1)
    digest = hashlib.sha256(batch["token_ids_BCT"].reshape(-1)[real].tobytes())
    for key in TEACHER_BATCH_KEYS:
        digest.update(batch[key].tobytes())
    return {
        "sequence_ids": meta["sequence_ids"],
        "token_count": int(real.sum()),
        "teacher_target_count": int(batch["teacher_mask_BR"].sum()),
        "actual_token_and_teacher_bytes_sha256": digest.hexdigest(),
    }


baseline = None
shapes = {}
for T in (8192, 2048, 512):
    iterator = make_planned_rollout_teacher_iterator(plan, segment_length=T)
    evidence = []
    # Entire first optimizer update, including all real Teacher arrays; one chain at a time.
    for micro in range(-(-plan["chains_per_update"] // PLANNED_MICROBATCH_CHAINS)):
        batch = next(iterator)
        evidence.append(digest(batch))
        shapes[str(T)] = {k: list(v.shape) for k, v in batch.items() if isinstance(v, np.ndarray)}
        assert batch[BATCH_PRETRAIN_METADATA_KEY]["common_update_idx"] == 0
        assert batch[BATCH_PRETRAIN_METADATA_KEY]["common_micro_idx"] == micro
        del batch
    state = iterator.get_state()
    uninterrupted = digest(next(iterator))
    restored = make_planned_rollout_teacher_iterator(plan, segment_length=T)
    restored.set_state(state)
    assert digest(next(restored)) == uninterrupted
    if baseline is None:
        baseline = evidence
    else:
        assert evidence == baseline, f"Actual loader bytes differ for T={T}"
    del iterator, restored
result = {
    "status": "PASS",
    "plan_sha256": plan["plan_sha256"],
    "cpu_job_id": os.environ["SLURM_JOB_ID"],
    "host": socket.gethostname(),
    "all_updates_metadata_verified": len(plan["updates"]),
    "actual_first_update_members": baseline,
    "shapes": shapes,
    "resume": "real Grain cursor after first update reproduces next complete batch for all arms",
}
with Path(args.output).open("x") as stream:
    json.dump(result, stream, indent=2)
print(
    json.dumps(
        {k: v for k, v in result.items() if k not in ("shapes", "actual_first_update_members")}
    )
)
