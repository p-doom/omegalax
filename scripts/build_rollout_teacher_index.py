"""Build, or validate an existing, rollout-teacher distillation pointer index.

Compute nodes only. An existing index (``out_dir/metadata.json``) is never rebuilt or
overwritten; it is validated metadata-only against the requested run parameters and the
source hashes, which keeps job restarts fast.
"""

from __future__ import annotations

import os
import socket
import sys

_job, _node, _host = (
    os.environ.get("SLURM_JOB_ID"),
    os.environ.get("SLURMD_NODENAME"),
    socket.gethostname(),
)
if not _job or not _node or _node.split(".")[0] != _host.split(".")[0]:
    sys.exit(f"Refusing to run outside a Slurm compute node: job={_job} node={_node} host={_host}")

import json
from pathlib import Path

import omegalax
from absl import app, flags

from omegalax.data.pretrain_data_set import COMPILED_METADATA_FILENAME
from omegalax.data.rollout_teacher_pipeline import (
    ROLLOUT_TEACHER_SELECTIONS,
    build_rollout_teacher_index,
    load_rollout_teacher_index,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_IMPORTED_REPO_ROOT = Path(omegalax.__file__).resolve().parents[1]
if _IMPORTED_REPO_ROOT != _REPO_ROOT:
    raise RuntimeError(
        f"Entrypoint/package repo mismatch: script={_REPO_ROOT}, imported_omegalax={_IMPORTED_REPO_ROOT}"
    )

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_root", None, "rollout_teacher_v1 dataset root.", required=True)
flags.DEFINE_string(
    "out_dir", None, "Index output directory (written nowhere else).", required=True
)
flags.DEFINE_enum(
    "selection", None, list(ROLLOUT_TEACHER_SELECTIONS), "Label selection.", required=True
)
flags.DEFINE_integer("segment_length", None, "Local segment length T in tokens.", required=True)
flags.DEFINE_list("segment_buckets", None, "Strictly increasing segment counts C.", required=True)


def main(_) -> None:
    out_dir = Path(FLAGS.out_dir).expanduser().resolve()
    config = dict(
        selection=FLAGS.selection,
        segment_length=FLAGS.segment_length,
        segment_buckets=[int(value) for value in FLAGS.segment_buckets],
    )
    if (out_dir / COMPILED_METADATA_FILENAME).exists():
        metadata = load_rollout_teacher_index(out_dir, **config)
        if (
            str(Path(FLAGS.dataset_root).expanduser().resolve())
            != metadata["source"]["dataset_root"]
        ):
            raise ValueError(
                f"Existing index at {out_dir} was built from {metadata['source']['dataset_root']}"
            )
        action = "validated existing"
    else:
        build_rollout_teacher_index(FLAGS.dataset_root, out_dir, **config)
        metadata = load_rollout_teacher_index(out_dir, **config)
        action = "built"
    summary = {
        split: {
            key: value[key]
            for key in ("num_samples", "num_context_tokens", "num_targets", "label_counts")
        }
        | {"bucket_samples": [bucket["num_samples"] for bucket in value["buckets"]]}
        for split, value in metadata["splits"].items()
    }
    print(f"{action} rollout-teacher index {out_dir} index_binding={metadata['index_binding']}")
    print(
        json.dumps(
            {"target_capacities": metadata["target_capacities"], "splits": summary}, sort_keys=True
        )
    )


if __name__ == "__main__":
    app.run(main)
