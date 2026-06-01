"""Print per-epoch step counts for a compiled Grain chunk-index dataset.

Pure metadata read — no JAX/HF imports, instant startup. Use this to size
``--num_steps`` against a dataset before launching ``train_vlm_sft.py``.

Example:
    uv run -- python scripts/grain_dataset_info.py \\
        --data_path /path/to/compiled_chunk_index/train \\
        --batch_size 2 --dp_size 1 --fsdp_size 1 --grad_accum_steps 1
"""

from __future__ import annotations

import json
from pathlib import Path

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", None, "Path to a compiled Grain chunk-index dataset directory.", required=True)
flags.DEFINE_integer("batch_size", None, "Global batch size across all JAX processes.", required=True)
flags.DEFINE_integer("dp_size", 1, "Data parallelism size.")
flags.DEFINE_integer("fsdp_size", 1, "FSDP parallelism size.")
flags.DEFINE_integer("grad_accum_steps", 1, "Gradient accumulation steps.")
flags.DEFINE_integer("num_steps", None, "If set, also report how many epochs --num_steps corresponds to.")


def main(_) -> None:
    metadata_path = Path(FLAGS.data_path).expanduser().resolve() / "metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(f"No metadata.json found at {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    num_records = int(metadata["num_records"])

    dp = FLAGS.dp_size * FLAGS.fsdp_size
    if FLAGS.batch_size <= 0:
        raise SystemExit(f"--batch_size must be > 0, got {FLAGS.batch_size}")
    if FLAGS.batch_size % dp != 0:
        raise SystemExit(
            f"--batch_size={FLAGS.batch_size} must be divisible by dp_size*fsdp_size={dp}"
        )
    per_process_batch = FLAGS.batch_size // dp

    records_per_epoch = num_records // dp
    micro_batches_per_epoch = records_per_epoch // per_process_batch
    steps_per_epoch = micro_batches_per_epoch // FLAGS.grad_accum_steps

    print(f"Compiled Grain dataset: {metadata_path.parent}")
    print(f"  num_records:                 {num_records}")
    print(f"  global_batch_size:           {FLAGS.batch_size}")
    print(f"  dp_size * fsdp_size:         {dp}")
    print(f"  per_process_batch_size:      {per_process_batch}")
    print(f"  grad_accum_steps:            {FLAGS.grad_accum_steps}")
    print()
    print("Per epoch (drop_remainder=True at both shard and batch):")
    print(f"  records_per_process:         {records_per_epoch}")
    print(f"  micro_batches_per_process:   {micro_batches_per_epoch}")
    print(f"  optimizer_steps:             {steps_per_epoch}")

    if records_per_epoch <= 0:
        print()
        print(f"WARNING: num_records={num_records} < dp={dp}; cannot shard with drop_remainder.")
    elif micro_batches_per_epoch <= 0:
        print()
        print(
            f"WARNING: per_process_batch_size={per_process_batch} > "
            f"records_per_process={records_per_epoch}; cannot form a full batch."
        )

    if FLAGS.num_steps is not None and steps_per_epoch > 0:
        full_epochs = FLAGS.num_steps // steps_per_epoch
        leftover_steps = FLAGS.num_steps % steps_per_epoch
        print()
        print(f"For --num_steps={FLAGS.num_steps}:")
        print(f"  full_epochs:                 {full_epochs}")
        print(f"  + leftover_steps:            {leftover_steps}")


if __name__ == "__main__":
    app.run(main)
