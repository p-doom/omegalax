"""Compile a JSONL SFT dataset into Grain-friendly ArrayRecord shards."""

from __future__ import annotations

from absl import app, flags

from omegalax.data.grain_pipeline import compile_jsonl_to_arrayrecord

FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", None, "Path to the raw JSONL dataset.", required=True)
flags.DEFINE_string("out_dir", None, "Output directory for ArrayRecord shards.", required=True)
flags.DEFINE_integer("messages_per_record", 128, "Maximum contiguous messages to store in one payload block.")
flags.DEFINE_integer("records_per_shard", 10_000, "Records per output shard.")
flags.DEFINE_bool("overwrite", False, "Overwrite existing output directory.")


def main(_) -> None:
    out_dir = compile_jsonl_to_arrayrecord(
        FLAGS.data_path,
        FLAGS.out_dir,
        messages_per_record=FLAGS.messages_per_record,
        records_per_shard=FLAGS.records_per_shard,
        overwrite=FLAGS.overwrite,
    )
    print(out_dir)


if __name__ == "__main__":
    app.run(main)
