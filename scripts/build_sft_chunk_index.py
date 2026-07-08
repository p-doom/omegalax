"""Build an offline chunk index for a compiled canonical SFT dataset."""

from __future__ import annotations

import json

from absl import app, flags
from transformers import AutoImageProcessor, AutoTokenizer

from omegalax.data.grain_pipeline import build_chunk_index
from omegalax.data.qwen3_encoding import make_message_length_fn
from omegalax.registry import resolve_hf_repo_id

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_path", None, "Path to a canonical compiled payload-block dataset.", required=True
)
flags.DEFINE_string("out_dir", None, "Output directory for the chunk-index dataset.", required=True)
flags.DEFINE_string(
    "model_id", None, "Model id used to resolve the default tokenizer.", required=True
)
flags.DEFINE_string("tokenizer", None, "HF tokenizer name/path (defaults to --model_id).")
flags.DEFINE_string(
    "processor", None, "HF repo to read image config from when the dataset contains images."
)
flags.DEFINE_string(
    "preprocessor_config",
    None,
    "Path to JSON file whose keys override default image processor config.",
)
flags.DEFINE_integer("max_length", None, "Maximum sequence length.", required=True)
flags.DEFINE_integer("records_per_shard", 100_000, "Records per output shard.")
flags.DEFINE_bool("overwrite", False, "Overwrite existing output directory.")
flags.DEFINE_integer(
    "num_workers", 2, "Number of parallel workers for message length measurement.", lower_bound=2
)
flags.DEFINE_string(
    "system_message_text",
    "",
    "If non-empty, prepend a text-only system message with this content to "
    "every emitted chunk. Persisted in chunk-index metadata; the iterator "
    "injects it at resolve time and the per-chunk token budget is reduced "
    "by the system message's measured length.",
)
flags.DEFINE_enum(
    "overflow_mode",
    "split",
    ["split", "truncate", "drop"],
    "How to handle a conversation whose turns exceed the token budget. "
    "'split': pack into multiple consecutive chunks at turn boundaries "
    "(no turns dropped). 'truncate': keep only the first fitting chunk and "
    "drop the overflowing turn plus the rest of the conversation. 'drop': "
    "discard the whole conversation if it does not fit in a single chunk. "
    "Truncation stats are written to truncation_stats.json.",
)
flags.DEFINE_string(
    "message_lengths_path",
    None,
    "Path to a message_lengths.jsonl cache (see scripts/measure_message_lengths.py). "
    "If set and present, per-message token lengths are loaded from it and the "
    "tokenizer pass is skipped; if set and absent, lengths are measured and "
    "written there. Lets repeated builds over the same payload (different "
    "max_length / overflow_mode) avoid re-tokenizing.",
)


def main(_) -> None:
    tokenizer_name = FLAGS.tokenizer or resolve_hf_repo_id(FLAGS.model_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    image_processor = None
    processor_name = None
    if FLAGS.processor:
        processor_name = FLAGS.processor
        ip_kwargs: dict = {}
        if FLAGS.preprocessor_config:
            with open(FLAGS.preprocessor_config) as f:
                ip_kwargs = json.load(f)
        image_processor = AutoImageProcessor.from_pretrained(
            processor_name, use_fast=False, **ip_kwargs
        )

    system_message = None
    if FLAGS.system_message_text:
        system_message = {
            "role": "system",
            "content": [{"type": "text", "text": FLAGS.system_message_text}],
        }

    out_dir = build_chunk_index(
        FLAGS.data_path,
        FLAGS.out_dir,
        max_length=FLAGS.max_length,
        measure_message=make_message_length_fn(tokenizer, image_processor),
        records_per_shard=FLAGS.records_per_shard,
        overwrite=FLAGS.overwrite,
        num_workers=FLAGS.num_workers,
        system_message=system_message,
        overflow_mode=FLAGS.overflow_mode,
        message_lengths_path=FLAGS.message_lengths_path,
        profile_metadata={
            "model_id": FLAGS.model_id,
            "tokenizer": tokenizer_name,
            "processor": processor_name,
            "preprocessor_config": FLAGS.preprocessor_config,
        },
    )
    print(out_dir)


if __name__ == "__main__":
    app.run(main)
