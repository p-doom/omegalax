"""Build self-contained inline SFT records straight from a raw chat.jsonl.

Payload-free analog of build_sft_chunk_index.py: skips the grain payload (stage
05) entirely. Reads chat.jsonl directly, bins each conversation's turns into
<= max_length token chunks, and writes ArrayRecord shards whose records ARE the
training examples (message slices with ar:// image refs preserved) -- not
pointers into a shared payload. The stage 01 master image store is unchanged;
records reference it by ar:// exactly as chat.jsonl does.
"""

from __future__ import annotations

import json

from absl import app, flags
from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer

from omegalax.data.grain_pipeline import build_records_from_chat
from omegalax.data.qwen3_encoding import make_message_length_fn
from omegalax.registry import resolve_hf_repo_id

FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", None, "Path to a raw chat.jsonl dataset.", required=True)
flags.DEFINE_string(
    "out_dir", None, "Output directory for the inline-records dataset.", required=True
)
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
flags.DEFINE_enum(
    "overflow_mode",
    "drop",
    ["split", "truncate", "drop"],
    "How to handle a conversation whose turns exceed the token budget. "
    "'drop' (default): discard the whole conversation if it does not fit in a "
    "single chunk. 'split': pack into multiple consecutive chunks at turn "
    "boundaries (no turns dropped). 'truncate': keep only the first fitting "
    "chunk and drop the overflowing turn plus the rest of the conversation. "
    "Truncation stats are written to truncation_stats.json.",
)
flags.DEFINE_float(
    "val_fraction",
    0.0,
    "Recording-level val fraction used only to compute the train/val split when "
    "--split is set.",
)
flags.DEFINE_string(
    "split",
    None,
    "If set (e.g. 'train' or 'val'), emit only conversations whose "
    "recording-level split (from --val_fraction over the row's recording_id) "
    "matches. Omit to emit all conversations.",
)


def main(_) -> None:
    tokenizer_name = FLAGS.tokenizer or resolve_hf_repo_id(FLAGS.model_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model_type = AutoConfig.from_pretrained(resolve_hf_repo_id(FLAGS.model_id)).model_type

    image_processor = None
    if FLAGS.processor:
        ip_kwargs: dict = {}
        if FLAGS.preprocessor_config:
            with open(FLAGS.preprocessor_config) as f:
                ip_kwargs = json.load(f)
        image_processor = AutoImageProcessor.from_pretrained(
            FLAGS.processor, use_fast=False, **ip_kwargs
        )

    measure_message = make_message_length_fn(tokenizer, image_processor, model_type)
    out_dir = build_records_from_chat(
        FLAGS.data_path,
        FLAGS.out_dir,
        max_length=FLAGS.max_length,
        measure_message=measure_message,
        records_per_shard=FLAGS.records_per_shard,
        overwrite=FLAGS.overwrite,
        num_workers=FLAGS.num_workers,
        overflow_mode=FLAGS.overflow_mode,
        val_fraction=FLAGS.val_fraction,
        split=FLAGS.split,
    )
    print(out_dir)


if __name__ == "__main__":
    app.run(main)
