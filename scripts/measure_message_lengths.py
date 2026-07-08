"""Measure per-message token lengths for a compiled payload, once.

Tokenization is the only tokenizer/processor-bound step of chunk-index
building and is independent of max_length / overflow_mode / system_message.
This script runs that pass once and writes a ``message_lengths.jsonl`` cache;
``build_sft_chunk_index.py --message_lengths_path=<cache>`` then reuses it for
every sequence length, so changing max_length no longer re-tokenizes.

Mirrors build_sft_chunk_index.py's tokenizer/processor setup so the measured
lengths are byte-identical to what the in-line measurement would produce.
"""

from __future__ import annotations

import json
from pathlib import Path

from absl import app, flags
from transformers import AutoImageProcessor, AutoTokenizer

from omegalax.data.grain_pipeline import MESSAGE_LENGTHS_FILENAME, measure_message_lengths
from omegalax.data.qwen3_encoding import make_message_length_fn
from omegalax.registry import resolve_hf_repo_id

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_path", None, "Path to a canonical compiled payload-block dataset.", required=True
)
flags.DEFINE_string(
    "out_dir",
    None,
    f"Output directory; the cache is written to <out_dir>/{MESSAGE_LENGTHS_FILENAME}.",
    required=True,
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
flags.DEFINE_integer(
    "num_workers", 2, "Number of parallel workers for message length measurement.", lower_bound=2
)


def main(_) -> None:
    tokenizer_name = FLAGS.tokenizer or resolve_hf_repo_id(FLAGS.model_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    image_processor = None
    if FLAGS.processor:
        ip_kwargs: dict = {}
        if FLAGS.preprocessor_config:
            with open(FLAGS.preprocessor_config) as f:
                ip_kwargs = json.load(f)
        image_processor = AutoImageProcessor.from_pretrained(
            FLAGS.processor, use_fast=False, **ip_kwargs
        )

    out_dir = Path(FLAGS.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = measure_message_lengths(
        FLAGS.data_path,
        out_dir / MESSAGE_LENGTHS_FILENAME,
        measure_message=make_message_length_fn(tokenizer, image_processor),
        num_workers=FLAGS.num_workers,
    )
    print(out_path)


if __name__ == "__main__":
    app.run(main)
