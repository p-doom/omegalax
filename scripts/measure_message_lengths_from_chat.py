"""Measure per-message token lengths for a raw chat.jsonl, once (payload-free).

Payload-free analog of measure_message_lengths.py: tokenizes a raw chat.jsonl
directly (no intermediate grain payload) and writes a ``message_lengths.jsonl``
cache keyed by ``(conv_idx, msg_offset)``. Per-message lengths are the only
tokenizer/processor-bound product of record building and are independent of
max_length / overflow_mode / split, so running this once lets every
build_sft_records_from_chat.py build over the same chat reuse the cache (via
--message_lengths_path) instead of re-tokenizing per sequence length.

Mirrors build_sft_records_from_chat.py's tokenizer/processor setup so the
measured lengths are byte-identical to what the in-line measurement produces.
"""

from __future__ import annotations

import json
from pathlib import Path

from absl import app, flags
from transformers import AutoImageProcessor, AutoTokenizer

from omegalax.data.artifact_contract import make_measurement_contract
from omegalax.data.grain_pipeline import (
    MESSAGE_LENGTHS_FILENAME,
    measure_message_lengths_from_chat,
)
from omegalax.data.collator_qwen3 import make_message_length_fn
from omegalax.registry import resolve_hf_repo_id

FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", None, "Path to a raw chat.jsonl dataset.", required=True)
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
flags.DEFINE_string(
    "producer_sha",
    None,
    "Exact Omegalax Git SHA whose renderer and preprocessing code produced the cache.",
    required=True,
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
    measure_message = make_message_length_fn(tokenizer, image_processor)
    measurement_contract = make_measurement_contract(
        producer_sha=FLAGS.producer_sha,
        tokenizer=tokenizer,
        tokenizer_source=tokenizer_name,
        image_processor=image_processor,
        processor_source=FLAGS.processor,
        renderer_config=measure_message.renderer_config,
        preprocessor_config_path=FLAGS.preprocessor_config,
    )
    out_path = measure_message_lengths_from_chat(
        FLAGS.data_path,
        out_dir / MESSAGE_LENGTHS_FILENAME,
        measure_message=measure_message,
        measurement_contract=measurement_contract,
        num_workers=FLAGS.num_workers,
    )
    print(out_path)


if __name__ == "__main__":
    app.run(main)
