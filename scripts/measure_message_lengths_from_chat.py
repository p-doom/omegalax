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

import tempfile
from pathlib import Path

from absl import app, flags
from transformers import AutoImageProcessor, AutoTokenizer

from omegalax.data.artifact_contract import make_measurement_contract
from omegalax.data.collator_qwen3 import make_message_length_fn
from omegalax.data.grain_pipeline import (
    MESSAGE_LENGTHS_FILENAME,
    measure_message_lengths_from_chat,
)
from omegalax.vlm.local_snapshot import open_local_vlm_snapshot

FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", None, "Path to a raw chat.jsonl dataset.", required=True)
flags.DEFINE_string(
    "out_dir",
    None,
    f"Output directory; the cache is written to <out_dir>/{MESSAGE_LENGTHS_FILENAME}.",
    required=True,
)
flags.DEFINE_string(
    "model_snapshot",
    None,
    "Absolute sealed local VLM snapshot directory.",
    required=True,
)
flags.DEFINE_integer(
    "num_workers", 2, "Number of parallel workers for message length measurement.", lower_bound=2
)


def _run(identity_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(identity_dir, local_files_only=True)
    image_processor = AutoImageProcessor.from_pretrained(
        identity_dir,
        use_fast=False,
        local_files_only=True,
    )

    out_dir = Path(FLAGS.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    measure_message = make_message_length_fn(tokenizer, image_processor)
    measurement_contract = make_measurement_contract(
        tokenizer=tokenizer,
        image_processor=image_processor,
        preprocessor_config_path=None,
    )
    out_path = measure_message_lengths_from_chat(
        FLAGS.data_path,
        out_dir / MESSAGE_LENGTHS_FILENAME,
        measure_message=measure_message,
        measurement_contract=measurement_contract,
        num_workers=FLAGS.num_workers,
    )
    print(out_path)


def main(_) -> None:
    with (
        open_local_vlm_snapshot(FLAGS.model_snapshot) as snapshot,
        tempfile.TemporaryDirectory(prefix="omegalax-vlm-identity-") as identity_tmp,
    ):
        identity_dir = Path(identity_tmp)
        snapshot.copy_identity_assets(identity_dir)
        _run(identity_dir)


if __name__ == "__main__":
    app.run(main)
