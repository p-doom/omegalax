"""Data loading and collation utilities for SFT training."""

from omegalax.data.collator_qwen3 import TextSFTCollator, VLMSFTCollator
from omegalax.data.grain_pipeline import (
    MixSource,
    build_chunk_index,
    compile_jsonl_to_arrayrecord,
    make_grain_iterator,
    pop_source_ids,
)

__all__ = [
    "MixSource",
    "TextSFTCollator",
    "VLMSFTCollator",
    "build_chunk_index",
    "compile_jsonl_to_arrayrecord",
    "make_grain_iterator",
    "pop_source_ids",
]
