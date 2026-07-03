"""Data loading and collation utilities."""

from omegalax.data.collator_qwen3 import TextSFTCollator, VLMSFTCollator
from omegalax.data.grain_pipeline import (
    MixSource,
    build_chunk_index,
    compile_jsonl_to_arrayrecord,
    make_grain_iterator,
    pop_source_ids,
)
from omegalax.data.pretrain_data_set import (
    BATCH_PRETRAIN_METADATA_KEY,
    DEFAULT_DATA_SET_SPLIT,
    DOC_CHAIN_FORMAT,
    DataSetReader,
    DataSetRecord,
    WindowMetadata,
    build_window_arrays,
    deserialize_data_set_record,
    iter_document_window_metadata,
    load_data_set_metadata,
    pop_pretrain_metadata,
    resolve_data_set_buckets,
    resolve_pretrain_dp,
)
from omegalax.data.pretrain_iid_pipeline import make_iid_iterator
from omegalax.data.pretrain_statepassing import (
    STATEPASSING_WINDOW_INDEX_FORMAT,
    build_statepassing_window_index,
    make_statepassing_iterator,
)

__all__ = [
    "BATCH_PRETRAIN_METADATA_KEY",
    "DEFAULT_DATA_SET_SPLIT",
    "DOC_CHAIN_FORMAT",
    "DataSetReader",
    "DataSetRecord",
    "WindowMetadata",
    "MixSource",
    "STATEPASSING_WINDOW_INDEX_FORMAT",
    "TextSFTCollator",
    "VLMSFTCollator",
    "build_chunk_index",
    "build_statepassing_window_index",
    "build_window_arrays",
    "compile_jsonl_to_arrayrecord",
    "deserialize_data_set_record",
    "iter_document_window_metadata",
    "load_data_set_metadata",
    "make_grain_iterator",
    "make_iid_iterator",
    "make_statepassing_iterator",
    "pop_pretrain_metadata",
    "pop_source_ids",
    "resolve_data_set_buckets",
    "resolve_pretrain_dp",
]
