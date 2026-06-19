"""Data loading and collation utilities."""

from omegalax.data.collator_qwen3 import TextSFTCollator, VLMSFTCollator
from omegalax.data.grain_pipeline import (
    MixSource,
    build_chunk_index,
    compile_jsonl_to_arrayrecord,
    make_grain_iterator,
    pop_source_ids,
)
from omegalax.data.pretrain_doc_chain import (
    BATCH_PRETRAIN_METADATA_KEY,
    DEFAULT_DOC_CHAIN_SPLIT,
    DOC_CHAIN_FORMAT,
    DocChainReader,
    DocChainRecord,
    DocPairRef,
    deserialize_doc_chain,
    iter_document_pair_refs,
    load_doc_chain_metadata,
    pop_pretrain_metadata,
    resolve_doc_chain_buckets,
    resolve_pretrain_dp,
)
from omegalax.data.pretrain_iid_pipeline import (
    build_iid_chunk_index,
    make_iid_iterator,
)
from omegalax.data.pretrain_statepassing import (
    STATEPASSING_PAIR_INDEX_FORMAT,
    build_statepassing_pair_index,
    make_statepassing_iterator,
)

__all__ = [
    "BATCH_PRETRAIN_METADATA_KEY",
    "DEFAULT_DOC_CHAIN_SPLIT",
    "DOC_CHAIN_FORMAT",
    "DocChainReader",
    "DocChainRecord",
    "DocPairRef",
    "MixSource",
    "STATEPASSING_PAIR_INDEX_FORMAT",
    "TextSFTCollator",
    "VLMSFTCollator",
    "build_chunk_index",
    "build_iid_chunk_index",
    "build_statepassing_pair_index",
    "compile_jsonl_to_arrayrecord",
    "deserialize_doc_chain",
    "iter_document_pair_refs",
    "load_doc_chain_metadata",
    "make_grain_iterator",
    "make_iid_iterator",
    "make_statepassing_iterator",
    "pop_pretrain_metadata",
    "pop_source_ids",
    "resolve_doc_chain_buckets",
    "resolve_pretrain_dp",
]
