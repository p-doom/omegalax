"""Helpers for pretraining tests that sample the real FineWeb pretraining data."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from absl.testing import absltest
from array_record.python.array_record_module import ArrayRecordReader, ArrayRecordWriter

from omegalax.data.pretrain_data_set import (
    ARRAY_RECORD_SUFFIX,
    COMPILED_METADATA_FILENAME,
    DOC_CHAIN_BINARY_MAGIC,
    DOC_CHAIN_DATASET_VERSION,
    DOC_CHAIN_FORMAT,
)

REAL_PRETRAIN_ROOT = Path(
    os.environ.get(
        "OMEGALAX_PRETRAIN_TEST_DATA",
        "/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/datasets/fineweb_edu_dedup_30b_8kto32k",
    )
)


def require_real_root(testcase: absltest.TestCase) -> Path:
    if not REAL_PRETRAIN_ROOT.exists():
        testcase.skipTest(f"real pretrain dataset is not available: {REAL_PRETRAIN_ROOT}")
    return REAL_PRETRAIN_ROOT


def require_real_split_leaves(testcase: absltest.TestCase, split: str) -> list[Path]:
    split_dir = require_real_root(testcase) / split
    if not split_dir.exists():
        testcase.skipTest(f"real pretrain split is not available: {split_dir}")
    leaves = sorted(
        child
        for child in split_dir.iterdir()
        if child.is_dir() and (child / COMPILED_METADATA_FILENAME).exists()
    )
    if not leaves:
        testcase.skipTest(f"real pretrain split has no data-set leaves: {split_dir}")
    return leaves


def test_temp_dir() -> tempfile.TemporaryDirectory:
    root = Path("/fast/home/salan.isaqzoi/.cache/omegalax_pretrain_tests")
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root, ignore_cleanup_errors=True)


test_temp_dir.__test__ = False


def write_real_binary_mini_root_dataset(
    testcase: absltest.TestCase,
    out_dir: str | Path,
    *,
    splits: Sequence[str] = ("train",),
    copy_from_split: str | None = None,
    num_records_per_leaf: int = 1,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_name = f"part-00000{ARRAY_RECORD_SUFFIX}"

    for split in splits:
        for input_leaf in require_real_split_leaves(testcase, copy_from_split or split):
            leaf_rel = Path(split) / input_leaf.name
            leaf_out_dir = out_dir / leaf_rel
            leaf_out_dir.mkdir(parents=True, exist_ok=True)

            input_metadata = json.loads((input_leaf / COMPILED_METADATA_FILENAME).read_text())
            input_shard = input_leaf / input_metadata["shard_paths"][0]
            if not input_shard.exists():
                testcase.skipTest(f"real pretrain shard is not available: {input_shard}")

            reader = ArrayRecordReader(str(input_shard))
            writer = ArrayRecordWriter(str(leaf_out_dir / shard_name), "group_size:1")
            try:
                for _ in range(num_records_per_leaf):
                    payload = reader.read()
                    testcase.assertTrue(payload.startswith(DOC_CHAIN_BINARY_MAGIC))
                    writer.write(payload)
            finally:
                writer.close()

            leaf_metadata = dict(input_metadata)
            leaf_metadata.update(
                {
                    "version": DOC_CHAIN_DATASET_VERSION,
                    "dataset_format": DOC_CHAIN_FORMAT,
                    "num_records": num_records_per_leaf,
                    "num_shards": 1,
                    "shard_paths": [shard_name],
                    "split": split,
                }
            )
            (leaf_out_dir / COMPILED_METADATA_FILENAME).write_text(
                json.dumps(leaf_metadata, indent=2) + "\n"
            )

    return out_dir
