"""Tests for pretraining data-set helpers against the real corpus."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest
import numpy as np

from omegalax.data.pretrain_data_set import (
    DOC_CHAIN_BINARY_HEADER,
    DOC_CHAIN_BINARY_MAGIC,
    DOC_CHAIN_FORMAT,
    DEFAULT_EOS_ID,
    MAX_PRETRAIN_POSITIONS,
    DataSetReader,
    build_chunk_arrays,
    calculate_samples_per_process,
    deserialize_data_set_record,
    iter_document_pair_refs,
    load_data_set_metadata,
    make_pretrain_index_record_dataset,
    num_pretrain_positions,
    num_pretrain_records_usable,
    resolve_data_set_buckets,
    resolve_pretrain_dp,
    rewrite_data_set_root_path,
    write_json_arrayrecord_dataset,
)
from tests.pretrain_real_data_test_utils import require_real_root, test_temp_dir


class PretrainDataSetTest(absltest.TestCase):
    def test_binary_data_set_record_deserialize(self):
        token_ids = np.asarray([1, 2, 3], dtype=np.int32)
        header = json.dumps(
            {
                "dataset_format": DOC_CHAIN_FORMAT,
                "doc_id": "doc-bin",
                "doc_token_count": 3,
                "source_dataset": "old-builder",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload = (
            DOC_CHAIN_BINARY_MAGIC
            + DOC_CHAIN_BINARY_HEADER.pack(len(header), token_ids.size)
            + header
            + token_ids.tobytes()
        )

        record = deserialize_data_set_record(payload)

        self.assertEqual(record.doc_id, "doc-bin")
        self.assertEqual(record.doc_token_count, 3)
        self.assertEqual(record.metadata["source_dataset"], "old-builder")
        self.assertNotIn("dataset_format", record.metadata)
        np.testing.assert_array_equal(record.token_ids, token_ids)

    def test_bad_dataset_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported doc-chain format"):
            deserialize_data_set_record(
                {
                    "dataset_format": "not_data_set",
                    "doc_id": "bad",
                    "token_ids": [1, 2, 3],
                    "doc_token_count": 3,
                }
            )

    def test_padding_and_masks_cover_only_real_tokens(self):
        arrays = build_chunk_arrays(
            np.asarray([7, 8, 9], dtype=np.int32),
            segment_length=5,
            pad_id=0,
        )

        np.testing.assert_array_equal(
            arrays["token_ids_T"],
            np.asarray([7, 8, 9, 0, 0], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            arrays["attention_mask_T"],
            np.asarray([1, 1, 1, 0, 0], dtype=np.int32),
        )
        np.testing.assert_array_equal(arrays["loss_mask_T"], arrays["attention_mask_T"])

    def test_pair_retention_drops_short_tail_and_keeps_long_tail(self):
        def pair_ranges(length: int) -> list[tuple[int, int, int]]:
            payload = {
                "doc_id": f"doc-{length}",
                "token_ids": list(range(length)),
                "doc_token_count": length,
            }
            record = deserialize_data_set_record(payload)
            return [
                (pair.start, pair.mid, pair.end)
                for pair in iter_document_pair_refs(record, segment_length=4)
            ]

        self.assertEqual(pair_ranges(4), [])
        self.assertEqual(pair_ranges(5), [(0, 4, 5)])
        self.assertEqual(pair_ranges(8), [(0, 4, 8)])
        self.assertEqual(pair_ranges(9), [(0, 4, 8)])
        self.assertEqual(pair_ranges(12), [(0, 4, 8)])
        self.assertEqual(pair_ranges(13), [(0, 4, 8), (8, 12, 13)])

    def test_default_eos_id_marks_retained_end_when_short_tail_is_dropped(self):
        record = deserialize_data_set_record(
            {
                "doc_id": "doc-eos",
                "token_ids": [1, 2, 3, 4, 5, 6, 7, 8, DEFAULT_EOS_ID],
                "doc_token_count": 9,
            }
        )
        pair = next(iter_document_pair_refs(record, segment_length=4))
        arrays = build_chunk_arrays(
            record.token_ids,
            start=pair.mid,
            end=pair.end,
            segment_length=4,
            eos_token_idx=pair.eos_token_idx,
        )

        self.assertEqual(pair.eos_token_idx, 7)
        np.testing.assert_array_equal(
            arrays["token_ids_T"],
            np.asarray([5, 6, 7, DEFAULT_EOS_ID], dtype=np.int32),
        )

    def test_real_leaf_metadata_accepts_dataset_format_alias(self):
        root = require_real_root(self)
        bucket_path = resolve_data_set_buckets(root, split="val")[-1]
        metadata = load_data_set_metadata(bucket_path)

        self.assertEqual(metadata["dataset_format"], DOC_CHAIN_FORMAT)
        self.assertEqual(metadata["split"], "val")
        self.assertEqual(metadata["bucket"], bucket_path.name.removeprefix("bucket_"))
        self.assertGreater(metadata["num_records"], 0)
        self.assertNotIn("format", metadata)

    def test_real_root_and_split_resolve_to_bucket_leaves(self):
        root = require_real_root(self)

        def expected_bucket_paths(split: str) -> list[Path]:
            return sorted(
                (
                    path.resolve()
                    for path in (root / split).iterdir()
                    if path.is_dir() and path.name.startswith("bucket_")
                ),
                key=lambda path: int(path.name.removeprefix("bucket_").removesuffix("k")),
            )

        train_buckets = resolve_data_set_buckets(root, split="train")
        val_buckets = resolve_data_set_buckets(root, split="val")

        self.assertEqual(train_buckets, expected_bucket_paths("train"))
        self.assertEqual(val_buckets, expected_bucket_paths("val"))
        self.assertGreater(len(train_buckets), 1)
        self.assertGreater(len(val_buckets), 1)

    def test_direct_leaf_path_is_not_a_data_set_root(self):
        root = require_real_root(self)
        bucket_path = resolve_data_set_buckets(root, split="val")[0]

        with self.assertRaisesRegex(ValueError, "split directory"):
            resolve_data_set_buckets(bucket_path)

    def test_source_path_rewrite_is_noop_without_env(self):
        with test_temp_dir() as tmp:
            root = Path(tmp) / "source"
            root.mkdir(parents=True)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("OMEGALAX_PRETRAIN_SOURCE_ROOT", None)
                os.environ.pop("OMEGALAX_PRETRAIN_LOCAL_ROOT", None)

                self.assertEqual(rewrite_data_set_root_path(root), root.resolve())

    def test_source_path_rewrite_replaces_source_root(self):
        with test_temp_dir() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            local_root = root / "local"
            source_path = source_root / "dataset"
            local_path = local_root / "dataset"
            source_path.mkdir(parents=True)
            local_path.mkdir(parents=True)

            self.assertEqual(
                rewrite_data_set_root_path(
                    source_path,
                    source_root=source_root,
                    local_root=local_root,
                ),
                local_path.resolve(),
            )

    def test_source_path_rewrite_requires_both_roots(self):
        with test_temp_dir() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "OMEGALAX_PRETRAIN_SOURCE_ROOT": str(Path(tmp) / "source"),
                },
                clear=False,
            ):
                os.environ.pop("OMEGALAX_PRETRAIN_LOCAL_ROOT", None)

                with self.assertRaisesRegex(ValueError, "must be set together"):
                    rewrite_data_set_root_path(Path(tmp) / "source")

    def test_source_path_rewrite_rejects_paths_outside_source_root(self):
        with test_temp_dir() as tmp:
            root = Path(tmp)
            outside_path = root / "outside"
            outside_path.mkdir(parents=True)
            local_root = root / "local"
            local_root.mkdir()

            with self.assertRaisesRegex(ValueError, "outside"):
                rewrite_data_set_root_path(
                    outside_path,
                    source_root=root / "source",
                    local_root=local_root,
                )

    def test_source_path_rewrite_rejects_missing_local_path(self):
        with test_temp_dir() as tmp:
            root = Path(tmp)
            source_path = root / "source" / "dataset"
            source_path.mkdir(parents=True)
            local_root = root / "local"
            local_root.mkdir()

            with self.assertRaisesRegex(ValueError, "does not exist"):
                rewrite_data_set_root_path(
                    source_path,
                    source_root=root / "source",
                    local_root=local_root,
                )

    def test_real_reader_loads_from_root_and_split(self):
        root = require_real_root(self)
        expected_bucket_names = [path.name for path in resolve_data_set_buckets(root, split="val")]
        reader = DataSetReader(root, split="val")
        record = reader.read(0, 0)

        self.assertEqual(reader.split, "val")
        self.assertEqual(reader.bucket_names, expected_bucket_names)
        self.assertGreater(record.doc_token_count, 0)
        self.assertEqual(record.metadata["split"], "val")
        self.assertEqual(
            record.metadata["bucket"], expected_bucket_names[0].removeprefix("bucket_")
        )

    def test_resolve_pretrain_dp_matches_grain_convention(self):
        self.assertEqual(
            resolve_pretrain_dp(dp_size=2, fsdp_size=3, process_index=7),
            (6, 1),
        )

    def test_pretrain_dp_assignment_counts_records(self):
        self.assertEqual(
            calculate_samples_per_process(num_records=10, dp_size=3, dp_index=0),
            4,
        )
        self.assertEqual(
            calculate_samples_per_process(num_records=10, dp_size=3, dp_index=1),
            3,
        )
        self.assertEqual(
            calculate_samples_per_process(num_records=2, dp_size=4, dp_index=3),
            0,
        )

    def test_pretrain_dp_usable_records_align_to_global_steps(self):
        self.assertEqual(
            num_pretrain_records_usable(
                num_records=10,
                dp_size=3,
                records_per_local_batch=2,
            ),
            6,
        )
        self.assertEqual(
            num_pretrain_records_usable(
                num_records=12,
                dp_size=3,
                records_per_local_batch=2,
            ),
            12,
        )

    def test_pretrain_position_count_supports_finite_and_infinite_epochs(self):
        self.assertEqual(
            num_pretrain_positions(epoch_samples_per_process=3, num_epochs=2),
            6,
        )
        self.assertEqual(
            num_pretrain_positions(epoch_samples_per_process=3, num_epochs=None),
            MAX_PRETRAIN_POSITIONS,
        )

    def test_pretrain_index_record_dataset_maps_positions_to_records(self):
        with test_temp_dir() as tmp:
            index = write_json_arrayrecord_dataset(
                ({"record": i} for i in range(5)),
                Path(tmp) / "index",
                records_per_shard=2,
                overwrite=False,
                metadata={"format": "test_index"},
            )
            dataset, total_samples_per_process = make_pretrain_index_record_dataset(
                index_shard_paths=sorted(index.glob("*.array_record")),
                num_records=5,
                num_epochs=2,
                dp_size=2,
                dp_index=1,
                shuffle=False,
                seed=0,
                shuffle_rounds=4,
            )

            self.assertEqual(total_samples_per_process, 2)
            self.assertLen(dataset, 4)
            self.assertEqual([dataset[i]["record"] for i in range(4)], [1, 3, 1, 3])


if __name__ == "__main__":
    absltest.main()
