"""Tests for pair-sampled statepassing iteration against the real corpus."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest
import numpy as np

from omegalax.data.pretrain_doc_chain import (
    iter_json_arrayrecord_records,
    load_arrayrecord_metadata,
    pop_pretrain_metadata,
    resolve_doc_chain_buckets,
)
from omegalax.data.pretrain_statepassing import (
    STATEPASSING_PAIR_INDEX_FORMAT,
    build_statepassing_pair_index,
    make_statepassing_iterator,
)
from tests.pretrain_real_data_test_utils import (
    test_temp_dir,
    write_real_binary_mini_root_dataset,
)


class PretrainStatepassingTest(absltest.TestCase):
    def _root(self) -> Path:
        root = getattr(self, "_real_subset_root", None)
        if root is None:
            tmp = self.enter_context(test_temp_dir())
            root = write_real_binary_mini_root_dataset(
                self,
                Path(tmp) / "docs",
                splits=("train",),
                copy_from_split="val",
            )
            self._real_subset_root = root
        return root

    def _bucket_paths(self, split: str = "train") -> list[Path]:
        return resolve_doc_chain_buckets(self._root(), split=split)

    def _index(self) -> Path:
        index = getattr(self, "_statepassing_pair_index", None)
        if index is None:
            tmp = self.enter_context(test_temp_dir())
            index = build_statepassing_pair_index(
                self._root(),
                Path(tmp) / "statepassing_pair_index",
                segment_length=4096,
                split="train",
                records_per_shard=1000,
            )
            self._statepassing_pair_index = index
        return index

    def _read_index_records(self, index_path: Path) -> list[dict]:
        return [record for _, record in iter_json_arrayrecord_records(index_path)]

    def _iterator(self, **kwargs):
        return make_statepassing_iterator(
            self._index(),
            batch_size=kwargs.pop("batch_size", 2),
            segment_length=kwargs.pop("segment_length", 4096),
            shuffle=kwargs.pop("shuffle", False),
            num_epochs=kwargs.pop("num_epochs", 1),
            process_index=kwargs.pop("process_index", 0),
            grain_workers=kwargs.pop("grain_workers", 0),
            **kwargs,
        )

    def _signature(self, batch: dict) -> tuple[list[str], list[int], list[list[int]]]:
        return (
            list(batch["metadata"]["doc_ids"]),
            batch["metadata"]["pair_idx_B"].tolist(),
            batch["chunk_idx_BS"].tolist(),
        )

    def _pair_keys(self, iterator) -> set[tuple[int, int, int]]:
        keys, _ = self._pair_keys_and_batches(iterator)
        return keys

    def _pair_keys_and_batches(self, iterator) -> tuple[set[tuple[int, int, int]], int]:
        keys = set()
        num_batches = 0
        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                return keys, num_batches
            num_batches += 1
            metadata = batch["metadata"]
            keys.update(
                zip(
                    metadata["bucket_idx_B"].tolist(),
                    metadata["record_idx_B"].tolist(),
                    metadata["pair_idx_B"].tolist(),
                    strict=True,
                )
            )

    def test_real_pair_index_points_to_bucket_ranges(self):
        index = self._index()

        metadata = load_arrayrecord_metadata(index)
        records = self._read_index_records(index)
        bucket_names = [path.name for path in self._bucket_paths()]

        self.assertEqual(metadata["format"], STATEPASSING_PAIR_INDEX_FORMAT)
        self.assertEqual(metadata["doc_chain_root"], str(self._root().resolve()))
        self.assertEqual(metadata["split"], "train")
        self.assertEqual(metadata["segment_length"], 4096)
        self.assertEqual(metadata["bucket_names"], bucket_names)
        self.assertNotIn("bucket_paths", metadata)
        self.assertGreater(len(metadata["bucket_names"]), 1)
        self.assertEqual(metadata["bucket_record_counts"], [1 for _ in bucket_names])
        self.assertGreater(metadata["num_pairs"], 0)
        self.assertEqual(metadata["num_records"], len(records))
        self.assertEqual(
            {record["bucket_idx"] for record in records},
            set(range(len(metadata["bucket_names"]))),
        )
        first = records[0]
        self.assertEqual(first["bucket_idx"], 0)
        self.assertEqual(first["record_idx"], 0)
        self.assertEqual(first["pair_idx"], 0)
        self.assertEqual(first["start"], 0)
        self.assertEqual(first["mid"], 4096)
        self.assertGreater(first["end"], first["mid"])

    def test_real_batch_is_pair_by_two_segments_by_t(self):
        batch = next(self._iterator())

        self.assertNotIn("token_ids_BT", batch)
        self.assertEqual(batch["token_ids_BST"].shape, (1, 2, 4096))
        self.assertEqual(batch["attention_mask_BST"].shape, (1, 2, 4096))
        self.assertEqual(batch["loss_mask_BST"].shape, (1, 2, 4096))
        self.assertEqual(batch["chunk_idx_BS"].shape, (1, 2))
        self.assertEqual(batch["reset_state_BS"].tolist(), [[True, False]])
        self.assertLen(batch["metadata"]["doc_ids"], 1)

    def test_real_binary_statepassing_index_and_iterator_match_training_path(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            root = write_real_binary_mini_root_dataset(
                self,
                tmpdir / "binary_docs",
                splits=("val",),
            )
            index = build_statepassing_pair_index(
                root,
                tmpdir / "binary_statepassing_pair_index",
                segment_length=4096,
                split="val",
                records_per_shard=1000,
            )
            batch = next(
                make_statepassing_iterator(
                    index,
                    batch_size=2,
                    segment_length=4096,
                    shuffle=False,
                    num_epochs=1,
                    process_index=0,
                    grain_workers=0,
                )
            )

            metadata = load_arrayrecord_metadata(index)
            self.assertEqual(metadata["format"], STATEPASSING_PAIR_INDEX_FORMAT)
            self.assertEqual(metadata["doc_chain_root"], str(root.resolve()))
            self.assertEqual(metadata["split"], "val")
            self.assertEqual(
                metadata["bucket_names"],
                [path.name for path in resolve_doc_chain_buckets(root, split="val")],
            )
            self.assertGreater(len(metadata["bucket_names"]), 1)
            self.assertGreater(metadata["num_pairs"], 0)
            self.assertEqual(batch["token_ids_BST"].shape, (1, 2, 4096))
            self.assertEqual(batch["reset_state_BS"].tolist(), [[True, False]])
            self.assertGreater(int(np.asarray(batch["attention_mask_BST"]).sum()), 0)

    def test_single_item_index_sequence_uses_indexed_iterator(self):
        batch = next(
            make_statepassing_iterator(
                [self._index()],
                batch_size=2,
                segment_length=4096,
                shuffle=False,
                num_epochs=1,
                process_index=0,
                grain_workers=0,
            )
        )

        self.assertEqual(batch["token_ids_BST"].shape, (1, 2, 4096))

    def test_indexed_iterator_rewrites_doc_chain_root_to_local_root(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            root = write_real_binary_mini_root_dataset(
                self,
                tmpdir / "docs",
                splits=("train",),
                copy_from_split="val",
            )
            index = build_statepassing_pair_index(
                root,
                tmpdir / "statepassing_pair_index",
                segment_length=4096,
                split="train",
                records_per_shard=1000,
            )
            local_root = tmpdir / "local_docs"
            shutil.copytree(root, local_root)
            shutil.rmtree(root)

            with mock.patch.dict(
                os.environ,
                {
                    "OMEGALAX_PRETRAIN_SOURCE_ROOT": str(root.resolve()),
                    "OMEGALAX_PRETRAIN_LOCAL_ROOT": str(local_root.resolve()),
                },
                clear=False,
            ):
                iterator = make_statepassing_iterator(
                    index,
                    batch_size=2,
                    segment_length=4096,
                    shuffle=False,
                    num_epochs=1,
                    process_index=0,
                    grain_workers=0,
                )

            batch = next(iterator)
            self.assertEqual(batch["token_ids_BST"].shape, (1, 2, 4096))
            self.assertLen(batch["metadata"]["doc_ids"], 1)

    def test_indexed_iterator_rejects_bucket_name_mismatch(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            root = write_real_binary_mini_root_dataset(
                self,
                tmpdir / "docs",
                splits=("train",),
                copy_from_split="val",
            )
            index = build_statepassing_pair_index(
                root,
                tmpdir / "statepassing_pair_index",
                segment_length=4096,
                split="train",
                records_per_shard=1000,
            )
            metadata_path = index / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["bucket_names"] = list(reversed(metadata["bucket_names"]))
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

            iterator = make_statepassing_iterator(
                index,
                batch_size=2,
                segment_length=4096,
                shuffle=False,
                num_epochs=1,
                process_index=0,
                grain_workers=0,
            )

            with self.assertRaisesRegex(ValueError, "bucket_names"):
                next(iterator)

    def test_index_shuffle_covers_usable_dp_shards_without_overlap(self):
        records = self._read_index_records(self._index())
        expected = {
            (record["bucket_idx"], record["record_idx"], record["pair_idx"]) for record in records
        }
        batch_size = 2
        dp_size = 3
        records_per_local_batch = batch_size // 2
        usable_records = (
            (len(records) // (dp_size * records_per_local_batch))
            * dp_size
            * records_per_local_batch
        )

        rank_keys = []
        rank_batches = []
        for dp_index in range(dp_size):
            keys, num_batches = self._pair_keys_and_batches(
                make_statepassing_iterator(
                    self._index(),
                    batch_size=batch_size,
                    segment_length=4096,
                    shuffle=True,
                    seed=17,
                    num_epochs=1,
                    dp_size=dp_size,
                    dp_index=dp_index,
                    process_index=0,
                    grain_workers=0,
                )
            )
            rank_keys.append(keys)
            rank_batches.append(num_batches)

        self.assertTrue(all(num_batches == rank_batches[0] for num_batches in rank_batches))
        union = set().union(*rank_keys)
        for i, keys in enumerate(rank_keys):
            self.assertTrue(keys <= expected)
            for other_keys in rank_keys[i + 1 :]:
                self.assertEmpty(keys & other_keys)
        self.assertLen(union, usable_records)

    def test_real_batch_size_counts_segments_but_batches_pairs(self):
        batch = next(self._iterator(batch_size=4))

        self.assertEqual(batch["token_ids_BST"].shape, (2, 2, 4096))
        self.assertEqual(batch["chunk_idx_BS"].shape, (2, 2))
        self.assertLen(batch["metadata"]["doc_ids"], 2)
        self.assertTrue(all(pair_idx >= 0 for pair_idx in batch["metadata"]["pair_idx_B"]))

    def test_real_metadata_can_be_popped_before_sharding(self):
        batch = next(self._iterator())
        metadata = pop_pretrain_metadata(batch)

        self.assertLen(metadata["doc_ids"], 1)
        self.assertNotIn("metadata", batch)
        for value in batch.values():
            self.assertTrue(hasattr(value, "ndim"))

    def test_odd_segment_batch_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be even"):
            self._iterator(batch_size=3)

    def test_directory_inputs_require_statepassing_pair_index(self):
        with self.assertRaisesRegex(ValueError, "requires a statepassing pair index"):
            make_statepassing_iterator(
                self._root(),
                batch_size=2,
                segment_length=4096,
                shuffle=False,
                num_epochs=1,
                process_index=0,
                grain_workers=0,
            )

    def test_file_inputs_require_statepassing_pair_index(self):
        with self.assertRaisesRegex(ValueError, "requires a statepassing pair index"):
            make_statepassing_iterator(
                self._bucket_paths()[0] / "part-00000.array_record",
                batch_size=2,
                segment_length=4096,
                shuffle=False,
                num_epochs=1,
                process_index=0,
                grain_workers=0,
            )

    def test_real_infinite_epochs_are_deterministic(self):
        it0 = self._iterator(shuffle=True, seed=5, num_epochs=None)
        it1 = self._iterator(shuffle=True, seed=5, num_epochs=None)

        replay0 = [self._signature(next(it0)) for _ in range(5)]
        replay1 = [self._signature(next(it1)) for _ in range(5)]

        self.assertEqual(replay0, replay1)

    def test_real_attention_mask_has_tokens(self):
        batch = next(self._iterator())

        self.assertGreater(int(np.asarray(batch["attention_mask_BST"]).sum()), 0)
        np.testing.assert_array_equal(batch["loss_mask_BST"], batch["attention_mask_BST"])


if __name__ == "__main__":
    absltest.main()
