"""Tests for IID pretraining indexing against the real corpus."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest
from array_record.python.array_record_module import ArrayRecordReader, ArrayRecordWriter
import numpy as np

from omegalax.data.pretrain_data_set import (
    DOC_CHAIN_BINARY_HEADER,
    DOC_CHAIN_BINARY_MAGIC,
    DEFAULT_EOS_ID,
    iter_json_arrayrecord_records,
    load_arrayrecord_metadata,
    resolve_data_set_buckets,
)
from omegalax.data.pretrain_iid_pipeline import build_iid_chunk_index, make_iid_iterator
from tests.pretrain_real_data_test_utils import (
    test_temp_dir,
    write_real_binary_mini_root_dataset,
)


class PretrainIidTest(absltest.TestCase):
    def _bucket_names(self, root: Path, split: str) -> list[str]:
        return [path.name for path in resolve_data_set_buckets(root, split=split)]

    def _build_real_index(self, tmpdir: Path, *, segment_length: int = 4096) -> tuple[Path, Path]:
        root = write_real_binary_mini_root_dataset(
            self,
            tmpdir / "docs",
            splits=("train",),
            copy_from_split="val",
        )
        index = build_iid_chunk_index(
            root,
            tmpdir / "index",
            segment_length=segment_length,
            split="train",
            records_per_shard=1000,
        )
        return index, root

    def _read_index_records(self, index_path: Path) -> list[dict]:
        return [record for _, record in iter_json_arrayrecord_records(index_path)]

    def _signature(self, batch: dict) -> tuple[list[str], list[int], list[list[int]]]:
        return (
            list(batch["metadata"]["doc_ids"]),
            batch["chunk_idx_B"].tolist(),
            batch["token_ids_BT"][:, :4].tolist(),
        )

    def _chunk_keys(self, iterator) -> set[tuple[int, int, int]]:
        keys, _ = self._chunk_keys_and_batches(iterator)
        return keys

    def _chunk_keys_and_batches(self, iterator) -> tuple[set[tuple[int, int, int]], int]:
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
                    batch["chunk_idx_B"].tolist(),
                    strict=True,
                )
            )

    def test_real_iid_index_points_to_bucket_ranges(self):
        with test_temp_dir() as tmp:
            index, root = self._build_real_index(Path(tmp))

            metadata = load_arrayrecord_metadata(index)
            records = self._read_index_records(index)
            bucket_names = self._bucket_names(root, "train")

            self.assertEqual(metadata["segment_length"], 4096)
            self.assertEqual(metadata["eos_id"], DEFAULT_EOS_ID)
            self.assertEqual(metadata["data_set_root"], str(root.resolve()))
            self.assertEqual(metadata["split"], "train")
            self.assertEqual(metadata["bucket_names"], bucket_names)
            self.assertNotIn("bucket_paths", metadata)
            self.assertGreater(len(metadata["bucket_names"]), 1)
            self.assertEqual(metadata["bucket_record_counts"], [1 for _ in bucket_names])
            self.assertGreater(metadata["num_pairs"], 0)
            self.assertGreater(metadata["num_chunks"], 0)
            self.assertEqual(metadata["num_records"], len(records))
            self.assertEqual(
                {record["bucket_idx"] for record in records},
                set(range(len(metadata["bucket_names"]))),
            )
            first = records[0]
            self.assertEqual(first["bucket_idx"], 0)
            self.assertEqual(first["record_idx"], 0)
            self.assertEqual(first["segment_in_pair"], 0)
            self.assertEqual(first["chunk_idx"], 0)
            self.assertEqual(first["start"], 0)
            self.assertEqual(first["end"], 4096)

    def test_real_iid_iterator_batch_shape_and_metadata(self):
        with test_temp_dir() as tmp:
            index, _ = self._build_real_index(Path(tmp))
            batch = next(
                make_iid_iterator(
                    index,
                    batch_size=2,
                    segment_length=4096,
                    shuffle=False,
                    num_epochs=1,
                    process_index=0,
                    grain_workers=0,
                )
            )

            self.assertEqual(batch["token_ids_BT"].shape, (2, 4096))
            self.assertEqual(batch["attention_mask_BT"].shape, (2, 4096))
            self.assertEqual(batch["loss_mask_BT"].shape, (2, 4096))
            self.assertLen(batch["metadata"]["doc_ids"], 2)
            np.testing.assert_array_equal(
                batch["metadata"]["bucket_idx_B"],
                np.asarray([0, 0], dtype=np.int32),
            )
            np.testing.assert_array_equal(
                batch["metadata"]["segment_in_pair_B"],
                np.asarray([0, 1], dtype=np.int32),
            )

    def test_iid_iterator_rewrites_data_set_root_to_local_root(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            index, root = self._build_real_index(tmpdir)
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
                iterator = make_iid_iterator(
                    index,
                    batch_size=2,
                    segment_length=4096,
                    shuffle=False,
                    num_epochs=1,
                    process_index=0,
                    grain_workers=0,
                )

            batch = next(iterator)
            self.assertEqual(batch["token_ids_BT"].shape, (2, 4096))
            self.assertLen(batch["metadata"]["doc_ids"], 2)

    def test_iid_iterator_rejects_data_set_record_doc_id_mismatch(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            index, root = self._build_real_index(tmpdir)
            local_root = tmpdir / "local_docs"
            shutil.copytree(root, local_root)
            metadata = load_arrayrecord_metadata(index)
            local_leaf = local_root.resolve() / metadata["split"] / metadata["bucket_names"][0]
            shard = next(local_leaf.rglob("*.array_record"))
            reader = ArrayRecordReader(str(shard))
            payload = reader.read()
            pos = len(DOC_CHAIN_BINARY_MAGIC)
            header_len, token_count = DOC_CHAIN_BINARY_HEADER.unpack(
                payload[pos : pos + DOC_CHAIN_BINARY_HEADER.size]
            )
            pos += DOC_CHAIN_BINARY_HEADER.size
            header = json.loads(payload[pos : pos + header_len].decode("utf-8"))
            pos += header_len
            header["doc_id"] = f"stale-{header['doc_id']}"
            header_bytes = json.dumps(
                header,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            rewritten_payload = (
                DOC_CHAIN_BINARY_MAGIC
                + DOC_CHAIN_BINARY_HEADER.pack(len(header_bytes), token_count)
                + header_bytes
                + payload[pos:]
            )
            tmp_shard = shard.with_name(f"{shard.name}.tmp")
            writer = ArrayRecordWriter(str(tmp_shard), "group_size:1")
            try:
                writer.write(rewritten_payload)
            finally:
                writer.close()
            tmp_shard.replace(shard)

            with mock.patch.dict(
                os.environ,
                {
                    "OMEGALAX_PRETRAIN_SOURCE_ROOT": str(root.resolve()),
                    "OMEGALAX_PRETRAIN_LOCAL_ROOT": str(local_root.resolve()),
                },
                clear=False,
            ):
                iterator = make_iid_iterator(
                    index,
                    batch_size=2,
                    segment_length=4096,
                    shuffle=False,
                    num_epochs=1,
                    process_index=0,
                    grain_workers=0,
                )

                with self.assertRaisesRegex(ValueError, "IID chunk index does not match"):
                    next(iterator)

    def test_iid_iterator_rejects_bucket_name_mismatch(self):
        with test_temp_dir() as tmp:
            index, _ = self._build_real_index(Path(tmp))
            metadata_path = index / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["bucket_names"] = list(reversed(metadata["bucket_names"]))
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

            iterator = make_iid_iterator(
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

    def test_real_binary_iid_val_index_uses_validation_split(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            root = write_real_binary_mini_root_dataset(
                self,
                tmpdir / "binary_docs",
                splits=("val",),
            )
            index = build_iid_chunk_index(
                root,
                tmpdir / "binary_index",
                segment_length=4096,
                split="val",
                records_per_shard=1000,
            )
            batch = next(
                make_iid_iterator(
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
            self.assertEqual(metadata["data_set_root"], str(root.resolve()))
            self.assertEqual(metadata["split"], "val")
            self.assertEqual(metadata["eos_id"], DEFAULT_EOS_ID)
            self.assertEqual(metadata["bucket_names"], self._bucket_names(root, "val"))
            self.assertGreater(len(metadata["bucket_names"]), 1)
            self.assertGreater(metadata["num_chunks"], 0)
            self.assertEqual(batch["token_ids_BT"].shape, (2, 4096))
            self.assertGreater(int(np.asarray(batch["attention_mask_BT"]).sum()), 0)
            self.assertLen(batch["metadata"]["doc_ids"], 2)

    def test_real_iid_shuffle_is_deterministic(self):
        with test_temp_dir() as tmp:
            index, _ = self._build_real_index(Path(tmp), segment_length=4096)
            it0 = make_iid_iterator(
                index,
                batch_size=1,
                segment_length=4096,
                shuffle=True,
                seed=0,
                num_epochs=1,
                process_index=0,
                grain_workers=0,
            )
            it1 = make_iid_iterator(
                index,
                batch_size=1,
                segment_length=4096,
                shuffle=True,
                seed=0,
                num_epochs=1,
                process_index=0,
                grain_workers=0,
            )

            order0 = [int(next(it0)["chunk_idx_B"][0]) for _ in range(4)]
            order1 = [int(next(it1)["chunk_idx_B"][0]) for _ in range(4)]
            self.assertEqual(order0, order1)

    def test_real_iid_infinite_epochs_are_deterministic(self):
        with test_temp_dir() as tmp:
            index, _ = self._build_real_index(Path(tmp), segment_length=4096)
            it0 = make_iid_iterator(
                index,
                batch_size=2,
                segment_length=4096,
                shuffle=True,
                seed=11,
                num_epochs=None,
                process_index=0,
                grain_workers=0,
            )
            it1 = make_iid_iterator(
                index,
                batch_size=2,
                segment_length=4096,
                shuffle=True,
                seed=11,
                num_epochs=None,
                process_index=0,
                grain_workers=0,
            )

            replay0 = [self._signature(next(it0)) for _ in range(4)]
            replay1 = [self._signature(next(it1)) for _ in range(4)]

            self.assertEqual(replay0, replay1)

    def test_iid_index_shuffle_covers_usable_dp_shards_without_overlap(self):
        with test_temp_dir() as tmp:
            index, _ = self._build_real_index(Path(tmp), segment_length=4096)
            records = self._read_index_records(index)
            expected = {
                (record["bucket_idx"], record["record_idx"], record["chunk_idx"])
                for record in records
            }
            batch_size = 2
            dp_size = 3
            usable_records = (len(records) // (dp_size * batch_size)) * dp_size * batch_size

            rank_keys = []
            rank_batches = []
            for dp_index in range(dp_size):
                keys, num_batches = self._chunk_keys_and_batches(
                    make_iid_iterator(
                        index,
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

    def test_iid_index_shuffle_covers_all_records_for_single_rank(self):
        with test_temp_dir() as tmp:
            index, _ = self._build_real_index(Path(tmp), segment_length=4096)
            records = self._read_index_records(index)
            expected = {
                (record["bucket_idx"], record["record_idx"], record["chunk_idx"])
                for record in records
            }

            rank0 = self._chunk_keys(
                make_iid_iterator(
                    index,
                    batch_size=1,
                    segment_length=4096,
                    shuffle=True,
                    seed=17,
                    num_epochs=1,
                    dp_size=1,
                    dp_index=0,
                    process_index=0,
                    grain_workers=0,
                )
            )

            self.assertEqual(rank0, expected)


if __name__ == "__main__":
    absltest.main()
