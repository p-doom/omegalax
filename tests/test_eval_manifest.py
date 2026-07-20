"""Tests for deterministic full-document eval manifests."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from absl.testing import absltest
from array_record.python.array_record_module import ArrayRecordWriter
import numpy as np

from omegalax.data.pretrain_data_set import (
    DOC_CHAIN_BINARY_HEADER,
    DOC_CHAIN_BINARY_MAGIC,
    DOC_CHAIN_FORMAT,
)
from omegalax.evals.manifest import (
    FullDocumentLoader,
    build_full_document_manifest,
    load_full_document_manifest,
    validate_manifest_resume_compatibility,
)
from tests.pretrain_real_data_test_utils import test_temp_dir


class EvalManifestTest(absltest.TestCase):
    def _write_val_root(
        self,
        root: Path,
        *,
        doc_id_prefix: str = "",
        token_offset: int = 0,
        split: str = "val",
    ) -> dict[str, np.ndarray]:
        tokens_by_doc_id: dict[str, np.ndarray] = {}
        bucket_lengths = {
            "bucket_2k": range(1, 5),
            "bucket_8k": range(5, 9),
            "bucket_16k": range(9, 13),
            "bucket_24k": range(13, 17),
        }
        for bucket_name, doc_lengths in bucket_lengths.items():
            bucket_path = root / split / bucket_name
            bucket_path.mkdir(parents=True)
            payloads = []
            for doc_num_chunks in doc_lengths:
                num_docs = 1 if doc_num_chunks == 1 else 6
                for doc_idx in range(num_docs):
                    doc_token_count = (doc_num_chunks - 1) * 4 + 1 + doc_idx % 3
                    doc_id = f"{doc_id_prefix}l{doc_num_chunks}-doc-{doc_idx}"
                    token_ids = np.arange(
                        token_offset + doc_num_chunks * 1000 + doc_idx * 100,
                        token_offset + doc_num_chunks * 1000 + doc_idx * 100 + doc_token_count,
                        dtype=np.int32,
                    )
                    tokens_by_doc_id[doc_id] = token_ids
                    header = json.dumps(
                        {
                            "dataset_format": DOC_CHAIN_FORMAT,
                            "doc_id": doc_id,
                            "doc_token_count": doc_token_count,
                            "token_dtype": "int32",
                            "split": split,
                            "bucket": bucket_name.removeprefix("bucket_"),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    payloads.append(
                        DOC_CHAIN_BINARY_MAGIC
                        + DOC_CHAIN_BINARY_HEADER.pack(len(header), token_ids.size)
                        + header
                        + token_ids.tobytes()
                    )

            split_idx = len(payloads) // 2
            shard_paths = []
            for shard_idx, shard_payloads in enumerate(
                (payloads[:split_idx], payloads[split_idx:])
            ):
                shard_name = f"part-{shard_idx:05d}.array_record"
                shard_paths.append(shard_name)
                writer = ArrayRecordWriter(str(bucket_path / shard_name), "group_size:1")
                try:
                    for payload in shard_payloads:
                        writer.write(payload)
                finally:
                    writer.close()

            (bucket_path / "metadata.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "dataset_format": DOC_CHAIN_FORMAT,
                        "split": split,
                        "bucket": bucket_name.removeprefix("bucket_"),
                        "segment_length_default": 4,
                        "num_records": len(payloads),
                        "num_shards": len(shard_paths),
                        "shard_paths": shard_paths,
                    },
                    indent=2,
                )
                + "\n"
            )
        return tokens_by_doc_id

    def _build(
        self,
        root: Path,
        output_path: Path,
        *,
        seed: int,
        sample_cap: int,
    ):
        return build_full_document_manifest(
            root,
            output_path,
            split="val",
            chunk_length=4,
            seed=seed,
            sample_cap=sample_cap,
            min_doc_chunks=2,
            max_doc_chunks=16,
        )

    def test_manifest_selection_ranking_and_donors_are_deterministic(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            root = tmpdir / "raw"
            self._write_val_root(root)

            same_a = self._build(root, tmpdir / "same-a.json", seed=0, sample_cap=2)
            same_b = self._build(root, tmpdir / "same-b.json", seed=0, sample_cap=2)
            other_seed_small = self._build(
                root,
                tmpdir / "other-small.json",
                seed=1,
                sample_cap=2,
            )
            other_seed_all = self._build(
                root,
                tmpdir / "other-all.json",
                seed=1,
                sample_cap=10,
            )
            extended = self._build(root, tmpdir / "extended.json", seed=0, sample_cap=4)
            odd = self._build(root, tmpdir / "odd.json", seed=0, sample_cap=5)
            all_available = self._build(root, tmpdir / "all.json", seed=0, sample_cap=10)

            changed_root = tmpdir / "changed-raw"
            self._write_val_root(changed_root, doc_id_prefix="changed-")
            changed_dataset = self._build(
                changed_root,
                tmpdir / "changed.json",
                seed=0,
                sample_cap=2,
            )
            changed_tokens_root = tmpdir / "changed-tokens-raw"
            self._write_val_root(changed_tokens_root, token_offset=100_000)
            changed_tokens = self._build(
                changed_tokens_root,
                tmpdir / "changed-tokens.json",
                seed=0,
                sample_cap=2,
            )
            copied_root = tmpdir / "copied-raw"
            self._write_val_root(copied_root)
            copied_dataset = self._build(
                copied_root,
                tmpdir / "copied.json",
                seed=0,
                sample_cap=2,
            )

            self.assertEqual(
                (tmpdir / "same-a.json").read_bytes(),
                (tmpdir / "same-b.json").read_bytes(),
            )
            self.assertEqual(same_a, same_b)
            self.assertEqual(same_a.dataset_hash, other_seed_all.dataset_hash)
            self.assertEqual(same_a.dataset_hash, extended.dataset_hash)
            self.assertEqual(same_a.dataset_hash, copied_dataset.dataset_hash)
            self.assertNotEqual(same_a.dataset_hash, changed_dataset.dataset_hash)
            self.assertNotEqual(same_a.dataset_hash, changed_tokens.dataset_hash)
            self.assertNotEqual(same_a.manifest_hash, other_seed_small.manifest_hash)
            self.assertNotEqual(same_a.manifest_hash, extended.manifest_hash)
            self.assertRegex(same_a.dataset_hash, r"^sha256:[0-9a-f]{64}$")
            self.assertRegex(same_a.manifest_hash, r"^sha256:[0-9a-f]{64}$")
            self.assertNotIn(1, {doc.doc_num_chunks for doc in extended.documents})
            for manifest, selected in (
                (same_a, 2),
                (extended, 4),
                (odd, 5),
                (all_available, 6),
            ):
                self.assertEqual(
                    [
                        (count.doc_num_chunks, count.available, count.selected)
                        for count in manifest.counts_by_length
                    ],
                    [(doc_num_chunks, 6, selected) for doc_num_chunks in range(2, 17)],
                )

            capped_seed_changes = 0
            for doc_num_chunks in range(2, 17):
                small_docs = [
                    doc for doc in same_a.documents if doc.doc_num_chunks == doc_num_chunks
                ]
                large_docs = [
                    doc for doc in extended.documents if doc.doc_num_chunks == doc_num_chunks
                ]
                odd_docs = [doc for doc in odd.documents if doc.doc_num_chunks == doc_num_chunks]
                other_small_docs = [
                    doc
                    for doc in other_seed_small.documents
                    if doc.doc_num_chunks == doc_num_chunks
                ]
                other_all_docs = [
                    doc for doc in other_seed_all.documents if doc.doc_num_chunks == doc_num_chunks
                ]
                all_docs = [
                    doc for doc in all_available.documents if doc.doc_num_chunks == doc_num_chunks
                ]

                self.assertEqual(
                    [doc.doc_id for doc in small_docs],
                    [doc.doc_id for doc in large_docs[:2]],
                )
                self.assertEqual(
                    [doc.doc_id for doc in large_docs],
                    [doc.doc_id for doc in all_docs[:4]],
                )
                self.assertEqual(
                    [doc.doc_id for doc in other_small_docs],
                    [doc.doc_id for doc in other_all_docs[:2]],
                )
                self.assertEqual([doc.sample_rank for doc in large_docs], [0, 1, 2, 3])
                self.assertLen(small_docs, 2)
                self.assertLen(large_docs, 4)
                self.assertLen(odd_docs, 5)
                self.assertLen(other_small_docs, 2)
                self.assertLen(other_all_docs, 6)
                self.assertLen(all_docs, 6)
                self.assertEqual(
                    {doc.doc_id for doc in all_docs},
                    {doc.doc_id for doc in other_all_docs},
                )
                self.assertNotEqual(
                    [doc.doc_id for doc in all_docs],
                    [doc.doc_id for doc in other_all_docs],
                )
                if [doc.doc_id for doc in small_docs] != [doc.doc_id for doc in other_small_docs]:
                    capped_seed_changes += 1
                self.assertEqual(
                    [doc.donor_doc_id for doc in large_docs],
                    [
                        large_docs[1].doc_id,
                        large_docs[0].doc_id,
                        large_docs[3].doc_id,
                        large_docs[2].doc_id,
                    ],
                )
                self.assertEqual(odd_docs[0].donor_doc_id, odd_docs[1].doc_id)
                self.assertEqual(odd_docs[1].donor_doc_id, odd_docs[0].doc_id)
                self.assertEqual(
                    {doc.donor_doc_id for doc in odd_docs[-3:]},
                    {doc.doc_id for doc in odd_docs[-3:]},
                )
                self.assertTrue(all(doc.doc_id != doc.donor_doc_id for doc in odd_docs))

            self.assertGreater(capped_seed_changes, 0)

            validate_manifest_resume_compatibility(same_a, extended)
            validate_manifest_resume_compatibility(same_a, odd)
            validate_manifest_resume_compatibility(same_a, all_available)
            validate_manifest_resume_compatibility(extended, all_available)
            validate_manifest_resume_compatibility(same_a, copied_dataset)
            with self.assertRaisesRegex(ValueError, "donor"):
                validate_manifest_resume_compatibility(extended, odd)
            with self.assertRaisesRegex(ValueError, "donor"):
                validate_manifest_resume_compatibility(odd, all_available)
            for incompatible_dataset in (changed_dataset, changed_tokens):
                with self.assertRaisesRegex(ValueError, "dataset_hash"):
                    validate_manifest_resume_compatibility(
                        same_a,
                        incompatible_dataset,
                    )
            with self.assertRaisesRegex(ValueError, "seed"):
                validate_manifest_resume_compatibility(same_a, other_seed_small)
            for field, value, error in (
                ("split", "train", "split"),
                ("chunk_length", 8, "chunk_length"),
                ("min_doc_chunks", 3, "length"),
                ("max_doc_chunks", 15, "length"),
            ):
                with self.subTest(incompatible_manifest_field=field):
                    with self.assertRaisesRegex(ValueError, error):
                        validate_manifest_resume_compatibility(
                            same_a,
                            dataclasses.replace(same_a, **{field: value}),
                        )

            prefix_documents = list(extended.documents)
            first, second = prefix_documents[:2]
            prefix_documents[:2] = [
                dataclasses.replace(second, sample_rank=0),
                dataclasses.replace(first, sample_rank=1),
            ]
            incompatible_prefix = dataclasses.replace(
                extended,
                documents=tuple(prefix_documents),
            )
            with self.assertRaisesRegex(ValueError, "prefix"):
                validate_manifest_resume_compatibility(same_a, incompatible_prefix)

            by_id = {doc.doc_id: doc for doc in extended.documents}
            for doc in extended.documents:
                self.assertEqual(doc.doc_num_chunks, (doc.doc_token_count + 3) // 4)
                self.assertNotEqual(doc.doc_id, doc.donor_doc_id)
                self.assertEqual(doc.doc_num_chunks, by_id[doc.donor_doc_id].doc_num_chunks)

            hidden_root = tmpdir / "hidden-raw"
            root.rename(hidden_root)
            self.assertEqual(load_full_document_manifest(tmpdir / "same-a.json"), same_a)

    def test_state_usage_v1_document_and_chunk_totals_are_frozen(self):
        sample_counts = {
            2: 500,
            3: 500,
            4: 500,
            5: 500,
            6: 500,
            7: 500,
            8: 500,
            9: 500,
            10: 500,
            11: 500,
            12: 500,
            13: 500,
            14: 381,
            15: 301,
            16: 213,
        }
        self.assertEqual(
            [sample_counts[doc_num_chunks] for doc_num_chunks in range(2, 14)],
            [500] * 12,
        )
        self.assertEqual(
            [sample_counts[doc_num_chunks] for doc_num_chunks in range(14, 17)],
            [381, 301, 213],
        )
        self.assertEqual(sum(sample_counts.values()), 6_895)
        self.assertEqual(
            sum(doc_num_chunks * count for doc_num_chunks, count in sample_counts.items()),
            58_257,
        )

    def test_loader_returns_complete_target_and_one_donor_chain(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            root = tmpdir / "raw"
            tokens_by_doc_id = self._write_val_root(root)
            built = self._build(root, tmpdir / "manifest.json", seed=0, sample_cap=4)

            manifest = load_full_document_manifest(tmpdir / "manifest.json")
            self.assertEqual(manifest, built)
            loader = FullDocumentLoader(manifest)

            loaded_bucket_indices = set()
            for entry in manifest.documents:
                if entry.bucket_idx in loaded_bucket_indices or entry.doc_token_count % 4 != 1:
                    continue
                target, donor = loader.load_pair(entry)
                loaded_bucket_indices.add(entry.bucket_idx)

                self.assertEqual(target.doc_id, entry.doc_id)
                self.assertEqual(donor.doc_id, entry.donor_doc_id)
                self.assertEqual(target.token_ids_CT.shape, (entry.doc_num_chunks, 4))
                self.assertEqual(target.attention_mask_CT.shape, target.token_ids_CT.shape)
                np.testing.assert_array_equal(
                    target.loss_mask_CT,
                    target.attention_mask_CT,
                )
                self.assertEqual(target.chunk_idx_C.tolist(), list(range(entry.doc_num_chunks)))
                self.assertEqual(target.attention_mask_CT[-1].tolist(), [1, 0, 0, 0])
                np.testing.assert_array_equal(
                    target.token_ids_CT[target.attention_mask_CT.astype(bool)],
                    tokens_by_doc_id[entry.doc_id],
                )

                self.assertEqual(target.doc_num_chunks, donor.doc_num_chunks)
                self.assertEqual(donor.chunk_idx_C.tolist(), list(range(entry.doc_num_chunks)))
                np.testing.assert_array_equal(
                    donor.token_ids_CT[donor.attention_mask_CT.astype(bool)],
                    tokens_by_doc_id[entry.donor_doc_id],
                )
                self.assertEqual(
                    donor.attention_mask_CT[-1].sum(),
                    donor.doc_token_count - (donor.doc_num_chunks - 1) * 4,
                )
            self.assertEqual(loaded_bucket_indices, {0, 1, 2, 3})

            raw_manifest = json.loads((tmpdir / "manifest.json").read_text())
            for field, value in (
                ("doc_id", "tampered-doc"),
                ("sample_rank", 99),
                ("record_idx", 99),
                ("donor_doc_id", "tampered-donor"),
            ):
                tampered = json.loads(json.dumps(raw_manifest))
                tampered["documents"][0][field] = value
                tampered_path = tmpdir / f"tampered-{field}.json"
                tampered_path.write_text(json.dumps(tampered))
                with self.assertRaisesRegex(ValueError, "manifest hash"):
                    load_full_document_manifest(tampered_path)

            tampered = json.loads(json.dumps(raw_manifest))
            tampered["seed"] = 9
            tampered_path = tmpdir / "tampered-seed.json"
            tampered_path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                load_full_document_manifest(tampered_path)

            def write_rehashed_manifest(payload, name):
                payload_without_hash = {
                    key: value for key, value in payload.items() if key != "manifest_hash"
                }
                digest = hashlib.sha256(
                    json.dumps(
                        payload_without_hash,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                payload_without_hash["manifest_hash"] = f"sha256:{digest}"
                path = tmpdir / name
                path.write_text(json.dumps(payload_without_hash))
                return path

            invalid_counts = json.loads(json.dumps(raw_manifest))
            invalid_counts["counts_by_length"].append(invalid_counts["counts_by_length"][0])
            with self.assertRaisesRegex(ValueError, "counts"):
                load_full_document_manifest(
                    write_rehashed_manifest(invalid_counts, "invalid-counts.json")
                )

            invalid_donor = json.loads(json.dumps(raw_manifest))
            first_length_docs = invalid_donor["documents"][:4]
            for target_idx, donor_idx in enumerate((2, 3, 0, 1)):
                first_length_docs[target_idx]["donor_bucket_idx"] = first_length_docs[donor_idx][
                    "bucket_idx"
                ]
                first_length_docs[target_idx]["donor_record_idx"] = first_length_docs[donor_idx][
                    "record_idx"
                ]
                first_length_docs[target_idx]["donor_doc_id"] = first_length_docs[donor_idx][
                    "doc_id"
                ]
            with self.assertRaisesRegex(ValueError, "donor assignment"):
                load_full_document_manifest(
                    write_rehashed_manifest(invalid_donor, "invalid-donor.json")
                )

            invalid_split = json.loads(json.dumps(raw_manifest))
            invalid_split["split"] = "train"
            with self.assertRaisesRegex(ValueError, "split"):
                load_full_document_manifest(
                    write_rehashed_manifest(invalid_split, "invalid-split.json")
                )

            self._write_val_root(root, split="train")
            with self.assertRaisesRegex(ValueError, "Only split='val'"):
                build_full_document_manifest(
                    root,
                    tmpdir / "train-manifest.json",
                    split="train",
                    chunk_length=4,
                    seed=0,
                    sample_cap=4,
                    min_doc_chunks=2,
                    max_doc_chunks=16,
                )


if __name__ == "__main__":
    absltest.main()
