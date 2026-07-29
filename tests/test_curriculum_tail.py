"""Production-path tests for the historical curriculum-tail bundle builder."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest

from omegalax.data.curriculum_tail import build_curriculum_c4_tail_bundle
from omegalax.data.pretrain_data_set import (
    DOC_CHAIN_FORMAT,
    iter_json_arrayrecord_records,
    load_arrayrecord_metadata,
    write_json_arrayrecord_dataset,
)
from omegalax.data.pretrain_iid_pipeline import make_iid_iterator
from omegalax.data.pretrain_statepassing import (
    STATEPASSING_CURRICULUM_INDEX_FORMAT,
    STATEPASSING_WINDOW_INDEX_FORMAT,
    make_statepassing_iterator,
)
from tests.pretrain_real_data_test_utils import test_temp_dir


def _window_record(*, doc_id: str, record_idx: int, start_chunk: int, c: int, eos=None):
    return {
        "bucket_idx": 0,
        "record_idx": record_idx,
        "doc_id": doc_id,
        "window_idx": start_chunk // c,
        "start_chunk": start_chunk,
        "num_segments": c,
        "doc_token_count": 8,
        "doc_num_chunks": 8,
        "eos_token_idx": eos,
    }


def _iterator_state(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "next_index_in_cycle": 0,
                "next_index_in_datasets": 1,
                "iterators_in_use_indices": [0],
                "iterators_in_use_states": [{"next_index": 0}],
                "exhausted": [0],
            }
        )
    )
    return path


class CurriculumTailTest(absltest.TestCase):
    def test_builder_and_real_iterators_preserve_c4_c8_tokens_eos_and_counts(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            docs = tmpdir / "docs"
            write_json_arrayrecord_dataset(
                (
                    {
                        "dataset_format": DOC_CHAIN_FORMAT,
                        "doc_id": "doc-c4",
                        "token_ids": list(range(1, 9)),
                        "doc_token_count": 8,
                    },
                    {
                        "dataset_format": DOC_CHAIN_FORMAT,
                        "doc_id": "doc-c8",
                        "token_ids": list(range(11, 19)),
                        "doc_token_count": 8,
                    },
                ),
                docs / "train" / "bucket_2k",
                records_per_shard=10,
                overwrite=False,
                metadata={"dataset_format": DOC_CHAIN_FORMAT},
            )
            curriculum = tmpdir / "curriculum"
            c4_path = curriculum / "c4" / "train"
            c8_path = curriculum / "c8" / "train"
            common = {
                "format": STATEPASSING_WINDOW_INDEX_FORMAT,
                "data_set_root": str(docs),
                "split": "train",
                "bucket_names": ["bucket_2k"],
                "chunk_length": 1,
                "eos_id": 63,
                "num_residual_chunks": 0,
                "num_bucket_records": 2,
                "bucket_record_counts": [2],
            }
            write_json_arrayrecord_dataset(
                (
                    _window_record(doc_id="doc-c4", record_idx=0, start_chunk=0, c=4),
                    _window_record(doc_id="doc-c4", record_idx=0, start_chunk=4, c=4),
                ),
                c4_path,
                records_per_shard=10,
                overwrite=False,
                metadata=common | {"num_segments": 4, "num_windows": 2},
            )
            write_json_arrayrecord_dataset(
                (_window_record(doc_id="doc-c8", record_idx=1, start_chunk=0, c=8, eos=6),),
                c8_path,
                records_per_shard=10,
                overwrite=False,
                metadata=common | {"num_segments": 8, "num_windows": 1},
            )
            curriculum.mkdir(exist_ok=True)
            (curriculum / "metadata.json").write_text(
                json.dumps(
                    {
                        "format": STATEPASSING_CURRICULUM_INDEX_FORMAT,
                        "data_set_root": str(docs),
                        "bucket_names": ["bucket_2k"],
                        "allocation_order": [8, 4],
                        "train_order": [4, 8],
                        "chunk_length": 1,
                        "eos_id": 63,
                        "trim_batch_size": 8,
                        "trim_grad_accum_steps": 1,
                        "splits": {
                            "train": {
                                "phases": {
                                    "4": {"path": "c4/train", "phase_steps": 1},
                                    "8": {"path": "c8/train", "phase_steps": 1},
                                }
                            }
                        },
                    }
                )
            )
            c4_state = _iterator_state(tmpdir / "c4-state.json")
            c8_state = _iterator_state(tmpdir / "c8-state.json")

            tail = build_curriculum_c4_tail_bundle(
                curriculum,
                tmpdir / "tail",
                c4_iterator_state=c4_state,
                c8_iterator_state=c8_state,
                c4_steps=1,
                c8_steps=1,
                trim_batch_size=8,
                trim_grad_accum_steps=1,
                records_per_shard=10,
            )

            metadata = load_arrayrecord_metadata(tail)
            state_records = [record for _, record in iter_json_arrayrecord_records(tail / "train")]
            iid_records = [
                record for _, record in iter_json_arrayrecord_records(tail / "iid" / "train")
            ]
            self.assertEqual(metadata["statepassing_steps"], 2)
            self.assertEqual(metadata["iid_steps"], 2)
            self.assertLen(state_records, 4)
            self.assertLen(iid_records, 16)
            c8_records = [record for record in state_records if record["source_phase"] == "c8"]
            self.assertEqual(
                [(record["start_chunk"], record["eos_token_idx"]) for record in c8_records],
                [(0, None), (4, 6)],
            )
            self.assertEqual(
                [
                    (record["start_chunk"], record["eos_token_idx"])
                    for record in iid_records
                    if record["eos_token_idx"] is not None
                ],
                [(6, 6)],
            )

            state_iter = make_statepassing_iterator(
                tail / "train",
                batch_size=8,
                chunk_length=1,
                eos_id=63,
                shuffle=False,
                num_epochs=1,
                dp_size=1,
                fsdp_size=1,
                process_index=0,
                grain_workers=0,
            )
            iid_iter = make_iid_iterator(
                tail / "iid" / "train",
                batch_size=8,
                chunk_length=1,
                eos_id=63,
                shuffle=False,
                num_epochs=1,
                dp_size=1,
                fsdp_size=1,
                process_index=0,
                grain_workers=0,
            )
            state_tokens = []
            iid_tokens = []
            while True:
                try:
                    state_tokens.extend(next(state_iter)["token_ids_BCT"].reshape(-1).tolist())
                except StopIteration:
                    break
            while True:
                try:
                    iid_tokens.extend(next(iid_iter)["token_ids_BT"].reshape(-1).tolist())
                except StopIteration:
                    break
            self.assertEqual(sorted(state_tokens), sorted(iid_tokens))
            self.assertEqual(state_tokens.count(63), 1)
            self.assertEqual(iid_tokens.count(63), 1)


if __name__ == "__main__":
    absltest.main()
