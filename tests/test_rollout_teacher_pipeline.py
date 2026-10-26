"""Tests the rollout-teacher pointer index, its metadata-only validation and the Grain iterator.

Fixture roots are written with the real record serializer and ArrayRecord writer; the
real probe builds the production index from the real dataset metadata and reads a
bounded number of real batches.
"""

import os
import socket

assert os.environ.get("SLURM_JOB_ID")
assert os.environ.get("SLURMD_NODENAME") == socket.gethostname().split(".")[0]
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import gc
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter

import omegalax
from omegalax.data import rollout_teacher as RT
from omegalax.data import rollout_teacher_pipeline as RTP
from omegalax.data.pretrain_data_set import iter_json_arrayrecord_records, pop_pretrain_metadata

REPO = Path(__file__).resolve().parents[1]
REAL_ROOT = Path("/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/datasets/rollout_teacher_v1")
BINDING = "scoring-binding-test"
T = 8
BUCKETS = [1, 2, 4]
SAMPLES = [
    dict(
        sequence_id="a",
        split="train",
        status="completed",
        correct=True,
        token_count=8,
        target_positions=range(2, 8),
    ),
    dict(
        sequence_id="b",
        split="train",
        status="completed",
        correct=True,
        token_count=9,
        target_positions=range(3, 9),
    ),
    dict(
        sequence_id="c",
        split="train",
        status="completed",
        correct=False,
        token_count=20,
        target_positions=[*range(1, 5), *range(14, 20)],
        block_rows=3,
    ),
    dict(
        sequence_id="d",
        split="train",
        status="completed",
        correct=False,
        token_count=32,
        target_positions=range(5, 32),
        block_rows=10,
    ),
    dict(
        sequence_id="e",
        split="train",
        status="truncated",
        correct=None,
        token_count=40,
        target_positions=range(30, 40),
    ),
    dict(
        sequence_id="f",
        split="train",
        status="completed",
        correct=None,
        token_count=6,
        target_positions=range(1, 6),
    ),
    dict(
        sequence_id="i",
        split="train",
        status="completed",
        correct=True,
        token_count=6,
        target_positions=range(4, 6),
    ),
    dict(
        sequence_id="j",
        split="train",
        status="completed",
        correct=True,
        token_count=7,
        target_positions=range(1, 7),
    ),
    dict(
        sequence_id="k",
        split="train",
        status="completed",
        correct=False,
        token_count=16,
        target_positions=range(2, 16),
    ),
    dict(
        sequence_id="g",
        split="val",
        status="completed",
        correct=True,
        token_count=12,
        target_positions=range(6, 12),
    ),
    dict(
        sequence_id="h",
        split="val",
        status="completed",
        correct=False,
        token_count=5,
        target_positions=range(1, 5),
    ),
]
SAMPLE = {sample["sequence_id"]: sample for sample in SAMPLES}
SELECTED = {
    "correct": {"train": ["a", "b", "i", "j"], "val": ["g"]},
    "incorrect": {"train": ["c", "d", "k"], "val": ["h"]},
    "graded": {"train": ["a", "b", "c", "d", "i", "j", "k"], "val": ["g", "h"]},
}
ITER = dict(
    selection="graded", segment_length=T, segment_buckets=BUCKETS, fsdp_size=1, grain_workers=0
)


def synthetic_targets(positions):
    """Teacher rows encoding their position; unique Top512, special and tail probabilities sum to 1."""
    positions = np.asarray(positions, dtype=np.int32)
    row = positions[:, None].astype(np.int64)
    ids = ((row * 7919 + np.arange(RT.TOP_K) * 97) % RT.VOCAB_SIZE).astype(np.int32)
    logits = (row * 1e-3 - np.arange(RT.TOP_K) * 1e-2).astype(np.float32)
    inside = ids[:, :, None] == np.asarray(RT.SPECIAL_IDS)[None, None, :]
    overlap = inside.any(axis=1)
    special = np.where(
        overlap,
        np.where(inside, logits[:, :, None], 0).sum(axis=1),
        logits[:, -1:] - 1.0 - np.arange(len(RT.SPECIAL_IDS)) * 1e-2,
    ).astype(np.float32)
    logz_tail = logits[:, -1].astype(np.float64) + 2.0
    stored = np.concatenate([logits, np.where(overlap, -np.inf, special)], axis=1).astype(
        np.float64
    )
    return {
        "target_positions": positions,
        "top512_ids": ids,
        "top512_logits": logits,
        "special_logits": special,
        "logz_all": np.logaddexp(np.logaddexp.reduce(stored, axis=1), logz_tail).astype(np.float32),
        "logz_tail": logz_tail.astype(np.float32),
    }


def sequence_tokens(index, n):
    return (1 + (np.arange(n) * 7919 + 13 * index) % 99999).astype(np.int32)


def bucket_of(sequence_id):
    return min(c for c in BUCKETS if c * T >= SAMPLE[sequence_id]["token_count"])


def sample_positions(sequence_id):
    return np.asarray(SAMPLE[sequence_id]["target_positions"], dtype=np.int32)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_fixture_root(root, samples):
    """Write a complete rollout_teacher_v1 root with the real record serializer and ArrayRecord writer.

    ``samples`` are dicts with ``sequence_id``, ``split``, ``status`` and ``correct`` (the
    snapshot result fields: "completed" with True/False is correct/incorrect, anything else
    unlabeled), ``token_count``, sorted ``target_positions`` in [1, token_count) and optional
    ``block_rows`` (target rows per record, default 1024). Sample i stores the token IDs
    ``sequence_tokens(i, token_count)`` and teacher rows ``synthetic_targets(target_positions)``.
    Writes manifest, catalog, prepared index and label snapshot with consistent hashes.
    """
    root = Path(root)
    (root / "sequences/00").mkdir(parents=True)
    (root / "targets").mkdir()
    writer = ArrayRecordWriter(str(root / "sequences/00/fixture.array_record"), "group_size:1")
    prepared, catalog, attempts = [], [], []
    for index, sample in enumerate(samples):
        sid, n, block_rows = (
            sample["sequence_id"],
            sample["token_count"],
            sample.get("block_rows", 1024),
        )
        positions = np.asarray(sample["target_positions"], dtype=np.int32)
        runs = (
            np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
            if positions.size
            else []
        )
        turns = [
            dict(
                start=int(run[0]),
                end=int(run[-1]) + 1,
                call=call,
                turn_kind="complete",
                eligible=True,
                audit_reason=None,
            )
            for call, run in enumerate(runs)
        ]
        alias = dict(attempt_id=f"attempt-{sid}", sequence_id=sid, task_key=f"swe_smith:task-{sid}")
        metadata = dict(
            sequence_id=sid, family="swe_smith", turns=turns, aliases=[alias], stop_positions=[]
        )
        writer.write(
            RT.encode_record(
                "sequence",
                metadata,
                dict(token_ids=sequence_tokens(index, n), target_positions=positions),
            )
        )
        full = synthetic_targets(positions)
        target_writer = ArrayRecordWriter(str(root / f"targets/{sid}.array_record"), "group_size:1")
        records = []
        for block_index, start in enumerate(range(0, len(positions), block_rows)):
            block = {name: array[start : start + block_rows] for name, array in full.items()}
            target_writer.write(
                RT.encode_record(
                    "targets",
                    dict(sequence_id=sid, block_index=block_index, binding=BINDING),
                    block,
                )
            )
            records.append(
                dict(
                    record_index=block_index,
                    row_start=start,
                    row_stop=start + len(block["target_positions"]),
                    position_min=int(block["target_positions"][0]),
                    position_max=int(block["target_positions"][-1]),
                )
            )
        target_writer.close()
        entry = dict(
            sequence_id=sid,
            split=sample["split"],
            family="swe_smith",
            task_key=alias["task_key"],
            token_count=n,
            target_count=len(positions),
            aliases=[alias],
            path="sequences/00/fixture.array_record",
            record_index=index,
        )
        prepared.append(entry)
        catalog.append(
            dict(entry, targets_path=f"targets/{sid}.array_record", target_records=records)
        )
        attempts.append(
            dict(
                attempt_id=alias["attempt_id"],
                task_key=alias["task_key"],
                state="included",
                result=dict(status=sample["status"], correct=sample["correct"]),
            )
        )
    writer.close()
    snapshot_sha = write_json(root / "prepared/snapshot.json", dict(attempts=attempts))
    prepared_sha = write_json(
        root / "prepared.json",
        dict(
            binding_sha256="builder-binding-test", snapshot_sha256=snapshot_sha, sequences=prepared
        ),
    )
    manifest = dict(
        format=RT.FORMAT,
        vocab_size=RT.VOCAB_SIZE,
        special_ids=list(RT.SPECIAL_IDS),
        binding_sha256="builder-binding-test",
        teacher_binding_sha256="teacher-binding-test",
        teacher=dict(revision="teacher-revision-test"),
        snapshot=dict(path="prepared/snapshot.json", sha256=snapshot_sha),
        prepared=dict(path="prepared.json", sha256=prepared_sha),
    )
    write_json(root / RT.MANIFEST_FILENAME, manifest)
    write_json(
        root / RT.CATALOG_FILENAME,
        dict(
            format=RT.FORMAT,
            binding=BINDING,
            complete=True,
            dataset_binding=RT.dataset_source_binding(manifest),
            sequences=catalog,
        ),
    )
    return root


def take(iterator, count):
    return [next(iterator) for _ in range(count)]


def assert_batches_equal(test, left, right):
    test.assertEqual(sorted(left), sorted(right))
    for key in left:
        if key == "metadata":
            test.assertEqual(sorted(left[key]), sorted(right[key]))
            for name, value in left[key].items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, right[key][name], err_msg=name)
                else:
                    test.assertEqual(value, right[key][name], name)
        else:
            test.assertEqual(left[key].dtype, right[key].dtype, key)
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def check_batch(test, batch, split):
    """Shape, mask and row checks of one batch against the fixture definition; returns valid sequence ids."""
    meta = batch["metadata"]
    C, R = meta["num_segments"], meta["target_capacity"]
    B = batch["token_ids_BCT"].shape[0]
    test.assertEqual(C, BUCKETS[meta["bucket_idx"]])
    expected = {
        "token_ids_BCT": (np.int32, (B, C, T)),
        "attention_mask_BCT": (np.int32, (B, C, T)),
        "loss_mask_BCT": (np.int32, (B, C, T)),
        "reset_state_BC": (np.bool_, (B, C)),
        "chunk_idx_BC": (np.int32, (B, C)),
        "teacher_positions_BR": (np.int32, (B, R)),
        "teacher_mask_BR": (np.bool_, (B, R)),
        "teacher_top512_ids_BRK": (np.int32, (B, R, RT.TOP_K)),
        "teacher_top512_logits_BRK": (np.float32, (B, R, RT.TOP_K)),
        "teacher_special_logits_BRS": (np.float32, (B, R, len(RT.SPECIAL_IDS))),
        "teacher_logz_all_BR": (np.float32, (B, R)),
        "teacher_logz_tail_BR": (np.float32, (B, R)),
    }
    test.assertEqual(sorted(batch), sorted([*expected, "metadata"]))
    for key, (dtype, shape) in expected.items():
        test.assertEqual((batch[key].dtype, batch[key].shape), (np.dtype(dtype), shape), key)
    np.testing.assert_array_equal(batch["reset_state_BC"], np.arange(C)[None, :].repeat(B, 0) == 0)
    np.testing.assert_array_equal(batch["chunk_idx_BC"], np.arange(C)[None, :].repeat(B, 0))
    seen = []
    for row in range(B):
        tokens = batch["token_ids_BCT"][row].reshape(-1)
        attention = batch["attention_mask_BCT"][row].reshape(-1)
        loss = batch["loss_mask_BCT"][row].reshape(-1)
        mask = batch["teacher_mask_BR"][row]
        sid = meta["sequence_ids"][row]
        if sid is None:
            test.assertFalse(meta["valid_sample_B"][row])
            test.assertEqual((meta["sample_idx_B"][row], meta["labels"][row]), (-1, None))
            test.assertFalse(attention.any() or loss.any() or mask.any() or tokens.any())
            for key in RTP.TEACHER_BATCH_KEYS:
                test.assertFalse(batch[key][row].any(), key)
            continue
        index = [sample["sequence_id"] for sample in SAMPLES].index(sid)
        n = SAMPLES[index]["token_count"]
        positions = sample_positions(sid)
        m = len(positions)
        test.assertEqual(SAMPLES[index]["split"], split)
        test.assertEqual(meta["valid_segments_B"][row], -(-n // T))
        test.assertEqual(C, bucket_of(sid))
        np.testing.assert_array_equal(tokens[:n], sequence_tokens(index, n))
        test.assertFalse(tokens[n:].any())
        np.testing.assert_array_equal(attention, np.arange(C * T) < n)
        np.testing.assert_array_equal(np.flatnonzero(loss), positions)
        np.testing.assert_array_equal(mask, np.arange(R) < m)
        hidden = batch["teacher_positions_BR"][row]
        np.testing.assert_array_equal(hidden[:m], positions - 1)
        test.assertFalse(hidden[m:].any())
        np.testing.assert_array_equal(
            meta["teacher_target_positions_BR"][row], np.r_[positions, np.full(R - m, -1)]
        )
        expected_rows = synthetic_targets(positions)
        for key, name in (
            ("teacher_top512_ids_BRK", "top512_ids"),
            ("teacher_top512_logits_BRK", "top512_logits"),
            ("teacher_special_logits_BRS", "special_logits"),
            ("teacher_logz_all_BR", "logz_all"),
            ("teacher_logz_tail_BR", "logz_tail"),
        ):
            np.testing.assert_array_equal(batch[key][row, :m], expected_rows[name], err_msg=key)
            test.assertFalse(batch[key][row, m:].any(), key)
        # Target j is predicted by hidden j-1; segment-boundary targets j = s*T come from the
        # last hidden of segment s-1 and the chain end has no invented target.
        np.testing.assert_array_equal(tokens[hidden[:m] + 1], sequence_tokens(index, n)[positions])
        test.assertTrue(np.all(hidden[:m] < n - 1))
        seen.append(sid)
    return seen


class RolloutTeacherIndexTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "rollout_teacher_v1"
        write_fixture_root(self.root, SAMPLES)

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, name="index", selection="graded", buckets=BUCKETS):
        return RTP.build_rollout_teacher_index(
            self.root,
            self.tmp / name,
            selection=selection,
            segment_length=T,
            segment_buckets=buckets,
        )

    def test_normal_import_uses_working_copy(self):
        self.assertEqual(Path(omegalax.__file__).resolve(), REPO / "omegalax/__init__.py")
        self.assertEqual(
            Path(RTP.__file__).resolve(), REPO / "omegalax/data/rollout_teacher_pipeline.py"
        )

    def test_build_selects_labels_preserves_splits_and_smallest_buckets(self):
        for selection, expected in SELECTED.items():
            with self.subTest(selection):
                index = self.build(selection, selection)
                metadata = RTP.load_rollout_teacher_index(
                    index, selection=selection, segment_length=T, segment_buckets=BUCKETS
                )
                self.assertEqual(sorted(metadata["splits"]), ["train", "val"])
                max_targets = [0] * len(BUCKETS)
                for split, sids in expected.items():
                    records = [record for _, record in iter_json_arrayrecord_records(index / split)]
                    self.assertEqual(
                        [r["sequence_id"] for r in records],
                        sorted(sids, key=lambda s: (bucket_of(s), s)),
                    )
                    self.assertEqual([r["sample_idx"] for r in records], list(range(len(records))))
                    split_metadata = metadata["splits"][split]
                    for record in records:
                        sample = SAMPLE[record["sequence_id"]]
                        self.assertEqual(record["split"], split)
                        self.assertEqual(
                            record["label"], "correct" if sample["correct"] else "incorrect"
                        )
                        self.assertEqual(record["num_segments"], -(-sample["token_count"] // T))
                        self.assertEqual(
                            BUCKETS[record["bucket_idx"]], bucket_of(record["sequence_id"])
                        )
                        self.assertEqual(
                            record["target_count"], len(sample_positions(record["sequence_id"]))
                        )
                        bucket = split_metadata["buckets"][record["bucket_idx"]]
                        self.assertLessEqual(bucket["record_start"], record["sample_idx"])
                        self.assertLess(
                            record["sample_idx"], bucket["record_start"] + bucket["num_samples"]
                        )
                        max_targets[record["bucket_idx"]] = max(
                            max_targets[record["bucket_idx"]], record["target_count"]
                        )
                    self.assertEqual(split_metadata["num_samples"], len(sids))
                    self.assertEqual(sum(split_metadata["label_counts"].values()), len(sids))
                    self.assertEqual(
                        split_metadata["num_targets"], sum(r["target_count"] for r in records)
                    )
                self.assertEqual(
                    metadata["target_capacities"], [1024 if value else 0 for value in max_targets]
                )
                self.assertEqual(metadata["source"]["scoring_binding"], BINDING)
        # No correct sample needs 4 segments: the empty bucket gets capacity 0 and no batches.
        metadata = RTP.load_rollout_teacher_index(
            self.tmp / "correct", selection="correct", segment_length=T, segment_buckets=BUCKETS
        )
        self.assertEqual(metadata["target_capacities"][2], 0)

    def test_build_rejects_too_long_chain_and_nonempty_output_without_dropping(self):
        # e (truncated, 5 segments) is never selected and does not block any build.
        self.build("small_correct", "correct", buckets=[1, 2])
        with self.assertRaisesRegex(
            ValueError, "2 selected samples need more than 2 segments.*largest needs 4"
        ):
            self.build("small_graded", "graded", buckets=[1, 2])
        self.assertFalse((self.tmp / "small_graded").exists())
        (self.tmp / "busy").mkdir()
        (self.tmp / "busy/file").write_text("x")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.build("busy")
        for selection, buckets in (("all", BUCKETS), ("graded", [2, 1]), ("graded", [])):
            with self.subTest(selection=selection, buckets=buckets), self.assertRaises(ValueError):
                self.build("bad", selection, buckets)

    def test_build_rejects_label_snapshot_not_bound_by_manifest(self):
        path = self.root / "prepared/snapshot.json"
        snapshot = json.loads(path.read_text())
        snapshot["attempts"][4]["result"] = dict(status="completed", correct=True)
        path.write_text(json.dumps(snapshot))
        with self.assertRaisesRegex(ValueError, "does not match the manifest sha256"):
            self.build()

    def test_load_is_metadata_only_and_rejects_changed_index_source_or_config(self):
        index = self.build()
        config = dict(selection="graded", segment_length=T, segment_buckets=BUCKETS)
        # Without any sequence/target record on disk, loading and iterator setup still succeed.
        for name in ("sequences", "targets"):
            (self.root / name).rename(self.tmp / name)
        metadata = RTP.load_rollout_teacher_index(index, **config)
        iterator = RTP.make_rollout_teacher_iterator(
            index, split="train", batch_size=4, shuffle=True, dp_size=1, process_index=0, **ITER
        )
        with self.assertRaises(Exception):
            next(iterator)
        for name in ("sequences", "targets"):
            (self.tmp / name).rename(self.root / name)
        self.assertEqual(RTP.load_rollout_teacher_index(index, **config), metadata)

        for key, value in (
            ("selection", "correct"),
            ("segment_length", 16),
            ("segment_buckets", [1, 2, 8]),
        ):
            with self.subTest(key), self.assertRaisesRegex(ValueError, f"{key} is"):
                RTP.load_rollout_teacher_index(index, **dict(config, **{key: value}))

        def check_changed(path, pattern):
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(path.name), self.assertRaisesRegex(ValueError, pattern):
                RTP.load_rollout_teacher_index(index, **config)
            path.write_bytes(original)

        check_changed(self.root / RT.CATALOG_FILENAME, "catalog_sha256")
        check_changed(self.root / "prepared/snapshot.json", "snapshot_sha256")
        check_changed(self.root / RT.MANIFEST_FILENAME, "manifest_sha256")
        check_changed(
            index / metadata["splits"]["train"]["shard_paths"][0], "shards of split 'train'"
        )
        edited = dict(metadata, target_capacities=[2048, 2048, 2048])
        (index / "metadata.json").write_text(json.dumps(edited))
        with self.assertRaisesRegex(ValueError, "index_binding"):
            RTP.load_rollout_teacher_index(index, **config)
        (index / "metadata.json").unlink()
        with self.assertRaisesRegex(ValueError, "metadata does not exist"):
            RTP.load_rollout_teacher_index(index, **config)


class RolloutTeacherIteratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.root = cls.tmp / "rollout_teacher_v1"
        write_fixture_root(cls.root, SAMPLES)
        cls.index = RTP.build_rollout_teacher_index(
            cls.root,
            cls.tmp / "index",
            selection="graded",
            segment_length=T,
            segment_buckets=BUCKETS,
        )
        cls.metadata = RTP.load_rollout_teacher_index(
            cls.index, selection="graded", segment_length=T, segment_buckets=BUCKETS
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def iterator(self, split="train", **kwargs):
        options = dict(ITER, batch_size=4, shuffle=True, dp_size=1, process_index=0)
        options.update(kwargs)
        return RTP.make_rollout_teacher_iterator(self.index, split=split, **options)

    def test_one_epoch_yields_every_target_row_exactly_once_with_masked_padding(self):
        for split in ("train", "val"):
            with self.subTest(split):
                epoch_batches = RTP.rollout_teacher_epoch_batches(
                    self.metadata, split=split, batch_size=4, dp_size=1, fsdp_size=1
                )
                batches = list(self.iterator(split, num_epochs=1))
                self.assertEqual(len(batches), epoch_batches)
                seen = [sid for batch in batches for sid in check_batch(self, batch, split)]
                self.assertEqual(sorted(seen), SELECTED["graded"][split])
                # Residual batches keep fully masked padding slots instead of dropping or repeating samples.
                padded = sum(int((~batch["metadata"]["valid_sample_B"]).sum()) for batch in batches)
                self.assertEqual(padded, {"train": 1, "val": 4}[split])
                # Fixture teacher rows are normalized grouped distributions (unique IDs plus tail).
                batch = batches[0]
                valid = batch["teacher_mask_BR"]
                overlap = (
                    batch["teacher_top512_ids_BRK"][..., :, None] == np.asarray(RT.SPECIAL_IDS)
                ).any(axis=-2)
                logz = batch["teacher_logz_all_BR"][..., None].astype(np.float64)
                mass = (
                    np.exp(batch["teacher_top512_logits_BRK"] - logz).sum(-1)
                    + np.where(
                        overlap, 0.0, np.exp(batch["teacher_special_logits_BRS"] - logz)
                    ).sum(-1)
                    + np.exp(batch["teacher_logz_tail_BR"] - logz[..., 0])
                )
                np.testing.assert_allclose(mass[valid], 1.0, rtol=1e-5)
                metadata = pop_pretrain_metadata(batches[0])
                self.assertEqual(metadata["num_segments"], batches[0]["token_ids_BCT"].shape[1])
                self.assertTrue(all(isinstance(value, np.ndarray) for value in batches[0].values()))
        # Train buckets C=1,2,4 hold 3,2,2 samples; local batches of 4,2,1 samples -> 1+1+2 batches.
        self.assertEqual(
            RTP.rollout_teacher_epoch_batches(
                self.metadata, split="train", batch_size=4, dp_size=1, fsdp_size=1
            ),
            4,
        )

    def test_data_shards_share_the_bucket_schedule_and_partition_each_epoch(self):
        shards = [self.iterator(dp_size=2, process_index=d, num_epochs=3) for d in range(2)]
        epoch_batches = RTP.rollout_teacher_epoch_batches(
            self.metadata, split="train", batch_size=4, dp_size=2, fsdp_size=1
        )
        self.assertEqual(epoch_batches, 3)
        orders = []
        for epoch in range(3):
            seen = []
            for step in range(epoch_batches):
                pair = [next(shard) for shard in shards]
                metas = [batch["metadata"] for batch in pair]
                self.assertEqual(
                    {
                        (m["epoch"], m["global_batch_idx"], m["bucket_idx"], m["target_capacity"])
                        for m in metas
                    },
                    {
                        (
                            epoch,
                            epoch * epoch_batches + step,
                            metas[0]["bucket_idx"],
                            metas[0]["target_capacity"],
                        )
                    },
                )
                self.assertEqual(
                    pair[0]["teacher_top512_ids_BRK"].shape, pair[1]["teacher_top512_ids_BRK"].shape
                )
                for batch in pair:
                    seen += check_batch(self, batch, "train")
            self.assertEqual(sorted(seen), SELECTED["graded"]["train"])
            orders.append(seen)
        for shard in shards:
            with self.assertRaises(StopIteration):
                next(shard)
        self.assertGreater(len({tuple(order) for order in orders}), 1)

    def test_order_is_deterministic_unshuffled_bucket_major_and_reshuffled_per_epoch(self):
        def ids(iterator, count):
            return [tuple(b["metadata"]["sequence_ids"]) for b in take(iterator, count)]

        self.assertEqual(
            ids(self.iterator(shuffle=False, num_epochs=1), 4),
            [("a", "i", "j", None), ("b", "k"), ("c",), ("d",)],
        )
        self.assertEqual(ids(self.iterator(seed=3), 12), ids(self.iterator(seed=3), 12))
        self.assertNotEqual(ids(self.iterator(seed=3), 12), ids(self.iterator(seed=4), 12))
        infinite = ids(self.iterator(seed=3), 40)
        epochs = [infinite[i : i + 4] for i in range(0, 40, 4)]
        self.assertTrue(
            all(
                sorted(s for batch in e for s in batch if s) == SELECTED["graded"]["train"]
                for e in epochs
            )
        )
        self.assertGreater(len({tuple(e) for e in epochs}), 1)

    def test_rejects_batch_size_split_and_shard_errors(self):
        cases = {
            "batch_not_divisible": dict(batch_size=6),
            "zero_batch": dict(batch_size=0),
            "unknown_split": dict(split="test"),
            "shard_index": dict(dp_size=2, dp_index=2),
            "epochs": dict(num_epochs=0),
        }
        for name, kwargs in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.iterator(**kwargs)

    def test_grain_state_resume_matches_uninterrupted_iteration(self):
        for workers in (0, 2):
            with self.subTest(grain_workers=workers):
                reference = take(self.iterator(seed=5, grain_workers=workers), 9)
                interrupted = self.iterator(seed=5, grain_workers=workers)
                take(interrupted, 5)
                state = interrupted.get_state()
                resumed = self.iterator(seed=5, grain_workers=workers)
                resumed.set_state(state)
                for left, right in zip(take(resumed, 4), reference[5:]):
                    assert_batches_equal(self, left, right)

    def test_production_orbax_checkpoint_restores_iterator_position(self):
        import jax.numpy as jnp
        from omegalax.trainers import checkpoint_utils
        from omegalax.trainers.text import _make_checkpoint_manager

        reference = take(self.iterator(seed=7, dp_size=2, process_index=1), 6)
        iterator = self.iterator(seed=7, dp_size=2, process_index=1)
        take(iterator, 4)
        manager = _make_checkpoint_manager(self.tmp / "ckpt", save_interval=None)
        train_state = {"step": jnp.asarray(4, dtype=jnp.int32)}
        manager.save(
            4, args=checkpoint_utils.make_grain_save_args(train_state, iterator), force=True
        )
        manager.wait_until_finished()
        fresh = self.iterator(seed=7, dp_size=2, process_index=1)
        restored = manager.restore(
            4, args=checkpoint_utils.make_grain_restore_args(train_state, fresh)
        )
        manager.close()
        resumed = checkpoint_utils.restored_input_iter(restored)
        for left, right in zip(take(resumed, 2), reference[4:]):
            assert_batches_equal(self, left, right)


@unittest.skipUnless(REAL_ROOT.is_dir(), f"real dataset is not available: {REAL_ROOT}")
class RealRolloutTeacherProbeTest(unittest.TestCase):
    """Production index over the real metadata; reads only a few real batches."""

    def test_real_graded_index_counts_and_bounded_batches_match_reader(self):
        config = dict(selection="graded", segment_length=4096, segment_buckets=[8, 16, 32])
        with tempfile.TemporaryDirectory() as tmp:
            index = RTP.build_rollout_teacher_index(REAL_ROOT, Path(tmp) / "index", **config)
            metadata = RTP.load_rollout_teacher_index(index, **config)
            splits = metadata["splits"]
            # Counts of the independent length audit (sample_lengths_20261023T154651/lengths.json).
            self.assertEqual(sum(s["label_counts"]["correct"] for s in splits.values()), 11413)
            self.assertEqual(sum(s["label_counts"]["incorrect"] for s in splits.values()), 8571)
            self.assertEqual(
                (splits["train"]["num_targets"], splits["val"]["num_targets"]), (209241511, 4305484)
            )
            self.assertEqual(metadata["target_capacities"], [24576, 46080, 50176])

            for split, shuffle in (("val", False), ("train", True)):
                dataset = RT.TeacherDataset(REAL_ROOT, split)
                self.assertEqual(metadata["source"]["scoring_binding"], dataset.binding)
                threads = set(threading.enumerate())
                # One read thread and prefetch 1 keep real reads at most one batch ahead.
                iterator = RTP.make_rollout_teacher_iterator(
                    index,
                    split=split,
                    batch_size=32,
                    shuffle=shuffle,
                    seed=0,
                    num_epochs=1,
                    dp_size=1,
                    fsdp_size=1,
                    process_index=0,
                    grain_workers=0,
                    grain_read_threads=1,
                    grain_read_prefetch_buffer_size=1,
                    **config,
                )
                batch = next(iterator)
                meta = batch["metadata"]
                C, R = meta["num_segments"], meta["target_capacity"]
                self.assertEqual(R, metadata["target_capacities"][meta["bucket_idx"]])
                for row, sid in enumerate(meta["sequence_ids"]):
                    if sid is None:
                        continue
                    _, sequence = dataset.get_sequence(sid)
                    targets = dataset.get_targets(sid)
                    n, m = len(sequence["token_ids"]), len(targets["target_positions"])
                    self.assertEqual(C, min(c for c in config["segment_buckets"] if c * 4096 >= n))
                    np.testing.assert_array_equal(
                        batch["token_ids_BCT"][row].reshape(-1)[:n], sequence["token_ids"]
                    )
                    self.assertEqual(int(batch["attention_mask_BCT"][row].sum()), n)
                    np.testing.assert_array_equal(
                        np.flatnonzero(batch["loss_mask_BCT"][row].reshape(-1)),
                        targets["target_positions"],
                    )
                    np.testing.assert_array_equal(
                        batch["teacher_positions_BR"][row, :m], targets["target_positions"] - 1
                    )
                    self.assertEqual(int(batch["teacher_mask_BR"][row].sum()), m)
                    for key, name in (
                        ("teacher_top512_ids_BRK", "top512_ids"),
                        ("teacher_top512_logits_BRK", "top512_logits"),
                        ("teacher_special_logits_BRS", "special_logits"),
                        ("teacher_logz_all_BR", "logz_all"),
                        ("teacher_logz_tail_BR", "logz_tail"),
                    ):
                        np.testing.assert_array_equal(
                            batch[key][row, :m], targets[name], err_msg=key
                        )
                    boundary = targets["target_positions"] % 4096 == 0
                    np.testing.assert_array_equal(
                        batch["teacher_positions_BR"][row, :m][boundary] % 4096, 4095
                    )
                    self.assertIn(meta["labels"][row], ("correct", "incorrect"))
                # Let the in-flight prefetch finish so no index shard stays open on NFS at cleanup.
                iterator.close()
                del iterator
                for thread in set(threading.enumerate()) - threads:
                    thread.join(timeout=300)
                gc.collect()


if __name__ == "__main__":
    unittest.main()
