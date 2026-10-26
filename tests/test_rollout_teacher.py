"""Tests the real rollout_teacher_v1 record format, dataset reader and KL reference."""

import os
import socket

assert os.environ.get("SLURM_JOB_ID")
assert os.environ.get("SLURMD_NODENAME") == socket.gethostname().split(".")[0]
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import copy
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from array_record.python.array_record_module import ArrayRecordReader, ArrayRecordWriter

REPO = Path(__file__).resolve().parents[1]
if importlib.util.find_spec("jax") is None:
    # Torch builder/scorer environment: load the data-only module by file, as they do.
    jax = None
    SPEC = importlib.util.spec_from_file_location(
        "rollout_teacher_data", REPO / "omegalax/data/rollout_teacher.py"
    )
    RT = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(RT)
else:
    import jax
    import omegalax
    from omegalax.data import rollout_teacher as RT

requires_jax = unittest.skipIf(jax is None, "needs the OmegaLAX/JAX environment")
WRITER_OPTIONS = "group_size:1,uncompressed"
BINDING = "scoring-binding-test"
SPECIAL = np.asarray(RT.SPECIAL_IDS, dtype=np.int32)


def sequence_fixture(sequence_id="seq-a"):
    """48 tokens: complete turn, masked turn, final truncated turn; stops at 19 and 29."""
    tokens = (np.arange(48) * 7919 % 100000).astype(np.int32)
    tokens[19], tokens[29] = 248046, 248044
    turns = [
        dict(start=10, end=20, call=0, turn_kind="complete", eligible=True, audit_reason=None),
        dict(
            start=24,
            end=30,
            call=1,
            turn_kind="complete",
            eligible=False,
            audit_reason="completion_count_mismatch",
        ),
        dict(
            start=32, end=48, call=2, turn_kind="final_truncated", eligible=True, audit_reason=None
        ),
    ]
    metadata = dict(
        sequence_id=sequence_id,
        family="swe_smith",
        turns=turns,
        aliases=[dict(root="a1", attempt=0, result_status="resolved")],
        stop_positions=[19, 29],
    )
    return metadata, dict(token_ids=tokens, target_positions=np.r_[10:20, 32:48].astype(np.int32))


def synthetic_targets(positions):
    """Valid production-shaped rows whose values encode their position."""
    positions = np.asarray(positions, dtype=np.int32)
    row = positions[:, None].astype(np.int64)
    return {
        "target_positions": positions,
        "top512_ids": ((row * 7919 + np.arange(RT.TOP_K) * 97) % RT.VOCAB_SIZE).astype(np.int32),
        "top512_logits": (row * 1e-3 - np.arange(RT.TOP_K) * 1e-2).astype(np.float32),
        "special_logits": (row - np.arange(len(SPECIAL))).astype(np.float32),
        "logz_all": (positions + 0.5).astype(np.float32),
        "logz_tail": (positions - 0.5).astype(np.float32),
    }


def compact_targets(rows, positions, k=RT.TOP_K, special_ids=SPECIAL):
    """Compact full teacher rows like the scorer: Top-K by (logit desc, ID asc), specials, logZ, tail."""
    full = np.asarray(rows, dtype=np.float64)
    ids = np.argsort(-full, axis=-1, kind="stable")[:, :k]
    tail = np.ones(full.shape, dtype=bool)
    np.put_along_axis(tail, ids, False, axis=-1)
    tail[:, special_ids] = False
    return {
        "target_positions": np.asarray(positions, dtype=np.int32),
        "top512_ids": ids.astype(np.int32),
        "top512_logits": np.take_along_axis(full, ids, axis=-1).astype(np.float32),
        "special_logits": full[:, special_ids].astype(np.float32),
        "logz_all": np.logaddexp.reduce(full, axis=-1).astype(np.float32),
        "logz_tail": np.logaddexp.reduce(np.where(tail, full, -np.inf), axis=-1).astype(np.float32),
    }


def grouped_reference(teacher, student, k, special_ids):
    """Independent float64 reference with explicit ID sets; returns KL rows and d KL / d student."""
    kls, grads = [], []
    for t, s in zip(np.asarray(teacher, np.float64), np.asarray(student, np.float64)):
        p, q = np.exp(t - np.logaddexp.reduce(t)), np.exp(s - np.logaddexp.reduce(s))
        stored = set(np.argsort(-t, kind="stable")[:k].tolist()) | set(special_ids.tolist())
        groups = [[i] for i in sorted(stored)] + [[i for i in range(len(t)) if i not in stored]]
        big_p = np.array([p[g].sum() for g in groups])
        big_q = np.array([q[g].sum() for g in groups])
        kls.append(np.sum(big_p * np.log(big_p / big_q)))
        ratio = np.empty(len(t))
        for g, pg, qg in zip(groups, big_p, big_q):
            ratio[g] = pg / qg
        grads.append(q * (1.0 - ratio))
    return np.array(kls), np.array(grads)


def write_dataset(root, sequences, corrupt_block=None):
    """Real root: manifest, one sequence shard, one target file per sequence, catalog."""
    (root / "targets").mkdir(parents=True)
    manifest = dict(
        format=RT.FORMAT,
        vocab_size=RT.VOCAB_SIZE,
        special_ids=list(RT.SPECIAL_IDS),
        binding_sha256="prepared-binding-test",
        snapshot={"sha256": "snapshot-test"},
        prepared={"sha256": "prepared-index-test"},
    )
    (root / RT.MANIFEST_FILENAME).write_text(json.dumps(manifest))
    entries = []
    writer = ArrayRecordWriter(str(root / "sequences-00000.array_record"), WRITER_OPTIONS)
    for index, (split, metadata, arrays, blocks) in enumerate(sequences):
        writer.write(RT.encode_record("sequence", metadata, arrays))
        sequence_id = metadata["sequence_id"]
        targets_path = f"targets/{sequence_id}.array_record"
        target_writer = ArrayRecordWriter(str(root / targets_path), WRITER_OPTIONS)
        records, row = [], 0
        for block_index, block in enumerate(blocks):
            payload = RT.encode_record(
                "targets",
                dict(sequence_id=sequence_id, block_index=block_index, binding=BINDING),
                block,
            )
            target_writer.write(
                payload[:-4] if (sequence_id, block_index) == corrupt_block else payload
            )
            positions = block["target_positions"]
            records.append(
                dict(
                    record_index=block_index,
                    row_start=row,
                    row_stop=row + len(positions),
                    position_min=int(positions[0]),
                    position_max=int(positions[-1]),
                )
            )
            row += len(positions)
        target_writer.close()
        entries.append(
            dict(
                sequence_id=sequence_id,
                path="sequences-00000.array_record",
                record_index=index,
                token_count=len(arrays["token_ids"]),
                target_count=len(arrays["target_positions"]),
                family=metadata["family"],
                task_key="task",
                split=split,
                aliases=metadata["aliases"],
                targets_path=targets_path,
                target_records=records,
            )
        )
    writer.close()
    (root / RT.CATALOG_FILENAME).write_text(
        json.dumps(
            dict(
                format=RT.FORMAT,
                binding=BINDING,
                dataset_binding=RT.dataset_source_binding(manifest),
                complete=True,
                sequences=entries,
            )
        )
    )


def with_header(payload, edit):
    start = len(RT.MAGIC) + 8
    (size,) = struct.unpack_from("<Q", payload, len(RT.MAGIC))
    header = json.loads(payload[start : start + size])
    edit(header)
    raw = json.dumps(header).encode()
    return RT.MAGIC + struct.pack("<Q", len(raw)) + raw + payload[start + size :]


def assert_arrays_equal(test, actual, expected):
    test.assertEqual(list(actual), list(expected))
    for name in expected:
        test.assertEqual(actual[name].dtype, expected[name].dtype, name)
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


class RecordFormatTest(unittest.TestCase):
    def test_roundtrip_is_exact_and_deterministic(self):
        metadata, arrays = sequence_fixture()
        payload = RT.encode_record("sequence", metadata, arrays)
        self.assertEqual(payload, RT.encode_record("sequence", copy.deepcopy(metadata), arrays))
        kind, decoded_metadata, decoded = RT.decode_record(payload)
        self.assertEqual((kind, decoded_metadata), ("sequence", metadata))
        assert_arrays_equal(self, decoded, arrays)

        block = synthetic_targets(arrays["target_positions"])
        block_metadata = dict(sequence_id="seq-a", block_index=0, binding=BINDING)
        kind, decoded_metadata, decoded = RT.decode_record(
            RT.encode_record("targets", block_metadata, block)
        )
        self.assertEqual((kind, decoded_metadata), ("targets", block_metadata))
        assert_arrays_equal(self, decoded, block)

    def test_decode_rejects_corrupt_payloads(self):
        metadata, arrays = sequence_fixture()
        sequence = RT.encode_record("sequence", metadata, arrays)
        targets = RT.encode_record(
            "targets",
            dict(sequence_id="seq-a", block_index=0, binding=BINDING),
            synthetic_targets(arrays["target_positions"]),
        )
        data_start = len(sequence) - arrays["token_ids"].nbytes - arrays["target_positions"].nbytes
        cases = {
            "magic": b"OMXDC01\n" + sequence[8:],
            "truncated": sequence[:-1],
            "trailing_bytes": sequence + b"\0" * 4,
            "format_version": with_header(
                sequence, lambda h: h.update(format="omegalax_doc_chain_v1")
            ),
            "kind_vs_arrays": with_header(sequence, lambda h: h.update(kind="targets")),
            "big_endian_dtype": with_header(
                sequence, lambda h: h["arrays"][0].__setitem__(1, ">i4")
            ),
            "shape_vs_offsets": with_header(
                targets, lambda h: h["arrays"][1][2].__setitem__(1, 511)
            ),
            "nan_metadata": with_header(
                sequence, lambda h: h["metadata"].update(family=float("nan"))
            ),
            "token_out_of_vocab": (
                sequence[:data_start]
                + np.int32(RT.VOCAB_SIZE).tobytes()
                + sequence[data_start + 4 :]
            ),
            "nan_logz_tail": targets[:-4] + np.float32(np.nan).tobytes(),
        }
        for name, payload in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                RT.decode_record(payload)

    def test_encode_rejects_invalid_records(self):
        base_metadata, base_arrays = sequence_fixture()
        base_block = synthetic_targets(base_arrays["target_positions"])

        def sequence_case(edit):
            metadata, arrays = (
                copy.deepcopy(base_metadata),
                {k: v.copy() for k, v in base_arrays.items()},
            )
            edit(metadata, arrays)
            return "sequence", metadata, arrays

        def targets_case(edit):
            metadata = dict(sequence_id="seq-a", block_index=0, binding=BINDING)
            arrays = {k: v.copy() for k, v in base_block.items()}
            edit(metadata, arrays)
            return "targets", metadata, arrays

        def tie_with_descending_ids(m, a):
            a["top512_ids"][0, :2] = a["top512_ids"][0, 1::-1]
            a["top512_logits"][0, 1] = a["top512_logits"][0, 0]

        cases = {
            "int64_tokens": sequence_case(
                lambda m, a: a.update(token_ids=a["token_ids"].astype(np.int64))
            ),
            "big_endian_tokens": sequence_case(
                lambda m, a: a.update(token_ids=a["token_ids"].astype(">i4"))
            ),
            "extra_array": sequence_case(lambda m, a: a.update(labels=a["token_ids"])),
            "missing_aliases": sequence_case(lambda m, a: m.pop("aliases")),
            "masked_turn_with_targets": sequence_case(
                lambda m, a: m["turns"][1].update(eligible=True)
            ),
            "masked_turn_without_reason": sequence_case(
                lambda m, a: m["turns"][1].update(audit_reason=None)
            ),
            "overlapping_turns": sequence_case(lambda m, a: m["turns"][1].update(start=19)),
            "unknown_turn_kind": sequence_case(
                lambda m, a: m["turns"][0].update(turn_kind="partial")
            ),
            "stop_not_eos": sequence_case(lambda m, a: m.update(stop_positions=[18])),
            "stop_outside_turn": sequence_case(
                lambda m, a: (m.update(stop_positions=[5]), a["token_ids"].__setitem__(5, 248046))
            ),
            "position_zero": targets_case(lambda m, a: a["target_positions"].__setitem__(0, 0)),
            "positions_unsorted": targets_case(
                lambda m, a: a["target_positions"].__setitem__(1, 10)
            ),
            "top_k_511": targets_case(
                lambda m, a: a.update(
                    top512_ids=a["top512_ids"][:, :511], top512_logits=a["top512_logits"][:, :511]
                )
            ),
            "duplicate_top_id": targets_case(
                lambda m, a: a["top512_ids"][0].__setitem__(1, a["top512_ids"][0, 0])
            ),
            "unsorted_top_logits": targets_case(
                lambda m, a: a["top512_logits"][0].__setitem__(1, 1e3)
            ),
            "tie_with_descending_ids": targets_case(tie_with_descending_ids),
            "top_id_out_of_vocab": targets_case(
                lambda m, a: a["top512_ids"][0].__setitem__(-1, RT.VOCAB_SIZE)
            ),
            "nonfinite_special": targets_case(
                lambda m, a: a["special_logits"][0].__setitem__(0, np.inf)
            ),
            "row_count_mismatch": targets_case(lambda m, a: a.update(logz_all=a["logz_all"][:-1])),
            "missing_binding": targets_case(lambda m, a: m.pop("binding")),
            "block_too_large": (
                "targets",
                dict(sequence_id="seq-a", block_index=0, binding=BINDING),
                synthetic_targets(np.arange(1, RT.TARGET_BLOCK_ROWS + 2)),
            ),
        }
        for name, (kind, metadata, arrays) in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                RT.encode_record(kind, metadata, arrays)


class TeacherDatasetTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "rollout_teacher_v1"

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip_windows_splits_and_selective_block_reads(self):
        positions = np.arange(50, 2550, dtype=np.int32)
        full = synthetic_targets(positions)
        blocks = [
            {k: v[i : i + RT.TARGET_BLOCK_ROWS] for k, v in full.items()}
            for i in range(0, len(positions), RT.TARGET_BLOCK_ROWS)
        ]
        long_arrays = dict(
            token_ids=(np.arange(2600) % 1000).astype(np.int32), target_positions=positions
        )

        def long_metadata(sequence_id):
            turn = dict(
                start=50,
                end=2550,
                call=3,
                turn_kind="final_truncated",
                eligible=True,
                audit_reason=None,
            )
            return dict(
                sequence_id=sequence_id,
                family="r2e_gym",
                turns=[turn],
                aliases=[dict(root="q")],
                stop_positions=[],
            )

        short_metadata, short_arrays = sequence_fixture()
        write_dataset(
            self.root,
            [
                ("train", long_metadata("seq-long"), long_arrays, blocks),
                (
                    "val",
                    short_metadata,
                    short_arrays,
                    [synthetic_targets(short_arrays["target_positions"])],
                ),
                ("train", long_metadata("seq-corrupt"), long_arrays, blocks),
            ],
            corrupt_block=("seq-corrupt", 0),
        )

        reader = ArrayRecordReader(str(self.root / "targets/seq-long.array_record"))
        self.assertIn("group_size:1,", reader.writer_options_string())
        self.assertIn(",uncompressed,", reader.writer_options_string())
        reader.close()

        self.assertEqual(len(RT.TeacherDataset(self.root)), 3)
        val = RT.TeacherDataset(self.root, "val")
        self.assertEqual(len(val), 1)
        metadata, arrays = val[0]
        self.assertEqual(metadata, short_metadata)
        assert_arrays_equal(self, arrays, short_arrays)

        train = RT.TeacherDataset(self.root, "train")
        self.assertEqual([e["sequence_id"] for e in train.sequences], ["seq-long", "seq-corrupt"])
        metadata, arrays = train.get_sequence("seq-long")
        self.assertEqual(metadata, long_metadata("seq-long"))
        assert_arrays_equal(self, arrays, long_arrays)
        assert_arrays_equal(self, train.get_targets("seq-long"), full)
        # Window crosses both 1024-row block boundaries (positions 1073|1074 and 2097|2098).
        keep = (positions >= 1000) & (positions < 2200)
        assert_arrays_equal(
            self, train.get_targets("seq-long", 1000, 2200), {k: v[keep] for k, v in full.items()}
        )
        empty = train.get_targets("seq-long", 0, 50)
        self.assertEqual(
            [empty[k].shape for k in empty],
            [(0,), (0, RT.TOP_K), (0, RT.TOP_K), (0, len(SPECIAL)), (0,), (0,)],
        )

        # Block 0 of seq-corrupt is truncated on disk: only windows touching it may fail.
        keep = positions >= 1074
        assert_arrays_equal(
            self, train.get_targets("seq-corrupt", 1074), {k: v[keep] for k, v in full.items()}
        )
        with self.assertRaises(ValueError):
            train.get_targets("seq-corrupt", 0, 60)

    def test_rejects_unbound_or_inconsistent_roots(self):
        metadata, arrays = sequence_fixture()
        write_dataset(
            self.root,
            [("train", metadata, arrays, [synthetic_targets(arrays["target_positions"])])],
        )
        manifest_path, catalog_path = (
            self.root / RT.MANIFEST_FILENAME,
            self.root / RT.CATALOG_FILENAME,
        )
        manifest, catalog = (
            json.loads(manifest_path.read_text()),
            json.loads(catalog_path.read_text()),
        )
        RT.TeacherDataset(self.root).get_targets("seq-a")

        def rewrite(path, value, edit):
            value = copy.deepcopy(value)
            edit(value)
            path.write_text(json.dumps(value))

        rewrite(catalog_path, catalog, lambda c: c.update(binding="other-scoring-binding"))
        dataset = RT.TeacherDataset(self.root)
        dataset.get_sequence("seq-a")
        with self.assertRaises(ValueError):
            dataset.get_targets("seq-a")
        init_cases = {
            "incomplete_after_drain": (catalog_path, catalog, lambda c: c.update(complete=False)),
            "different_snapshot": (
                manifest_path,
                manifest,
                lambda m: m["snapshot"].update(sha256="other-snapshot"),
            ),
            "different_source_binding": (
                catalog_path,
                catalog,
                lambda c: c.update(dataset_binding="other-source"),
            ),
            "catalog_target_count": (
                catalog_path,
                catalog,
                lambda c: c["sequences"][0].update(target_count=27),
            ),
            "catalog_escape_path": (
                catalog_path,
                catalog,
                lambda c: c["sequences"][0].update(path="../x"),
            ),
            "manifest_specials": (
                manifest_path,
                manifest,
                lambda m: m.update(special_ids=m["special_ids"][:-1]),
            ),
            "manifest_vocab": (manifest_path, manifest, lambda m: m.update(vocab_size=248077)),
        }
        for name, (path, value, edit) in init_cases.items():
            rewrite(catalog_path, catalog, lambda c: None)
            rewrite(manifest_path, manifest, lambda m: None)
            rewrite(path, value, edit)
            with self.subTest(name), self.assertRaises(ValueError):
                RT.TeacherDataset(self.root)


@requires_jax
class KLReferenceTest(unittest.TestCase):
    def test_normal_import_uses_working_copy(self):
        self.assertEqual(Path(omegalax.__file__).resolve(), REPO / "omegalax/__init__.py")
        self.assertEqual(Path(RT.__file__).resolve(), REPO / "omegalax/data/rollout_teacher.py")

    def test_matches_grouped_reference_with_special_overlap_and_gradients(self):
        rng = np.random.default_rng(0)
        special = np.asarray([3, 10, 20, 40, 41, 63], dtype=np.int32)
        teacher = (2 * rng.standard_normal((32, 64))).astype(np.float32)
        # Specials 10 and 40 always inside Top-8 as well, without dominating the mass.
        teacher[:, [10, 40]] = teacher.max(axis=1, keepdims=True) + [0.5, 0.25]
        student = (teacher + rng.standard_normal(teacher.shape)).astype(np.float32)
        targets = compact_targets(teacher, np.arange(1, 33), k=8, special_ids=special)
        inside = (targets["top512_ids"][:, :, None] == special[None, None, :]).any(axis=1)
        self.assertTrue(inside[:, [1, 3]].all())
        self.assertTrue((~inside).any(axis=1).all())

        def loss(logits):
            return RT.top512_special_tail_kl(logits, targets, special_ids=special).sum()

        expected_kl, expected_grad = grouped_reference(teacher, student, 8, special)
        actual_kl = np.asarray(RT.top512_special_tail_kl(student, targets, special_ids=special))
        self.assertGreater(expected_kl.min(), 1e-2)
        np.testing.assert_allclose(actual_kl, expected_kl, rtol=1e-4, atol=1e-6)
        grad = np.asarray(jax.grad(loss)(student))
        self.assertTrue(np.isfinite(grad).all())
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-3, atol=1e-6)

        # Teacher == student: KL and its gradient vanish.
        self.assertLess(
            np.abs(
                np.asarray(RT.top512_special_tail_kl(teacher, targets, special_ids=special))
            ).max(),
            1e-5,
        )
        grad = np.asarray(jax.grad(loss)(teacher))
        self.assertTrue(np.isfinite(grad).all())
        self.assertLess(np.abs(grad).max(), 1e-5)

    def test_segment_boundary_targets_use_previous_hidden(self):
        metadata, arrays = sequence_fixture()
        positions = arrays["target_positions"]
        # Student full logits per hidden position; teacher row j is the same model's hidden j-1.
        hidden_logits = 3 * np.random.default_rng(1).standard_normal(
            (48, RT.VOCAB_SIZE), dtype=np.float32
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rollout_teacher_v1"
            write_dataset(
                root,
                [
                    (
                        "train",
                        metadata,
                        arrays,
                        [compact_targets(hidden_logits[positions - 1], positions)],
                    )
                ],
            )
            dataset = RT.TeacherDataset(root, "train")
            seen = []
            for a in range(0, 48, 16):
                segment = hidden_logits[
                    a : a + 16
                ]  # hidden states a..a+15 of one statepassing segment
                targets = dataset.get_targets("seq-a", a + 1, a + 17)
                kl = RT.top512_special_tail_kl(
                    segment[targets["target_positions"] - 1 - a], targets
                )
                self.assertLess(float(np.abs(kl).max()), 1e-5)
                naive = dataset.get_targets("seq-a", a, a + 16)  # wrong shift: row j from hidden j
                wrong = RT.top512_special_tail_kl(segment[naive["target_positions"] - a], naive)
                self.assertGreater(float(np.min(wrong)), 1.0)
                seen.extend(targets["target_positions"].tolist())
        # Every target exactly once; 16 and 32 come from the last hidden of the previous segment.
        self.assertEqual(seen, positions.tolist())
        self.assertTrue({16, 32} <= set(seen))


if __name__ == "__main__":
    unittest.main()
