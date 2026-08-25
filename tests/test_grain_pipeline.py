"""Tests for Grain-backed inline-record building and iteration."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from absl.testing import absltest

from omegalax.data.grain_pipeline import (
    _write_chat_message_lengths,
    build_records_from_chat,
    make_grain_iterator,
    make_grain_multiprocessing_options,
    make_grain_read_options,
    resolve_arrayrecord_paths,
)
from omegalax.trainers import checkpoint_utils


def _batch_starts(examples):
    return {
        "starts": np.asarray([int(ex["messages"][0]["content"]) for ex in examples], dtype=np.int32)
    }


# build_records_from_chat measures messages in a `spawn` multiprocessing pool, so
# the measure_message callable must be picklable (importable by qualified name) --
# a local lambda is not. This module-level stand-in counts every message as one
# token.
def _measure_one(message):
    return 1


def _measure_declared(message):
    return message["measurement"]


_TEST_MEASUREMENT_CONTRACT = {
    "version": 1,
    "tokenizer_sha256": "a" * 64,
    "processor_sha256": None,
    "preprocessor_sha256": None,
}


def _measured(role, content, length=1, vision_tokens=0):
    return {
        "role": role,
        "content": content,
        "measurement": {
            "length": length,
            "supervised_tokens": length if role == "assistant" else 0,
            "vision_tokens": vision_tokens,
            "vision_patches": vision_tokens * 4,
            "num_images": int(vision_tokens > 0),
            "image_grid_thw": [[1, 2, 2]] if vision_tokens else [],
        },
    }


class GrainPipelineTest(absltest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _expected_session_id(self, path: Path, line_num: int) -> str:
        return f"{path.stem}-{line_num:09d}"

    def _read_records(self, records_dir: Path) -> list[dict]:
        count = json.loads((records_dir / "metadata.json").read_text())["num_records"]
        iterator = make_grain_iterator(
            records_dir,
            batch_size=1,
            batch_fn=lambda batch: batch[0],
            shuffle=False,
            seed=0,
            read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
            multiprocessing_options=make_grain_multiprocessing_options(
                num_workers=0, per_worker_buffer_size=1
            ),
            dp_size=1,
            fsdp_size=1,
        )
        return [next(iterator) for _ in range(count)]

    def test_malformed_message_list_rejects_under_optimized_python(self):
        code = """
import sys
from pathlib import Path
from omegalax.data.grain_pipeline import _iter_chat_conversations
try:
    list(_iter_chat_conversations(Path(sys.argv[1])))
except TypeError as error:
    if "Expected 'messages' to be a list" not in str(error):
        raise
else:
    raise RuntimeError('malformed messages were accepted')
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "chat.jsonl"
            source.write_text(json.dumps({"messages": {"role": "user"}}) + "\n")
            for optimized in (False, True):
                command = [sys.executable]
                if optimized:
                    command.append("-O")
                command.extend(["-c", code, str(source)])
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def _build_split(
        self,
        tmpdir,
        messages,
        max_length,
        marker=None,
        split_unit_ends=None,
        measure=_measure_one,
    ):
        src = Path(tmpdir) / "train.jsonl"
        cache = Path(tmpdir) / "message_lengths.jsonl"
        row = {"messages": messages}
        if marker is not None:
            row["_omegalax_carry_messages"] = marker
        if split_unit_ends is not None:
            row["_omegalax_split_unit_ends"] = split_unit_ends
        self._write_jsonl(src, [row])
        _write_chat_message_lengths(
            cache,
            {(0, i): measure(message) for i, message in enumerate(messages)},
            src,
            _TEST_MEASUREMENT_CONTRACT,
        )
        records_dir = build_records_from_chat(
            src,
            Path(tmpdir) / "records",
            max_length=max_length,
            measure_message=measure,
            records_per_shard=8,
            overflow_mode="split",
            message_lengths_path=cache,
            measurement_contract=_TEST_MEASUREMENT_CONTRACT,
        )
        return records_dir, self._read_records(records_dir)

    def test_system_turn_is_budgeted_and_carried_into_the_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "train.jsonl"
            system_message = {"role": "system", "content": "SYS"}
            self._write_jsonl(
                src,
                [
                    {
                        "messages": [
                            system_message,
                            {"role": "user", "content": "10"},
                            {"role": "assistant", "content": "11"},
                        ],
                    },
                ],
            )

            records_dir = build_records_from_chat(
                src,
                Path(tmpdir) / "records",
                max_length=3,
                measure_message=_measure_one,
                records_per_shard=8,
                measurement_contract=_TEST_MEASUREMENT_CONTRACT,
            )

            iterator = make_grain_iterator(
                records_dir,
                batch_size=1,
                batch_fn=lambda batch: batch[0],
                shuffle=False,
                seed=0,
                read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
                multiprocessing_options=make_grain_multiprocessing_options(
                    num_workers=0, per_worker_buffer_size=1
                ),
                dp_size=1,
                fsdp_size=1,
            )
            record = next(iterator)
            self.assertEqual(record["messages"][0], system_message)
            self.assertEqual(record["_omegalax_measured_length"], 3)

    def test_truncation_stats_spend_the_whole_window_and_balance(self):
        # A 78%-dropped build was read as a slicing bug because the summary
        # claimed `effective_max = max_length - system_tokens`. There is no such
        # reservation: the system turn is message 0 of the conversation and is
        # budgeted like any other, so the window is spent in full and
        # kept + dropped has to come back to what was measured.
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "train.jsonl"
            self._write_jsonl(
                src,
                [
                    {
                        "messages": [
                            {"role": "system", "content": "SYS"},
                            {"role": "user", "content": "10"},
                            {"role": "assistant", "content": "11"},
                            {"role": "user", "content": "12"},
                            {"role": "assistant", "content": "13"},
                        ],
                    },
                ],
            )

            records_dir = build_records_from_chat(
                src,
                Path(tmpdir) / "records",
                max_length=3,
                measure_message=_measure_one,
                records_per_shard=8,
                overflow_mode="truncate",
                measurement_contract=_TEST_MEASUREMENT_CONTRACT,
            )

            stats = json.loads((records_dir / "truncation_stats.json").read_text())
            self.assertEqual(stats["effective_max"], 3)
            self.assertEqual(stats["max_length"], 3)
            self.assertEqual(stats["tokens"]["total_measured"], 5)
            self.assertEqual(stats["tokens"]["kept"], 3)
            self.assertEqual(stats["tokens"]["dropped"], 2)
            self.assertEqual(
                stats["tokens"]["kept"] + stats["tokens"]["dropped"],
                stats["tokens"]["total_measured"],
            )

    def test_make_grain_iterator_requires_inline_records_dataset(self):
        # Legacy pre-inline-records datasets still sit on disk; iterating one would
        # feed the trainer chunk descriptors instead of examples.
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy = Path(tmpdir) / "legacy"
            legacy.mkdir()
            (legacy / "metadata.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "num_records": 1,
                        "num_shards": 1,
                        "shard_paths": ["part-00000.array_record"],
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "incomplete or has unsupported version"):
                make_grain_iterator(
                    legacy,
                    batch_size=1,
                    batch_fn=lambda batch: batch[0],
                    shuffle=False,
                    seed=0,
                    read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
                    multiprocessing_options=make_grain_multiprocessing_options(
                        num_workers=0, per_worker_buffer_size=1
                    ),
                    dp_size=1,
                    fsdp_size=1,
                )

    def test_build_records_from_chat_splits_oversized_conversation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "train.jsonl"
            self._write_jsonl(
                src,
                [
                    {
                        "messages": [
                            {"role": "user", "content": "10"},
                            {"role": "assistant", "content": "11"},
                            {"role": "user", "content": "12"},
                            {"role": "assistant", "content": "13"},
                            {"role": "user", "content": "14"},
                            {"role": "assistant", "content": "15"},
                        ],
                    },
                ],
            )

            records_dir = build_records_from_chat(
                src,
                Path(tmpdir) / "records",
                max_length=2,
                measure_message=_measure_one,
                records_per_shard=8,
                overflow_mode="split",
                measurement_contract=_TEST_MEASUREMENT_CONTRACT,
            )

            iterator = make_grain_iterator(
                records_dir,
                batch_size=1,
                batch_fn=lambda batch: batch[0],
                shuffle=False,
                seed=0,
                read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
                multiprocessing_options=make_grain_multiprocessing_options(
                    num_workers=0, per_worker_buffer_size=1
                ),
                dp_size=1,
                fsdp_size=1,
            )
            records = [next(iterator) for _ in range(3)]
            self.assertEqual([len(record["messages"]) for record in records], [2, 2, 2])
            self.assertEqual(
                [record["messages"][0]["content"] for record in records], ["10", "12", "14"]
            )
            expected_session_id = self._expected_session_id(src, 1)
            self.assertEqual(
                [record["_omegalax_session_id"] for record in records], [expected_session_id] * 3
            )

    def test_split_carries_a_marked_noninitial_nonsystem_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            messages = [
                {"role": role, "content": str(i)}
                for i, role in enumerate(["user", "assistant"] * 4)
            ]
            records_dir, records = self._build_split(tmpdir, messages, 3, marker=[2])
            self.assertEqual(
                [[message["content"] for message in record["messages"]] for record in records],
                [["0", "1", "2"], ["2", "3", "4"], ["2", "5", "6"], ["2", "7"]],
            )
            self.assertTrue(all("_omegalax_carry_messages" not in record for record in records))
            carry = json.loads((records_dir / "truncation_stats.json").read_text())["carry"]
            self.assertEqual(carry["carried_tokens"], 3)
            self.assertEqual(carry["respent_fraction"], round(3 / 11, 6))

    def test_split_reserves_carry_budget_and_truncates_when_it_cannot_fit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lengths = [2, 1, 1, 1, 4, 1]
            messages = [
                {
                    "role": "assistant" if i % 2 else "user",
                    "content": str(i),
                    "measurement": length,
                }
                for i, length in enumerate(lengths)
            ]
            records_dir, records = self._build_split(
                tmpdir, messages, 4, marker=[0], measure=_measure_declared
            )
            self.assertEqual(
                [[message["content"] for message in record["messages"]] for record in records],
                [["0", "1", "2"], ["0", "3"]],
            )
            self.assertTrue(all(record["_omegalax_measured_length"] <= 4 for record in records))
            stats = json.loads((records_dir / "truncation_stats.json").read_text())
            self.assertEqual(stats["sessions"]["truncated_single_message"], 1)
            self.assertEqual(stats["messages"]["dropped"], 2)
            self.assertEqual(stats["tokens"]["dropped"], 5)
            self.assertEqual(stats["supervision"]["dropped"], 1)

    def test_split_reports_dropped_unsupervised_slices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            messages = [
                {"role": "user", "content": "0", "measurement": 1},
                {"role": "assistant", "content": "1", "measurement": 1},
                {"role": "user", "content": "2", "measurement": 2},
                {"role": "assistant", "content": "3", "measurement": 1},
            ]
            records_dir, _ = self._build_split(tmpdir, messages, 2, measure=_measure_declared)
            stats = json.loads((records_dir / "truncation_stats.json").read_text())
            self.assertEqual(stats["tokens"]["dropped"], 2)
            self.assertEqual(
                stats["supervision"],
                {
                    "basis": "assistant_message_length_estimate",
                    "total_measured": 2,
                    "kept": 2,
                    "dropped": 0,
                    "repeated": 0,
                    "emitted": 2,
                    "dropped_fraction": 0.0,
                },
            )

    def test_split_keeps_marked_units_indivisible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            messages = [
                _measured("system", "context"),
                _measured("user", "observation-1"),
                _measured("assistant", "action-1"),
                _measured("user", "observation-2"),
                _measured("assistant", "action-2"),
                _measured("user", "observation-3"),
                _measured("assistant", "action-3"),
            ]
            _, records = self._build_split(
                tmpdir,
                messages,
                4,
                marker=[0],
                split_unit_ends=[1, 3, 5, 7],
                measure=_measure_declared,
            )
            self.assertEqual(
                [[message["content"] for message in record["messages"]] for record in records],
                [
                    ["context", "observation-1", "action-1"],
                    ["context", "observation-2", "action-2"],
                    ["context", "observation-3", "action-3"],
                ],
            )
            for record in records:
                noncarry = record["messages"][1:]
                self.assertEqual([message["role"] for message in noncarry], ["user", "assistant"])
                self.assertNotIn("_omegalax_split_unit_ends", record)

    def test_split_prefix_truncates_before_an_impossible_unit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            messages = [
                _measured("system", "context", 2),
                _measured("user", "observation-1"),
                _measured("assistant", "action-1"),
                _measured("user", "observation-2", 2),
                _measured("assistant", "action-2"),
            ]
            records_dir, records = self._build_split(
                tmpdir,
                messages,
                4,
                marker=[0],
                split_unit_ends=[1, 3, 5],
                measure=_measure_declared,
            )
            self.assertEqual(
                [[message["content"] for message in record["messages"]] for record in records],
                [["context", "observation-1", "action-1"]],
            )
            stats = json.loads((records_dir / "truncation_stats.json").read_text())
            self.assertEqual(stats["messages"]["dropped"], 2)
            self.assertEqual(stats["tokens"]["dropped"], 3)
            self.assertEqual(stats["supervision"]["dropped"], 1)

    def test_split_reports_repeated_supervision_from_carried_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            messages = [
                _measured("user", "observation-1"),
                _measured("assistant", "action-1"),
                _measured("user", "observation-2"),
                _measured("assistant", "action-2"),
                _measured("user", "observation-3"),
                _measured("assistant", "action-3"),
            ]
            records_dir, records = self._build_split(
                tmpdir,
                messages,
                3,
                marker=[1],
                split_unit_ends=[2, 4, 6],
                measure=_measure_declared,
            )
            self.assertEqual(len(records), 3)
            stats = json.loads((records_dir / "truncation_stats.json").read_text())
            self.assertEqual(
                stats["supervision"],
                {
                    "basis": "loss_mask",
                    "total_measured": 3,
                    "kept": 3,
                    "dropped": 0,
                    "repeated": 2,
                    "emitted": 5,
                    "dropped_fraction": 0.0,
                },
            )

    def test_split_rejects_unit_boundaries_that_do_not_partition_the_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            messages = [
                _measured("user", "observation"),
                _measured("assistant", "action"),
            ]
            with self.assertRaisesRegex(ValueError, "exclusive message offsets ending at 2"):
                self._build_split(
                    tmpdir,
                    messages,
                    4,
                    split_unit_ends=[1],
                    measure=_measure_declared,
                )

    def test_carried_image_measurements_remain_aligned_with_emitted_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image = [{"type": "image", "image": "frame.png"}]
            messages = [
                _measured("user", "0"),
                _measured("assistant", "1"),
                _measured("user", image, 2, vision_tokens=1),
                _measured("assistant", "3"),
                _measured("user", "4"),
                _measured("assistant", "5"),
            ]
            records_dir, records = self._build_split(
                tmpdir, messages, 4, marker=[2], measure=_measure_declared
            )
            self.assertEqual([record["_omegalax_measured_length"] for record in records], [4, 4, 3])
            self.assertTrue(
                all(sum(m["content"] == image for m in r["messages"]) == 1 for r in records)
            )
            token_stats = json.loads((records_dir / "token_stats.json").read_text())
            self.assertEqual(token_stats["per_chunk"]["vision_tokens"]["sum"], 3)
            self.assertEqual(token_stats["per_chunk"]["num_images"]["sum"], 3)

    def test_grain_iterator_checkpoint_restore_on_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "train.jsonl"
            self._write_jsonl(
                src,
                [
                    {
                        "messages": [
                            {"role": "user", "content": "10"},
                            {"role": "assistant", "content": "11"},
                            {"role": "user", "content": "12"},
                            {"role": "assistant", "content": "13"},
                            {"role": "user", "content": "14"},
                            {"role": "assistant", "content": "15"},
                            {"role": "user", "content": "16"},
                            {"role": "assistant", "content": "17"},
                        ],
                    },
                ],
            )
            records_dir = build_records_from_chat(
                src,
                Path(tmpdir) / "records",
                max_length=2,
                measure_message=_measure_one,
                records_per_shard=8,
                overflow_mode="split",
                measurement_contract=_TEST_MEASUREMENT_CONTRACT,
            )

            iterator = make_grain_iterator(
                records_dir,
                batch_size=2,
                batch_fn=_batch_starts,
                shuffle=False,
                seed=0,
                read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
                multiprocessing_options=make_grain_multiprocessing_options(
                    num_workers=0, per_worker_buffer_size=1
                ),
                dp_size=1,
                fsdp_size=1,
            )
            first_batch = next(iterator)
            self.assertEqual(first_batch["starts"].tolist(), [10, 12])

            save_dir = Path(tmpdir) / "ckpt"
            handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
            handler_registry.add(
                "train_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler
            )
            handler_registry.add(
                "train_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler
            )
            checkpoint_utils.register_grain_iterator_handler(handler_registry)
            manager = ocp.CheckpointManager(
                save_dir,
                options=ocp.CheckpointManagerOptions(
                    save_interval_steps=1, cleanup_tmp_directories=True
                ),
                handler_registry=handler_registry,
            )

            train_state = {"step": np.asarray(1, dtype=np.int32)}
            manager.save(1, args=checkpoint_utils.make_grain_save_args(train_state, iterator))
            manager.wait_until_finished()

            expected_next = next(iterator)
            self.assertEqual(expected_next["starts"].tolist(), [14, 16])

            restored_iterator = make_grain_iterator(
                records_dir,
                batch_size=2,
                batch_fn=_batch_starts,
                shuffle=False,
                seed=0,
                read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
                multiprocessing_options=make_grain_multiprocessing_options(
                    num_workers=0, per_worker_buffer_size=1
                ),
                dp_size=1,
                fsdp_size=1,
            )
            abstract_state = {"step": jax.ShapeDtypeStruct((), jnp.int32)}
            restored = manager.restore(
                1,
                args=checkpoint_utils.make_grain_restore_args(abstract_state, restored_iterator),
            )
            next_after_restore = next(restored["input_iter"])
            self.assertEqual(next_after_restore["starts"].tolist(), [14, 16])
            manager.close()

    def test_make_grain_iterator_shards_by_jax_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "train.jsonl"
            self._write_jsonl(
                src,
                [
                    {
                        "messages": [
                            {"role": "user", "content": "10"},
                            {"role": "assistant", "content": "11"},
                            {"role": "user", "content": "12"},
                            {"role": "assistant", "content": "13"},
                            {"role": "user", "content": "14"},
                            {"role": "assistant", "content": "15"},
                            {"role": "user", "content": "16"},
                            {"role": "assistant", "content": "17"},
                        ],
                    },
                ],
            )
            records_dir = build_records_from_chat(
                src,
                Path(tmpdir) / "records",
                max_length=2,
                measure_message=_measure_one,
                records_per_shard=8,
                overflow_mode="split",
                measurement_contract=_TEST_MEASUREMENT_CONTRACT,
            )

            with mock.patch("jax.process_index", return_value=0):
                iterator0 = make_grain_iterator(
                    records_dir,
                    batch_size=1,
                    batch_fn=lambda batch: batch[0],
                    shuffle=False,
                    seed=0,
                    read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
                    multiprocessing_options=make_grain_multiprocessing_options(
                        num_workers=0, per_worker_buffer_size=1
                    ),
                    dp_size=2,
                    fsdp_size=1,
                )
                records0 = [next(iterator0) for _ in range(2)]

            with mock.patch("jax.process_index", return_value=1):
                iterator1 = make_grain_iterator(
                    records_dir,
                    batch_size=1,
                    batch_fn=lambda batch: batch[0],
                    shuffle=False,
                    seed=0,
                    read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
                    multiprocessing_options=make_grain_multiprocessing_options(
                        num_workers=0, per_worker_buffer_size=1
                    ),
                    dp_size=2,
                    fsdp_size=1,
                )
                records1 = [next(iterator1) for _ in range(2)]

            starts0 = [record["messages"][0]["content"] for record in records0]
            starts1 = [record["messages"][0]["content"] for record in records1]
            self.assertEqual(starts0, ["10", "12"])
            self.assertEqual(starts1, ["14", "16"])
            self.assertEmpty(set(starts0).intersection(starts1))

    def test_make_grain_iterator_global_shuffle_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "train.jsonl"
            rows = []
            for value in range(8):
                rows.append(
                    {
                        "messages": [
                            {"role": "user", "content": str(value)},
                            {"role": "assistant", "content": "ok"},
                        ],
                    }
                )
            self._write_jsonl(src, rows)

            records_dir = build_records_from_chat(
                src,
                Path(tmpdir) / "records",
                # max_length=2 keeps the user+assistant pair in one chunk, so the
                # tracking value stays at messages[0] and the assistant-turn filter
                # is satisfied (one chunk per session).
                max_length=2,
                measure_message=_measure_one,
                records_per_shard=8,
                measurement_contract=_TEST_MEASUREMENT_CONTRACT,
            )

            def collect_process_order(process_index: int, seed: int) -> list[str]:
                with mock.patch("jax.process_index", return_value=process_index):
                    iterator = make_grain_iterator(
                        records_dir,
                        batch_size=1,
                        batch_fn=lambda batch: batch[0],
                        shuffle=True,
                        seed=seed,
                        read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
                        multiprocessing_options=make_grain_multiprocessing_options(
                            num_workers=0, per_worker_buffer_size=1
                        ),
                        dp_size=2,
                        fsdp_size=1,
                    )
                    return [next(iterator)["messages"][0]["content"] for _ in range(4)]

            process0_seed0 = collect_process_order(0, seed=0)
            process1_seed0 = collect_process_order(1, seed=0)
            process0_seed0_repeat = collect_process_order(0, seed=0)
            process1_seed0_repeat = collect_process_order(1, seed=0)
            process0_seed1 = collect_process_order(0, seed=1)
            process1_seed1 = collect_process_order(1, seed=1)

            self.assertEqual(process0_seed0, process0_seed0_repeat)
            self.assertEqual(process1_seed0, process1_seed0_repeat)
            self.assertEmpty(set(process0_seed0).intersection(process1_seed0))
            self.assertEqual(set(process0_seed0).union(process1_seed0), {str(i) for i in range(8)})
            self.assertNotEqual(process0_seed0 + process1_seed0, [str(i) for i in range(8)])
            self.assertNotEqual(process0_seed0 + process1_seed0, process0_seed1 + process1_seed1)

    def test_resolve_arrayrecord_paths_rejects_raw_jsonl_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "train.jsonl"
            self._write_jsonl(src, [{"messages": [{"role": "user", "content": "a"}]}])

            with self.assertRaisesRegex(ValueError, "metadata does not exist"):
                resolve_arrayrecord_paths(src)


if __name__ == "__main__":
    absltest.main()
