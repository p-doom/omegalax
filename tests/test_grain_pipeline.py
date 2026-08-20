"""Tests for Grain-backed inline-record building and iteration."""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from omegalax.data.grain_pipeline import (
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


class GrainPipelineTest(absltest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _expected_session_id(self, path: Path, line_num: int) -> str:
        return f"{path.stem}-{line_num:09d}"

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

            with self.assertRaisesRegex(ValueError, "inline-records dataset"):
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

            with self.assertRaisesRegex(ValueError, "compiled Grain shard"):
                resolve_arrayrecord_paths(src)


if __name__ == "__main__":
    absltest.main()
