"""Tests for Grain-backed payload-block compilation and chunk-index iteration."""

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
    build_chunk_index,
    compile_jsonl_to_arrayrecord,
    make_grain_iterator,
    make_grain_multiprocessing_options,
    make_grain_read_options,
    resolve_arrayrecord_paths,
)
from omegalax.trainers import checkpoint_utils


def _batch_starts(examples):
    return {"starts": np.asarray([int(ex["messages"][0]["content"]) for ex in examples], dtype=np.int32)}


class GrainPipelineTest(absltest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _expected_session_id(self, path: Path, line_num: int) -> str:
        return f"{path.stem}-{line_num:09d}"

    def test_compile_jsonl_to_arrayrecord_blocks_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "train.jsonl"
            self._write_jsonl(
                src,
                [
                    {
                        "messages": [
                            {"role": "user", "content": "a"},
                            {"role": "assistant", "content": "b"},
                            {"role": "user", "content": "c"},
                            {"role": "assistant", "content": "d"},
                            {"role": "user", "content": "e"},
                        ],
                    },
                ],
            )

            out_dir = compile_jsonl_to_arrayrecord(
                src,
                Path(tmpdir) / "compiled",
                messages_per_record=2,
                records_per_shard=1,
            )
            shard_paths = resolve_arrayrecord_paths(out_dir)
            self.assertLen(shard_paths, 3)

    def test_make_grain_iterator_requires_chunk_index_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "train.jsonl"
            self._write_jsonl(
                src,
                [
                    {
                        "messages": [{"role": "user", "content": "a"}],
                    },
                ],
            )
            payload = compile_jsonl_to_arrayrecord(src, Path(tmpdir) / "payload", records_per_shard=1)

            with self.assertRaisesRegex(ValueError, "chunk-index dataset"):
                make_grain_iterator(
                    payload,
                    batch_size=1,
                    batch_fn=lambda batch: batch[0],
                    shuffle=False,
                    seed=0,
                    read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
                    multiprocessing_options=make_grain_multiprocessing_options(num_workers=0, per_worker_buffer_size=1),
                )

    def test_build_chunk_index_splits_across_payload_blocks(self):
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
                        ],
                    },
                ],
            )

            payload = compile_jsonl_to_arrayrecord(
                src,
                Path(tmpdir) / "payload",
                messages_per_record=2,
                records_per_shard=8,
            )
            chunked = build_chunk_index(
                payload,
                Path(tmpdir) / "chunked",
                max_length=2,
                measure_message=lambda message: 1,
                records_per_shard=8,
            )

            iterator = make_grain_iterator(
                chunked,
                batch_size=1,
                batch_fn=lambda batch: batch[0],
                shuffle=False,
                seed=0,
                read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
                multiprocessing_options=make_grain_multiprocessing_options(num_workers=0, per_worker_buffer_size=1),
            )
            records = [next(iterator) for _ in range(3)]
            self.assertEqual([len(record["messages"]) for record in records], [2, 2, 1])
            self.assertEqual([record["messages"][0]["content"] for record in records], ["10", "12", "14"])
            expected_session_id = self._expected_session_id(src, 1)
            self.assertEqual([record["_omegalax_session_id"] for record in records], [expected_session_id] * 3)

    def test_grain_iterator_checkpoint_restore_on_chunk_index(self):
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
            payload = compile_jsonl_to_arrayrecord(
                src,
                Path(tmpdir) / "payload",
                messages_per_record=2,
                records_per_shard=8,
            )
            chunked = build_chunk_index(
                payload,
                Path(tmpdir) / "chunked",
                max_length=2,
                measure_message=lambda message: 1,
                records_per_shard=8,
            )

            iterator = make_grain_iterator(
                chunked,
                batch_size=2,
                batch_fn=_batch_starts,
                shuffle=False,
                seed=0,
                read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
                multiprocessing_options=make_grain_multiprocessing_options(num_workers=0, per_worker_buffer_size=1),
            )
            first_batch = next(iterator)
            self.assertEqual(first_batch["starts"].tolist(), [10, 12])

            save_dir = Path(tmpdir) / "ckpt"
            handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
            handler_registry.add("train_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)
            handler_registry.add("train_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
            checkpoint_utils.register_grain_iterator_handler(handler_registry)
            manager = ocp.CheckpointManager(
                save_dir,
                options=ocp.CheckpointManagerOptions(save_interval_steps=1, cleanup_tmp_directories=True),
                handler_registry=handler_registry,
            )

            train_state = {"step": np.asarray(1, dtype=np.int32)}
            manager.save(1, args=checkpoint_utils.make_grain_save_args(train_state, iterator))
            manager.wait_until_finished()

            expected_next = next(iterator)
            self.assertEqual(expected_next["starts"].tolist(), [14, 16])

            restored_iterator = make_grain_iterator(
                chunked,
                batch_size=2,
                batch_fn=_batch_starts,
                shuffle=False,
                seed=0,
                read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
                multiprocessing_options=make_grain_multiprocessing_options(num_workers=0, per_worker_buffer_size=1),
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
            payload = compile_jsonl_to_arrayrecord(
                src,
                Path(tmpdir) / "payload",
                messages_per_record=2,
                records_per_shard=8,
            )
            chunked = build_chunk_index(
                payload,
                Path(tmpdir) / "chunked",
                max_length=2,
                measure_message=lambda message: 1,
                records_per_shard=8,
            )

            with mock.patch("jax.process_count", return_value=2):
                with mock.patch("jax.process_index", return_value=0):
                    iterator0 = make_grain_iterator(
                        chunked,
                        batch_size=1,
                        batch_fn=lambda batch: batch[0],
                        shuffle=False,
                        seed=0,
                        read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
                        multiprocessing_options=make_grain_multiprocessing_options(num_workers=0, per_worker_buffer_size=1),
                    )
                    records0 = [next(iterator0) for _ in range(2)]

                with mock.patch("jax.process_index", return_value=1):
                    iterator1 = make_grain_iterator(
                        chunked,
                        batch_size=1,
                        batch_fn=lambda batch: batch[0],
                        shuffle=False,
                        seed=0,
                        read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
                        multiprocessing_options=make_grain_multiprocessing_options(num_workers=0, per_worker_buffer_size=1),
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
                        ],
                    }
                )
            self._write_jsonl(src, rows)

            payload = compile_jsonl_to_arrayrecord(
                src,
                Path(tmpdir) / "payload",
                messages_per_record=1,
                records_per_shard=8,
            )
            chunked = build_chunk_index(
                payload,
                Path(tmpdir) / "chunked",
                max_length=1,
                measure_message=lambda message: 1,
                records_per_shard=8,
            )

            def collect_process_order(process_index: int, seed: int) -> list[str]:
                with mock.patch("jax.process_count", return_value=2):
                    with mock.patch("jax.process_index", return_value=process_index):
                        iterator = make_grain_iterator(
                            chunked,
                            batch_size=1,
                            batch_fn=lambda batch: batch[0],
                            shuffle=True,
                            seed=seed,
                            read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
                            multiprocessing_options=make_grain_multiprocessing_options(
                                num_workers=0, per_worker_buffer_size=1
                            ),
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
