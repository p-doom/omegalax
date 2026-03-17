"""Tests for Grain-backed payload-block compilation and chunk-index iteration."""

import json
import os
import tempfile
from pathlib import Path

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

    def test_compile_jsonl_to_arrayrecord_blocks_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "train.jsonl"
            self._write_jsonl(
                src,
                [
                    {
                        "session_id": 1,
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
                        "session_id": 1,
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
                        "session_id": 7,
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
                measure_messages=lambda messages: len(messages),
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
            self.assertEqual([record["_omegalax_session_id"] for record in records], ["7", "7", "7"])

    def test_grain_iterator_checkpoint_restore_on_chunk_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "train.jsonl"
            self._write_jsonl(
                src,
                [
                    {
                        "session_id": "session-0",
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
                measure_messages=lambda messages: len(messages),
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
            self.assertEqual(expected_next["starts"].tolist(), [14])

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
            self.assertEqual(next_after_restore["starts"].tolist(), [14])
            manager.close()

    def test_resolve_arrayrecord_paths_rejects_raw_jsonl_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "train.jsonl"
            self._write_jsonl(src, [{"session_id": 1, "messages": [{"role": "user", "content": "a"}]}])

            with self.assertRaisesRegex(ValueError, "compiled Grain shard"):
                resolve_arrayrecord_paths(src)


if __name__ == "__main__":
    absltest.main()
