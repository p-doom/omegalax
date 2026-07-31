"""Tests for inline-records dataset building and Grain iteration."""

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
    load_compiled_metadata,
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


# build_records_from_chat measures messages in a `spawn` multiprocessing pool and
# ships measure_message to each worker via the pool initializer, so it must be
# picklable (importable by qualified name) -- a local lambda is not. This
# module-level stand-in counts every message as one token.
def _measure_one(message):
    return 1


_FAST_ITER_OPTS = dict(
    read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
    multiprocessing_options=make_grain_multiprocessing_options(
        num_workers=0, per_worker_buffer_size=1
    ),
)


class GrainPipelineTest(absltest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _expected_session_id(self, path: Path, line_num: int) -> str:
        return f"{path.stem}-{line_num:09d}"

    def _write_chat(self, tmpdir: Path, values: list[str], name: str = "chat") -> Path:
        """One conversation whose turns alternate user(value)/assistant."""
        src = tmpdir / f"{name}.jsonl"
        messages = []
        for value in values:
            messages.append({"role": "user", "content": value})
            messages.append({"role": "assistant", "content": "ok"})
        self._write_jsonl(src, [{"messages": messages}])
        return src

    def _build(self, src: Path, out_dir: Path, *, max_length: int = 2) -> Path:
        """Build an inline-records dataset from a chat.jsonl.

        ``max_length=2`` with ``_measure_one`` keeps each user+assistant pair in
        its own chunk, so every chunk carries an assistant turn (satisfying the
        builder's assistant-turn filter) and the tracking value stays readable at
        ``messages[0]``. ``overflow_mode="split"`` (not the ``"drop"`` default) is
        what turns a multi-turn conversation into several chunks instead of
        discarding it for exceeding the budget.
        """
        return build_records_from_chat(
            src,
            out_dir,
            max_length=max_length,
            measure_message=_measure_one,
            records_per_shard=8,
            num_workers=1,
            overflow_mode="split",
        )

    def test_build_records_from_chat_emits_self_contained_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = self._write_chat(tmp, ["10", "12", "14"])
            out = self._build(src, tmp / "records")

            metadata = load_compiled_metadata(out)
            # Inline records ARE the training examples -- no payload indirection.
            self.assertTrue(metadata["inline_records"])
            self.assertNotIn("payload_path", metadata)
            self.assertEqual(metadata["max_length"], 2)
            self.assertEqual(metadata["num_records"], 3)

    def test_make_grain_iterator_rejects_non_inline_records_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = self._write_chat(tmp, ["10"])
            out = self._build(src, tmp / "records")

            # Strip the inline-records marker to emulate a legacy chunk-index
            # dataset; the iterator must refuse it rather than mis-read records.
            meta_path = out / "metadata.json"
            metadata = json.loads(meta_path.read_text())
            del metadata["inline_records"]
            meta_path.write_text(json.dumps(metadata))

            with self.assertRaisesRegex(ValueError, "inline-records dataset"):
                make_grain_iterator(
                    out,
                    batch_size=1,
                    batch_fn=lambda batch: batch[0],
                    shuffle=False,
                    seed=0,
                    dp_size=1,
                    fsdp_size=1,
                    **_FAST_ITER_OPTS,
                )

    def test_build_records_from_chat_splits_conversation_into_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = self._write_chat(tmp, ["10", "12", "14"])
            out = self._build(src, tmp / "records")

            iterator = make_grain_iterator(
                out,
                batch_size=1,
                batch_fn=lambda batch: batch[0],
                shuffle=False,
                seed=0,
                dp_size=1,
                fsdp_size=1,
                **_FAST_ITER_OPTS,
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
            tmp = Path(tmpdir)
            src = self._write_chat(tmp, ["10", "12", "14", "16"])
            out = self._build(src, tmp / "records")

            iterator = make_grain_iterator(
                out,
                batch_size=2,
                batch_fn=_batch_starts,
                shuffle=False,
                seed=0,
                dp_size=1,
                fsdp_size=1,
                **_FAST_ITER_OPTS,
            )
            first_batch = next(iterator)
            self.assertEqual(first_batch["starts"].tolist(), [10, 12])

            save_dir = tmp / "ckpt"
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
                out,
                batch_size=2,
                batch_fn=_batch_starts,
                shuffle=False,
                seed=0,
                dp_size=1,
                fsdp_size=1,
                **_FAST_ITER_OPTS,
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
            tmp = Path(tmpdir)
            src = self._write_chat(tmp, ["10", "12", "14", "16"])
            out = self._build(src, tmp / "records")

            def collect(process_index: int) -> list[str]:
                with mock.patch("jax.process_index", return_value=process_index):
                    iterator = make_grain_iterator(
                        out,
                        batch_size=1,
                        batch_fn=lambda batch: batch[0],
                        shuffle=False,
                        seed=0,
                        dp_size=2,
                        fsdp_size=1,
                        **_FAST_ITER_OPTS,
                    )
                    return [next(iterator)["messages"][0]["content"] for _ in range(2)]

            starts0 = collect(0)
            starts1 = collect(1)
            self.assertEqual(starts0, ["10", "12"])
            self.assertEqual(starts1, ["14", "16"])
            self.assertEmpty(set(starts0).intersection(starts1))

    def test_make_grain_iterator_global_shuffle_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "chat.jsonl"
            # One conversation per tracking value, each a single user+assistant
            # pair -> exactly one chunk per conversation at max_length=2.
            self._write_jsonl(
                src,
                [
                    {
                        "messages": [
                            {"role": "user", "content": str(value)},
                            {"role": "assistant", "content": "ok"},
                        ],
                    }
                    for value in range(8)
                ],
            )
            out = self._build(src, tmp / "records")

            def collect_process_order(process_index: int, seed: int) -> list[str]:
                with mock.patch("jax.process_index", return_value=process_index):
                    iterator = make_grain_iterator(
                        out,
                        batch_size=1,
                        batch_fn=lambda batch: batch[0],
                        shuffle=True,
                        seed=seed,
                        dp_size=2,
                        fsdp_size=1,
                        **_FAST_ITER_OPTS,
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
