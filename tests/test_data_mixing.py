"""Tests for multi-source data mixing in ``make_grain_iterator``."""

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest
import numpy as np
import orbax.checkpoint as ocp

from omegalax.data.grain_pipeline import (
    BATCH_SOURCE_IDS_KEY,
    MixSource,
    build_chunk_index,
    compile_jsonl_to_arrayrecord,
    make_grain_iterator,
    make_grain_multiprocessing_options,
    make_grain_read_options,
    pop_source_ids,
)
from omegalax.trainers import checkpoint_utils


def _batch_starts(examples):
    out = {
        "starts": np.asarray(
            [int(ex["messages"][0]["content"]) for ex in examples], dtype=np.int32,
        ),
    }
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _build_chunked_source(
    tmpdir: Path, *, name: str, contents: list[str],
) -> Path:
    """Compile a single-message-per-session JSONL into a chunk-index dataset."""
    src = tmpdir / f"{name}.jsonl"
    _write_jsonl(
        src,
        [
            {"messages": [{"role": "user", "content": value}]}
            for value in contents
        ],
    )
    payload = compile_jsonl_to_arrayrecord(
        src,
        tmpdir / f"{name}_payload",
        messages_per_record=1,
        records_per_shard=8,
    )
    chunked = build_chunk_index(
        payload,
        tmpdir / f"{name}_chunked",
        max_length=1,
        measure_message=lambda _msg: 1,
        records_per_shard=8,
    )
    return chunked


_FAST_OPTS = dict(
    read_options=make_grain_read_options(num_threads=1, prefetch_buffer_size=1),
    multiprocessing_options=make_grain_multiprocessing_options(num_workers=0, per_worker_buffer_size=1),
)


class DataMixingTest(absltest.TestCase):
    def test_single_source_passes_through_unchanged(self):
        """A scalar path or single MixSource produces identical output."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            src = _build_chunked_source(tmpdir, name="a", contents=[str(i) for i in range(8)])

            it_path = make_grain_iterator(
                src, batch_size=2, batch_fn=_batch_starts,
                shuffle=False, seed=0, num_epochs=1, **_FAST_OPTS,
            )
            it_mix = make_grain_iterator(
                [MixSource(path=src, weight=1.0)], batch_size=2, batch_fn=_batch_starts,
                shuffle=False, seed=0, num_epochs=1, **_FAST_OPTS,
            )
            from_path = [next(it_path)["starts"].tolist() for _ in range(4)]
            from_mix = [next(it_mix)["starts"].tolist() for _ in range(4)]
            self.assertEqual(from_path, from_mix)

    def test_source_ids_attached_when_mixing(self):
        """The mixing iterator surfaces per-example source ids in each batch."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            a = _build_chunked_source(tmpdir, name="a", contents=[str(i) for i in range(0, 8)])
            b = _build_chunked_source(tmpdir, name="b", contents=[str(i) for i in range(100, 108)])

            it = make_grain_iterator(
                [MixSource(path=a, weight=0.5), MixSource(path=b, weight=0.5)],
                batch_size=4, batch_fn=_batch_starts,
                shuffle=False, seed=0, num_epochs=None, **_FAST_OPTS,
            )
            batch = next(it)
            self.assertIn(BATCH_SOURCE_IDS_KEY, batch)
            self.assertEqual(batch[BATCH_SOURCE_IDS_KEY].shape, (4,))
            self.assertEqual(batch[BATCH_SOURCE_IDS_KEY].dtype, np.int32)
            # Sanity: source 0 examples come from `a` (values <100), source 1 from `b` (>=100).
            for sid, start in zip(batch[BATCH_SOURCE_IDS_KEY], batch["starts"]):
                if int(sid) == 0:
                    self.assertLess(int(start), 100)
                else:
                    self.assertGreaterEqual(int(start), 100)

    def test_realized_mix_ratio_matches_weights(self):
        """Empirical per-source frequency converges to configured weights."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            a = _build_chunked_source(tmpdir, name="a", contents=[str(i) for i in range(64)])
            b = _build_chunked_source(tmpdir, name="b", contents=[str(i) for i in range(100, 164)])

            it = make_grain_iterator(
                [MixSource(path=a, weight=0.7), MixSource(path=b, weight=0.3)],
                batch_size=8, batch_fn=_batch_starts,
                shuffle=True, seed=42, num_epochs=None, **_FAST_OPTS,
            )
            counts = {0: 0, 1: 0}
            n_batches = 64
            for _ in range(n_batches):
                batch = next(it)
                for sid in batch[BATCH_SOURCE_IDS_KEY].tolist():
                    counts[sid] += 1
            total = counts[0] + counts[1]
            frac_a = counts[0] / total
            # Tolerance: 8*64 = 512 examples, std of binomial fraction ~= sqrt(0.21/512) ~= 0.02.
            # Allow 5σ on either side to keep the test robust.
            self.assertAlmostEqual(frac_a, 0.7, delta=0.10)

    def test_pop_source_ids_helper_strips_metadata(self):
        """``pop_source_ids`` removes the key so it never reaches the JIT cache."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            a = _build_chunked_source(tmpdir, name="a", contents=[str(i) for i in range(4)])
            it = make_grain_iterator(
                [MixSource(path=a, weight=1.0)],
                batch_size=2, batch_fn=_batch_starts,
                shuffle=False, seed=0, num_epochs=1, **_FAST_OPTS,
            )
            batch = next(it)
            self.assertIn(BATCH_SOURCE_IDS_KEY, batch)
            sids = pop_source_ids(batch)
            self.assertIsNotNone(sids)
            self.assertEqual(sids.shape, (2,))
            self.assertNotIn(BATCH_SOURCE_IDS_KEY, batch)

    def test_pop_source_ids_returns_none_if_absent(self):
        plain = {"x": np.zeros(2)}
        self.assertIsNone(pop_source_ids(plain))
        self.assertEqual(set(plain.keys()), {"x"})

    def test_zero_weight_source_never_sampled(self):
        """A weight=0 source is excluded from the realized mix."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            a = _build_chunked_source(tmpdir, name="a", contents=[str(i) for i in range(8)])
            b = _build_chunked_source(tmpdir, name="b", contents=[str(i) for i in range(100, 108)])
            it = make_grain_iterator(
                [MixSource(path=a, weight=1.0), MixSource(path=b, weight=0.0)],
                batch_size=4, batch_fn=_batch_starts,
                shuffle=True, seed=0, num_epochs=None, **_FAST_OPTS,
            )
            seen_b = False
            for _ in range(8):
                batch = next(it)
                if 1 in batch[BATCH_SOURCE_IDS_KEY].tolist():
                    seen_b = True
                    break
            self.assertFalse(seen_b)

    def test_negative_weight_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            a = _build_chunked_source(tmpdir, name="a", contents=["x"])
            with self.assertRaisesRegex(ValueError, "non-negative"):
                make_grain_iterator(
                    [MixSource(path=a, weight=-0.5), MixSource(path=a, weight=1.5)],
                    batch_size=1, batch_fn=_batch_starts,
                    shuffle=False, seed=0, num_epochs=1, **_FAST_OPTS,
                )

    def test_all_zero_weights_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            a = _build_chunked_source(tmpdir, name="a", contents=["x"])
            with self.assertRaisesRegex(ValueError, "weights must be > 0"):
                make_grain_iterator(
                    [MixSource(path=a, weight=0.0), MixSource(path=a, weight=0.0)],
                    batch_size=1, batch_fn=_batch_starts,
                    shuffle=False, seed=0, num_epochs=1, **_FAST_OPTS,
                )

    def test_mixing_different_max_length_rejected(self):
        """Mixing chunk indices built with different max_length is refused."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            # Build two payloads, then chunk-index them with different max_lengths.
            src_a = tmpdir / "a.jsonl"
            _write_jsonl(src_a, [{"messages": [{"role": "user", "content": "x"}]}])
            payload_a = compile_jsonl_to_arrayrecord(
                src_a, tmpdir / "a_payload", messages_per_record=1, records_per_shard=8,
            )
            chunked_a_short = build_chunk_index(
                payload_a, tmpdir / "a_chunked_short", max_length=1,
                measure_message=lambda _m: 1, records_per_shard=8,
            )
            chunked_a_long = build_chunk_index(
                payload_a, tmpdir / "a_chunked_long", max_length=2,
                measure_message=lambda _m: 1, records_per_shard=8,
            )
            with self.assertRaisesRegex(ValueError, "different max_length"):
                make_grain_iterator(
                    [MixSource(path=chunked_a_short), MixSource(path=chunked_a_long)],
                    batch_size=1, batch_fn=_batch_starts,
                    shuffle=False, seed=0, num_epochs=1, **_FAST_OPTS,
                )

    def test_checkpoint_restore_preserves_mix(self):
        """Save/restore mid-stream resumes the same mixed sequence (and source ids)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            a = _build_chunked_source(tmpdir, name="a", contents=[str(i) for i in range(16)])
            b = _build_chunked_source(tmpdir, name="b", contents=[str(i) for i in range(100, 116)])
            sources = [MixSource(path=a, weight=0.6), MixSource(path=b, weight=0.4)]

            it = make_grain_iterator(
                sources, batch_size=2, batch_fn=_batch_starts,
                shuffle=True, seed=7, num_epochs=None, **_FAST_OPTS,
            )
            first = next(it)["starts"].tolist()

            save_dir = tmpdir / "ckpt"
            registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
            registry.add("train_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)
            registry.add("train_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
            checkpoint_utils.register_grain_iterator_handler(registry)
            manager = ocp.CheckpointManager(
                save_dir,
                options=ocp.CheckpointManagerOptions(save_interval_steps=1, cleanup_tmp_directories=True),
                handler_registry=registry,
            )
            train_state = {"step": np.asarray(1, dtype=np.int32)}
            manager.save(1, args=checkpoint_utils.make_grain_save_args(train_state, it))
            manager.wait_until_finished()

            expected = [next(it)["starts"].tolist() for _ in range(4)]

            restored_it = make_grain_iterator(
                sources, batch_size=2, batch_fn=_batch_starts,
                shuffle=True, seed=7, num_epochs=None, **_FAST_OPTS,
            )
            abstract = {"step": np.asarray(0, dtype=np.int32)}
            restored = manager.restore(
                1, args=checkpoint_utils.make_grain_restore_args(abstract, restored_it),
            )
            restored_it = checkpoint_utils.restored_input_iter(restored)
            replay = [next(restored_it)["starts"].tolist() for _ in range(4)]

            self.assertEqual(replay, expected)
            self.assertNotEqual(first, expected[0])  # we did advance past the save point


if __name__ == "__main__":
    absltest.main()
