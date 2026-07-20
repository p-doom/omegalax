"""Tests for resumable evaluation metric storage."""

from __future__ import annotations

from dataclasses import fields, replace
import json
import os
from pathlib import Path
from unittest import mock

from absl.testing import absltest
import pyarrow as pa
import pyarrow.parquet as pq

from omegalax.evals.aggregation import SummaryPoint
from omegalax.evals.storage import EvalRunIdentity, EvalRunStore, MetricRow
from tests.pretrain_real_data_test_utils import test_temp_dir


_CONDITIONS = {
    "gdn": ("true_gdn", "zero_gdn", "shuffled_gdn"),
    "conv": ("true_conv", "zero_conv", "shuffled_conv"),
}


class EvalStorageTest(absltest.TestCase):
    def _eval_config(self) -> dict:
        config = {
            "c_train": 4,
            "document_cap": 32,
            "effective_document_cap": 5,
            "experiments": ["gdn", "conv"],
            "metric_contract": "nll_sum_token_count_v1",
            "conditions_by_experiment": {
                experiment: list(conditions) for experiment, conditions in _CONDITIONS.items()
            },
            "population_counts": {"2": 8, "3": 5},
            "expected_shards": {
                experiment: {condition: [0] for condition in conditions}
                for experiment, conditions in _CONDITIONS.items()
            },
        }
        config["expected_shards"]["gdn"]["true_gdn"] = [2, 7, 19]
        config["expected_shards"]["conv"]["shuffled_conv"] = [5, 13]
        return config

    def _identity(self, checkpoint_root: Path, *, step: int = 17) -> EvalRunIdentity:
        return EvalRunIdentity(
            dataset_hash="sha256:dataset",
            manifest_hash="sha256:manifest",
            checkpoint_root=str(checkpoint_root.resolve()),
            checkpoint_step=step,
            code_hash="git:abc123",
            eval_config=self._eval_config(),
        )

    def _row(
        self,
        experiment: str = "gdn",
        condition: str = "true_gdn",
        *,
        nll_sum: float = 3.5,
    ) -> MetricRow:
        return MetricRow(
            experiment=experiment,
            condition=condition,
            bucket_idx=2,
            record_idx=11,
            doc_id="doc-11",
            doc_num_chunks=4,
            chunk_position=3,
            nll_sum=nll_sum,
            token_count=2,
        )

    def _snapshot(self, run_dir: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(run_dir)): path.read_bytes()
            for path in sorted(run_dir.rglob("*"))
            if path.is_file()
        }

    def test_open_creates_canonical_component_layout_and_eval_config(self):
        with test_temp_dir() as tmp:
            checkpoint_root = Path(tmp) / "checkpoint"
            identity = self._identity(checkpoint_root)

            store = EvalRunStore.open(checkpoint_root, 17, identity)

            self.assertEqual(
                store.run_dir,
                checkpoint_root / "evals" / "state_usage_v1" / "step_17",
            )
            self.assertEqual(store.config_path, store.run_dir / "eval_config.json")
            self.assertEqual(store.status_path, store.run_dir / "status.json")
            for relative_dir in (
                "raw/gdn",
                "raw/conv",
                "summary",
                "plots/gdn",
                "plots/conv",
                "comparisons",
            ):
                self.assertTrue((store.run_dir / relative_dir).is_dir(), relative_dir)
            self.assertEqual(
                {path.name for path in (store.run_dir / "raw").iterdir()},
                {"gdn", "conv"},
            )
            self.assertEqual(
                {path.name for path in (store.run_dir / "plots").iterdir()},
                {"gdn", "conv"},
            )

            config = json.loads(store.config_path.read_text())
            stored_identity = config.get("identity", config)
            for name in (
                "dataset_hash",
                "manifest_hash",
                "checkpoint_root",
                "checkpoint_step",
                "code_hash",
                "eval_config",
            ):
                self.assertEqual(stored_identity[name], getattr(identity, name))
            self.assertFalse(json.loads(store.status_path.read_text())["complete"])

    def test_mark_complete_requires_every_shard_declared_by_eval_config(self):
        with test_temp_dir() as tmp:
            checkpoint_root = Path(tmp) / "checkpoint"
            store = EvalRunStore.open(checkpoint_root, 17, self._identity(checkpoint_root))
            expected = [
                (experiment, condition, shard_id)
                for experiment, conditions in self._eval_config()["expected_shards"].items()
                for condition, shard_ids in conditions.items()
                for shard_id in shard_ids
            ]

            with self.assertRaisesRegex(ValueError, "missing|incomplete"):
                store.mark_complete()
            self.assertFalse(json.loads(store.status_path.read_text())["complete"])

            missing = ("gdn", "true_gdn", 19)
            written_paths = set()
            for experiment, condition, shard_id in expected:
                if (experiment, condition, shard_id) == missing:
                    continue
                path = store.write_shard(
                    experiment,
                    condition,
                    shard_id,
                    (self._row(experiment, condition),),
                )
                self.assertEqual(path.parent, store.run_dir / "raw" / experiment)
                self.assertNotIn(path, written_paths)
                written_paths.add(path)
                self.assertTrue(store.shard_is_complete(experiment, condition, shard_id))
            with self.assertRaisesRegex(ValueError, "missing|incomplete|19"):
                store.mark_complete()
            self.assertFalse(json.loads(store.status_path.read_text())["complete"])

            experiment, condition, shard_id = missing
            final_path = store.write_shard(
                experiment,
                condition,
                shard_id,
                (self._row(experiment, condition),),
            )
            self.assertNotIn(final_path, written_paths)
            self.assertTrue(store.shard_is_complete(experiment, condition, shard_id))
            store.mark_complete()
            self.assertTrue(json.loads(store.status_path.read_text())["complete"])
            reopened = EvalRunStore.open(checkpoint_root, 17, self._identity(checkpoint_root))
            self.assertTrue(json.loads(reopened.status_path.read_text())["complete"])

    def test_parquet_schema_filtering_and_compatible_resume(self):
        with test_temp_dir() as tmp:
            checkpoint_root = Path(tmp) / "checkpoint"
            identity = self._identity(checkpoint_root)
            store = EvalRunStore.open(checkpoint_root, 17, identity)
            true_rows = (
                self._row(),
                replace(self._row(), record_idx=12, doc_id="doc-12"),
            )

            self.assertFalse(store.shard_is_complete("gdn", "true_gdn", 2))
            shard_path = store.write_shard("gdn", "true_gdn", 2, true_rows)
            self.assertTrue(store.shard_is_complete("gdn", "true_gdn", 2))
            self.assertEqual(shard_path.parent, store.run_dir / "raw" / "gdn")
            self.assertEqual(shard_path.suffix, ".parquet")

            required_fields = {
                "experiment",
                "condition",
                "bucket_idx",
                "record_idx",
                "doc_id",
                "doc_num_chunks",
                "chunk_position",
                "nll_sum",
                "token_count",
            }
            metric_fields = {field.name for field in fields(MetricRow)}
            parquet_fields = set(pq.read_schema(shard_path).names)
            self.assertEqual(metric_fields, required_fields)
            self.assertEqual(parquet_fields, required_fields)

            original_bytes = shard_path.read_bytes()
            original_status = store.status_path.read_bytes()
            reopened = EvalRunStore.open(checkpoint_root, 17, identity)
            self.assertTrue(reopened.shard_is_complete("gdn", "true_gdn", 2))
            with (
                mock.patch.object(
                    pq,
                    "write_table",
                    side_effect=AssertionError("complete shard must not be rewritten"),
                ) as parquet_write,
                mock.patch(
                    "os.replace",
                    side_effect=AssertionError("complete shard must not be republished"),
                ) as atomic_replace,
            ):
                reused_path = reopened.write_shard("gdn", "true_gdn", 2, true_rows)
            parquet_write.assert_not_called()
            atomic_replace.assert_not_called()
            self.assertEqual(reused_path, shard_path)
            self.assertEqual(reused_path.read_bytes(), original_bytes)
            self.assertEqual(reopened.status_path.read_bytes(), original_status)

            zero_row = self._row(condition="zero_gdn", nll_sum=4.5)
            reopened.write_shard("gdn", "zero_gdn", 0, (zero_row,))
            self.assertEqual(reopened.read_rows(condition="true_gdn"), true_rows)
            self.assertEqual(
                reopened.read_rows(experiment="gdn", condition="zero_gdn"),
                (zero_row,),
            )
            self.assertCountEqual(reopened.read_rows(), (*true_rows, zero_row))

    def test_interrupted_pyarrow_write_leaves_no_final_or_corrupt_shard(self):
        with test_temp_dir() as tmp:
            checkpoint_root = Path(tmp) / "checkpoint"
            store = EvalRunStore.open(checkpoint_root, 17, self._identity(checkpoint_root))
            status_before = store.status_path.read_bytes()

            observed_temp_paths = []

            def write_partial_then_fail(_table, where, *args, **kwargs):
                del args, kwargs
                raw_path = where if isinstance(where, (str, os.PathLike)) else where.name
                path = Path(raw_path)
                observed_temp_paths.append(path)
                self.assertEqual(path.parent, store.run_dir / "raw" / "gdn")
                self.assertNotEqual(path.suffix, ".parquet")
                self.assertEmpty(tuple(path.parent.glob("*.parquet")))
                path.write_bytes(b"partial parquet data")
                raise OSError("injected parquet write failure")

            with (
                mock.patch.object(pq, "write_table", side_effect=write_partial_then_fail),
                self.assertRaisesRegex(OSError, "injected parquet write failure"),
            ):
                store.write_shard("gdn", "true_gdn", 2, (self._row(),))

            self.assertLen(observed_temp_paths, 1)
            self.assertFalse(store.shard_is_complete("gdn", "true_gdn", 2))
            self.assertEmpty(
                [path for path in (store.run_dir / "raw" / "gdn").rglob("*") if path.is_file()]
            )
            self.assertEmpty(store.read_rows(experiment="gdn", condition="true_gdn"))
            self.assertEqual(store.status_path.read_bytes(), status_before)

            with mock.patch("os.replace", wraps=os.replace) as atomic_replace:
                shard_path = store.write_shard("gdn", "true_gdn", 2, (self._row(),))
            shard_replaces = [
                call for call in atomic_replace.call_args_list if Path(call.args[1]) == shard_path
            ]
            self.assertLen(shard_replaces, 1)
            source, destination = (Path(value) for value in shard_replaces[0].args[:2])
            self.assertEqual(source.parent, destination.parent)
            self.assertNotEqual(source, destination)
            self.assertEqual(pq.read_table(shard_path).num_rows, 1)
            self.assertEqual(
                store.read_rows(experiment="gdn", condition="true_gdn"),
                (self._row(),),
            )

            reopened = EvalRunStore.open(
                checkpoint_root,
                17,
                self._identity(checkpoint_root),
            )
            self.assertTrue(reopened.shard_is_complete("gdn", "true_gdn", 2))
            with (
                mock.patch.object(
                    pq,
                    "write_table",
                    side_effect=AssertionError("reopened complete shard must not be rewritten"),
                ) as parquet_write,
                mock.patch(
                    "os.replace",
                    side_effect=AssertionError("reopened complete shard must not be republished"),
                ) as atomic_replace,
            ):
                self.assertEqual(
                    reopened.write_shard("gdn", "true_gdn", 2, (self._row(),)),
                    shard_path,
                )
            parquet_write.assert_not_called()
            atomic_replace.assert_not_called()

    def test_summary_is_atomically_persisted_and_roundtrips_after_reopen(self):
        with test_temp_dir() as tmp:
            checkpoint_root = Path(tmp) / "checkpoint"
            identity = self._identity(checkpoint_root)
            store = EvalRunStore.open(checkpoint_root, 17, identity)
            points = (
                SummaryPoint(
                    experiment="gdn",
                    metric="nll",
                    view="beyond_horizon",
                    doc_num_chunks=None,
                    chunk_position=2,
                    condition="true_gdn",
                    value=2.25,
                    token_count=37.5,
                ),
                SummaryPoint(
                    experiment="gdn",
                    metric="gdn_state_gain",
                    view="exact_length",
                    doc_num_chunks=4,
                    chunk_position=2,
                    condition=None,
                    value=0.125,
                    token_count=19.0,
                ),
            )
            summary_path = store.run_dir / "summary" / "metrics.parquet"

            observed_temp_paths = []

            def write_partial_then_fail(_table, where, *args, **kwargs):
                del args, kwargs
                raw_path = where if isinstance(where, (str, os.PathLike)) else where.name
                path = Path(raw_path)
                observed_temp_paths.append(path)
                self.assertEqual(path.parent, summary_path.parent)
                self.assertNotEqual(path, summary_path)
                path.write_bytes(b"partial summary")
                raise OSError("injected summary write failure")

            with (
                mock.patch.object(pq, "write_table", side_effect=write_partial_then_fail),
                self.assertRaisesRegex(OSError, "injected summary write failure"),
            ):
                store.write_summary(points)

            self.assertLen(observed_temp_paths, 1)
            self.assertFalse(summary_path.exists())
            self.assertEmpty(tuple(summary_path.parent.iterdir()))

            with mock.patch("os.replace", wraps=os.replace) as atomic_replace:
                self.assertEqual(store.write_summary(points), summary_path)
            summary_replaces = [
                call for call in atomic_replace.call_args_list if Path(call.args[1]) == summary_path
            ]
            self.assertLen(summary_replaces, 1)
            source, destination = (Path(value) for value in summary_replaces[0].args[:2])
            self.assertEqual(source.parent, destination.parent)
            self.assertNotEqual(source, destination)
            expected_summary_fields = {
                "experiment",
                "metric",
                "view",
                "doc_num_chunks",
                "chunk_position",
                "condition",
                "value",
                "token_count",
            }
            self.assertEqual(
                {field.name for field in fields(SummaryPoint)},
                expected_summary_fields,
            )
            self.assertEqual(
                set(pq.read_schema(summary_path).names),
                expected_summary_fields,
            )
            summary_schema = pq.read_schema(summary_path)
            expected_arrow_types = {
                "experiment": pa.string(),
                "metric": pa.string(),
                "view": pa.string(),
                "doc_num_chunks": pa.int64(),
                "chunk_position": pa.int64(),
                "condition": pa.string(),
                "value": pa.float64(),
                "token_count": pa.float64(),
            }
            expected_nullability = {
                "experiment": False,
                "metric": False,
                "view": False,
                "doc_num_chunks": True,
                "chunk_position": False,
                "condition": True,
                "value": False,
                "token_count": False,
            }
            for field_name in expected_summary_fields:
                self.assertEqual(
                    summary_schema.field(field_name).type,
                    expected_arrow_types[field_name],
                )
                self.assertEqual(
                    summary_schema.field(field_name).nullable,
                    expected_nullability[field_name],
                )

            original_bytes = summary_path.read_bytes()
            reopened = EvalRunStore.open(checkpoint_root, 17, identity)
            self.assertEqual(reopened.read_summary(), points)
            with (
                mock.patch.object(
                    pq,
                    "write_table",
                    side_effect=AssertionError("persisted summary must not be rewritten"),
                ) as parquet_write,
                mock.patch(
                    "os.replace",
                    side_effect=AssertionError("persisted summary must not be republished"),
                ) as atomic_replace,
            ):
                self.assertEqual(reopened.write_summary(points), summary_path)
            parquet_write.assert_not_called()
            atomic_replace.assert_not_called()
            self.assertEqual(summary_path.read_bytes(), original_bytes)

            updated_points = (
                replace(points[0], value=2.5, token_count=41.25),
                replace(points[1], value=0.25, token_count=23.0),
            )
            with (
                mock.patch(
                    "os.replace",
                    side_effect=OSError("injected summary publish failure"),
                ),
                self.assertRaisesRegex(OSError, "injected summary publish failure"),
            ):
                reopened.write_summary(updated_points)
            self.assertEqual(summary_path.read_bytes(), original_bytes)
            self.assertEqual(reopened.read_summary(), points)
            self.assertEqual(
                {path for path in summary_path.parent.iterdir() if path.is_file()},
                {summary_path},
            )

            with mock.patch("os.replace", wraps=os.replace) as atomic_replace:
                self.assertEqual(reopened.write_summary(updated_points), summary_path)
            update_replaces = [
                call for call in atomic_replace.call_args_list if Path(call.args[1]) == summary_path
            ]
            self.assertLen(update_replaces, 1)
            source, destination = (Path(value) for value in update_replaces[0].args[:2])
            self.assertEqual(source.parent, destination.parent)
            self.assertNotEqual(source, destination)
            self.assertNotEqual(summary_path.read_bytes(), original_bytes)

            updated_store = EvalRunStore.open(checkpoint_root, 17, identity)
            self.assertEqual(updated_store.read_summary(), updated_points)
            self.assertEqual(
                set(pq.read_schema(summary_path).names),
                expected_summary_fields,
            )
            self.assertEqual(pq.read_schema(summary_path), summary_schema)

    def test_shard_and_identity_conflicts_preserve_existing_files_and_status(self):
        with test_temp_dir() as tmp:
            checkpoint_root = Path(tmp) / "checkpoint"
            identity = self._identity(checkpoint_root)
            store = EvalRunStore.open(checkpoint_root, 17, identity)
            row = self._row()
            shard_path = store.write_shard("gdn", "true_gdn", 2, (row,))
            snapshot = self._snapshot(store.run_dir)

            with self.assertRaises(ValueError):
                store.write_shard(
                    "gdn",
                    "true_gdn",
                    2,
                    (replace(row, nll_sum=99.0),),
                )
            self.assertEqual(self._snapshot(store.run_dir), snapshot)
            self.assertEqual(
                shard_path.read_bytes(),
                snapshot[str(shard_path.relative_to(store.run_dir))],
            )

            with self.assertRaises(ValueError):
                store.write_shard(
                    "gdn",
                    "zero_gdn",
                    0,
                    (replace(row, condition="true_gdn"),),
                )
            self.assertEqual(self._snapshot(store.run_dir), snapshot)

            changed_config = self._eval_config()
            changed_config["c_train"] = 6
            conflicts = {
                "dataset_hash": replace(identity, dataset_hash="sha256:different"),
                "manifest_hash": replace(identity, manifest_hash="sha256:different"),
                "checkpoint_root": replace(
                    identity,
                    checkpoint_root=str((Path(tmp) / "other").resolve()),
                ),
                "checkpoint_step": replace(identity, checkpoint_step=18),
                "code_hash": replace(identity, code_hash="git:different"),
                "eval_config": replace(identity, eval_config=changed_config),
            }
            for field_name, conflicting_identity in conflicts.items():
                with self.subTest(field=field_name):
                    with self.assertRaises(ValueError):
                        EvalRunStore.open(checkpoint_root, 17, conflicting_identity)
                    self.assertEqual(self._snapshot(store.run_dir), snapshot)


if __name__ == "__main__":
    absltest.main()
