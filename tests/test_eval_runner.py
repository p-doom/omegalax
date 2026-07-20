"""Tests for checkpoint-eval execution, resume, plotting, and comparison."""

from __future__ import annotations

from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=1"
).strip()

from absl.testing import absltest
from array_record.python.array_record_module import ArrayRecordWriter
import jax
import numpy as np
import orbax.checkpoint as ocp
import pyarrow.parquet as pq

from omegalax import export as export_lib
from omegalax.data.pretrain_data_set import (
    DOC_CHAIN_BINARY_HEADER,
    DOC_CHAIN_BINARY_MAGIC,
    DOC_CHAIN_FORMAT,
)
from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.evals import runner
from omegalax.evals.aggregation import aggregate_metrics
from omegalax.evals.manifest import build_full_document_manifest
from omegalax.evals.runner import (
    CheckpointEvalSpec,
    applicable_experiments,
    compare_checkpoint_results,
    result_dir_for_checkpoint,
    run_evals,
)
from omegalax.evals.storage import EvalRunIdentity, EvalRunStore, MetricRow
from omegalax.models.params_utils import save_hf_config
from omegalax.text import api as text_api
from omegalax.trainers import text as text_trainer
from tests.eval_test_utils import tiny_hybrid_config
from tests.pretrain_real_data_test_utils import test_temp_dir


_CONDITIONS = {
    "gdn": ("true_gdn", "zero_gdn", "shuffled_gdn"),
    "conv": ("true_conv", "zero_conv", "shuffled_conv"),
}
_METRIC_CONTRACT = "nll_sum_token_count_v1"
_PLOT_TYPES = ("in_horizon", "beyond_horizon", "exact_length", "heatmap")
_METRICS = {
    "gdn": ("nll", "gdn_state_gain", "gdn_semantic_gain"),
    "conv": ("nll", "conv_state_gain", "conv_semantic_gain"),
}


def _spec(checkpoint: str | Path, **overrides) -> CheckpointEvalSpec:
    values = {
        "c_train": 4,
        "pass_gdn_state": True,
        "gdn_layer_limit": None,
        "pass_conv_state": True,
        "pass_rope_positions": True,
        "pad_id": 0,
        "eos_id": 2,
    }
    values.update(overrides)
    return CheckpointEvalSpec(checkpoint, **values)


def _resolved(root: Path, step: int = 17):
    return SimpleNamespace(
        root=root.resolve(),
        step=step,
        step_path=(root / f"{step:06d}").resolve(),
        config_path=(root / "config.json").resolve(),
    )


def _write_val_root(
    root: Path,
    *,
    document_lengths: tuple[int, ...],
    chunk_length: int,
) -> None:
    bucket_path = root / "val" / "bucket_2k"
    bucket_path.mkdir(parents=True)
    shard_name = "part-00000.array_record"
    writer = ArrayRecordWriter(str(bucket_path / shard_name), "group_size:1")
    try:
        for doc_idx, doc_num_chunks in enumerate(document_lengths):
            start = 3 + doc_idx * 6
            token_ids = np.arange(
                start,
                start + doc_num_chunks * chunk_length,
                dtype=np.int32,
            )
            token_ids[-1] = 2
            header = json.dumps(
                {
                    "dataset_format": DOC_CHAIN_FORMAT,
                    "doc_id": f"doc-{doc_idx}",
                    "doc_token_count": int(token_ids.size),
                    "token_dtype": "int32",
                    "split": "val",
                    "bucket": "2k",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            writer.write(
                DOC_CHAIN_BINARY_MAGIC
                + DOC_CHAIN_BINARY_HEADER.pack(len(header), token_ids.size)
                + header
                + token_ids.tobytes()
            )
    finally:
        writer.close()

    (bucket_path / "metadata.json").write_text(
        json.dumps(
            {
                "version": 1,
                "dataset_format": DOC_CHAIN_FORMAT,
                "split": "val",
                "bucket": "2k",
                "segment_length_default": chunk_length,
                "num_records": len(document_lengths),
                "num_shards": 1,
                "shard_paths": [shard_name],
            },
            indent=2,
        )
        + "\n"
    )


def _build_manifest(
    tmpdir: Path,
    *,
    documents_per_length: int,
    lengths: tuple[int, ...] = (2, 3),
    chunk_length: int = 2,
):
    dataset_root = tmpdir / "raw"
    _write_val_root(
        dataset_root,
        document_lengths=tuple(length for length in lengths for _ in range(documents_per_length)),
        chunk_length=chunk_length,
    )
    manifest_path = tmpdir / "manifest.json"
    manifest = build_full_document_manifest(
        dataset_root,
        manifest_path,
        split="val",
        chunk_length=chunk_length,
        seed=0,
        sample_cap=documents_per_length,
        min_doc_chunks=min(lengths),
        max_doc_chunks=max(lengths),
    )
    return manifest_path, manifest


def _save_checkpoint(root: Path, *, step: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    model, cfg = text_api.init_model(
        tiny_hybrid_config(),
        jax.random.key(0),
        tp_size=1,
        fsdp_size=1,
        dp_size=1,
    )
    save_hf_config(export_lib.model_config_to_hf_dict(cfg), root)
    with mesh_rules(ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)):
        optimizer = text_trainer.build_optimizer(
            model,
            text_trainer.TrainConfig(num_steps=1),
        )
    train_state = text_trainer._train_state(optimizer, jax.random.key(17))
    registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    registry.add("train_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)
    manager = ocp.CheckpointManager(
        root,
        options=ocp.CheckpointManagerOptions(step_format_fixed_length=6),
        handler_registry=registry,
    )
    try:
        manager.save(
            step,
            args=ocp.args.Composite(train_state=ocp.args.PyTreeSave(train_state)),
        )
        manager.wait_until_finished()
    finally:
        manager.close()


def _eval_config(run_dir: Path) -> dict[str, object]:
    raw = json.loads((run_dir / "eval_config.json").read_text())
    return dict(raw.get("identity", raw)["eval_config"])


def _stored_identity(run_dir: Path) -> EvalRunIdentity:
    raw = json.loads((run_dir / "eval_config.json").read_text())
    return EvalRunIdentity(**dict(raw.get("identity", raw)))


def _read_metric_rows(run_dir: Path) -> tuple[MetricRow, ...]:
    return tuple(
        MetricRow(**row)
        for path in (run_dir / "raw").rglob("*.parquet")
        for row in pq.read_table(path).to_pylist()
    )


def _file_snapshot(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _identity(
    checkpoint_root: Path,
    *,
    step: int,
    c_train: int,
    dataset_hash: str = "sha256:dataset",
    manifest_hash: str = "sha256:manifest",
    document_cap: int | None = None,
    effective_document_cap: int = 2,
    metric_contract: str = _METRIC_CONTRACT,
    experiments: tuple[str, ...] = ("gdn",),
    pass_gdn_state: bool = True,
    gdn_layer_limit: int | None = None,
    pass_conv_state: bool = True,
    pass_rope_positions: bool = True,
    pad_id: int = 0,
    eos_id: int = 2,
    tp_size: int = 1,
    fsdp_size: int = 1,
    dp_size: int = 1,
    batch_size: int = 2,
    population_counts: dict[int, int] | None = None,
) -> EvalRunIdentity:
    if population_counts is None:
        population_counts = {2: effective_document_cap}
        shard_ids = list(range(effective_document_cap))
    else:
        shard_ids = [
            doc_num_chunks * 100 + doc_idx
            for doc_num_chunks, count in population_counts.items()
            for doc_idx in range(count)
        ]
    conditions = {experiment: list(_CONDITIONS[experiment]) for experiment in experiments}
    expected_shards = {
        experiment: {condition: shard_ids for condition in _CONDITIONS[experiment]}
        for experiment in experiments
    }
    return EvalRunIdentity(
        dataset_hash=dataset_hash,
        manifest_hash=manifest_hash,
        checkpoint_root=str(checkpoint_root.resolve()),
        checkpoint_step=step,
        code_hash="git:test",
        eval_config={
            "c_train": c_train,
            "pass_gdn_state": pass_gdn_state,
            "gdn_layer_limit": gdn_layer_limit,
            "pass_conv_state": pass_conv_state,
            "pass_rope_positions": pass_rope_positions,
            "pad_id": pad_id,
            "eos_id": eos_id,
            "tp_size": tp_size,
            "fsdp_size": fsdp_size,
            "dp_size": dp_size,
            "batch_size": batch_size,
            "document_cap": document_cap,
            "effective_document_cap": effective_document_cap,
            "experiments": list(experiments),
            "population_counts": {
                str(doc_num_chunks): count for doc_num_chunks, count in population_counts.items()
            },
            "metric_contract": metric_contract,
            "conditions_by_experiment": conditions,
            "expected_shards": expected_shards,
        },
    )


def _rows(experiment: str, condition: str, shard_id: int, *, all_chunks: bool = True):
    condition_offset = {
        "true_gdn": 0.0,
        "zero_gdn": 0.4,
        "shuffled_gdn": 0.2,
        "true_conv": 0.0,
        "zero_conv": 0.3,
        "shuffled_conv": 0.1,
    }[condition]
    doc_num_chunks = shard_id // 100 if shard_id >= 100 else 2
    chunk_positions = tuple(range(1, doc_num_chunks + 1)) if all_chunks else (1,)
    return tuple(
        MetricRow(
            experiment=experiment,
            condition=condition,
            bucket_idx=0,
            record_idx=shard_id,
            doc_id=f"doc-{shard_id}",
            doc_num_chunks=doc_num_chunks,
            chunk_position=chunk_position,
            nll_sum=(2.0 + (condition_offset if chunk_position > 1 else 0.0) + chunk_position / 10)
            * 2,
            token_count=2,
        )
        for chunk_position in chunk_positions
    )


def _write_expected(store: EvalRunStore, *, all_chunks: bool = True) -> None:
    config = _eval_config(store.run_dir)
    for experiment, conditions in dict(config["expected_shards"]).items():
        for condition, shard_ids in dict(conditions).items():
            for raw_shard_id in shard_ids:
                shard_id = int(raw_shard_id)
                if store.shard_is_complete(experiment, condition, shard_id):
                    continue
                store.write_shard(
                    experiment,
                    condition,
                    shard_id,
                    _rows(experiment, condition, shard_id, all_chunks=all_chunks),
                )


def _complete_store(root: Path, identity: EvalRunIdentity) -> EvalRunStore:
    store = EvalRunStore.open(root, identity.checkpoint_step, identity)
    _write_expected(store)
    store.mark_complete()
    return store


def _required_plot_stems(
    *,
    experiment: str,
    lengths: tuple[int, ...],
    plot_types: tuple[str, ...],
) -> set[str]:
    stems = set()
    for plot_type in plot_types:
        if plot_type in ("in_horizon", "beyond_horizon"):
            stems.update(f"{plot_type}_{metric}" for metric in _METRICS[experiment])
        elif plot_type == "exact_length":
            stems.update(
                f"exact_length_L{length}_{metric}"
                for length in lengths
                for metric in _METRICS[experiment]
            )
        elif plot_type == "heatmap":
            stems.update(f"heatmap_nll_{condition}" for condition in _CONDITIONS[experiment])
            stems.update(f"heatmap_{metric}" for metric in _METRICS[experiment] if metric != "nll")
        else:
            raise AssertionError(f"Unexpected plot type in test: {plot_type}")
    return stems


def _assert_plot_inventory(
    testcase: absltest.TestCase,
    root: Path,
    *,
    experiment: str,
    lengths: tuple[int, ...],
    plot_types: tuple[str, ...],
) -> None:
    required_stems = _required_plot_stems(
        experiment=experiment,
        lengths=lengths,
        plot_types=plot_types,
    )
    formats_by_stem: dict[str, set[str]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        testcase.assertIn(path.suffix, {".pdf", ".svg", ".png"}, msg=str(path))
        relative_stem = str(path.relative_to(root).with_suffix(""))
        formats_by_stem.setdefault(relative_stem, set()).add(path.suffix)
    testcase.assertEqual(set(formats_by_stem), required_stems)
    for stem in required_stems:
        testcase.assertEqual(formats_by_stem[stem], {".pdf", ".svg", ".png"})
        for suffix in (".pdf", ".svg", ".png"):
            path = root / f"{stem}{suffix}"
            testcase.assertTrue(path.is_file(), msg=str(path))
            testcase.assertGreater(path.stat().st_size, 0, msg=str(path))


def _assert_external_plot_root(testcase: absltest.TestCase, root: Path) -> None:
    files = tuple(path for path in root.rglob("*") if path.is_file())
    testcase.assertNotEmpty(files)
    for path in files:
        testcase.assertIn(path.suffix, {".pdf", ".svg", ".png"}, msg=str(path))
        testcase.assertTrue({"raw", "summary"}.isdisjoint(path.relative_to(root).parts))


def _assert_common_scale_inventory(
    testcase: absltest.TestCase,
    summary_dir: Path,
    *,
    lengths: tuple[int, ...],
) -> None:
    expected = {
        f"{experiment}_{'nll_all_conditions' if metric == 'nll' else metric}"
        "_exact_length_scale.json": (experiment, metric)
        for experiment, metrics in _METRICS.items()
        for metric in metrics
    }
    actual = {path.name: path for path in summary_dir.glob("*_exact_length_scale.json")}
    testcase.assertEqual(set(actual), set(expected))
    for name, (experiment, metric) in expected.items():
        metadata = json.loads(actual[name].read_text())
        testcase.assertEqual(metadata["experiment"], experiment)
        testcase.assertEqual(metadata["metric"], metric)
        testcase.assertEqual(metadata["view"], "exact_length")
        testcase.assertEqual(metadata["doc_num_chunks"], list(lengths))
        testcase.assertIsNone(metadata.get("condition"))
        if metric == "nll":
            testcase.assertEqual(
                set(metadata["conditions"]),
                set(_CONDITIONS[experiment]),
            )
        limits = tuple(float(value) for value in metadata["y_limits"])
        testcase.assertLen(limits, 2)
        testcase.assertTrue(all(np.isfinite(value) for value in limits))
        testcase.assertLess(limits[0], limits[1])
        if metric != "nll":
            testcase.assertAlmostEqual(-limits[0], limits[1])


def _inode_map(root: Path) -> dict[Path, tuple[int, int]]:
    return {
        path.relative_to(root): (stat.st_ino, stat.st_mtime_ns)
        for path in root.rglob("*.parquet")
        for stat in (path.stat(),)
    }


def _assert_inodes_preserved(
    testcase: absltest.TestCase,
    before: dict[Path, tuple[int, int]],
    after: dict[Path, tuple[int, int]],
) -> None:
    testcase.assertLessEqual(set(before), set(after))
    for relative_path, inode_and_mtime in before.items():
        testcase.assertEqual(
            after[relative_path],
            inode_and_mtime,
            msg=str(relative_path),
        )


class EvalRunnerTest(absltest.TestCase):
    def test_explicit_spec_applicability_and_canonical_result_path(self):
        signature = inspect.signature(CheckpointEvalSpec)
        for name in (
            "c_train",
            "pass_gdn_state",
            "gdn_layer_limit",
            "pass_conv_state",
            "pass_rope_positions",
            "pad_id",
            "eos_id",
        ):
            self.assertIs(signature.parameters[name].default, inspect.Parameter.empty)

        gdn_only = _spec(
            "/runs/misleading_component_name",
            c_train=1,
            pass_gdn_state=False,
            pass_conv_state=False,
            pass_rope_positions=False,
        )
        self.assertEqual(tuple(applicable_experiments(gdn_only)), ("gdn",))
        all_components = replace(
            gdn_only,
            checkpoint="/runs/name_claims_iid",
            pass_conv_state=True,
            pass_rope_positions=True,
        )
        self.assertEqual(set(applicable_experiments(all_components)), {"gdn", "conv"})

        with test_temp_dir() as tmp:
            resolved = _resolved(Path(tmp) / "checkpoint", step=123)
            self.assertEqual(
                result_dir_for_checkpoint(resolved),
                resolved.root / "evals" / "state_usage_v1" / "step_123",
            )
            with mock.patch.object(runner, "resolve_checkpoint", return_value=resolved):
                with self.assertRaisesRegex(ValueError, "(?i)plot.*type"):
                    run_evals(
                        (gdn_only,),
                        mode="plot",
                        plot_types=("in_horizon", "confidence_interval"),
                    )

    def test_run_evals_dispatches_all_subset_plot_and_compare_for_one_or_many(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            specs = (_spec(tmpdir / "c4"), _spec(tmpdir / "c6", c_train=6))
            resolved = {
                spec.checkpoint: _resolved(Path(spec.checkpoint), idx)
                for idx, spec in enumerate(specs, 3)
            }
            manifest_path = tmpdir / "manifest.json"
            roots = (tmpdir / "plots-a", tmpdir / "plots-b")

            with (
                mock.patch.object(runner, "run_checkpoint_eval") as run_one,
                mock.patch.object(runner, "plot_checkpoint_results") as plot_one,
                mock.patch.object(runner, "compare_checkpoint_results") as compare,
                mock.patch.object(
                    runner,
                    "resolve_checkpoint",
                    side_effect=lambda checkpoint: resolved[checkpoint],
                ),
            ):
                run_evals(
                    specs,
                    mode="all",
                    manifest_path=manifest_path,
                    plot_output_roots=roots,
                    document_cap=2,
                    plot_types=("in_horizon", "heatmap"),
                    tp_size=1,
                    fsdp_size=1,
                    dp_size=4,
                    batch_size=16,
                )
                self.assertEqual(run_one.call_count, 2)
                for spec, call in zip(specs, run_one.call_args_list, strict=True):
                    self.assertEqual(call.args[0], spec)
                    self.assertIsNone(call.kwargs["experiments"])
                    self.assertEqual(call.kwargs["document_cap"], 2)
                    self.assertEqual(call.kwargs["plot_output_roots"], roots)
                    self.assertEqual(call.kwargs["plot_types"], ("in_horizon", "heatmap"))
                    self.assertEqual(
                        tuple(
                            call.kwargs[name]
                            for name in ("tp_size", "fsdp_size", "dp_size", "batch_size")
                        ),
                        (1, 1, 4, 16),
                    )
                plot_one.assert_not_called()
                compare.assert_not_called()

                run_one.reset_mock()
                run_evals(
                    specs,
                    mode="subset",
                    manifest_path=manifest_path,
                    experiments=("gdn", "conv"),
                    plot_types=("exact_length",),
                    document_cap=None,
                )
                self.assertEqual(run_one.call_count, 2)
                for call in run_one.call_args_list:
                    self.assertEqual(call.kwargs["experiments"], ("gdn", "conv"))
                    self.assertEqual(call.kwargs["plot_types"], ("exact_length",))
                    self.assertIsNone(call.kwargs["document_cap"])

                run_one.reset_mock()
                plot_one.reset_mock()
                run_evals(
                    specs,
                    mode="plot",
                    experiments=("gdn",),
                    plot_types=("beyond_horizon", "heatmap"),
                    plot_output_roots=roots,
                )
                run_one.assert_not_called()
                self.assertEqual(
                    [call.args[0] for call in plot_one.call_args_list],
                    [result_dir_for_checkpoint(resolved[spec.checkpoint]) for spec in specs],
                )
                for call in plot_one.call_args_list:
                    self.assertEqual(call.kwargs["experiments"], ("gdn",))
                    self.assertEqual(call.kwargs["plot_types"], ("beyond_horizon", "heatmap"))

                run_one.reset_mock()
                compare.reset_mock()
                run_evals(
                    specs,
                    mode="compare",
                    experiments=("gdn",),
                    plot_types=("in_horizon",),
                    comparison_name="c4_vs_c6",
                    plot_output_roots=roots,
                )
                run_one.assert_not_called()
                compare.assert_called_once_with(
                    tuple(result_dir_for_checkpoint(resolved[spec.checkpoint]) for spec in specs),
                    experiments=("gdn",),
                    plot_types=("in_horizon",),
                    comparison_name="c4_vs_c6",
                    plot_output_roots=roots,
                )

    def test_synthetic_chunk_one_values_match_across_every_condition(self):
        for experiment, conditions in _CONDITIONS.items():
            chunk_one = [_rows(experiment, condition, 0)[0] for condition in conditions]
            self.assertLen({row.token_count for row in chunk_one}, 1)
            self.assertLen({row.nll_sum for row in chunk_one}, 1)
            if len(conditions) > 1:
                self.assertGreater(
                    len({_rows(experiment, condition, 0)[1].nll_sum for condition in conditions}),
                    1,
                )

    def test_incomplete_chunk_or_condition_is_not_plottable(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            cases = []
            truncated_root = tmpdir / "truncated"
            truncated_store = EvalRunStore.open(
                truncated_root,
                7,
                _identity(truncated_root, step=7, c_train=4),
            )
            _write_expected(truncated_store, all_chunks=False)
            cases.append((truncated_root, 7, "chunk|row|incomplete"))

            missing_root = tmpdir / "missing-condition"
            missing_store = EvalRunStore.open(
                missing_root,
                9,
                _identity(missing_root, step=9, c_train=4),
            )
            for condition in ("true_gdn", "zero_gdn"):
                for shard_id in (0, 1):
                    missing_store.write_shard(
                        "gdn",
                        condition,
                        shard_id,
                        _rows("gdn", condition, shard_id),
                    )
            cases.append((missing_root, 9, "shuffled_gdn|condition|incomplete"))

            for checkpoint_root, step, message in cases:
                before = _inode_map(checkpoint_root)
                with self.subTest(checkpoint=checkpoint_root.name):
                    with mock.patch.object(
                        runner,
                        "resolve_checkpoint",
                        return_value=_resolved(checkpoint_root, step=step),
                    ):
                        with self.assertRaisesRegex(ValueError, f"(?i)({message})"):
                            run_evals(
                                (_spec(checkpoint_root),),
                                mode="plot",
                                experiments=("gdn",),
                                plot_types=("exact_length",),
                            )
                    self.assertEqual(_inode_map(checkpoint_root), before)

    def test_beyond_horizon_plot_is_skipped_when_no_document_exceeds_c_train(self):
        with test_temp_dir() as tmp:
            checkpoint_root = Path(tmp) / "checkpoint-c8"
            store = _complete_store(
                checkpoint_root,
                _identity(
                    checkpoint_root,
                    step=7,
                    c_train=8,
                    population_counts={2: 2, 8: 2},
                ),
            )

            runner.plot_checkpoint_results(
                store.run_dir,
                experiments=("gdn",),
                plot_types=_PLOT_TYPES,
            )

            _assert_plot_inventory(
                self,
                store.run_dir / "plots" / "gdn",
                experiment="gdn",
                lengths=(2, 8),
                plot_types=("in_horizon", "exact_length", "heatmap"),
            )

    def test_non_donor_closed_document_cap_is_rejected_before_inference(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            manifest_path, manifest = _build_manifest(
                tmpdir / "dataset",
                documents_per_length=4,
            )
            for length in (2, 3):
                prefix = {
                    document.doc_id
                    for document in manifest.documents
                    if document.doc_num_chunks == length and document.sample_rank < 3
                }
                self.assertTrue(
                    any(
                        document.donor_doc_id not in prefix
                        for document in manifest.documents
                        if document.doc_num_chunks == length and document.sample_rank < 3
                    )
                )

            checkpoint_root = tmpdir / "checkpoint-without-model-files"
            resolved = _resolved(checkpoint_root, step=5)
            with mock.patch.object(runner, "resolve_checkpoint", return_value=resolved):
                with self.assertRaisesRegex(
                    ValueError,
                    "(?i)(donor|closed|document.cap|prefix)",
                ):
                    run_evals(
                        (_spec(checkpoint_root, c_train=2),),
                        mode="subset",
                        manifest_path=manifest_path,
                        experiments=("gdn",),
                        plot_types=("heatmap",),
                        document_cap=3,
                        tp_size=1,
                        fsdp_size=1,
                        dp_size=1,
                        batch_size=2,
                    )
            self.assertEmpty(tuple(checkpoint_root.rglob("*.parquet")))

    def test_tiny_pipeline_executes_all_conditions_and_resumes_without_rewrites(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            checkpoint_root = tmpdir / "checkpoint-c2"
            _save_checkpoint(checkpoint_root, step=5)
            manifest_path, manifest = _build_manifest(
                tmpdir / "dataset",
                documents_per_length=4,
            )
            external_root = tmpdir / "published"
            spec = _spec(checkpoint_root, c_train=2)
            topology = {
                "tp_size": 1,
                "fsdp_size": 1,
                "dp_size": 1,
                "batch_size": 2,
            }

            run_evals(
                (spec,),
                mode="subset",
                manifest_path=manifest_path,
                experiments=("gdn",),
                plot_types=("heatmap",),
                plot_output_roots=(external_root,),
                document_cap=2,
                **topology,
            )

            run_dir = checkpoint_root / "evals" / "state_usage_v1" / "step_5"
            pilot_config = _eval_config(run_dir)
            expected_identity = {
                "c_train": 2,
                "pass_gdn_state": True,
                "gdn_layer_limit": None,
                "pass_conv_state": True,
                "pass_rope_positions": True,
                "pad_id": 0,
                "eos_id": 2,
                "resolution_source": "manual_flags",
                "training_contract_hash": None,
                **topology,
            }
            for name, value in expected_identity.items():
                self.assertEqual(pilot_config[name], value, msg=name)
            self.assertEqual(pilot_config["document_cap"], 2)
            self.assertEqual(pilot_config["effective_document_cap"], 2)
            self.assertEqual(pilot_config["experiments"], ["gdn"])
            pilot_rows = tuple(
                MetricRow(**row)
                for path in (run_dir / "raw").rglob("*.parquet")
                for row in pq.read_table(path).to_pylist()
            )
            for length in (2, 3):
                self.assertLen(
                    {
                        row.doc_id
                        for row in pilot_rows
                        if row.condition == "true_gdn" and row.doc_num_chunks == length
                    },
                    2,
                )

            pilot_inodes = _inode_map(run_dir / "raw")
            run_evals(
                (spec,),
                mode="subset",
                manifest_path=manifest_path,
                experiments=("gdn", "conv"),
                plot_types=("heatmap",),
                plot_output_roots=(external_root,),
                document_cap=2,
                **topology,
            )
            component_inodes = _inode_map(run_dir / "raw")
            _assert_inodes_preserved(self, pilot_inodes, component_inodes)
            component_config = _eval_config(run_dir)
            self.assertEqual(set(component_config["experiments"]), set(_CONDITIONS))
            self.assertEqual(component_config["effective_document_cap"], 2)

            shutil.rmtree(run_dir / "summary")
            shutil.rmtree(run_dir / "plots")
            shutil.rmtree(external_root)
            run_evals(
                (spec,),
                mode="all",
                manifest_path=manifest_path,
                plot_types=("heatmap",),
                plot_output_roots=(external_root,),
                document_cap=None,
                **topology,
            )
            full_inodes = _inode_map(run_dir / "raw")
            _assert_inodes_preserved(self, component_inodes, full_inodes)
            self.assertGreater(len(full_inodes), len(component_inodes))
            full_config = _eval_config(run_dir)
            self.assertIsNone(full_config["document_cap"])
            self.assertEqual(full_config["effective_document_cap"], 4)
            self.assertEqual(set(full_config["experiments"]), set(_CONDITIONS))
            self.assertEqual(full_config["metric_contract"], _METRIC_CONTRACT)
            self.assertTrue(json.loads((run_dir / "status.json").read_text())["complete"])
            full_rows = _read_metric_rows(run_dir)
            expected_summary = aggregate_metrics(
                full_rows,
                population_counts={
                    count.doc_num_chunks: count.available for count in manifest.counts_by_length
                },
                c_train=2,
            )
            full_store = EvalRunStore.open(
                checkpoint_root,
                5,
                _stored_identity(run_dir),
            )
            self.assertEqual(full_store.read_summary(), expected_summary)
            self.assertEmpty(tuple((run_dir / "summary").glob("*_exact_length_scale.json")))
            self.assertNotEmpty(
                tuple(path for path in (run_dir / "summary").rglob("*") if path.is_file())
            )
            for experiment in _CONDITIONS:
                _assert_plot_inventory(
                    self,
                    run_dir / "plots" / experiment,
                    experiment=experiment,
                    lengths=(2, 3),
                    plot_types=("heatmap",),
                )

            immutable_changes = (
                ("c_train", replace(spec, c_train=3), topology),
                ("pass_gdn_state", replace(spec, pass_gdn_state=False), topology),
                ("gdn_layer_limit", replace(spec, gdn_layer_limit=1), topology),
                ("pass_conv_state", replace(spec, pass_conv_state=False), topology),
                ("pass_rope_positions", replace(spec, pass_rope_positions=False), topology),
                ("pad_id", replace(spec, pad_id=1), topology),
                ("eos_id", replace(spec, eos_id=3), topology),
                ("tp_size", spec, {**topology, "tp_size": 2}),
                ("fsdp_size", spec, {**topology, "fsdp_size": 2}),
                ("dp_size", spec, {**topology, "dp_size": 2}),
                ("batch_size", spec, {**topology, "batch_size": 4}),
            )
            for name, changed_spec, changed_topology in immutable_changes:
                with self.subTest(immutable=name):
                    with self.assertRaisesRegex(
                        ValueError,
                        "(?i)(identity|immutable|eval.config|config.*conflict)",
                    ):
                        run_evals(
                            (changed_spec,),
                            mode="subset",
                            manifest_path=manifest_path,
                            experiments=("gdn",),
                            plot_types=("heatmap",),
                            document_cap=None,
                            **changed_topology,
                        )
                    self.assertEqual(_inode_map(run_dir / "raw"), full_inodes)
                    self.assertEqual(_eval_config(run_dir), full_config)

            full_summary_snapshot = _file_snapshot(run_dir / "summary")
            full_status_bytes = (run_dir / "status.json").read_bytes()
            run_evals(
                (spec,),
                mode="subset",
                manifest_path=manifest_path,
                experiments=("gdn",),
                plot_types=("heatmap",),
                document_cap=2,
                **topology,
            )
            self.assertEqual(_inode_map(run_dir / "raw"), full_inodes)
            self.assertEqual(_eval_config(run_dir), full_config)
            self.assertEqual(_file_snapshot(run_dir / "summary"), full_summary_snapshot)
            self.assertEqual((run_dir / "status.json").read_bytes(), full_status_bytes)

            resolved = _resolved(checkpoint_root, step=5)
            shutil.rmtree(resolved.step_path)
            resolved.config_path.unlink()
            shutil.rmtree(run_dir / "summary")
            shutil.rmtree(run_dir / "plots")
            shutil.rmtree(external_root)
            with mock.patch.object(runner, "resolve_checkpoint", return_value=resolved):
                run_evals(
                    (spec,),
                    mode="plot",
                    experiments=("gdn", "conv"),
                    plot_types=("heatmap",),
                    plot_output_roots=(external_root,),
                )
            self.assertEqual(_inode_map(run_dir / "raw"), full_inodes)
            self.assertEqual(_eval_config(run_dir), full_config)
            regenerated_store = EvalRunStore.open(
                checkpoint_root,
                5,
                _stored_identity(run_dir),
            )
            self.assertEqual(regenerated_store.read_summary(), expected_summary)
            self.assertEmpty(tuple((run_dir / "summary").glob("*_exact_length_scale.json")))
            self.assertNotEmpty(
                tuple(path for path in (run_dir / "summary").rglob("*") if path.is_file())
            )
            for experiment in _CONDITIONS:
                _assert_plot_inventory(
                    self,
                    run_dir / "plots" / experiment,
                    experiment=experiment,
                    lengths=(2, 3),
                    plot_types=("heatmap",),
                )
                self.assertEmpty(tuple((run_dir / "plots" / experiment).glob("in_horizon_*")))

            with mock.patch.object(runner, "resolve_checkpoint", return_value=resolved):
                run_evals(
                    (spec,),
                    mode="plot",
                    experiments=("gdn", "conv"),
                    plot_types=_PLOT_TYPES,
                    plot_output_roots=(external_root,),
                )
            self.assertEqual(_inode_map(run_dir / "raw"), full_inodes)
            self.assertEqual(_eval_config(run_dir), full_config)
            for experiment in _CONDITIONS:
                for root in (
                    run_dir / "plots" / experiment,
                    external_root / checkpoint_root.name / "step_5" / experiment,
                ):
                    _assert_plot_inventory(
                        self,
                        root,
                        experiment=experiment,
                        lengths=(2, 3),
                        plot_types=_PLOT_TYPES,
                    )
            _assert_common_scale_inventory(
                self,
                run_dir / "summary",
                lengths=(2, 3),
            )
            _assert_external_plot_root(self, external_root)

            rows = _read_metric_rows(run_dir)
            self.assertEqual(
                {(row.experiment, row.condition) for row in rows},
                {
                    (experiment, condition)
                    for experiment, conditions in _CONDITIONS.items()
                    for condition in conditions
                },
            )
            expected_rows_per_condition = sum(
                count.selected * count.doc_num_chunks for count in manifest.counts_by_length
            )
            for experiment, conditions in _CONDITIONS.items():
                for condition in conditions:
                    self.assertLen(
                        [
                            row
                            for row in rows
                            if row.experiment == experiment and row.condition == condition
                        ],
                        expected_rows_per_condition,
                    )
            for doc_id in {row.doc_id for row in rows}:
                chunk_one = [
                    row for row in rows if row.doc_id == doc_id and row.chunk_position == 1
                ]
                self.assertLen(chunk_one, 6)
                self.assertLen({row.token_count for row in chunk_one}, 1)
                np.testing.assert_allclose(
                    [row.nll_sum for row in chunk_one],
                    chunk_one[0].nll_sum,
                    rtol=1e-6,
                    atol=1e-6,
                )
            self.assertEqual(
                aggregate_metrics(
                    rows,
                    population_counts={
                        count.doc_num_chunks: count.available for count in manifest.counts_by_length
                    },
                    c_train=2,
                ),
                expected_summary,
            )

    def test_three_checkpoint_comparison_covers_all_components_and_common_plot_types(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            checkpoint_roots = tuple(tmpdir / f"checkpoint-{idx}" for idx in range(3))
            model_configs = (
                {"c_train": 2, "pass_gdn_state": True, "gdn_layer_limit": None},
                {"c_train": 3, "pass_gdn_state": False, "gdn_layer_limit": 1},
                {"c_train": 4, "pass_gdn_state": True, "gdn_layer_limit": 2},
            )
            stores = tuple(
                _complete_store(
                    checkpoint_root,
                    _identity(
                        checkpoint_root,
                        step=step,
                        experiments=("gdn", "conv"),
                        population_counts={2: 2, 3: 2, 4: 2, 5: 2},
                        **model_config,
                    ),
                )
                for step, checkpoint_root, model_config in zip(
                    range(2, 5),
                    checkpoint_roots,
                    model_configs,
                    strict=True,
                )
            )
            specs = tuple(
                _spec(checkpoint_root, **model_config)
                for checkpoint_root, model_config in zip(
                    checkpoint_roots,
                    model_configs,
                    strict=True,
                )
            )
            resolved = {
                spec.checkpoint: _resolved(Path(spec.checkpoint), step)
                for spec, step in zip(specs, range(2, 5), strict=True)
            }
            roots = (tmpdir / "paper-a", tmpdir / "paper-b")
            common_plot_types = ("in_horizon", "beyond_horizon", "exact_length")

            with mock.patch.object(
                runner,
                "resolve_checkpoint",
                side_effect=lambda checkpoint: resolved[checkpoint],
            ):
                run_evals(
                    specs,
                    mode="compare",
                    experiments=("gdn", "conv"),
                    plot_types=common_plot_types,
                    comparison_name="three_models",
                    plot_output_roots=roots,
                )

            for root in (
                *(store.run_dir / "comparisons" / "three_models" for store in stores),
                *(plot_root / "comparisons" / "three_models" for plot_root in roots),
            ):
                for experiment in _CONDITIONS:
                    _assert_plot_inventory(
                        self,
                        root / experiment,
                        experiment=experiment,
                        lengths=(2, 3, 4, 5),
                        plot_types=common_plot_types,
                    )
            for root in roots:
                _assert_external_plot_root(self, root)

            run_dirs = tuple(store.run_dir for store in stores)
            with self.assertRaisesRegex(ValueError, "(?i)(heatmap|comparison)"):
                compare_checkpoint_results(
                    run_dirs,
                    experiments=("gdn",),
                    plot_types=("heatmap",),
                    comparison_name="no_multi_heatmap",
                )

            gdn_only_root = tmpdir / "gdn-only"
            gdn_only = _complete_store(
                gdn_only_root,
                _identity(
                    gdn_only_root,
                    step=9,
                    c_train=2,
                    experiments=("gdn",),
                    population_counts={2: 2, 3: 2, 4: 2, 5: 2},
                ),
            )
            compare_checkpoint_results(
                (stores[0].run_dir, gdn_only.run_dir),
                experiments=("gdn",),
                plot_types=("in_horizon",),
                comparison_name="shared_gdn_subset",
            )
            for run_dir in (stores[0].run_dir, gdn_only.run_dir):
                _assert_plot_inventory(
                    self,
                    run_dir / "comparisons" / "shared_gdn_subset" / "gdn",
                    experiment="gdn",
                    lengths=(2, 3, 4, 5),
                    plot_types=("in_horizon",),
                )
            with self.assertRaisesRegex(ValueError, "(?i)(conv|component|experiment|missing)"):
                compare_checkpoint_results(
                    (stores[0].run_dir, gdn_only.run_dir),
                    experiments=("conv",),
                    plot_types=("in_horizon",),
                    comparison_name="missing_conv",
                )

            baseline_identity = _identity(tmpdir / "baseline", step=1, c_train=4)
            baseline = _complete_store(tmpdir / "baseline", baseline_identity)
            conflicts = (
                (
                    replace(
                        baseline_identity,
                        checkpoint_root=str((tmpdir / "dataset").resolve()),
                        dataset_hash="sha256:other",
                    ),
                    "dataset",
                ),
                (
                    replace(
                        baseline_identity,
                        checkpoint_root=str((tmpdir / "manifest").resolve()),
                        manifest_hash="sha256:other",
                    ),
                    "manifest",
                ),
                (
                    replace(
                        baseline_identity,
                        checkpoint_root=str((tmpdir / "cap").resolve()),
                        eval_config={
                            **baseline_identity.eval_config,
                            "effective_document_cap": 1,
                            "population_counts": {"2": 1},
                            "expected_shards": {
                                "gdn": {condition: [0] for condition in _CONDITIONS["gdn"]}
                            },
                        },
                    ),
                    "cap",
                ),
                (
                    replace(
                        baseline_identity,
                        checkpoint_root=str((tmpdir / "metric").resolve()),
                        eval_config={
                            **baseline_identity.eval_config,
                            "metric_contract": "different_v2",
                        },
                    ),
                    "metric",
                ),
            )
            for conflict_identity, label in conflicts:
                conflicting = _complete_store(
                    Path(conflict_identity.checkpoint_root),
                    conflict_identity,
                )
                with self.subTest(conflict=label):
                    with self.assertRaisesRegex(ValueError, f"(?i){label}"):
                        compare_checkpoint_results(
                            (baseline.run_dir, conflicting.run_dir),
                            experiments=("gdn",),
                            plot_types=("in_horizon",),
                            comparison_name=f"bad_{label}",
                        )


if __name__ == "__main__":
    absltest.main()
