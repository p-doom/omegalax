"""Reference tests for the checkpoint state-usage eval pipeline on the real code path."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

from absl.testing import absltest
from array_record.python.array_record_module import ArrayRecordWriter
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pyarrow as pa
import pyarrow.parquet as pq

from omegalax import export as export_lib
from omegalax.data.pretrain_data_set import (
    COMPILED_METADATA_FILENAME,
    DOC_CHAIN_BINARY_HEADER,
    DOC_CHAIN_BINARY_MAGIC,
    DOC_CHAIN_FORMAT,
)
from omegalax.data.pretrain_statepassing import (
    STATEPASSING_CURRICULUM_INDEX_FORMAT,
    STATEPASSING_FIXED_C_INDEX_FORMAT,
)
from omegalax.data.rollout_teacher_pipeline import ROLLOUT_TEACHER_INDEX_FORMAT
from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.evals import experiments
from omegalax.evals.aggregation import SummaryPoint
from omegalax.evals.executor import ChainBatch, make_run_segment
from omegalax.evals.experiments import run_conv_experiment, run_gdn_experiment
from omegalax.evals.full_context import (
    compare_relative_nll,
    full_context_arrays,
    make_run_full_context,
)
from omegalax.evals.manifest import FullDocumentLoader, build_full_document_manifest
from omegalax.evals.runner import (
    CheckpointEvalRequest,
    resolve_checkpoint_eval_request,
    resolve_checkpoint_eval_requests,
    run_evals,
)
from omegalax.models.params_utils import save_hf_config
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api
from omegalax.text.chat import StatePassingConfig
from omegalax.trainers import text as text_trainer
from omegalax.trainers.pretrain import (
    _position_ids_zbt_from_chunk_idx,
    _select_conv_states_for_carry,
    _select_gdn_states_for_carry,
)
from omegalax.training_contract import (
    ManualEvalConfig,
    build_training_contract,
    ensure_training_contract,
    resolve_eval_config,
)
from tests.eval_test_utils import tiny_hybrid_config, two_document_chain_arrays
from tests.test_eval_experiments import _recording_run_chain
from tests.test_eval_runner import _build_manifest, _eval_config, _spec


_TARGET_SEED = 7
_REFERENCE_SEED = 11
_T = 9
_LEN2_COUNTS = (18, 14, 10, 17)
_LEN3_COUNTS = (27, 22, 19, 25, 21)
_TOKEN_COUNTS = _LEN2_COUNTS + _LEN3_COUNTS
_CONDITIONS = {
    "gdn": ("true_gdn", "zero_gdn", "shuffled_gdn"),
    "conv": ("true_conv", "zero_conv", "shuffled_conv"),
}


def _save_checkpoint(root: Path, *, step: int, seed: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    model, cfg = text_api.init_model(
        tiny_hybrid_config(), jax.random.key(seed), tp_size=1, fsdp_size=1, dp_size=1
    )
    save_hf_config(export_lib.model_config_to_hf_dict(cfg), root)
    with mesh_rules(ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)):
        optimizer = text_trainer.build_optimizer(model, text_trainer.TrainConfig(num_steps=1))
    train_state = text_trainer._train_state(optimizer, jax.random.key(17))
    registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    registry.add("train_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)
    manager = ocp.CheckpointManager(
        root,
        options=ocp.CheckpointManagerOptions(step_format_fixed_length=6),
        handler_registry=registry,
    )
    try:
        manager.save(step, args=ocp.args.Composite(train_state=ocp.args.PyTreeSave(train_state)))
        manager.wait_until_finished()
    finally:
        manager.close()


def _reference_model(seed: int):
    model, _ = text_api.init_model(
        tiny_hybrid_config(), jax.random.key(seed), tp_size=1, fsdp_size=1, dp_size=1
    )
    set_attn_backend(model, "xla")
    return model


def _build_ragged_manifest(tmpdir: Path, token_counts, *, chunk_length: int, sample_cap: int):
    rng = np.random.default_rng(0)
    bucket_path = tmpdir / "raw" / "val" / "bucket_2k"
    bucket_path.mkdir(parents=True)
    writer = ArrayRecordWriter(str(bucket_path / "part-00000.array_record"), "group_size:1")
    written = {}
    try:
        for doc_idx, token_count in enumerate(token_counts):
            token_ids = rng.integers(3, 64, size=token_count, dtype=np.int32)
            written[f"doc-{doc_idx}"] = token_ids
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
                "num_records": len(token_counts),
                "num_shards": 1,
                "shard_paths": ["part-00000.array_record"],
            }
        )
    )
    lengths = sorted({-(-count // chunk_length) for count in token_counts})
    manifest_path = tmpdir / "manifest.json"
    manifest = build_full_document_manifest(
        tmpdir / "raw",
        manifest_path,
        split="val",
        chunk_length=chunk_length,
        seed=0,
        sample_cap=sample_cap,
        min_doc_chunks=min(lengths),
        max_doc_chunks=max(lengths),
    )
    return manifest_path, manifest, written


def _read_rows(run_dir: Path, selection: str = "all") -> dict[tuple, tuple[float, int]]:
    rows = {}
    for path in sorted((run_dir / "raw" / selection).rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            key = (row["experiment"], row["condition"], row["doc_id"], row["chunk_position"])
            assert key not in rows, key
            rows[key] = (row["nll_sum"], row["token_count"])
    return rows


def _copy_checkpoint(source: Path, target: Path) -> Path:
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("evals"))
    return target


def _run_state_usage(root: Path, manifest_path: Path, *, batch_size: int, dp_size: int = 1):
    run_evals(
        (_spec(root, c_train=2),),
        mode="all",
        manifest_path=manifest_path,
        plot_types=("heatmap",),
        dp_size=dp_size,
        batch_size=batch_size,
    )
    return root / "evals" / "state_usage_v2" / "step_1"


def _write_index(directory: Path, metadata: dict) -> Path:
    directory.mkdir(parents=True)
    (directory / COMPILED_METADATA_FILENAME).write_text(json.dumps(metadata))
    return directory


def _assert_rows_close(testcase, actual, expected, *, atol):
    testcase.assertEqual(set(actual), set(expected))
    for key, (expected_nll, expected_count) in expected.items():
        actual_nll, actual_count = actual[key]
        testcase.assertEqual(actual_count, expected_count, key)
        testcase.assertLessEqual(abs(actual_nll - expected_nll), atol, key)


@nnx.jit
def _reference_forward(model, token_ids_BT, attention_mask_BT, position_ids_ZBT, gdn, conv):
    hidden, _, gdn_out, conv_out = text_api.forward_with_gdn_state(
        model,
        token_ids_BT,
        pad_id=0,
        cfg=model.text.cfg,
        attention_mask_BT=attention_mask_BT,
        initial_gdn_states=gdn,
        initial_conv_states=conv,
        position_ids_ZBT=position_ids_ZBT,
        return_conv_states=True,
    )
    return hidden, gdn_out, conv_out


def _reference_logp(model, document, segment_idx, gdn, conv, *, offset):
    chunk_length = document.token_ids_CT.shape[-1]
    positions = jnp.stack([jnp.arange(chunk_length, dtype=jnp.int32)[None] + offset] * 3)
    hidden, gdn_out, conv_out = _reference_forward(
        model,
        jnp.asarray(document.token_ids_CT[segment_idx][None]),
        jnp.asarray(document.attention_mask_CT[segment_idx][None]),
        positions,
        gdn,
        conv,
    )
    logits = np.asarray(hidden[0], dtype=np.float64) @ np.asarray(
        model.output_weight(), dtype=np.float64
    )
    logits -= logits.max(axis=-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True)), gdn_out, conv_out


def _reference_nll(logp, token_ids_T, mask_T) -> tuple[float, int]:
    targets = [t for t in range(1, len(token_ids_T)) if mask_T[t]]
    return -sum(logp[t - 1, int(token_ids_T[t])] for t in targets), len(targets)


def _reference_logps(model, manifest, experiment_names, *, pass_conv, pass_rope, layer_limit):
    """Batch-of-one python chain: {(experiment, condition, doc_id, chunk): (logp, document)}."""

    def limit(states):
        if states is None or layer_limit == 0:
            return None
        if layer_limit is None:
            return states
        return tuple(s if i < layer_limit else jnp.zeros_like(s) for i, s in enumerate(states))

    loader = FullDocumentLoader(manifest)
    entries = {entry.doc_id: entry for entry in manifest.documents}
    documents = {doc_id: loader.load_document(entry) for doc_id, entry in entries.items()}
    chunk_length = manifest.chunk_length
    true_logps, true_states = {}, {}
    for doc_id, document in documents.items():
        gdn = conv = None
        for segment_idx in range(entries[doc_id].doc_num_chunks):
            logp, gdn, conv = _reference_logp(
                model,
                document,
                segment_idx,
                limit(gdn),
                limit(conv) if pass_conv else None,
                offset=segment_idx * chunk_length if pass_rope else 0,
            )
            true_logps[doc_id, segment_idx] = logp
            true_states[doc_id, segment_idx] = (gdn, conv)

    out = {}
    for doc_id, document in documents.items():
        donor_id = entries[doc_id].donor_doc_id
        for segment_idx in range(entries[doc_id].doc_num_chunks):
            for experiment in experiment_names:
                true_condition, zero_condition, shuffled_condition = _CONDITIONS[experiment]
                out[experiment, true_condition, doc_id, segment_idx + 1] = (
                    true_logps[doc_id, segment_idx],
                    document,
                )
                if segment_idx == 0:
                    controls = {zero_condition: (None, None), shuffled_condition: (None, None)}
                else:
                    own_gdn, own_conv = true_states[doc_id, segment_idx - 1]
                    donor_gdn, donor_conv = true_states[donor_id, segment_idx - 1]
                    conv = limit(own_conv) if pass_conv else None
                    if experiment == "gdn":
                        controls = {
                            zero_condition: (None, conv),
                            shuffled_condition: (limit(donor_gdn), conv),
                        }
                    else:
                        controls = {
                            zero_condition: (limit(own_gdn), None),
                            shuffled_condition: (limit(own_gdn), limit(donor_conv)),
                        }
                for condition, (gdn, conv) in controls.items():
                    logp, _, _ = _reference_logp(
                        model,
                        document,
                        segment_idx,
                        gdn,
                        conv,
                        offset=segment_idx * chunk_length if pass_rope else 0,
                    )
                    out[experiment, condition, doc_id, segment_idx + 1] = (logp, document)
    return out


class EvalPipelineReferenceTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmp.name)
        cls.base_checkpoint = cls.tmpdir / "base"
        _save_checkpoint(cls.base_checkpoint, step=1, seed=_TARGET_SEED)
        cls.manifest_path, cls.manifest, cls.written = _build_ragged_manifest(
            cls.tmpdir / "data", _TOKEN_COUNTS, chunk_length=_T, sample_cap=5
        )
        cls.main_root = _copy_checkpoint(cls.base_checkpoint, cls.tmpdir / "main")
        cls.main_run_dir = _run_state_usage(cls.main_root, cls.manifest_path, batch_size=4)
        cls.main_rows = _read_rows(cls.main_run_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        super().tearDownClass()

    # T1: run_evals NLL and token counts equal a float64 batch-of-one reference.
    def test_pipeline_matches_independent_per_document_reference(self):
        self.assertLen(self.manifest.documents, len(_TOKEN_COUNTS))
        loader = FullDocumentLoader(self.manifest)
        for entry in self.manifest.documents:
            document = loader.load_document(entry)
            tokens = self.written[entry.doc_id]
            np.testing.assert_array_equal(document.token_ids_CT.reshape(-1)[: tokens.size], tokens)
            self.assertEqual(int(document.attention_mask_CT.sum()), tokens.size)
        reference = _reference_logps(
            _reference_model(_TARGET_SEED),
            self.manifest,
            ("gdn", "conv"),
            pass_conv=True,
            pass_rope=True,
            layer_limit=None,
        )
        expected = {
            key: _reference_nll(
                logp, document.token_ids_CT[key[3] - 1], document.loss_mask_CT[key[3] - 1]
            )
            for key, (logp, document) in reference.items()
        }
        _assert_rows_close(self, self.main_rows, expected, atol=5e-4)
        counts = {count for _, count in expected.values()}
        self.assertTrue({0, _T - 1}.issubset(counts) and len(counts) > 2, counts)
        for experiment, (true_condition, zero_condition, _) in _CONDITIONS.items():
            gap = max(
                abs(
                    self.main_rows[key][0]
                    - self.main_rows[(experiment, zero_condition) + key[2:]][0]
                )
                for key in self.main_rows
                if key[1] == true_condition and key[3] > 1
            )
            self.assertGreater(gap, 1e-3, experiment)

    # T4: eval hand-over (layer limit, conv passing, RoPE offsets) equals the trainer helpers.
    def test_eval_handover_equals_trainer_handover(self):
        arrays = two_document_chain_arrays()
        arrays["chunk_indices_BC"] = np.asarray([[3, 4, 5], [3, 4, 5]], dtype=np.int32)
        batch = ChainBatch(
            document_ids=("doc-0", "doc-1"),
            **{name: jnp.asarray(value) for name, value in arrays.items()},
        )
        chunk_length = arrays["token_ids_BCT"].shape[-1]
        model = _reference_model(_TARGET_SEED)
        run_segment = make_run_segment(ensure_mesh())

        def assert_same_states(actual, expected, context):
            if expected is None:
                self.assertIsNone(actual, context)
                return
            self.assertIsNotNone(actual, context)
            self.assertLen(actual, len(expected))
            for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(np.asarray(actual_leaf), np.asarray(expected_leaf))

        for layer_limit in (None, 0, 1, 2):
            for pass_conv in (True, False):
                for pass_rope in (True, False):
                    state_config = StatePassingConfig(True, layer_limit, pass_conv, pass_rope)
                    runs = [("gdn", run_gdn_experiment)]
                    if pass_conv:
                        runs.append(("conv", run_conv_experiment))
                    for experiment, run_experiment in runs:
                        recorded = {}
                        with mock.patch.object(
                            experiments, "run_chain", side_effect=_recording_run_chain(recorded)
                        ):
                            result = run_experiment(
                                model,
                                model.text.cfg,
                                batch,
                                state_config=state_config,
                                donor_indices_B=jnp.asarray([1, 0], dtype=jnp.int32),
                                pad_id=0,
                                checkpoint="tiny",
                                run_segment=run_segment,
                            )
                        true_condition = _CONDITIONS[experiment][0]
                        true_segments = result.conditions[true_condition].segments
                        for segment_idx, inputs in enumerate(recorded[true_condition]):
                            context = (experiment, layer_limit, pass_conv, pass_rope, segment_idx)
                            if segment_idx == 0:
                                expected_gdn = expected_conv = None
                            else:
                                previous = true_segments[segment_idx - 1].state
                                expected_gdn = _select_gdn_states_for_carry(
                                    previous.gdn_states,
                                    pass_gdn_state=True,
                                    gdn_layer_limit=layer_limit,
                                )
                                expected_conv = _select_conv_states_for_carry(
                                    previous.conv_states,
                                    pass_conv_state=pass_conv,
                                    gdn_layer_limit=layer_limit,
                                )
                            assert_same_states(inputs.gdn_states, expected_gdn, context)
                            assert_same_states(inputs.conv_states, expected_conv, context)
                        for condition in _CONDITIONS[experiment]:
                            for segment_idx, inputs in enumerate(recorded[condition]):
                                chunk_idx_B = arrays["chunk_indices_BC"][:, segment_idx]
                                expected_offsets = (
                                    np.asarray(
                                        _position_ids_zbt_from_chunk_idx(chunk_idx_B, chunk_length)
                                    )[0][:, 0]
                                    if pass_rope
                                    else np.zeros(2, dtype=np.int32)
                                )
                                np.testing.assert_array_equal(
                                    np.asarray(inputs.position_offsets_B), expected_offsets
                                )

    # T3: a document's rows do not depend on batch size, padding rows or batch neighbours.
    def test_rows_do_not_depend_on_batch_size_or_neighbours(self):
        for batch_size in (3, 8):
            root = _copy_checkpoint(self.base_checkpoint, self.tmpdir / f"batch_{batch_size}")
            run_dir = _run_state_usage(root, self.manifest_path, batch_size=batch_size)
            _assert_rows_close(self, _read_rows(run_dir), self.main_rows, atol=1e-4)

    # T2: two emulated devices (dp_size=2) give the same rows as one device.
    def test_two_devices_equal_one_device(self):
        root = _copy_checkpoint(self.base_checkpoint, self.tmpdir / "two_devices")
        output = self.tmpdir / "two_devices_rows.json"
        script = (
            "import sys, json\n"
            "from pathlib import Path\n"
            "import jax\n"
            "assert jax.device_count() == 2, jax.devices()\n"
            "from tests.test_eval_pipeline_reference import _read_rows, _run_state_usage\n"
            "run_dir = _run_state_usage(Path(sys.argv[1]), Path(sys.argv[2]), batch_size=4,"
            " dp_size=2)\n"
            "rows = _read_rows(run_dir)\n"
            "Path(sys.argv[3]).write_text(json.dumps([[*k, *v] for k, v in rows.items()]))\n"
        )
        env = os.environ.copy()
        env.update(
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=2",
            OMEGALAX_DELTANET_KERNEL="xla",
        )
        env.pop("JAX_LOCAL_DEVICE_IDS", None)
        env.pop("SLURM_JOB_ID", None)
        subprocess.run(
            [sys.executable, "-c", script, str(root), str(self.manifest_path), str(output)],
            check=True,
            cwd=Path(__file__).parents[1],
            env=env,
        )
        rows = {tuple(row[:4]): tuple(row[4:]) for row in json.loads(output.read_text())}
        self.assertLen({key[1] for key in rows}, 6)
        _assert_rows_close(self, rows, self.main_rows, atol=1e-4)

    # T8: deleted shards plus an interrupted status resume to the uninterrupted result.
    def test_resume_after_interruption_equals_uninterrupted_run(self):
        root = _copy_checkpoint(self.base_checkpoint, self.tmpdir / "resume")
        run_dir = _run_state_usage(root, self.manifest_path, batch_size=4)
        gdn_dir = run_dir / "raw" / "all" / "gdn"
        deleted = (
            sorted(gdn_dir.glob("shuffled_gdn_shard_*.parquet"))[0],
            sorted((run_dir / "raw" / "all" / "conv").glob("true_conv_shard_*.parquet"))[-1],
        )
        for path in deleted:
            path.unlink()
        leftover = gdn_dir / ".true_gdn_shard_0.parquet.abc123.tmp"
        leftover.write_bytes(b"partial")
        (run_dir / "status.json").write_text(json.dumps({"complete": False}))
        untouched = {
            path: (path.stat().st_ino, path.stat().st_mtime_ns)
            for path in (run_dir / "raw").rglob("*.parquet")
        }
        _run_state_usage(root, self.manifest_path, batch_size=4)
        self.assertTrue(json.loads((run_dir / "status.json").read_text())["complete"])
        self.assertTrue(all(path.is_file() for path in deleted))
        for path, stamp in untouched.items():
            self.assertEqual((path.stat().st_ino, path.stat().st_mtime_ns), stamp, path)
        _assert_rows_close(self, _read_rows(run_dir), self.main_rows, atol=1e-6)
        summary = pq.read_table(run_dir / "summary" / "all" / "metrics.parquet").to_pylist()
        main_summary = pq.read_table(
            self.main_run_dir / "summary" / "all" / "metrics.parquet"
        ).to_pylist()
        self.assertLen(summary, len(main_summary))
        for row, main_row in zip(summary, main_summary, strict=True):
            self.assertEqual(
                {k: v for k, v in row.items() if k != "value"},
                {k: v for k, v in main_row.items() if k != "value"},
            )
            self.assertLessEqual(abs(row["value"] - main_row["value"]), 1e-6)

    # T8: a zero-byte result shard is never aggregated into a complete result.
    def test_zero_byte_result_shard_is_not_aggregated(self):
        root = _copy_checkpoint(self.base_checkpoint, self.tmpdir / "zero_byte")
        run_dir = _run_state_usage(root, self.manifest_path, batch_size=4)
        sorted((run_dir / "raw" / "all" / "gdn").glob("zero_gdn_shard_*.parquet"))[0].write_bytes(
            b""
        )
        (run_dir / "status.json").write_text(json.dumps({"complete": False}))
        with self.assertRaisesRegex(ValueError, "0 bytes"):
            _run_state_usage(root, self.manifest_path, batch_size=4)
        self.assertFalse(json.loads((run_dir / "status.json").read_text())["complete"])

    # T11: segmented true-GDN eval equals one full forward for a GDN-only model.
    def test_segmented_eval_equals_full_forward_for_gdn_only_model(self):
        cfg = replace(
            tiny_hybrid_config(), num_hidden_layers=2, layer_types=("linear_attention",) * 2
        )
        model, cfg = text_api.init_model(
            cfg, jax.random.key(_TARGET_SEED), tp_size=1, fsdp_size=1, dp_size=1
        )
        set_attn_backend(model, "xla")
        arrays = two_document_chain_arrays()
        batch = ChainBatch(
            document_ids=("doc-0", "doc-1"),
            **{name: jnp.asarray(value) for name, value in arrays.items()},
        )
        segmented = run_gdn_experiment(
            model,
            cfg,
            batch,
            state_config=StatePassingConfig(True, None, True, True),
            donor_indices_B=jnp.asarray([1, 0], dtype=jnp.int32),
            pad_id=0,
            checkpoint="gdn-only",
        ).conditions["true_gdn"]
        run_full_context = make_run_full_context(ensure_mesh())
        for batch_idx in range(2):
            document = SimpleNamespace(
                token_ids_CT=arrays["token_ids_BCT"][batch_idx],
                attention_mask_CT=arrays["attention_mask_BCT"][batch_idx],
                loss_mask_CT=arrays["loss_mask_BCT"][batch_idx],
            )
            token_ids, attention_mask, loss_masks = full_context_arrays(document)
            nll_sums, token_counts = run_full_context(
                model,
                jnp.asarray(token_ids),
                jnp.asarray(attention_mask),
                jnp.asarray(loss_masks),
                pad_id=0,
            )
            nll_sums = np.asarray(nll_sums).reshape(-1)
            token_counts = np.asarray(token_counts).reshape(-1)
            for segment_idx, segment in enumerate(segmented.segments):
                self.assertEqual(
                    float(token_counts[segment_idx]), float(segment.token_count_B[batch_idx])
                )
                self.assertLessEqual(
                    abs(float(nll_sums[segment_idx]) - float(segment.nll_sum_B[batch_idx])), 1e-4
                )
        zero = run_gdn_experiment(
            model,
            cfg,
            batch,
            state_config=StatePassingConfig(True, 0, False, True),
            donor_indices_B=jnp.asarray([1, 0], dtype=jnp.int32),
            pad_id=0,
            checkpoint="gdn-only",
        ).conditions["true_gdn"]
        zero_gap = max(
            abs(float(a.nll_sum_B[b]) - float(z.nll_sum_B[b]))
            for a, z in zip(segmented.segments[1:], zero.segments[1:], strict=True)
            for b in range(2)
        )
        self.assertGreater(zero_gap, 1e-3)

    # T11: compare_relative_nll rejects differing token counts and missing points.
    def test_compare_relative_nll_rejects_misaligned_candidates(self):
        def write_summary(path: Path, points) -> Path:
            path.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([vars(point) for point in points]), path)
            return path

        def point(chunk_position, value, token_count):
            return SummaryPoint(
                "gdn", "nll", "exact_length", 3, chunk_position, "true_gdn", value, token_count
            )

        baseline = write_summary(
            self.tmpdir / "relative" / "baseline" / "metrics.parquet",
            (point(1, 3.0, 8.0), point(2, 2.5, 8.0)),
        )
        counts = write_summary(
            self.tmpdir / "relative" / "counts" / "metrics.parquet",
            (point(1, 3.0, 8.0), point(2, 2.0, 7.0)),
        )
        missing = write_summary(
            self.tmpdir / "relative" / "missing" / "metrics.parquet", (point(1, 3.0, 8.0),)
        )
        with self.assertRaisesRegex(ValueError, "token count differs"):
            compare_relative_nll(
                baseline_result=baseline,
                candidate_results=(("counts", counts),),
                output_dir=self.tmpdir / "relative" / "out_counts",
            )
        with self.assertRaisesRegex(ValueError, "lacks aligned true_gdn points"):
            compare_relative_nll(
                baseline_result=baseline,
                candidate_results=(("missing", missing),),
                output_dir=self.tmpdir / "relative" / "out_missing",
            )

    # T10: top-entropy rows use the reference model's top 20 % targets scored by the target model.
    def test_top_entropy_selection_matches_reference(self):
        chunk_length = 11
        tmpdir = self.tmpdir / "entropy"
        manifest_path, manifest, _ = _build_ragged_manifest(
            tmpdir / "data", (22, 16, 33, 25, 30), chunk_length=chunk_length, sample_cap=3
        )
        target_root = _copy_checkpoint(self.base_checkpoint, tmpdir / "target")
        reference_root = tmpdir / "reference"
        _save_checkpoint(reference_root, step=1, seed=_REFERENCE_SEED)
        run_evals(
            (_spec(target_root, c_train=2),),
            mode="subset",
            manifest_path=manifest_path,
            experiments=("gdn",),
            token_selections=("all", "top_entropy_20pct"),
            entropy_reference_checkpoint=reference_root / "000001",
            plot_types=("heatmap",),
            batch_size=4,
        )
        run_dir = target_root / "evals" / "state_usage_v2" / "step_1"
        reference_model = _reference_model(_REFERENCE_SEED)
        loader = FullDocumentLoader(manifest)
        selection_masks = {}
        for entry in manifest.documents:
            document = loader.load_document(entry)
            for segment_idx in range(entry.doc_num_chunks):
                logp, _, _ = _reference_logp(
                    reference_model, document, segment_idx, None, None, offset=0
                )
                entropies = -(np.exp(logp) * logp).sum(axis=-1)
                loss_mask = document.loss_mask_CT[segment_idx]
                targets = [t for t in range(1, chunk_length) if loss_mask[t]]
                mask = np.zeros(chunk_length, dtype=np.int32)
                if targets:
                    count = max(1, math.ceil(0.2 * len(targets)))
                    ranked = sorted(targets, key=lambda t: (-entropies[t - 1], t))
                    if count < len(targets):
                        gap = entropies[ranked[count - 1] - 1] - entropies[ranked[count] - 1]
                        self.assertGreater(gap, 1e-5, "fixture has an entropy near-tie")
                    mask[ranked[:count]] = 1
                selection_masks[entry.doc_id, segment_idx + 1] = mask
        reference = _reference_logps(
            _reference_model(_TARGET_SEED),
            manifest,
            ("gdn",),
            pass_conv=True,
            pass_rope=True,
            layer_limit=None,
        )
        expected = {
            key: _reference_nll(
                logp, document.token_ids_CT[key[3] - 1], selection_masks[key[2], key[3]]
            )
            for key, (logp, document) in reference.items()
        }
        all_rows = _read_rows(run_dir, "all")
        top_rows = _read_rows(run_dir, "top_entropy_20pct")
        _assert_rows_close(self, top_rows, expected, atol=5e-4)
        self.assertTrue(any(top_rows[key][1] < all_rows[key][1] for key in top_rows))
        for key, (_, count) in top_rows.items():
            all_count = all_rows[key][1]
            self.assertEqual(count, 0 if all_count == 0 else max(1, math.ceil(0.2 * all_count)))

    # T7: contract values reach the evaluated configuration and rows through the real pipeline.
    def test_contract_values_reach_evaluated_configuration(self):
        tmpdir = self.tmpdir / "contract"
        manifest_path, _ = _build_manifest(tmpdir / "data", documents_per_length=2)
        index = _write_index(
            tmpdir / "index",
            {
                "format": STATEPASSING_CURRICULUM_INDEX_FORMAT,
                "train_order": [2, 3],
                "splits": {"train": {"phases": {"2": {"phase_steps": 1}, "3": {"phase_steps": 5}}}},
            },
        )
        contract = build_training_contract(
            index,
            pass_gdn_state=True,
            gdn_layer_limit=1,
            pass_conv_state=True,
            pass_rope_positions=True,
            pad_id=0,
            eos_id=2,
        )
        contract_root = tmpdir / "with_contract"
        contract_hash = ensure_training_contract(contract_root, contract)
        shutil.copytree(
            self.base_checkpoint,
            contract_root,
            ignore=shutil.ignore_patterns("evals"),
            dirs_exist_ok=True,
        )
        explicit_root = _copy_checkpoint(self.base_checkpoint, tmpdir / "explicit")
        options = dict(
            mode="all", manifest_path=manifest_path, plot_types=("heatmap",), batch_size=2
        )
        run_evals((CheckpointEvalRequest(contract_root),), **options)
        run_evals(
            (_spec(explicit_root, c_train=2, gdn_layer_limit=1, pass_conv_state=True),),
            **options,
        )
        contract_run = contract_root / "evals" / "state_usage_v2" / "step_1"
        explicit_run = explicit_root / "evals" / "state_usage_v2" / "step_1"
        config = _eval_config(contract_run)
        expected = {
            "c_train": 2,
            "pass_gdn_state": True,
            "gdn_layer_limit": 1,
            "pass_conv_state": True,
            "pass_rope_positions": True,
            "pad_id": 0,
            "eos_id": 2,
            "resolution_source": "training_contract",
            "training_contract_hash": contract_hash,
        }
        self.assertEqual({name: config[name] for name in expected}, expected)
        self.assertEqual(config["experiments"], ["gdn", "conv"])
        _assert_rows_close(self, _read_rows(contract_run), _read_rows(explicit_run), atol=1e-6)

        matching = ManualEvalConfig(2, True, 1, True, True, 0, 2)
        self.assertEqual(
            resolve_checkpoint_eval_request(CheckpointEvalRequest(contract_root, matching)).c_train,
            2,
        )
        conflicts = dict(
            c_train=3,
            pass_gdn_state=False,
            gdn_layer_limit=None,
            pass_conv_state=False,
            pass_rope_positions=False,
            pad_id=1,
            eos_id=3,
        )
        for name, value in conflicts.items():
            with self.assertRaisesRegex(
                ValueError, f"conflict with training_contract.json for: {name}"
            ):
                resolve_checkpoint_eval_request(
                    CheckpointEvalRequest(contract_root, replace(matching, **{name: value}))
                )
        with self.assertRaisesRegex(ValueError, "no training_contract.json"):
            resolve_checkpoint_eval_request(CheckpointEvalRequest(explicit_root))

    # T5: a training contract with gdn_state_routing=last_to_first must be rejected by the eval.
    def test_non_default_routing_contract_is_rejected(self):
        # Regression V1: the eval ignored the trainer's last_to_first routing and fed every GDN layer its own state.
        tmpdir = self.tmpdir / "routing"
        index = _write_index(
            tmpdir / "index", {"format": STATEPASSING_FIXED_C_INDEX_FORMAT, "num_segments": 2}
        )
        contract = build_training_contract(
            index,
            pass_gdn_state=True,
            gdn_layer_limit=1,
            gdn_state_routing="last_to_first",
            pass_conv_state=False,
            pass_rope_positions=False,
            pad_id=0,
            eos_id=2,
        )
        root = tmpdir / "checkpoint"
        ensure_training_contract(root, contract)
        shutil.copytree(self.base_checkpoint, root, dirs_exist_ok=True)
        with self.assertRaisesRegex(ValueError, "routing"):
            resolve_eval_config(root, 1, None)
        with self.assertRaisesRegex(ValueError, "routing"):
            resolve_checkpoint_eval_request(CheckpointEvalRequest(root))

    # T6: training segment length and manifest chunk length must agree before inference.
    def test_segment_length_mismatch_is_rejected(self):
        # Regression V2: the eval never compared the training segment length with the manifest chunk length.
        tmpdir = self.tmpdir / "segment_length"
        manifest_path, manifest = _build_manifest(tmpdir / "data", documents_per_length=2)
        index = _write_index(
            tmpdir / "index",
            {
                "format": ROLLOUT_TEACHER_INDEX_FORMAT,
                "segment_buckets": [1, 2, 4],
                "segment_length": 4,
                "target_capacities": [8],
                "index_binding": "test",
                "selection": {"name": "test"},
                "source": {"dataset_binding": "test", "scoring_binding": "test"},
            },
        )
        contract = build_training_contract(
            index,
            pass_gdn_state=True,
            gdn_layer_limit=None,
            pass_conv_state=False,
            pass_rope_positions=False,
            pad_id=0,
            eos_id=2,
            teacher_distillation={"objective": "teacher_top512_special_tail_kl"},
        )
        self.assertNotEqual(
            contract["teacher_distillation"]["segment_length"], manifest.chunk_length
        )
        root = tmpdir / "checkpoint"
        ensure_training_contract(root, contract)
        shutil.copytree(self.base_checkpoint, root, dirs_exist_ok=True)
        with self.assertRaisesRegex(ValueError, "segment"):
            run_evals(
                (CheckpointEvalRequest(root),),
                mode="subset",
                manifest_path=manifest_path,
                experiments=("gdn",),
                plot_types=("heatmap",),
                batch_size=2,
            )
        self.assertFalse(tuple(root.glob("evals/state_usage_v2/*/raw/**/*.parquet")))

    # T9: the evaluated step and its stored c_train must belong together.
    def test_evaluated_step_and_c_train_belong_together(self):
        # Regression V3: c_train was resolved for the latest step at request time, the eval later ran on a newer step.
        tmpdir = self.tmpdir / "moving_step"
        manifest_path, _ = _build_manifest(tmpdir / "data", documents_per_length=2)
        index = _write_index(
            tmpdir / "index",
            {
                "format": STATEPASSING_CURRICULUM_INDEX_FORMAT,
                "train_order": [2, 3],
                "splits": {"train": {"phases": {"2": {"phase_steps": 1}, "3": {"phase_steps": 5}}}},
            },
        )
        contract = build_training_contract(
            index,
            pass_gdn_state=True,
            gdn_layer_limit=None,
            pass_conv_state=False,
            pass_rope_positions=False,
            pad_id=0,
            eos_id=2,
        )
        root = tmpdir / "checkpoint"
        ensure_training_contract(root, contract)
        _save_checkpoint(root, step=1, seed=_TARGET_SEED)
        specs = resolve_checkpoint_eval_requests((CheckpointEvalRequest(root),))
        _save_checkpoint(root, step=2, seed=_TARGET_SEED)
        run_evals(
            specs,
            mode="subset",
            manifest_path=manifest_path,
            experiments=("gdn",),
            plot_types=("heatmap",),
            batch_size=2,
        )
        run_dirs = sorted((root / "evals" / "state_usage_v2").glob("step_*"))
        self.assertLen(run_dirs, 1)
        step = int(run_dirs[0].name.removeprefix("step_"))
        self.assertEqual(
            _eval_config(run_dirs[0])["c_train"], resolve_eval_config(root, step, None).c_train
        )


if __name__ == "__main__":
    absltest.main()
