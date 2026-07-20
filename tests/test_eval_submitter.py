"""Tests for local-to-Slurm checkpoint-eval orchestration."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
import subprocess
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest

from omegalax.evals import runner
from omegalax.evals.runner import CheckpointEvalRequest, CheckpointEvalSpec
from omegalax.evals.storage import EvalRunIdentity, EvalRunStore, MetricRow
from omegalax.training_contract import (
    ManualEvalConfig,
    ensure_training_contract,
    training_contract_hash,
)
from scripts import submit_checkpoint_evals as eval_submitter
from tests.pretrain_real_data_test_utils import test_temp_dir


_CONDITIONS = ("true_gdn", "zero_gdn", "shuffled_gdn")
_PLOT_TYPES = ("in_horizon", "beyond_horizon", "exact_length", "heatmap")


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


def _request(checkpoint: str | Path, **overrides) -> CheckpointEvalRequest:
    spec = _spec(checkpoint, **overrides)
    return CheckpointEvalRequest(
        checkpoint,
        ManualEvalConfig(
            c_train=spec.c_train,
            pass_gdn_state=spec.pass_gdn_state,
            gdn_layer_limit=spec.gdn_layer_limit,
            pass_conv_state=spec.pass_conv_state,
            pass_rope_positions=spec.pass_rope_positions,
            pad_id=spec.pad_id,
            eos_id=spec.eos_id,
        ),
    )


def _normalized(script: str) -> str:
    return script.replace('"', "").replace("'", "")


def _resolved(root: Path, *, step: int):
    return SimpleNamespace(
        root=root.resolve(),
        step=step,
        step_path=(root / f"{step:06d}").resolve(),
        config_path=(root / "config.json").resolve(),
    )


def _partial_gdn_store(root: Path, *, step: int) -> EvalRunStore:
    identity = EvalRunIdentity(
        dataset_hash="sha256:dataset",
        manifest_hash="sha256:manifest",
        checkpoint_root=str(root.resolve()),
        checkpoint_step=step,
        code_hash="git:test",
        eval_config={
            "c_train": 4,
            "pass_gdn_state": True,
            "gdn_layer_limit": None,
            "pass_conv_state": True,
            "pass_rope_positions": True,
            "pad_id": 0,
            "eos_id": 2,
            "tp_size": 1,
            "fsdp_size": 1,
            "dp_size": 4,
            "batch_size": 16,
            "document_cap": None,
            "effective_document_cap": 2,
            "experiments": ["gdn"],
            "population_counts": {"2": 2},
            "metric_contract": "nll_sum_token_count_v1",
            "conditions_by_experiment": {"gdn": list(_CONDITIONS)},
            "expected_shards": {"gdn": {condition: [0, 1] for condition in _CONDITIONS}},
        },
    )
    store = EvalRunStore.open(root, step, identity)
    for condition in ("true_gdn", "zero_gdn"):
        for shard_id in (0, 1):
            store.write_shard(
                "gdn",
                condition,
                shard_id,
                (
                    MetricRow(
                        experiment="gdn",
                        condition=condition,
                        bucket_idx=0,
                        record_idx=shard_id,
                        doc_id=f"doc-{shard_id}",
                        doc_num_chunks=2,
                        chunk_position=1,
                        nll_sum=3.0,
                        token_count=2,
                    ),
                    MetricRow(
                        experiment="gdn",
                        condition=condition,
                        bucket_idx=0,
                        record_idx=shard_id,
                        doc_id=f"doc-{shard_id}",
                        doc_num_chunks=2,
                        chunk_position=2,
                        nll_sum=3.2,
                        token_count=2,
                    ),
                ),
            )
    return store


def _submitted(stdout: str):
    return SimpleNamespace(stdout=stdout, stderr="")


def _raw_inodes(store: EvalRunStore) -> dict[Path, tuple[int, int]]:
    return {
        path.relative_to(store.run_dir): (stat.st_ino, stat.st_mtime_ns)
        for path in (store.run_dir / "raw").rglob("*.parquet")
        for stat in (path.stat(),)
    }


def _cli_spec_flags(
    checkpoint: Path,
    *,
    c_train: int,
    pass_gdn_state: bool,
    gdn_layer_limit: int | None,
    pass_conv_state: bool,
    pass_rope_positions: bool,
    pad_id: int,
    eos_id: int,
) -> list[str]:
    return [
        f"--checkpoint={checkpoint}",
        f"--c_train={c_train}",
        f"--pass_gdn_state={str(pass_gdn_state).lower()}",
        f"--gdn_layer_limit={'all' if gdn_layer_limit is None else gdn_layer_limit}",
        f"--pass_conv_state={str(pass_conv_state).lower()}",
        f"--pass_rope_positions={str(pass_rope_positions).lower()}",
        f"--pad_id={pad_id}",
        f"--eos_id={eos_id}",
    ]


class EvalSubmitterTest(absltest.TestCase):
    def test_contract_request_is_prevalidated_and_rendered_without_legacy_flags(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            checkpoint = tmpdir / "checkpoint"
            contract = {
                "schema_version": 1,
                "training_index": {"path": "/index", "metadata_hash": "sha256:index"},
                "eval_statepassing_config": {
                    "pass_gdn_state": True,
                    "gdn_layer_limit": None,
                    "pass_conv_state": False,
                    "pass_rope_positions": True,
                    "pad_id": 0,
                    "eos_id": 2,
                },
                "horizon_by_step": [{"start_step": 1, "end_step": None, "c_train": 4}],
            }
            ensure_training_contract(checkpoint, contract)
            resolved = _resolved(checkpoint, step=10)
            with mock.patch.object(runner, "resolve_checkpoint", return_value=resolved):
                paths = eval_submitter.submit_checkpoint_evals(
                    (CheckpointEvalRequest(checkpoint),),
                    manifest_path=tmpdir / "manifest.json",
                    repo_root=tmpdir / "repo",
                    job_dir=tmpdir / "jobs",
                    log_dir=tmpdir / "logs",
                    partition="gpu",
                    qos=None,
                    time_limit="04:00:00",
                    mode="all",
                    experiments=None,
                    plot_types=("heatmap",),
                    plot_output_roots=(),
                    comparison_name=None,
                    pilot_cap=2,
                    dry_run=True,
                )

            script = _normalized(paths[0].read_text())
            self.assertEqual(
                json.loads((checkpoint / "training_contract.json").read_text()), contract
            )
            self.assertIn(f"--checkpoint={checkpoint}", script)
            for flag in eval_submitter._SPEC_FIELDS:
                self.assertNotIn(f"--{flag}=", script)
            with mock.patch.object(runner, "resolve_checkpoint", return_value=resolved):
                spec = runner.resolve_checkpoint_eval_request(CheckpointEvalRequest(checkpoint))
            self.assertEqual(spec.training_contract_hash, training_contract_hash(contract))

    def test_rendered_subset_job_has_fixed_topology_and_every_explicit_override(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            checkpoint = tmpdir / "misleading_iid_c1_name"
            plot_roots = (tmpdir / "plots-a", tmpdir / "plots-b")
            script = eval_submitter.render_eval_sbatch(
                _spec(
                    checkpoint,
                    c_train=6,
                    pass_gdn_state=False,
                    gdn_layer_limit=1,
                    pass_conv_state=False,
                    pass_rope_positions=True,
                    pad_id=7,
                    eos_id=9,
                ),
                manifest_path=tmpdir / "manifest.json",
                repo_root=tmpdir / "repo",
                log_dir=tmpdir / "logs",
                partition="gpu",
                qos="normal",
                time_limit="04:00:00",
                mode="subset",
                experiments=("gdn", "conv"),
                plot_types=("in_horizon", "heatmap"),
                plot_output_roots=plot_roots,
                comparison_name=None,
                pilot_cap=2,
            )
            normalized = _normalized(script)
            lower = normalized.lower()

            self.assertIn("#SBATCH --nodes=1", script)
            self.assertIn("#SBATCH --gres=gpu:4", script)
            self.assertIn("#SBATCH --cpus-per-task=24", script)
            self.assertIn("#SBATCH --mem=120G", script)
            self.assertRegex(script, r"#SBATCH --ntasks(?:-per-node)?=1")
            for flag, value in (
                ("tp_size", "1"),
                ("fsdp_size", "1"),
                ("dp_size", "4"),
                ("batch_size", "16"),
                ("c_train", "6"),
                ("gdn_layer_limit", "1"),
                ("pad_id", "7"),
                ("eos_id", "9"),
                ("mode", "subset"),
                ("document_cap", "2"),
            ):
                self.assertEqual(re.findall(rf"--{flag}=([^\s\\]+)", normalized), [value])
            self.assertEqual(16 % 4, 0)
            self.assertIn(f"--checkpoint={checkpoint}", normalized)
            self.assertIn(f"--manifest_path={tmpdir / 'manifest.json'}", normalized)
            self.assertIn("--pass_gdn_state=false", lower)
            self.assertIn("--pass_conv_state=false", lower)
            self.assertIn("--pass_rope_positions=true", lower)
            self.assertIn("--experiments=gdn,conv", normalized)
            self.assertIn("--plot_types=in_horizon,heatmap", normalized)
            for root in plot_roots:
                self.assertIn(str(root), normalized)
            self.assertIn("JAX_LOCAL_DEVICE_IDS=0,1,2,3", script)
            self.assertRegex(
                script,
                r"(?:scripts/run_checkpoint_evals\.py|scripts\.run_checkpoint_evals)",
            )
            self.assertNotIn("--gpus-per-task=1", script)
            self.assertNotIn("--gpu-bind=single:1", script)
            self.assertNotIn("--pilot_cap", normalized)
            self.assertNotIn("wandb", lower)
            self.assertNotIn("w&b", lower)

    def test_rendered_job_quotes_real_parenthesized_repo_path_and_is_bash_valid(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertIn("p(doom)", str(repo_root))
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            checkpoint = tmpdir / "checkpoint with spaces"
            manifest_path = tmpdir / "manifest with spaces.json"
            log_dir = tmpdir / "logs with spaces"
            plot_root = tmpdir / "plots with spaces"
            script = eval_submitter.render_eval_sbatch(
                _spec(checkpoint),
                manifest_path=manifest_path,
                repo_root=repo_root,
                log_dir=log_dir,
                partition="gpu",
                qos=None,
                time_limit="04:00:00",
                mode="subset",
                experiments=("gdn",),
                plot_types=("exact_length",),
                plot_output_roots=(plot_root,),
                comparison_name=None,
                pilot_cap=2,
            )

        self.assertIn(f"cd {shlex.quote(str(repo_root))}", script)
        for flag, path in (
            ("checkpoint", checkpoint),
            ("manifest_path", manifest_path),
            ("plot_output_root", plot_root),
        ):
            self.assertTrue(
                f"--{flag}={shlex.quote(str(path))}" in script
                or shlex.quote(f"--{flag}={path}") in script,
                msg=f"unquoted --{flag} in rendered script",
            )
        log_directives = tuple(
            line
            for line in script.splitlines()
            if line.startswith(("#SBATCH --output=", "#SBATCH --error="))
        )
        self.assertNotEmpty(log_directives)
        for directive in log_directives:
            fields = shlex.split(directive)
            self.assertLen(fields, 2, msg=directive)
            self.assertIn(str(log_dir), fields[1], msg=directive)
        syntax = subprocess.run(
            ("bash", "-n"),
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, msg=syntax.stderr)

    def test_full_job_omits_document_cap_and_forwards_none_layer_limit(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            script = eval_submitter.render_eval_sbatch(
                _spec(tmpdir / "checkpoint", gdn_layer_limit=None),
                manifest_path=tmpdir / "manifest.json",
                repo_root=tmpdir / "repo",
                log_dir=tmpdir / "logs",
                partition="gpu",
                qos=None,
                time_limit="04:00:00",
                mode="all",
                experiments=None,
                plot_types=_PLOT_TYPES,
                plot_output_roots=(),
                comparison_name=None,
                pilot_cap=None,
            )
            normalized = _normalized(script).lower()

            self.assertIn("--mode=all", normalized)
            self.assertIn("--gdn_layer_limit=all", normalized)
            self.assertIn("--plot_types=in_horizon,beyond_horizon,exact_length,heatmap", normalized)
            self.assertNotIn("--document_cap", normalized)
            self.assertNotIn("--pilot_cap", normalized)

    def test_submitter_creates_one_independent_job_per_spec_and_preserves_each_spec(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            specs = (
                _spec(
                    tmpdir / "checkpoint-a",
                    c_train=4,
                    pass_gdn_state=False,
                    gdn_layer_limit=None,
                    pass_conv_state=True,
                    pass_rope_positions=False,
                    pad_id=3,
                    eos_id=4,
                ),
                _spec(
                    tmpdir / "checkpoint-b",
                    c_train=6,
                    pass_gdn_state=True,
                    gdn_layer_limit=2,
                    pass_conv_state=False,
                    pass_rope_positions=True,
                    pad_id=5,
                    eos_id=6,
                ),
            )
            with mock.patch.object(
                eval_submitter.subprocess,
                "run",
                side_effect=(
                    _submitted("Submitted batch job 101\n"),
                    _submitted("Submitted batch job 102\n"),
                ),
            ) as run:
                eval_submitter.submit_checkpoint_evals(
                    specs,
                    manifest_path=tmpdir / "manifest.json",
                    repo_root=tmpdir / "repo",
                    job_dir=tmpdir / "jobs",
                    log_dir=tmpdir / "logs",
                    partition="gpu",
                    qos="normal",
                    time_limit="04:00:00",
                    mode="subset",
                    experiments=("gdn",),
                    plot_types=("beyond_horizon", "exact_length"),
                    plot_output_roots=(tmpdir / "published",),
                    comparison_name=None,
                    pilot_cap=2,
                    dry_run=False,
                )

            self.assertEqual(run.call_count, 2)
            scripts = sorted((tmpdir / "jobs").glob("*.sbatch"))
            self.assertLen(scripts, 2)
            contents = [_normalized(path.read_text()) for path in scripts]
            expected = {
                str(specs[0].checkpoint): {
                    "c_train": "4",
                    "pass_gdn_state": "false",
                    "gdn_layer_limit": "all",
                    "pass_conv_state": "true",
                    "pass_rope_positions": "false",
                    "pad_id": "3",
                    "eos_id": "4",
                },
                str(specs[1].checkpoint): {
                    "c_train": "6",
                    "pass_gdn_state": "true",
                    "gdn_layer_limit": "2",
                    "pass_conv_state": "false",
                    "pass_rope_positions": "true",
                    "pad_id": "5",
                    "eos_id": "6",
                },
            }
            for checkpoint, flags in expected.items():
                matches = [text for text in contents if f"--checkpoint={checkpoint}" in text]
                self.assertLen(matches, 1)
                lower = matches[0].lower()
                for name, value in flags.items():
                    self.assertIn(f"--{name}={value}", lower)
                self.assertIn("--mode=subset", lower)
                self.assertIn("--experiments=gdn", lower)
                self.assertIn("--plot_types=beyond_horizon,exact_length", lower)
                self.assertIn("--document_cap=2", lower)
                self.assertIn(str(tmpdir / "manifest.json"), matches[0])
                self.assertIn(str(tmpdir / "published"), matches[0])

            for call in run.call_args_list:
                command = call.args[0]
                self.assertEqual(command[0], "sbatch")
                self.assertLen(command, 2)
                self.assertNotIn("--dependency", " ".join(command))

    def test_public_submitter_cli_roundtrips_aligned_specs_and_dispatch_options(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            argv = [
                *_cli_spec_flags(
                    tmpdir / "checkpoint-a",
                    c_train=4,
                    pass_gdn_state=False,
                    gdn_layer_limit=None,
                    pass_conv_state=True,
                    pass_rope_positions=False,
                    pad_id=3,
                    eos_id=4,
                ),
                *_cli_spec_flags(
                    tmpdir / "checkpoint-b",
                    c_train=6,
                    pass_gdn_state=True,
                    gdn_layer_limit=2,
                    pass_conv_state=False,
                    pass_rope_positions=True,
                    pad_id=5,
                    eos_id=6,
                ),
                "--mode=subset",
                f"--manifest_path={tmpdir / 'manifest.json'}",
                "--experiments=gdn,conv",
                "--plot_types=in_horizon,heatmap",
                f"--plot_output_root={tmpdir / 'paper-a'}",
                f"--plot_output_root={tmpdir / 'paper-b'}",
                f"--repo_root={tmpdir / 'repo'}",
                f"--job_dir={tmpdir / 'jobs'}",
                f"--log_dir={tmpdir / 'logs'}",
                "--partition=gpu",
                "--qos=normal",
                "--time_limit=04:00:00",
                "--pilot_cap=2",
                "--dry_run",
            ]
            self.assertIsNotNone(eval_submitter.parse_submit_args(argv))
            with mock.patch.object(
                eval_submitter,
                "submit_checkpoint_evals",
            ) as submit:
                eval_submitter.submit_from_arguments(argv)

            submit.assert_called_once()
            specs = submit.call_args.args[0]
            self.assertEqual(
                specs,
                (
                    _request(
                        str(tmpdir / "checkpoint-a"),
                        c_train=4,
                        pass_gdn_state=False,
                        gdn_layer_limit=None,
                        pass_conv_state=True,
                        pass_rope_positions=False,
                        pad_id=3,
                        eos_id=4,
                    ),
                    _request(
                        str(tmpdir / "checkpoint-b"),
                        c_train=6,
                        pass_gdn_state=True,
                        gdn_layer_limit=2,
                        pass_conv_state=False,
                        pass_rope_positions=True,
                        pad_id=5,
                        eos_id=6,
                    ),
                ),
            )
            kwargs = submit.call_args.kwargs
            self.assertEqual(kwargs["mode"], "subset")
            self.assertEqual(kwargs["experiments"], ("gdn", "conv"))
            self.assertEqual(kwargs["plot_types"], ("in_horizon", "heatmap"))
            self.assertEqual(
                tuple(Path(path) for path in kwargs["plot_output_roots"]),
                (tmpdir / "paper-a", tmpdir / "paper-b"),
            )
            self.assertEqual(kwargs["pilot_cap"], 2)
            self.assertTrue(kwargs["dry_run"])

    def test_public_submitter_cli_accepts_one_checkpoint_all_mode(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            argv = [
                *_cli_spec_flags(
                    tmpdir / "checkpoint",
                    c_train=4,
                    pass_gdn_state=True,
                    gdn_layer_limit=None,
                    pass_conv_state=True,
                    pass_rope_positions=True,
                    pad_id=0,
                    eos_id=2,
                ),
                "--mode=all",
                f"--manifest_path={tmpdir / 'manifest.json'}",
                "--plot_types=in_horizon,beyond_horizon,exact_length,heatmap",
                f"--repo_root={tmpdir / 'repo'}",
                f"--job_dir={tmpdir / 'jobs'}",
                f"--log_dir={tmpdir / 'logs'}",
                "--partition=gpu",
                "--time_limit=04:00:00",
                "--dry_run",
            ]
            with mock.patch.object(
                eval_submitter,
                "submit_checkpoint_evals",
            ) as submit:
                eval_submitter.submit_from_arguments(argv)

            submit.assert_called_once()
            self.assertLen(submit.call_args.args[0], 1)
            self.assertEqual(submit.call_args.kwargs["mode"], "all")
            self.assertEqual(submit.call_args.kwargs["plot_types"], _PLOT_TYPES)

    def test_public_submitter_cli_rejects_misaligned_specs_and_invalid_mode(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            argv = _cli_spec_flags(
                tmpdir / "checkpoint",
                c_train=4,
                pass_gdn_state=True,
                gdn_layer_limit=None,
                pass_conv_state=True,
                pass_rope_positions=True,
                pad_id=0,
                eos_id=2,
            )
            argv.append("--c_train=6")
            argv.extend(("--mode=all", f"--manifest_path={tmpdir / 'manifest.json'}"))
            with self.assertRaisesRegex(ValueError, "(?i)(aligned|length|c_train)"):
                eval_submitter.submit_from_arguments(argv)

            one_spec = _cli_spec_flags(
                tmpdir / "checkpoint",
                c_train=4,
                pass_gdn_state=True,
                gdn_layer_limit=None,
                pass_conv_state=True,
                pass_rope_positions=True,
                pad_id=0,
                eos_id=2,
            )
            second_spec = _cli_spec_flags(
                tmpdir / "checkpoint-b",
                c_train=6,
                pass_gdn_state=False,
                gdn_layer_limit=1,
                pass_conv_state=False,
                pass_rope_positions=False,
                pad_id=3,
                eos_id=4,
            )
            aligned_base = [
                *one_spec,
                *second_spec,
                "--mode=plot",
                "--plot_types=heatmap",
                "--dry_run",
            ]
            for flag, extra_value in (
                ("c_train", "8"),
                ("pass_gdn_state", "true"),
                ("gdn_layer_limit", "all"),
                ("pass_conv_state", "true"),
                ("pass_rope_positions", "false"),
                ("pad_id", "9"),
                ("eos_id", "10"),
            ):
                with self.subTest(flag=flag):
                    with mock.patch.object(
                        eval_submitter,
                        "submit_checkpoint_evals",
                    ) as submit:
                        with self.assertRaisesRegex(
                            ValueError,
                            f"(?i)(aligned|length|{flag})",
                        ):
                            eval_submitter.submit_from_arguments(
                                [*aligned_base, f"--{flag}={extra_value}"]
                            )
                    submit.assert_not_called()

            with self.assertRaisesRegex(ValueError, "(?i)(two|2|multiple|compare)"):
                eval_submitter.submit_from_arguments(
                    [
                        *one_spec,
                        "--mode=compare",
                        "--experiments=gdn",
                        "--plot_types=in_horizon",
                        "--comparison_name=one_checkpoint",
                        "--dry_run",
                    ]
                )
            with self.assertRaisesRegex(ValueError, "(?i)comparison.*name"):
                eval_submitter.submit_from_arguments(
                    [
                        *one_spec,
                        *second_spec,
                        "--mode=compare",
                        f"--manifest_path={tmpdir / 'manifest.json'}",
                        "--experiments=gdn",
                        "--plot_types=in_horizon",
                        "--dry_run",
                    ]
                )
            for invalid_experiment in ("rope", "unknown"):
                with self.subTest(invalid_experiment=invalid_experiment):
                    with self.assertRaisesRegex(ValueError, "(?i)experiment"):
                        eval_submitter.submit_from_arguments(
                            [
                                *one_spec,
                                "--mode=plot",
                                f"--experiments=gdn,{invalid_experiment}",
                                "--plot_types=in_horizon",
                                "--dry_run",
                            ]
                        )
            with self.assertRaisesRegex(ValueError, "(?i)manifest"):
                eval_submitter.submit_from_arguments(
                    [
                        *one_spec,
                        "--mode=all",
                        "--plot_types=heatmap",
                        "--dry_run",
                    ]
                )

            with self.assertRaises(SystemExit):
                eval_submitter.parse_submit_args(
                    [*one_spec, "--mode=not-a-mode", "--plot_types=heatmap"]
                )
            with self.assertRaises(SystemExit):
                eval_submitter.parse_submit_args(
                    [
                        *one_spec,
                        "--mode=plot",
                        "--plot_types=in_horizon,confidence_interval",
                    ]
                )

    def test_full_dry_run_writes_one_script_without_subprocess_or_auto_pilot(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            with mock.patch.object(eval_submitter.subprocess, "run") as run:
                eval_submitter.submit_checkpoint_evals(
                    (_spec(tmpdir / "checkpoint"),),
                    manifest_path=tmpdir / "manifest.json",
                    repo_root=tmpdir / "repo",
                    job_dir=tmpdir / "jobs",
                    log_dir=tmpdir / "logs",
                    partition="gpu",
                    qos=None,
                    time_limit="04:00:00",
                    mode="all",
                    experiments=None,
                    plot_types=_PLOT_TYPES,
                    plot_output_roots=(),
                    comparison_name=None,
                    pilot_cap=None,
                    dry_run=True,
                )

            run.assert_not_called()
            scripts = tuple((tmpdir / "jobs").glob("*.sbatch"))
            self.assertLen(scripts, 1)
            content = _normalized(scripts[0].read_text()).lower()
            self.assertIn("--mode=all", content)
            self.assertNotIn("--document_cap", content)
            self.assertNotIn("afterok", content)

    def test_compare_dry_run_writes_scripts_without_subprocess_or_dependencies(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            specs = (
                _spec(
                    tmpdir / "c4",
                    pass_gdn_state=False,
                    gdn_layer_limit=None,
                    pass_conv_state=True,
                    pass_rope_positions=False,
                    pad_id=3,
                    eos_id=4,
                ),
                _spec(
                    tmpdir / "c6",
                    c_train=6,
                    pass_gdn_state=True,
                    gdn_layer_limit=2,
                    pass_conv_state=False,
                    pass_rope_positions=True,
                    pad_id=5,
                    eos_id=6,
                ),
            )
            with mock.patch.object(eval_submitter.subprocess, "run") as run:
                eval_submitter.submit_checkpoint_evals(
                    specs,
                    manifest_path=tmpdir / "manifest.json",
                    repo_root=tmpdir / "repo",
                    job_dir=tmpdir / "jobs",
                    log_dir=tmpdir / "logs",
                    partition="gpu",
                    qos=None,
                    time_limit="04:00:00",
                    mode="compare",
                    experiments=("gdn",),
                    plot_types=("in_horizon",),
                    plot_output_roots=(tmpdir / "paper",),
                    comparison_name="dry_c4_vs_c6",
                    pilot_cap=None,
                    dry_run=True,
                )

            run.assert_not_called()
            scripts = tuple((tmpdir / "jobs").glob("*.sbatch"))
            self.assertLen(scripts, len(specs) + 1)
            contents = tuple(path.read_text().lower() for path in scripts)
            self.assertLen([text for text in contents if "--mode=compare" in text], 1)
            for content in contents:
                self.assertNotIn("afterok", content)
                self.assertNotIn("--dependency", content)

    def test_plot_mode_with_missing_condition_fails_before_inference_or_sbatch(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            checkpoint_root = tmpdir / "checkpoint"
            _partial_gdn_store(checkpoint_root, step=7)
            spec = _spec(checkpoint_root)

            with (
                mock.patch.object(
                    eval_submitter,
                    "resolve_checkpoint",
                    return_value=_resolved(checkpoint_root, step=7),
                    create=True,
                ),
                mock.patch.object(
                    runner,
                    "resolve_checkpoint",
                    return_value=_resolved(checkpoint_root, step=7),
                ),
                mock.patch.object(eval_submitter.subprocess, "run") as submit,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "(?i)(shuffled_gdn|condition|incomplete)",
                ):
                    eval_submitter.submit_checkpoint_evals(
                        (spec,),
                        manifest_path=None,
                        repo_root=tmpdir / "repo",
                        job_dir=tmpdir / "jobs",
                        log_dir=tmpdir / "logs",
                        partition="gpu",
                        qos=None,
                        time_limit="04:00:00",
                        mode="plot",
                        experiments=("gdn",),
                        plot_types=("heatmap",),
                        plot_output_roots=(tmpdir / "published",),
                        comparison_name=None,
                        pilot_cap=None,
                        dry_run=False,
                    )

            submit.assert_not_called()

    def test_complete_plot_mode_submits_one_plot_only_job_with_external_roots(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            checkpoint_root = tmpdir / "checkpoint"
            store = _partial_gdn_store(checkpoint_root, step=7)
            for shard_id in (0, 1):
                store.write_shard(
                    "gdn",
                    "shuffled_gdn",
                    shard_id,
                    tuple(
                        MetricRow(
                            experiment="gdn",
                            condition="shuffled_gdn",
                            bucket_idx=0,
                            record_idx=shard_id,
                            doc_id=f"doc-{shard_id}",
                            doc_num_chunks=2,
                            chunk_position=chunk_position,
                            nll_sum=3.0 if chunk_position == 1 else 3.1,
                            token_count=2,
                        )
                        for chunk_position in (1, 2)
                    ),
                )
            store.mark_complete()
            raw_before = _raw_inodes(store)
            resolved = _resolved(checkpoint_root, step=7)
            with (
                mock.patch.object(
                    eval_submitter,
                    "resolve_checkpoint",
                    return_value=resolved,
                    create=True,
                ),
                mock.patch.object(runner, "resolve_checkpoint", return_value=resolved),
                mock.patch.object(
                    eval_submitter.subprocess,
                    "run",
                    return_value=_submitted("Submitted batch job 150\n"),
                ) as submit,
            ):
                eval_submitter.submit_checkpoint_evals(
                    (_spec(checkpoint_root),),
                    manifest_path=None,
                    repo_root=tmpdir / "repo",
                    job_dir=tmpdir / "jobs",
                    log_dir=tmpdir / "logs",
                    partition="gpu",
                    qos=None,
                    time_limit="04:00:00",
                    mode="plot",
                    experiments=("gdn",),
                    plot_types=("exact_length", "heatmap"),
                    plot_output_roots=(tmpdir / "paper",),
                    comparison_name=None,
                    pilot_cap=None,
                    dry_run=False,
                )

            self.assertEqual(_raw_inodes(store), raw_before)
            submit.assert_called_once()
            scripts = tuple((tmpdir / "jobs").glob("*.sbatch"))
            self.assertLen(scripts, 1)
            content = _normalized(scripts[0].read_text())
            self.assertIn("--mode=plot", content)
            self.assertIn("--experiments=gdn", content)
            self.assertIn("--plot_types=exact_length,heatmap", content)
            self.assertIn(str(tmpdir / "paper"), content)

    def test_compare_rejects_heatmap_only_before_writing_or_submitting_jobs(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            job_dir = tmpdir / "jobs"
            with mock.patch.object(eval_submitter.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "(?i)(heatmap|comparison)"):
                    eval_submitter.submit_checkpoint_evals(
                        (_spec(tmpdir / "c4"), _spec(tmpdir / "c6", c_train=6)),
                        manifest_path=tmpdir / "manifest.json",
                        repo_root=tmpdir / "repo",
                        job_dir=job_dir,
                        log_dir=tmpdir / "logs",
                        partition="gpu",
                        qos=None,
                        time_limit="04:00:00",
                        mode="compare",
                        experiments=("gdn",),
                        plot_types=("heatmap",),
                        plot_output_roots=(tmpdir / "paper",),
                        comparison_name="no_common_heatmap",
                        pilot_cap=None,
                        dry_run=False,
                    )

            run.assert_not_called()
            self.assertEmpty(tuple(job_dir.glob("*.sbatch")))

    def test_explicit_compare_submits_n_checkpoint_jobs_then_one_dependent_plot_job(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            specs = (
                _spec(
                    tmpdir / "c4",
                    pass_gdn_state=False,
                    gdn_layer_limit=None,
                    pass_conv_state=True,
                    pass_rope_positions=False,
                    pad_id=3,
                    eos_id=4,
                ),
                _spec(
                    tmpdir / "c6",
                    c_train=6,
                    pass_gdn_state=True,
                    gdn_layer_limit=2,
                    pass_conv_state=False,
                    pass_rope_positions=True,
                    pad_id=5,
                    eos_id=6,
                ),
            )
            submitted = (
                _submitted("Submitted batch job 201\n"),
                _submitted("Submitted batch job 202\n"),
                _submitted("Submitted batch job 203\n"),
            )
            with mock.patch.object(
                eval_submitter.subprocess,
                "run",
                side_effect=submitted,
            ) as run:
                eval_submitter.submit_checkpoint_evals(
                    specs,
                    manifest_path=tmpdir / "manifest.json",
                    repo_root=tmpdir / "repo",
                    job_dir=tmpdir / "jobs",
                    log_dir=tmpdir / "logs",
                    partition="gpu",
                    qos="normal",
                    time_limit="04:00:00",
                    mode="compare",
                    experiments=("gdn",),
                    plot_types=("in_horizon", "heatmap"),
                    plot_output_roots=(tmpdir / "paper-a", tmpdir / "paper-b"),
                    comparison_name="c4_vs_c6",
                    pilot_cap=None,
                    dry_run=False,
                )

            self.assertEqual(run.call_count, len(specs) + 1)
            for call in run.call_args_list[: len(specs)]:
                self.assertNotIn("--dependency", " ".join(call.args[0]))
            comparison_command = run.call_args_list[-1].args[0]
            self.assertIn("--dependency=afterok:201:202", comparison_command)

            scripts = sorted((tmpdir / "jobs").glob("*.sbatch"))
            self.assertLen(scripts, len(specs) + 1)
            comparison_scripts = [
                _normalized(path.read_text())
                for path in scripts
                if "--mode=compare" in _normalized(path.read_text())
            ]
            self.assertLen(comparison_scripts, 1)
            comparison = comparison_scripts[0]
            checkpoint_scripts = [
                _normalized(path.read_text())
                for path in scripts
                if "--mode=compare" not in _normalized(path.read_text())
            ]
            self.assertLen(checkpoint_scripts, len(specs))
            for script in checkpoint_scripts:
                self.assertIn("--mode=subset", script)
                self.assertIn("--experiments=gdn", script)
                self.assertIn("--plot_types=in_horizon,heatmap", script)
                self.assertNotIn("--document_cap", script)
            self.assertIn("--comparison_name=c4_vs_c6", comparison)
            self.assertIn("--plot_types=in_horizon", comparison)
            self.assertNotRegex(
                comparison,
                r"--plot_types=[^\s\\]*heatmap",
            )
            for flag, values in (
                ("c_train", ("4", "6")),
                ("pass_gdn_state", ("false", "true")),
                ("gdn_layer_limit", ("all", "2")),
                ("pass_conv_state", ("true", "false")),
                ("pass_rope_positions", ("false", "true")),
                ("pad_id", ("3", "5")),
                ("eos_id", ("4", "6")),
            ):
                self.assertEqual(
                    re.findall(rf"--{flag}=([^\s\\]+)", comparison.lower()),
                    list(values),
                )
            for spec in specs:
                self.assertIn(str(spec.checkpoint), comparison)
            for root in (tmpdir / "paper-a", tmpdir / "paper-b"):
                self.assertIn(str(root), comparison)
            self.assertNotIn("--document_cap", comparison)
            self.assertNotIn("wandb", comparison.lower())

    def test_eval_code_has_no_wandb_or_automatic_training_hook(self):
        repo_root = Path(__file__).resolve().parents[1]
        eval_sources = tuple(sorted((repo_root / "omegalax" / "evals").rglob("*.py"))) + tuple(
            sorted((repo_root / "scripts").glob("*eval*.py"))
        )
        self.assertNotEmpty(eval_sources)
        for path in eval_sources:
            source = path.read_text().lower()
            self.assertNotIn("wandb", source, msg=str(path))
            self.assertNotIn("w&b", source, msg=str(path))

        for relative_path in (
            "scripts/train_text_pretrain.py",
            "scripts/submit_text_pretrain_slurm.py",
            "scripts/babysit_text_pretrain_slurm.py",
            "omegalax/trainers/pretrain.py",
            "omegalax/trainers/text.py",
        ):
            source = (repo_root / relative_path).read_text().lower()
            for marker in (
                "omegalax.evals",
                "submit_checkpoint_evals",
                "run_checkpoint_evals",
                "run_checkpoint_eval",
                "run_evals",
            ):
                self.assertNotIn(marker, source, msg=relative_path)


if __name__ == "__main__":
    absltest.main()
