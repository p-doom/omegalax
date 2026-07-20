"""Tests for the local checkpoint-eval worker CLI contract."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

from absl.testing import absltest

from omegalax.evals.runner import CheckpointEvalRequest
from omegalax.training_contract import ManualEvalConfig
from scripts import run_checkpoint_evals as eval_worker
from tests.pretrain_real_data_test_utils import test_temp_dir


def _spec_flags(
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


class EvalWorkerCliTest(absltest.TestCase):
    def test_contract_request_omits_manual_flags_and_partial_legacy_flags_fail(self):
        argv = [
            "--checkpoint=/tmp/checkpoint",
            "--mode=plot",
            "--plot_types=heatmap",
        ]
        with mock.patch.object(eval_worker, "run_evals") as run:
            eval_worker.run_from_arguments(argv)
        self.assertEqual(
            run.call_args.args[0],
            (CheckpointEvalRequest("/tmp/checkpoint"),),
        )

        with mock.patch.object(eval_worker, "run_evals") as run:
            with self.assertRaisesRegex(ValueError, "all seven.*missing"):
                eval_worker.run_from_arguments([*argv, "--c_train=4"])
        run.assert_not_called()

    def test_worker_cli_initializes_absl_without_reparsing_argparse_flags(self):
        command = [
            sys.executable,
            "-c",
            """
import sys

from scripts import run_checkpoint_evals as worker
from tokamax._src import config as tokamax_config

try:
    worker.run_from_arguments(sys.argv[1:])
except ValueError as error:
    assert "requires manifest_path" in str(error), error
print(tokamax_config.autotuning_cache_miss_fallback.value)
""",
            "--checkpoint=/tmp/checkpoint",
            "--c_train=4",
            "--pass_gdn_state=true",
            "--gdn_layer_limit=all",
            "--pass_conv_state=false",
            "--pass_rope_positions=false",
            "--pad_id=0",
            "--eos_id=2",
            "--mode=all",
            "--plot_types=heatmap",
        ]
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        env["OMEGALAX_DELTANET_KERNEL"] = "xla"

        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "heuristics")

    def test_one_to_many_specs_roundtrip_booleans_all_and_every_selector(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            argv = [
                *_spec_flags(
                    tmpdir / "checkpoint-a",
                    c_train=4,
                    pass_gdn_state=False,
                    gdn_layer_limit=None,
                    pass_conv_state=True,
                    pass_rope_positions=False,
                    pad_id=3,
                    eos_id=4,
                ),
                *_spec_flags(
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
                "--plot_types=in_horizon,beyond_horizon,exact_length,heatmap",
                f"--plot_output_root={tmpdir / 'paper-a'}",
                f"--plot_output_root={tmpdir / 'paper-b'}",
                "--document_cap=2",
                "--tp_size=1",
                "--fsdp_size=1",
                "--dp_size=4",
                "--batch_size=16",
            ]
            self.assertIsNotNone(eval_worker.parse_worker_args(argv))
            with mock.patch.object(eval_worker, "run_evals") as run:
                eval_worker.run_from_arguments(argv)

            run.assert_called_once()
            self.assertEqual(
                run.call_args.args[0],
                (
                    CheckpointEvalRequest(
                        str(tmpdir / "checkpoint-a"),
                        ManualEvalConfig(
                            c_train=4,
                            pass_gdn_state=False,
                            gdn_layer_limit=None,
                            pass_conv_state=True,
                            pass_rope_positions=False,
                            pad_id=3,
                            eos_id=4,
                        ),
                    ),
                    CheckpointEvalRequest(
                        str(tmpdir / "checkpoint-b"),
                        ManualEvalConfig(
                            c_train=6,
                            pass_gdn_state=True,
                            gdn_layer_limit=2,
                            pass_conv_state=False,
                            pass_rope_positions=True,
                            pad_id=5,
                            eos_id=6,
                        ),
                    ),
                ),
            )
            kwargs = run.call_args.kwargs
            self.assertEqual(kwargs["mode"], "subset")
            self.assertEqual(Path(kwargs["manifest_path"]), tmpdir / "manifest.json")
            self.assertEqual(kwargs["experiments"], ("gdn", "conv"))
            self.assertEqual(
                kwargs["plot_types"],
                ("in_horizon", "beyond_horizon", "exact_length", "heatmap"),
            )
            self.assertEqual(
                tuple(Path(path) for path in kwargs["plot_output_roots"]),
                (tmpdir / "paper-a", tmpdir / "paper-b"),
            )
            self.assertEqual(kwargs["document_cap"], 2)
            self.assertEqual(
                tuple(kwargs[name] for name in ("tp_size", "fsdp_size", "dp_size", "batch_size")),
                (1, 1, 4, 16),
            )

    def test_inference_modes_require_manifest_and_aligned_spec_fields(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            base = _spec_flags(
                tmpdir / "checkpoint",
                c_train=4,
                pass_gdn_state=True,
                gdn_layer_limit=None,
                pass_conv_state=True,
                pass_rope_positions=True,
                pad_id=0,
                eos_id=2,
            )
            for mode in ("all", "subset"):
                with self.subTest(mode=mode):
                    with mock.patch.object(eval_worker, "run_evals") as run:
                        with self.assertRaisesRegex(ValueError, "(?i)manifest"):
                            eval_worker.run_from_arguments(
                                [
                                    *base,
                                    f"--mode={mode}",
                                    *(["--experiments=gdn"] if mode == "subset" else []),
                                    "--plot_types=exact_length",
                                ]
                            )
                    run.assert_not_called()

            with mock.patch.object(eval_worker, "run_evals") as run:
                with self.assertRaisesRegex(ValueError, "(?i)(aligned|length|c_train)"):
                    eval_worker.run_from_arguments(
                        [
                            *base,
                            "--c_train=6",
                            "--mode=all",
                            f"--manifest_path={tmpdir / 'manifest.json'}",
                            "--plot_types=heatmap",
                        ]
                    )
            run.assert_not_called()

            with mock.patch.object(eval_worker, "run_evals") as run:
                eval_worker.run_from_arguments(
                    [
                        *base,
                        "--mode=all",
                        f"--manifest_path={tmpdir / 'manifest.json'}",
                        "--plot_types=in_horizon,beyond_horizon,exact_length,heatmap",
                    ]
                )
            run.assert_called_once()
            self.assertEqual(run.call_args.kwargs["mode"], "all")
            self.assertIsNone(run.call_args.kwargs["experiments"])

    def test_every_override_list_must_align_and_unknown_experiments_are_rejected(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            first = _spec_flags(
                tmpdir / "checkpoint-a",
                c_train=4,
                pass_gdn_state=True,
                gdn_layer_limit=None,
                pass_conv_state=True,
                pass_rope_positions=True,
                pad_id=0,
                eos_id=2,
            )
            second = _spec_flags(
                tmpdir / "checkpoint-b",
                c_train=6,
                pass_gdn_state=False,
                gdn_layer_limit=1,
                pass_conv_state=False,
                pass_rope_positions=False,
                pad_id=3,
                eos_id=4,
            )
            base = [*first, *second, "--mode=plot", "--plot_types=heatmap"]
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
                    with mock.patch.object(eval_worker, "run_evals") as run:
                        with self.assertRaisesRegex(
                            ValueError,
                            f"(?i)(aligned|length|{flag})",
                        ):
                            eval_worker.run_from_arguments([*base, f"--{flag}={extra_value}"])
                    run.assert_not_called()

            for invalid_experiment in ("rope", "unknown"):
                with self.subTest(invalid_experiment=invalid_experiment):
                    with mock.patch.object(eval_worker, "run_evals") as run:
                        with self.assertRaisesRegex(ValueError, "(?i)experiment"):
                            eval_worker.run_from_arguments(
                                [
                                    *first,
                                    "--mode=plot",
                                    f"--experiments=gdn,{invalid_experiment}",
                                    "--plot_types=in_horizon",
                                ]
                            )
                    run.assert_not_called()

    def test_plot_needs_no_manifest_but_compare_requires_two_specs(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            first = _spec_flags(
                tmpdir / "checkpoint-a",
                c_train=4,
                pass_gdn_state=True,
                gdn_layer_limit=None,
                pass_conv_state=True,
                pass_rope_positions=True,
                pad_id=0,
                eos_id=2,
            )
            with mock.patch.object(eval_worker, "run_evals") as run:
                eval_worker.run_from_arguments(
                    [
                        *first,
                        "--mode=plot",
                        "--experiments=gdn",
                        "--plot_types=heatmap",
                    ]
                )
            run.assert_called_once()
            self.assertEqual(run.call_args.kwargs["mode"], "plot")
            self.assertIsNone(run.call_args.kwargs["manifest_path"])

            with mock.patch.object(eval_worker, "run_evals") as run:
                with self.assertRaisesRegex(ValueError, "(?i)(two|2|multiple|compare)"):
                    eval_worker.run_from_arguments(
                        [
                            *first,
                            "--mode=compare",
                            "--experiments=gdn",
                            "--plot_types=in_horizon",
                            "--comparison_name=one_is_not_a_comparison",
                        ]
                    )
            run.assert_not_called()

            second = _spec_flags(
                tmpdir / "checkpoint-b",
                c_train=6,
                pass_gdn_state=False,
                gdn_layer_limit=1,
                pass_conv_state=False,
                pass_rope_positions=False,
                pad_id=7,
                eos_id=8,
            )
            with mock.patch.object(eval_worker, "run_evals") as run:
                with self.assertRaisesRegex(ValueError, "(?i)comparison.*name"):
                    eval_worker.run_from_arguments(
                        [
                            *first,
                            *second,
                            "--mode=compare",
                            "--experiments=gdn",
                            "--plot_types=in_horizon",
                        ]
                    )
            run.assert_not_called()

            with mock.patch.object(eval_worker, "run_evals") as run:
                eval_worker.run_from_arguments(
                    [
                        *first,
                        *second,
                        "--mode=compare",
                        "--experiments=gdn",
                        "--plot_types=in_horizon",
                        "--comparison_name=c4_vs_c6",
                    ]
                )
            run.assert_called_once()
            self.assertLen(run.call_args.args[0], 2)
            self.assertEqual(run.call_args.kwargs["mode"], "compare")
            self.assertEqual(run.call_args.kwargs["comparison_name"], "c4_vs_c6")

    def test_mode_and_plot_type_are_closed_public_choices(self):
        with self.assertRaises(SystemExit):
            eval_worker.parse_worker_args([])
        with test_temp_dir() as tmp:
            base = _spec_flags(
                Path(tmp) / "checkpoint",
                c_train=4,
                pass_gdn_state=True,
                gdn_layer_limit=None,
                pass_conv_state=True,
                pass_rope_positions=True,
                pad_id=0,
                eos_id=2,
            )
            with self.assertRaises(SystemExit):
                eval_worker.parse_worker_args([*base, "--mode=not-a-mode", "--plot_types=heatmap"])
            with self.assertRaises(SystemExit):
                eval_worker.parse_worker_args(
                    [
                        *base,
                        "--mode=plot",
                        "--plot_types=in_horizon,confidence_interval",
                    ]
                )


if __name__ == "__main__":
    absltest.main()
