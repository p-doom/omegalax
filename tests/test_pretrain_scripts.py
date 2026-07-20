"""Tests for pretraining helper scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest
from absl.testing import flagsaver

from scripts import babysit_text_pretrain_slurm
from scripts import build_pretrain_indexes
from scripts import launch_text_pretrain_experiments
from scripts import submit_text_pretrain_slurm
from omegalax.trainers.pretrain import PretrainMode


class BuildPretrainIndexScriptTest(absltest.TestCase):
    def test_calls_window_builder_with_forwarded_args(self):
        with mock.patch.object(
            build_pretrain_indexes, "build_statepassing_window_index"
        ) as window_builder:
            window_builder.return_value = Path("/tmp/index")
            out = build_pretrain_indexes.build_pretrain_index(
                root="/data/root",
                out_dir="/indexes",
                split="train",
                chunk_length=123,
                num_segments=6,
                eos_id=456,
                records_per_shard=789,
                overwrite=True,
            )

        self.assertEqual(out, Path("/tmp/index"))
        window_builder.assert_called_once()
        args, kwargs = window_builder.call_args
        self.assertEqual(args[0], "/data/root")
        self.assertEqual(args[1], Path("/indexes").resolve() / "train")
        self.assertEqual(kwargs["chunk_length"], 123)
        self.assertEqual(kwargs["num_segments"], 6)
        self.assertEqual(kwargs["eos_id"], 456)
        self.assertEqual(kwargs["split"], "train")
        self.assertEqual(kwargs["records_per_shard"], 789)
        self.assertTrue(kwargs["overwrite"])

    def test_val_split_uses_val_index_dir(self):
        with mock.patch.object(
            build_pretrain_indexes, "build_statepassing_window_index"
        ) as statepassing_builder:
            statepassing_builder.return_value = Path("/tmp/statepassing")
            out = build_pretrain_indexes.build_pretrain_index(
                root="/data/root",
                out_dir="/indexes",
                split="val",
                chunk_length=4096,
                num_segments=6,
                eos_id=248046,
                records_per_shard=1000,
                overwrite=False,
            )

        self.assertEqual(out, Path("/tmp/statepassing"))
        statepassing_builder.assert_called_once()
        args, kwargs = statepassing_builder.call_args
        self.assertEqual(args[0], "/data/root")
        self.assertEqual(args[1], Path("/indexes").resolve() / "val")
        self.assertEqual(kwargs["split"], "val")
        self.assertEqual(kwargs["num_segments"], 6)

    def test_calls_curriculum_builder_with_forwarded_args(self):
        with mock.patch.object(
            build_pretrain_indexes, "build_statepassing_curriculum_indexes"
        ) as curriculum_builder:
            curriculum_builder.return_value = Path("/tmp/curriculum")
            out = build_pretrain_indexes.build_curriculum_pretrain_indexes(
                root="/data/root",
                out_dir="/indexes",
                splits=["train", "val"],
                chunk_length=123,
                allocation_order=[6, 4, 2],
                train_order=[2, 4, 6],
                max_tokens_by_num_segments={2: 1000},
                trim_batch_size=240,
                trim_grad_accum_steps=4,
                eos_id=456,
                records_per_shard=789,
                overwrite=True,
            )

        self.assertEqual(out, Path("/tmp/curriculum"))
        curriculum_builder.assert_called_once()
        args, kwargs = curriculum_builder.call_args
        self.assertEqual(args[0], "/data/root")
        self.assertEqual(args[1], "/indexes")
        self.assertEqual(kwargs["splits"], ["train", "val"])
        self.assertEqual(kwargs["chunk_length"], 123)
        self.assertEqual(kwargs["allocation_order"], [6, 4, 2])
        self.assertEqual(kwargs["train_order"], [2, 4, 6])
        self.assertEqual(kwargs["max_tokens_by_num_segments"], {2: 1000})
        self.assertEqual(kwargs["trim_batch_size"], 240)
        self.assertEqual(kwargs["trim_grad_accum_steps"], 4)
        self.assertEqual(kwargs["eos_id"], 456)
        self.assertEqual(kwargs["records_per_shard"], 789)
        self.assertTrue(kwargs["overwrite"])

    def test_curriculum_flag_parsers(self):
        self.assertEqual(
            build_pretrain_indexes.parse_curriculum_order("16,12,8,4", flag_name="x"),
            [16, 12, 8, 4],
        )
        self.assertEqual(
            build_pretrain_indexes.parse_curriculum_max_tokens(["2:25", "4:10,8:20"]),
            {2: 25, 4: 10, 8: 20},
        )

    def test_curriculum_max_tokens_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_pretrain_indexes.parse_curriculum_max_tokens(["2:25", "2:30"])

    def test_calls_fixed_c_builder_with_forwarded_args(self):
        with mock.patch.object(
            build_pretrain_indexes, "build_statepassing_fixed_c_indexes"
        ) as fixed_builder:
            fixed_builder.return_value = Path("/tmp/fixed")
            out = build_pretrain_indexes.build_fixed_c_pretrain_indexes(
                root="/data/root",
                out_dir="/indexes",
                splits=["train", "val"],
                chunk_length=123,
                num_segments=6,
                trim_batch_size=240,
                trim_grad_accum_steps=4,
                eos_id=456,
                records_per_shard=789,
                overwrite=True,
            )

        self.assertEqual(out, Path("/tmp/fixed"))
        fixed_builder.assert_called_once()
        args, kwargs = fixed_builder.call_args
        self.assertEqual(args[0], "/data/root")
        self.assertEqual(args[1], "/indexes")
        self.assertEqual(kwargs["splits"], ["train", "val"])
        self.assertEqual(kwargs["chunk_length"], 123)
        self.assertEqual(kwargs["num_segments"], 6)
        self.assertEqual(kwargs["trim_batch_size"], 240)
        self.assertEqual(kwargs["trim_grad_accum_steps"], 4)
        self.assertEqual(kwargs["eos_id"], 456)
        self.assertEqual(kwargs["records_per_shard"], 789)
        self.assertTrue(kwargs["overwrite"])


class TrainTextPretrainScriptTest(absltest.TestCase):
    def _run_train_script_probe(self, body: str) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, "-c", body],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return json.loads(result.stdout)

    def test_make_statepassing_iterator_uses_runtime_flags_and_batch(self):
        output = self._run_train_script_probe(
            """
import json
from unittest import mock
from absl.testing import flagsaver
from scripts import train_text_pretrain
from omegalax.trainers.pretrain import PretrainMode

train_text_pretrain.FLAGS(["probe", "--train_index_path=/idx"])

with flagsaver.flagsaver(
    seq_len=2048,
    pad_id=0,
    eos_id=248046,
    seed=123,
    iterator_dp_size=None,
    iterator_fsdp_size=None,
    dp_size=2,
    fsdp_size=4,
    grain_workers=3,
    grain_worker_buffer_size=5,
    grain_read_threads=7,
    grain_read_prefetch_buffer_size=11,
):
    with mock.patch.object(train_text_pretrain, "make_statepassing_iterator") as iterator:
        iterator.return_value = object()
        train_text_pretrain._make_iterator(
            "/idx", PretrainMode.STATEPASSING_BPTT, 12, shuffle=True
        )
        args, kwargs = iterator.call_args

print(json.dumps({
    "index_path": args[0],
    "batch_size": kwargs["batch_size"],
    "chunk_length": kwargs["chunk_length"],
    "seed": kwargs["seed"],
    "dp_size": kwargs["dp_size"],
    "fsdp_size": kwargs["fsdp_size"],
    "grain_workers": kwargs["grain_workers"],
    "shuffle": kwargs["shuffle"],
}))
"""
        )

        self.assertEqual(output["index_path"], "/idx")
        self.assertEqual(output["batch_size"], 12)
        self.assertEqual(output["chunk_length"], 2048)
        self.assertEqual(output["seed"], 123)
        self.assertEqual(output["dp_size"], 2)
        self.assertEqual(output["fsdp_size"], 4)
        self.assertEqual(output["grain_workers"], 3)
        self.assertTrue(output["shuffle"])

    def test_runtime_kwargs_forward_statepassing_flags(self):
        output = self._run_train_script_probe(
            """
import json
from absl.testing import flagsaver
from scripts import train_text_pretrain

train_text_pretrain.FLAGS(["probe", "--train_index_path=/idx"])

with flagsaver.flagsaver(
    bptt_chunks=4,
    pass_gdn_state=False,
    gdn_layer_limit=1,
    pass_rope_positions=True,
    pass_conv_state=True,
):
    kwargs = train_text_pretrain._runtime_kwargs()

print(json.dumps(kwargs))
"""
        )

        self.assertEqual(output["bptt_chunks"], 4)
        self.assertFalse(output["pass_gdn_state"])
        self.assertEqual(output["gdn_layer_limit"], 1)
        self.assertTrue(output["pass_rope_positions"])
        self.assertTrue(output["pass_conv_state"])

    def test_curriculum_runtime_config_accepts_matching_shape(self):
        output = self._run_train_script_probe(
            """
import json
from absl.testing import flagsaver
from scripts import train_text_pretrain

train_text_pretrain.FLAGS(["probe", "--train_index_path=/idx"])

with flagsaver.flagsaver(batch_size=240, grad_accum_steps=4):
    train_text_pretrain._validate_curriculum_runtime_config({
        "trim_batch_size": 240,
        "trim_grad_accum_steps": 4,
    })

print(json.dumps({"ok": True}))
"""
        )

        self.assertTrue(output["ok"])

    def test_curriculum_runtime_config_rejects_mismatched_shape(self):
        output = self._run_train_script_probe(
            """
import json
from absl.testing import flagsaver
from scripts import train_text_pretrain

train_text_pretrain.FLAGS(["probe", "--train_index_path=/idx"])

with flagsaver.flagsaver(batch_size=128, grad_accum_steps=4):
    try:
        train_text_pretrain._validate_curriculum_runtime_config({
            "trim_batch_size": 240,
            "trim_grad_accum_steps": 4,
        })
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
"""
        )

        self.assertIn("batch_size=240", output["error"])

    def test_fixed_c_runtime_config_rejects_mismatched_shape(self):
        output = self._run_train_script_probe(
            """
import json
from absl.testing import flagsaver
from scripts import train_text_pretrain

train_text_pretrain.FLAGS(["probe", "--train_index_path=/idx"])

with flagsaver.flagsaver(batch_size=240, grad_accum_steps=2):
    try:
        train_text_pretrain._validate_trimmed_index_runtime_config(
            {
                "trim_batch_size": 240,
                "trim_grad_accum_steps": 4,
            },
            label="Fixed-C",
        )
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
"""
        )

        self.assertIn("grad_accum_steps=4", output["error"])

    def test_fixed_c_bundle_path_helpers_select_mode_paths_and_steps(self):
        output = self._run_train_script_probe(
            """
import json
from pathlib import Path
from scripts import train_text_pretrain
from omegalax.trainers.pretrain import PretrainMode

train_text_pretrain.FLAGS(["probe", "--train_index_path=/idx"])

split_metadata = {
    "path": "train",
    "statepassing_steps": 12,
    "iid": {"path": "iid/train", "iid_steps": 36},
}
root = Path("/bundle")
print(json.dumps({
    "statepassing_path": str(train_text_pretrain._fixed_c_index_path(
        root, split_metadata, PretrainMode.STATEPASSING_BPTT
    )),
    "iid_path": str(train_text_pretrain._fixed_c_index_path(
        root, split_metadata, PretrainMode.IID_BASELINE
    )),
    "statepassing_steps": train_text_pretrain._fixed_c_steps(
        split_metadata, PretrainMode.STATEPASSING_NO_BPTT
    ),
    "iid_steps": train_text_pretrain._fixed_c_steps(
        split_metadata, PretrainMode.IID_BASELINE
    ),
}))
"""
        )

        self.assertEqual(output["statepassing_path"], "/bundle/train")
        self.assertEqual(output["iid_path"], "/bundle/iid/train")
        self.assertEqual(output["statepassing_steps"], 12)
        self.assertEqual(output["iid_steps"], 36)

    def test_eos_validation_rejects_wrong_dominant_id(self):
        with self.assertRaisesRegex(ValueError, "EOS sanity check failed"):
            build_pretrain_indexes.validate_eos_stats(
                {"sampled": 10, "dominant_id": 248044, "dominant_fraction": 0.9},
                eos_id=248046,
                min_fraction=0.8,
                split="train",
            )

    def test_eos_validation_accepts_matching_dominant_id(self):
        build_pretrain_indexes.validate_eos_stats(
            {"sampled": 10, "dominant_id": 248046, "dominant_fraction": 1.0},
            eos_id=248046,
            min_fraction=0.95,
            split="train",
        )


class LaunchTextPretrainExperimentsScriptTest(absltest.TestCase):
    def test_build_commands_uses_same_extra_args_and_distinct_modes(self):
        commands = launch_text_pretrain_experiments.build_commands(
            index_root="/idx",
            save_root="/runs",
            modes=["iid_baseline", "statepassing_no_bptt", "statepassing_bptt"],
            wandb_name_prefix="cmp",
            extra_args=["--batch_size=8", "--seed=123"],
        )

        self.assertLen(commands, 3)
        rendered = [" ".join(command) for command in commands]
        self.assertIn("--pretrain_mode=iid_baseline", rendered[0])
        self.assertIn("--pretrain_mode=statepassing_no_bptt", rendered[1])
        self.assertIn("--pretrain_mode=statepassing_bptt", rendered[2])
        for command in rendered:
            self.assertIn("--batch_size=8", command)
            self.assertIn("--seed=123", command)
            self.assertIn("--tp_size=1", command)
            self.assertIn("--fsdp_size=1", command)
            self.assertIn("--dp_size=1", command)
            self.assertIn(f"--train_index_path={Path('/idx').resolve()}", command)
            self.assertNotIn("--val_index_path=", command)
            self.assertIn("--save_dir=", command)


class SubmitTextPretrainSlurmScriptTest(absltest.TestCase):
    def test_validate_submit_shape_rejects_invalid_statepassing_batch(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            submit_text_pretrain_slurm.validate_submit_shape(
                batch_size=34, nodes=1, gpus_per_node=8
            )

    def test_validate_submit_shape_accepts_two_gpu_adjustment(self):
        self.assertEqual(
            submit_text_pretrain_slurm.validate_submit_shape(
                batch_size=32, nodes=1, gpus_per_node=2
            ),
            2,
        )

    def test_validate_submit_shape_rejects_fsdp_that_does_not_divide_hidden_size(self):
        with self.assertRaisesRegex(ValueError, "must divide hidden_size"):
            submit_text_pretrain_slurm.validate_submit_shape(
                batch_size=60, nodes=1, gpus_per_node=5
            )

    def test_validate_submit_shape_allows_single_process_dp_non_fsdp_divisor(self):
        self.assertEqual(
            submit_text_pretrain_slurm.validate_submit_shape(
                batch_size=60,
                nodes=1,
                gpus_per_node=5,
                single_process_per_run=True,
            ),
            5,
        )

    def test_parse_run_specs_validates_default_shape(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            submit_text_pretrain_slurm.parse_run_specs(
                [],
                nodes_per_run=1,
                default_gpus_per_node=3,
                default_batch_size=128,
                default_grad_accum_steps=4,
            )

    def test_parse_run_specs_allows_unequal_gpus_with_same_batch_and_accum(self):
        specs = submit_text_pretrain_slurm.parse_run_specs(
            [
                "iid_baseline:6:48:11",
                "statepassing_no_bptt:3:48:11",
                "statepassing_bptt:3:48:11",
            ],
            nodes_per_run=1,
            default_gpus_per_node=8,
            default_batch_size=128,
            default_grad_accum_steps=4,
        )

        self.assertEqual([spec.gpus_per_node for spec in specs], [6, 3, 3])
        self.assertEqual({spec.batch_size for spec in specs}, {48})
        self.assertEqual({spec.grad_accum_steps for spec in specs}, {11})

    def test_parse_run_specs_allows_selected_single_mode(self):
        specs = submit_text_pretrain_slurm.parse_run_specs(
            [],
            nodes_per_run=1,
            default_gpus_per_node=8,
            default_batch_size=192,
            default_grad_accum_steps=5,
            single_process_per_run=True,
            modes=[submit_text_pretrain_slurm.PretrainMode.STATEPASSING_BPTT],
        )

        self.assertLen(specs, 1)
        self.assertEqual(specs[0].mode, submit_text_pretrain_slurm.PretrainMode.STATEPASSING_BPTT)
        self.assertEqual(specs[0].batch_size, 192)
        self.assertEqual(specs[0].grad_accum_steps, 5)

    def test_parse_run_specs_allows_iid_only_without_statepassing_c_divisibility(self):
        specs = submit_text_pretrain_slurm.parse_run_specs(
            [],
            nodes_per_run=1,
            default_gpus_per_node=2,
            default_batch_size=10,
            default_grad_accum_steps=5,
            modes=[submit_text_pretrain_slurm.PretrainMode.IID_BASELINE],
        )

        self.assertLen(specs, 1)
        self.assertEqual(specs[0].mode, submit_text_pretrain_slurm.PretrainMode.IID_BASELINE)
        self.assertEqual(specs[0].batch_size, 10)

    def test_parse_run_specs_validates_all_curriculum_segment_lengths(self):
        with flagsaver.flagsaver(
            submit_curriculum_allocation_order="12,8,4,2,1",
            submit_curriculum_train_order="1,2,4,8,12",
        ):
            specs = submit_text_pretrain_slurm.parse_run_specs(
                [],
                nodes_per_run=1,
                default_gpus_per_node=8,
                default_batch_size=192,
                default_grad_accum_steps=5,
                single_process_per_run=True,
                modes=[submit_text_pretrain_slurm.PretrainMode.STATEPASSING_BPTT],
            )

            self.assertEqual(specs[0].batch_size, 192)

            with self.assertRaisesRegex(ValueError, "num_segments \\* total_tasks=96"):
                submit_text_pretrain_slurm.parse_run_specs(
                    [],
                    nodes_per_run=1,
                    default_gpus_per_node=8,
                    default_batch_size=240,
                    default_grad_accum_steps=4,
                    single_process_per_run=True,
                    modes=[submit_text_pretrain_slurm.PretrainMode.STATEPASSING_BPTT],
                )

    def test_parse_run_specs_rejects_unequal_batch_or_accum(self):
        with self.assertRaisesRegex(ValueError, "same batch_size"):
            submit_text_pretrain_slurm.parse_run_specs(
                [
                    "iid_baseline:7:112:6",
                    "statepassing_no_bptt:4:56:12",
                    "statepassing_bptt:3:48:14",
                ],
                nodes_per_run=1,
                default_gpus_per_node=8,
                default_batch_size=128,
                default_grad_accum_steps=4,
            )

    def test_submit_seq_len_controls_index_and_train_flags(self):
        with flagsaver.flagsaver(
            submit_seq_len=2048,
            submit_num_segments=6,
            submit_bptt_chunks=4,
            submit_pass_gdn_state=False,
            submit_gdn_layer_limit=1,
            submit_pass_rope_positions=True,
            submit_pass_conv_state=True,
        ):
            index_script = submit_text_pretrain_slurm.render_index_sbatch(
                repo_root="/repo",
                dataset_root="/data",
                index_root="/idx",
                log_dir="/logs",
                run_id="run",
                partition="standard",
                qos=None,
                time_limit="12:00:00",
                cpus_per_task=12,
                records_per_shard=100,
                eos_check_records=10,
                min_eos_fraction=0.95,
                overwrite=False,
            )
            train_flags = submit_text_pretrain_slurm._train_flags(
                mode=PretrainMode.IID_BASELINE,
                save_root=Path("/runs"),
                jax_cache_root=Path("/jax_cache"),
                run_id="run",
                total_devices=8,
                jax_processes=8,
                batch_size=128,
                grad_accum_steps=4,
                single_process_per_run=False,
                single_process_per_node=False,
            )

        self.assertIn("--chunk_length=2048", index_script)
        self.assertIn("--num_segments=6", index_script)
        self.assertIn("--fixed_trim_batch_size=128", index_script)
        self.assertIn("--fixed_trim_grad_accum_steps=4", index_script)
        self.assertNotIn("--pretrain_mode=", index_script)
        self.assertIn("--seq_len=2048", train_flags)
        self.assertNotIn("--num_segments=6", train_flags)
        self.assertIn("--train_index_path=${TRAIN_INDEX_ROOT}", train_flags)
        self.assertFalse(any(flag.startswith("--val_index_path") for flag in train_flags))
        self.assertIn("--bptt_chunks=4", train_flags)
        self.assertIn("--pass_gdn_state=False", train_flags)
        self.assertIn("--gdn_layer_limit=1", train_flags)
        self.assertIn("--pass_rope_positions=True", train_flags)
        self.assertIn("--pass_conv_state=True", train_flags)

    def test_train_flags_forward_run_specific_submit_overrides(self):
        with flagsaver.flagsaver(
            submit_model_id="/configs/xl",
            submit_lr_end_factor=0.25,
            submit_lr_schedule_steps=1234,
            submit_save_every=2000,
            submit_keep_latest=10,
            submit_grain_read_threads=4,
            submit_grain_read_prefetch_buffer_size=8,
            submit_grain_workers=16,
            submit_grain_worker_buffer_size=2,
        ):
            train_flags = submit_text_pretrain_slurm._train_flags(
                mode=PretrainMode.STATEPASSING_BPTT,
                save_root=Path("/runs"),
                jax_cache_root=Path("/jax_cache"),
                run_id="run",
                total_devices=16,
                jax_processes=16,
                batch_size=192,
                grad_accum_steps=5,
                single_process_per_run=False,
                single_process_per_node=False,
            )

        self.assertIn("--model_id=/configs/xl", train_flags)
        self.assertIn("--lr_end_factor=0.25", train_flags)
        self.assertIn("--lr_schedule_steps=1234", train_flags)
        self.assertIn("--save_every=2000", train_flags)
        self.assertIn("--keep_latest=10", train_flags)
        self.assertIn("--grain_read_threads=4", train_flags)
        self.assertIn("--grain_read_prefetch_buffer_size=8", train_flags)
        self.assertIn("--grain_workers=16", train_flags)
        self.assertIn("--grain_worker_buffer_size=2", train_flags)

    def test_render_train_sbatch_can_request_memory(self):
        script = submit_text_pretrain_slurm.render_train_sbatch(
            repo_root="/repo",
            source_root="/src",
            dataset_root="/src/data",
            index_root="/src/index",
            save_root="/runs",
            log_dir="/logs",
            jax_cache_root="/jax_cache",
            run_id="run",
            mode=PretrainMode.STATEPASSING_BPTT,
            partition="standard",
            qos="normal",
            time_limit="24:00:00",
            nodes=2,
            gpus_per_node=8,
            batch_size=192,
            grad_accum_steps=5,
            cpus_per_task=24,
            stage_to_scratch=False,
            run_pallas_tests=False,
            single_process_per_run=False,
            single_process_per_node=False,
            mem="120G",
        )

        self.assertIn("#SBATCH --mem=120G", script)

    def test_render_index_sbatch_uses_curriculum_flags(self):
        with flagsaver.flagsaver(
            submit_seq_len=2048,
            submit_num_segments=6,
            submit_curriculum_allocation_order="12,8,4,2,1",
            submit_curriculum_train_order="1,2,4,8,12",
        ):
            index_script = submit_text_pretrain_slurm.render_index_sbatch(
                repo_root="/repo",
                dataset_root="/data",
                index_root="/idx",
                log_dir="/logs",
                run_id="run",
                partition="standard",
                qos="normal",
                time_limit="12:00:00",
                cpus_per_task=12,
                records_per_shard=100,
                eos_check_records=10,
                min_eos_fraction=0.95,
                overwrite=False,
                trim_batch_size=192,
                trim_grad_accum_steps=5,
            )

        self.assertIn("--chunk_length=2048", index_script)
        self.assertIn("--curriculum_allocation_order=12,8,4,2,1", index_script)
        self.assertIn("--curriculum_train_order=1,2,4,8,12", index_script)
        self.assertIn("--curriculum_trim_batch_size=192", index_script)
        self.assertIn("--curriculum_trim_grad_accum_steps=5", index_script)
        self.assertNotIn("--fixed_trim_batch_size", index_script)
        self.assertNotIn("--num_segments=6", index_script)

    def test_indexes_ready_accepts_curriculum_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in (
                "c1/train",
                "c1/val",
                "c2/train",
                "c2/val",
                "iid/train",
                "iid/val",
            ):
                (root / path).mkdir(parents=True)
                (root / path / "metadata.json").write_text("{}")
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "format": submit_text_pretrain_slurm._STATEPASSING_CURRICULUM_INDEX_FORMAT,
                        "splits": {
                            "train": {
                                "phases": {
                                    "1": {"path": "c1/train"},
                                    "2": {"path": "c2/train"},
                                },
                                "iid": {"path": "iid/train"},
                            },
                            "val": {
                                "phases": {
                                    "1": {"path": "c1/val"},
                                    "2": {"path": "c2/val"},
                                },
                                "iid": {"path": "iid/val"},
                            },
                        },
                    }
                )
            )

            self.assertTrue(
                submit_text_pretrain_slurm.indexes_ready(
                    root,
                    expected_format=submit_text_pretrain_slurm._STATEPASSING_CURRICULUM_INDEX_FORMAT,
                )
            )
            self.assertFalse(
                submit_text_pretrain_slurm.indexes_ready(
                    root,
                    expected_format=submit_text_pretrain_slurm._STATEPASSING_FIXED_C_INDEX_FORMAT,
                )
            )

    def test_submit_defaults_use_current_2048_dataset(self):
        self.assertIn(
            "fineweb_edu_dedup_2048_2kto32k", str(submit_text_pretrain_slurm._DATASET_ROOT)
        )
        self.assertIn("2048_eos248046", str(submit_text_pretrain_slurm._INDEX_ROOT))

    def test_validate_submit_shape_respects_num_segments(self):
        with self.assertRaisesRegex(ValueError, "num_segments"):
            submit_text_pretrain_slurm.validate_submit_shape(
                batch_size=40,
                nodes=1,
                gpus_per_node=4,
                num_segments=6,
            )

    def test_render_train_sbatch_requests_gpus_mesh_and_preflight(self):
        script = submit_text_pretrain_slurm.render_train_sbatch(
            repo_root="/repo",
            source_root="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan",
            dataset_root=(
                "/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/"
                "datasets/fineweb_edu_dedup_30b_8kto32k"
            ),
            index_root=(
                "/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/pretrain_indexes/fineweb"
            ),
            save_root="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/runs",
            log_dir="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs",
            jax_cache_root="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/jax_cache",
            run_id="run",
            mode=PretrainMode.STATEPASSING_BPTT,
            partition="standard",
            qos="low",
            time_limit="24:00:00",
            nodes=1,
            gpus_per_node=8,
            batch_size=128,
            grad_accum_steps=4,
            cpus_per_task=12,
            stage_to_scratch=True,
            run_pallas_tests=True,
            single_process_per_run=False,
            single_process_per_node=False,
        )

        self.assertIn("#SBATCH --gres=gpu:8", script)
        self.assertIn("#SBATCH --qos=low", script)
        self.assertIn("#SBATCH --signal=USR1@1800", script)
        self.assertIn("#SBATCH --requeue", script)
        self.assertIn("#SBATCH --ntasks-per-node=8", script)
        self.assertIn("JAX_PLATFORMS=cuda", script)
        self.assertIn('--fsdp_size="8"', script)
        self.assertIn('--pretrain_mode="statepassing_bptt"', script)
        self.assertIn('--adam_beta2="0.95"', script)
        self.assertIn("OMEGALAX_PRETRAIN_SOURCE_ROOT", script)
        self.assertIn("JAX_LOCAL_DEVICE_IDS=0", script)
        self.assertIn("WANDB_MODE=online", script)
        self.assertIn("WANDB_DIR=", script)
        self.assertIn("pytest tests/test_gated_delta_rule_pallas.py", script)
        self.assertIn("WANDB_MODE=disabled", script)
        self.assertIn("--gpus-per-task=1", script)
        self.assertNotIn('CUDA_VISIBLE_DEVICES="${SLURM_LOCALID}"', script)

    def test_render_train_sbatch_omits_wandb_env_when_project_disabled(self):
        with flagsaver.flagsaver(submit_wandb_project=""):
            script = submit_text_pretrain_slurm.render_train_sbatch(
                repo_root="/repo",
                source_root="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan",
                dataset_root=(
                    "/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/"
                    "datasets/fineweb_edu_dedup_30b_8kto32k"
                ),
                index_root=(
                    "/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/pretrain_indexes/fineweb"
                ),
                save_root="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/runs",
                log_dir="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs",
                jax_cache_root="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/jax_cache",
                run_id="run",
                mode=PretrainMode.IID_BASELINE,
                partition="standard",
                qos=None,
                time_limit="24:00:00",
                nodes=1,
                gpus_per_node=8,
                batch_size=128,
                grad_accum_steps=4,
                cpus_per_task=12,
                stage_to_scratch=False,
                run_pallas_tests=False,
                single_process_per_run=False,
                single_process_per_node=False,
            )

        self.assertNotIn("WANDB_MODE=online", script)
        self.assertNotIn("--wandb_project", script)

    def test_render_train_sbatch_can_use_single_jax_process_for_local_gpus(self):
        script = submit_text_pretrain_slurm.render_train_sbatch(
            repo_root="/repo",
            source_root="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan",
            dataset_root=(
                "/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/"
                "datasets/fineweb_edu_dedup_30b_8kto32k"
            ),
            index_root=(
                "/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/pretrain_indexes/fineweb"
            ),
            save_root="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/runs",
            log_dir="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs",
            jax_cache_root="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/jax_cache",
            run_id="run",
            mode=PretrainMode.IID_BASELINE,
            partition="standard",
            qos="low",
            time_limit="24:00:00",
            nodes=1,
            gpus_per_node=5,
            batch_size=60,
            grad_accum_steps=8,
            cpus_per_task=40,
            stage_to_scratch=False,
            run_pallas_tests=False,
            single_process_per_run=True,
            single_process_per_node=False,
        )

        self.assertIn("#SBATCH --ntasks-per-node=1", script)
        self.assertIn("JAX_LOCAL_DEVICE_IDS=0,1,2,3,4", script)
        self.assertIn("srun --ntasks=1 --ntasks-per-node=1 --gres=gpu:5 bash", script)
        self.assertIn('--fsdp_size="1"', script)
        self.assertIn('--dp_size="5"', script)
        self.assertIn('--iterator_fsdp_size="1"', script)
        self.assertIn('--iterator_dp_size="1"', script)
        self.assertNotIn("--gpus-per-task=1", script)
        self.assertNotIn("--gpu-bind=single:1", script)

    def test_render_train_sbatch_can_use_one_jax_process_per_node(self):
        script = submit_text_pretrain_slurm.render_train_sbatch(
            repo_root="/repo",
            source_root="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan",
            dataset_root=(
                "/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/"
                "datasets/fineweb_edu_dedup_30b_8kto32k"
            ),
            index_root=(
                "/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/pretrain_indexes/fineweb"
            ),
            save_root="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/runs",
            log_dir="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs",
            jax_cache_root="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/jax_cache",
            run_id="run",
            mode=PretrainMode.IID_BASELINE,
            partition="standard",
            qos="low",
            time_limit="24:00:00",
            nodes=2,
            gpus_per_node=8,
            batch_size=192,
            grad_accum_steps=5,
            cpus_per_task=24,
            stage_to_scratch=False,
            run_pallas_tests=False,
            single_process_per_run=False,
            single_process_per_node=True,
        )

        self.assertIn("#SBATCH --ntasks-per-node=1", script)
        self.assertIn("JAX_LOCAL_DEVICE_IDS=0,1,2,3,4,5,6,7", script)
        self.assertIn("srun --ntasks=2 --ntasks-per-node=1 --gres=gpu:8 bash", script)
        self.assertIn('--fsdp_size="1"', script)
        self.assertIn('--dp_size="16"', script)
        self.assertIn('--iterator_fsdp_size="1"', script)
        self.assertIn('--iterator_dp_size="2"', script)
        self.assertNotIn("--gpus-per-task=1", script)
        self.assertNotIn("--gpu-bind=single:1", script)

    def test_train_flags_omits_empty_wandb_entity(self):
        with flagsaver.flagsaver(submit_wandb_entity="", submit_wandb_project="omegalax"):
            train_flags = submit_text_pretrain_slurm._train_flags(
                mode=PretrainMode.IID_BASELINE,
                save_root=Path("/runs"),
                jax_cache_root=Path("/jax_cache"),
                run_id="run",
                total_devices=8,
                jax_processes=8,
                batch_size=128,
                grad_accum_steps=4,
                single_process_per_run=False,
                single_process_per_node=False,
            )

        self.assertIn("--wandb_project=omegalax", train_flags)
        self.assertFalse(any(flag.startswith("--wandb_entity") for flag in train_flags))

    def test_train_flags_can_resume_existing_wandb_run(self):
        with flagsaver.flagsaver(submit_wandb_project="omegalax", submit_wandb_resume="allow"):
            train_flags = submit_text_pretrain_slurm._train_flags(
                mode=PretrainMode.IID_BASELINE,
                save_root=Path("/runs"),
                jax_cache_root=Path("/jax_cache"),
                run_id="run",
                total_devices=8,
                jax_processes=8,
                batch_size=128,
                grad_accum_steps=4,
                single_process_per_run=False,
                single_process_per_node=False,
                wandb_resume_id="abc123",
            )

        self.assertIn("--wandb_id=abc123", train_flags)
        self.assertIn("--wandb_resume=allow", train_flags)


class BabysitTextPretrainSlurmScriptTest(absltest.TestCase):
    def test_prompt_codex_uses_current_session_resume(self):
        with mock.patch.object(babysit_text_pretrain_slurm.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")

            with flagsaver.flagsaver(
                babysit_codex_session_id="thread-1",
                babysit_codex_model="gpt-5.6-terra",
                babysit_codex_reasoning_effort="high",
            ):
                ok, output = babysit_text_pretrain_slurm._prompt_codex(
                    mode="iid_baseline",
                    reason="job failed with state FAILED",
                    job_id="123",
                    state="FAILED",
                    script_path=Path("/logs/jobs/train_iid_baseline.sbatch"),
                    log_dir=Path("/logs"),
                    wiki_path=Path("/wiki/progress.md"),
                    tail_text="traceback",
                )

        self.assertTrue(ok)
        self.assertIn("ok", output)
        args, kwargs = run.call_args
        self.assertIn("codex", args[0])
        self.assertIn("exec", args[0])
        self.assertIn("resume", args[0])
        resume_index = args[0].index("resume")
        self.assertEqual(args[0][resume_index + 1], "thread-1")
        self.assertLess(args[0].index("-C"), args[0].index("resume"))
        self.assertNotIn("--ask-for-approval", args[0])
        self.assertIn("gpt-5.6-terra", args[0])
        self.assertIn('model_reasoning_effort="high"', args[0])
        self.assertIn("Do not blindly resubmit", kwargs["input"])

    def test_metrics_after_returns_every_new_step_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train_123.log").write_text(
                "time=x step=10 loss=2.0 grad_norm=0.2 train/lr=1e-4\n"
                "time=x step=20 loss=1.9 grad_norm=0.3 train/lr=9e-5\n"
            )

            metrics = babysit_text_pretrain_slurm._metrics_after(root, "123", 10)

        self.assertEqual([step for step, _ in metrics], [20])
        self.assertIn("step=20", metrics[0][1])

    def test_metric_check_only_flags_non_finite_values(self):
        self.assertFalse(
            babysit_text_pretrain_slurm._metric_has_non_finite_value(
                "step=20 loss=1.9 grad_norm=0.3 train/lr=0.0"
            )
        )
        self.assertTrue(
            babysit_text_pretrain_slurm._metric_has_non_finite_value(
                "step=21 loss=nan grad_norm=0.3 train/lr=9e-5"
            )
        )

    def test_metrics_after_is_not_limited_to_log_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train_123.log").write_text(
                "step=11 loss=nan grad_norm=0.2 train/lr=1e-4\n"
                + "noise\n" * 600
                + "step=12 loss=1.9 grad_norm=0.3 train/lr=9e-5\n"
            )

            metrics = babysit_text_pretrain_slurm._metrics_after(root, "123", 10)

        self.assertEqual([step for step, _ in metrics], [11, 12])

    def test_main_switches_from_startup_to_steady_polling(self):
        metric_11 = "step=11 loss=2.0 grad_norm=0.2 train/lr=1e-4"
        metric_12 = "step=12 loss=1.9 grad_norm=0.3 train/lr=9e-5"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "progress.md"
            with (
                flagsaver.flagsaver(
                    babysit_mode_job=["iid:123:/logs/train.sbatch"],
                    babysit_resume_step=["iid:10"],
                    babysit_log_dir=str(root),
                    babysit_wiki_path=str(wiki),
                    babysit_startup_steps=2,
                    babysit_startup_poll_seconds=60,
                    babysit_poll_seconds=1200,
                ),
                mock.patch.object(
                    babysit_text_pretrain_slurm,
                    "job_status",
                    side_effect=[
                        ("RUNNING", "sacct"),
                        ("RUNNING", "sacct"),
                        ("COMPLETED", "sacct"),
                    ],
                ),
                mock.patch.object(
                    babysit_text_pretrain_slurm,
                    "_metrics_after",
                    side_effect=[[(11, metric_11)], [(12, metric_12)], []],
                ),
                mock.patch.object(babysit_text_pretrain_slurm, "_alert_codex") as alert,
                mock.patch.object(babysit_text_pretrain_slurm.time, "sleep") as sleep,
            ):
                babysit_text_pretrain_slurm.main(None)

        self.assertEqual([call.args[0] for call in sleep.call_args_list], [60, 1200])
        alert.assert_not_called()

    def test_main_checks_final_metrics_before_accepting_completion(self):
        metric = "step=11 loss=nan grad_norm=0.2 train/lr=1e-4"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                flagsaver.flagsaver(
                    babysit_mode_job=["iid:123:/logs/train.sbatch"],
                    babysit_resume_step=["iid:10"],
                    babysit_log_dir=str(root),
                    babysit_wiki_path=str(root / "progress.md"),
                ),
                mock.patch.object(
                    babysit_text_pretrain_slurm,
                    "job_status",
                    return_value=("COMPLETED", "sacct"),
                ),
                mock.patch.object(
                    babysit_text_pretrain_slurm,
                    "_metrics_after",
                    return_value=[(11, metric)],
                ),
                mock.patch.object(babysit_text_pretrain_slurm, "_alert_codex") as alert,
            ):
                babysit_text_pretrain_slurm.main(None)

        self.assertIn("completed with non-finite metric", alert.call_args.kwargs["reason"])

    def test_main_unknown_state_does_not_reset_stall_timer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                flagsaver.flagsaver(
                    babysit_mode_job=["iid:123:/logs/train.sbatch"],
                    babysit_resume_step=["iid:10"],
                    babysit_log_dir=str(root),
                    babysit_wiki_path=str(root / "progress.md"),
                    babysit_stall_seconds=900,
                ),
                mock.patch.object(
                    babysit_text_pretrain_slurm,
                    "job_status",
                    side_effect=[
                        ("RUNNING", "sacct"),
                        ("UNKNOWN", "unknown"),
                        ("RUNNING", "sacct"),
                        ("COMPLETED", "sacct"),
                    ],
                ),
                mock.patch.object(
                    babysit_text_pretrain_slurm,
                    "_metrics_after",
                    return_value=[],
                ),
                mock.patch.object(babysit_text_pretrain_slurm, "_alert_codex") as alert,
                mock.patch.object(babysit_text_pretrain_slurm.time, "sleep"),
                mock.patch.object(
                    babysit_text_pretrain_slurm.time,
                    "time",
                    side_effect=[0, 0, 100, 901, 901, 902],
                ),
            ):
                babysit_text_pretrain_slurm.main(None)

        self.assertEqual(alert.call_count, 1)
        self.assertIn("no new logged optimizer step", alert.call_args.kwargs["reason"])

    def test_parse_resume_steps(self):
        with flagsaver.flagsaver(
            babysit_resume_step=["statepassing_bptt:31295", "iid_baseline:32875"]
        ):
            self.assertEqual(
                babysit_text_pretrain_slurm._parse_resume_steps(),
                {"statepassing_bptt": 31295, "iid_baseline": 32875},
            )

    def test_requeue_self_requeues_current_slurm_job(self):
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with (
            mock.patch.dict(os.environ, {"SLURM_JOB_ID": "123"}),
            mock.patch.object(babysit_text_pretrain_slurm, "_run", return_value=completed) as run,
            self.assertRaises(SystemExit) as raised,
        ):
            babysit_text_pretrain_slurm._requeue_self(None, None)

        self.assertEqual(raised.exception.code, 0)
        run.assert_called_once_with(["scontrol", "requeue", "123"])


if __name__ == "__main__":
    absltest.main()
