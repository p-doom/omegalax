"""Tests for pretraining helper scripts."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest
from absl.testing import flagsaver

from scripts import babysit_text_pretrain_slurm
from scripts import build_pretrain_indexes
from scripts import launch_text_pretrain_experiments
from scripts import monitor_text_pretrain_slurm
from scripts import submit_text_pretrain_slurm
from omegalax.trainers.pretrain import PretrainMode


class BuildPretrainIndexScriptTest(absltest.TestCase):
    def test_iid_mode_calls_iid_builder_with_forwarded_args(self):
        with mock.patch.object(build_pretrain_indexes, "build_iid_chunk_index") as iid_builder:
            with mock.patch.object(
                build_pretrain_indexes, "build_statepassing_pair_index"
            ) as statepassing_builder:
                iid_builder.return_value = Path("/tmp/index")
                out = build_pretrain_indexes.build_pretrain_index(
                    root="/data/root",
                    out_dir="/indexes",
                    pretrain_mode="iid_baseline",
                    split="train",
                    chunk_length=123,
                    eos_id=456,
                    records_per_shard=789,
                    overwrite=True,
                )

        self.assertEqual(out, Path("/tmp/index"))
        statepassing_builder.assert_not_called()
        iid_builder.assert_called_once()
        _, kwargs = iid_builder.call_args
        self.assertEqual(kwargs["chunk_length"], 123)
        self.assertEqual(kwargs["eos_id"], 456)
        self.assertEqual(kwargs["split"], "train")
        self.assertEqual(kwargs["records_per_shard"], 789)
        self.assertTrue(kwargs["overwrite"])

    def test_statepassing_modes_call_statepassing_builder(self):
        with mock.patch.object(build_pretrain_indexes, "build_iid_chunk_index") as iid_builder:
            with mock.patch.object(
                build_pretrain_indexes, "build_statepassing_pair_index"
            ) as statepassing_builder:
                statepassing_builder.return_value = Path("/tmp/statepassing")
                out = build_pretrain_indexes.build_pretrain_index(
                    root="/data/root",
                    out_dir="/indexes",
                    pretrain_mode="statepassing_bptt",
                    split="val",
                    chunk_length=4096,
                    eos_id=248046,
                    records_per_shard=1000,
                    overwrite=False,
                )

        self.assertEqual(out, Path("/tmp/statepassing"))
        iid_builder.assert_not_called()
        statepassing_builder.assert_called_once()
        args, kwargs = statepassing_builder.call_args
        self.assertEqual(args[0], "/data/root")
        self.assertEqual(args[1], Path("/indexes").resolve() / "val" / "statepassing_bptt")
        self.assertEqual(kwargs["split"], "val")

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
            self.assertIn("--train_index_path=", command)
            self.assertIn("--val_index_path=", command)
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
        )

        self.assertIn("#SBATCH --gres=gpu:8", script)
        self.assertIn("#SBATCH --qos=low", script)
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

    def test_train_flags_omits_empty_wandb_entity(self):
        with flagsaver.flagsaver(submit_wandb_entity="", submit_wandb_project="omegalax"):
            train_flags = submit_text_pretrain_slurm._train_flags(
                mode=PretrainMode.IID_BASELINE,
                save_root=Path("/runs"),
                jax_cache_root=Path("/jax_cache"),
                run_id="run",
                total_tasks=8,
                batch_size=128,
                grad_accum_steps=4,
                single_process_per_run=False,
            )

        self.assertIn("--wandb_project=omegalax", train_flags)
        self.assertFalse(any(flag.startswith("--wandb_entity") for flag in train_flags))

    def test_render_monitor_sbatch_records_jobs_and_wiki_path(self):
        script = submit_text_pretrain_slurm.render_monitor_sbatch(
            repo_root="/repo",
            log_dir="/logs/run",
            run_id="run",
            partition="standard",
            qos=None,
            time_limit="24:00:00",
            job_ids=["1", "2", "3"],
            wiki_path="/wiki/progress.md",
            poll_seconds=1200,
        )

        self.assertNotIn("#SBATCH --gres=gpu", script)
        self.assertIn("--monitor_job_ids=1,2,3", script)
        self.assertIn("--monitor_wiki_path=/wiki/progress.md", script)
        self.assertIn("--monitor_poll_seconds=1200", script)


class MonitorTextPretrainSlurmScriptTest(absltest.TestCase):
    def test_job_status_prefers_sacct_rows(self):
        with mock.patch.object(monitor_text_pretrain_slurm, "_run") as run:
            run.side_effect = [
                mock.Mock(stdout="123 COMPLETED\n", stderr=""),
            ]

            state, source = monitor_text_pretrain_slurm.job_status("123")

        self.assertEqual(state, "COMPLETED")
        self.assertEqual(source, "sacct")


class BabysitTextPretrainSlurmScriptTest(absltest.TestCase):
    def test_prompt_codex_uses_current_session_resume(self):
        with mock.patch.object(babysit_text_pretrain_slurm.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")

            ok, output = babysit_text_pretrain_slurm._prompt_codex(
                mode="iid_baseline",
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
        self.assertLess(args[0].index("-C"), args[0].index("resume"))
        self.assertNotIn("--ask-for-approval", args[0])
        self.assertIn("Do not blindly resubmit", kwargs["input"])

    def test_auto_resubmit_requires_successful_codex_prompt(self):
        with flagsaver.flagsaver(babysit_auto_resubmit=True):
            self.assertFalse(babysit_text_pretrain_slurm._should_auto_resubmit_after_codex(False))
            self.assertTrue(babysit_text_pretrain_slurm._should_auto_resubmit_after_codex(True))

        with flagsaver.flagsaver(babysit_auto_resubmit=False):
            self.assertFalse(babysit_text_pretrain_slurm._should_auto_resubmit_after_codex(True))


if __name__ == "__main__":
    absltest.main()
