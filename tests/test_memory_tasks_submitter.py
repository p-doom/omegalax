"""Tests for the controlled-memory Slurm renderer."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from absl.testing import absltest

from scripts import submit_memory_tasks_slurm
from tests.pretrain_real_data_test_utils import test_temp_dir


_PLAN_SHA256 = "a" * 64


class MemoryTaskSubmitterTest(absltest.TestCase):
    def _train_args(self, save_dir: Path) -> list[str]:
        return [
            "--task=exact",
            "--condition=state",
            f"--save_dir={save_dir}",
            f"--jax_cache_dir={save_dir.parent / 'jax_cache'}",
            f"--run_plan_sha256={_PLAN_SHA256}",
            "--success_criterion=criterion",
        ]

    def test_rendered_script_has_required_preflight_and_safe_command(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            script = submit_memory_tasks_slurm.render_sbatch(
                repo_root=tmpdir / "repo",
                python_bin=tmpdir / "python",
                job_name="memory-test",
                log_dir=tmpdir / "logs",
                time_limit="01:00:00",
                train_args=self._train_args(tmpdir / "run"),
                expected_commit="test-commit",
            )

        self.assertIn("#SBATCH --partition=standard", script)
        self.assertIn("#SBATCH --nodes=1", script)
        self.assertIn("#SBATCH --ntasks=1", script)
        self.assertIn("#SBATCH --cpus-per-task=24", script)
        self.assertIn("#SBATCH --mem=120G", script)
        self.assertIn("#SBATCH --gres=gpu:H100-SXM5:1", script)
        self.assertIn("#SBATCH --signal=B:USR1@1800", script)
        self.assertIn("#SBATCH --requeue", script)
        self.assertIn("PYTHONPATH", script)
        self.assertIn("omegalax.__file__", script)
        self.assertIn('"git", "status", "--porcelain"', script)
        self.assertIn("OMEGALAX_DELTANET_KERNEL=pallas", script)
        self.assertIn("JAX_PLATFORMS=cuda", script)
        self.assertIn("JAX_LOCAL_DEVICE_IDS=0", script)
        self.assertIn("XLA_PYTHON_CLIENT_MEM_FRACTION=0.95", script)
        self.assertIn("WANDB_MODE=online", script)
        self.assertIn(f"WANDB_DIR={tmpdir / 'logs' / 'wandb'}", script)
        self.assertIn(f"WANDB_CACHE_DIR={tmpdir / 'wandb_cache'}", script)
        self.assertIn('mkdir -p "$LOG_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"', script)
        self.assertIn("--text_attn_backend=mosaic_gpu", script)
        self.assertIn("--resume=if_present", script)
        self.assertIn("srun --cpu-bind=none", script)
        self.assertIn("trap on_usr1 USR1", script)
        self.assertIn('scontrol requeue "$SLURM_JOB_ID"', script)

    def test_default_cli_only_renders_and_never_submits(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            argv = [
                f"--repo_root={tmpdir / 'repo'}",
                f"--python_bin={tmpdir / 'python'}",
                "--job_name=memory-test",
                f"--job_script={tmpdir / 'job.sbatch'}",
                f"--log_dir={tmpdir / 'logs'}",
                "--expected_commit=test-commit",
                "--",
                *self._train_args(tmpdir / "run"),
            ]
            with mock.patch("subprocess.run") as submit:
                submit_memory_tasks_slurm.main(argv)
            submit.assert_not_called()
            self.assertTrue((tmpdir / "job.sbatch").is_file())
            self.assertTrue((tmpdir / "logs").is_dir())

    def test_renderer_rejects_non_sha256_run_plan_value(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            train_args = self._train_args(tmpdir / "run")
            train_args[4] = "--run_plan_sha256=not-a-hash"
            with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
                submit_memory_tasks_slurm.render_sbatch(
                    repo_root=tmpdir / "repo",
                    python_bin=tmpdir / "python",
                    job_name="memory-test",
                    log_dir=tmpdir / "logs",
                    time_limit="01:00:00",
                    train_args=train_args,
                    expected_commit="test-commit",
                )


if __name__ == "__main__":
    absltest.main()
