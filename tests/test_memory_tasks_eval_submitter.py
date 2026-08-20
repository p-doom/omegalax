"""Tests for the controlled-memory eval Slurm renderer."""

from __future__ import annotations

from pathlib import Path
import subprocess
from unittest import mock

from absl.testing import absltest

from scripts import submit_memory_tasks_eval_slurm
from tests.pretrain_real_data_test_utils import test_temp_dir


class MemoryTaskEvalSubmitterTest(absltest.TestCase):
    def _eval_args(self, checkpoint: Path) -> list[str]:
        return [
            f"--checkpoint={checkpoint}",
            "--task=exact",
            "--seed=1",
            "--segment_length=16",
            "--num_segments=2",
            "--vocab_size=1024",
            "--num_rule_examples=5",
            "--num_exact_bindings=3",
            "--relevance_cue=marked",
            "--num_pairs=2048",
            "--pairs_per_batch=256",
            f"--jax_cache_dir={checkpoint.parent / 'jax_cache'}",
            f"--wiki_mirror_dir={checkpoint.parent / 'wiki_mirror'}",
        ]

    def test_rendered_script_enforces_eval_runtime_and_resources(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            script = submit_memory_tasks_eval_slurm.render_sbatch(
                repo_root=tmpdir / "repo",
                python_bin=tmpdir / "python",
                job_name="memory-eval-test",
                log_dir=tmpdir / "logs",
                time_limit="02:00:00",
                eval_args=self._eval_args(tmpdir / "checkpoint"),
                expected_commit="test-commit",
            )

        self.assertIn("#SBATCH --partition=standard", script)
        self.assertIn("#SBATCH --nodes=1", script)
        self.assertIn("#SBATCH --ntasks=1", script)
        self.assertIn("#SBATCH --cpus-per-task=24", script)
        self.assertIn("#SBATCH --mem=120G", script)
        self.assertIn("#SBATCH --gres=gpu:H100-SXM5:1", script)
        self.assertIn("PYTHONPATH", script)
        self.assertIn("omegalax.__file__", script)
        self.assertIn('"git", "status", "--porcelain"', script)
        self.assertIn("JAX_PLATFORMS=cuda", script)
        self.assertIn("JAX_LOCAL_DEVICE_IDS=0", script)
        self.assertIn("XLA_PYTHON_CLIENT_MEM_FRACTION=0.95", script)
        self.assertIn("OMEGALAX_DELTANET_KERNEL=pallas", script)
        self.assertIn("jax.default_backend()", script)
        self.assertIn("_resolve_backend()", script)
        self.assertIn("--tp_size=1", script)
        self.assertIn("--fsdp_size=1", script)
        self.assertIn("--dp_size=1", script)
        self.assertIn("--text_attn_backend=mosaic_gpu", script)
        self.assertIn("srun --cpu-bind=none", script)
        self.assertIn('>"$EVAL_LOG" 2>&1\n', script)
        self.assertNotIn('>"$EVAL_LOG" 2>&1 &', script)
        preflight = script.split("<<'PY'\n", maxsplit=1)[1].split("\nPY\n", maxsplit=1)[0]
        compile(preflight, "rendered_eval_preflight", "exec")

    def test_renderer_requires_the_canonical_pair_manifest(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            eval_args = self._eval_args(tmpdir / "checkpoint")
            eval_args[9] = "--num_pairs=2"
            with self.assertRaisesRegex(ValueError, "num_pairs=2048"):
                submit_memory_tasks_eval_slurm.render_sbatch(
                    repo_root=tmpdir / "repo",
                    python_bin=tmpdir / "python",
                    job_name="memory-eval-test",
                    log_dir=tmpdir / "logs",
                    time_limit="02:00:00",
                    eval_args=eval_args,
                    expected_commit="test-commit",
                )

    def test_default_cli_only_renders_and_the_script_passes_bash_n(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            script_path = tmpdir / "eval.sbatch"
            argv = [
                f"--repo_root={tmpdir / 'repo'}",
                f"--python_bin={tmpdir / 'python'}",
                "--job_name=memory-eval-test",
                f"--job_script={script_path}",
                f"--log_dir={tmpdir / 'logs'}",
                "--expected_commit=test-commit",
                "--",
                *self._eval_args(tmpdir / "checkpoint"),
            ]
            with mock.patch("subprocess.run") as submit:
                submit_memory_tasks_eval_slurm.main(argv)
            submit.assert_not_called()
            self.assertTrue(script_path.is_file())
            subprocess.run(["bash", "-n", str(script_path)], check=True)


if __name__ == "__main__":
    absltest.main()
