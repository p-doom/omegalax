#!/bin/bash
#SBATCH --job-name=verify-fix-slowdown
#SBATCH --partition=standard
#SBATCH --gres=gpu:H100-SXM5:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=_bench_logs/%x_%j_%a.log
#SBATCH --array=0-3

cd /fast/home/franz.srambical/omegalax
echo "=== task=$SLURM_ARRAY_TASK_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

TESTS=(
  "tests.test_fix_slowdown_ports.CuDnnPackedVisionAttentionTest"
  "tests.test_fix_slowdown_ports.Qwen3TextAttnBackendSwapTest"
  "tests.test_fix_slowdown_ports.Qwen3_5PaddingNoOpTest"
  "tests.test_fix_slowdown_ports.Qwen3_5JitStabilityTest"
)
TEST="${TESTS[$SLURM_ARRAY_TASK_ID]}"
echo "=== running $TEST ==="
.venv/bin/python -m unittest "$TEST" 2>&1
RC=$?
echo "=== rc=$RC for $TEST ==="
exit $RC
