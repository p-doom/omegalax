#!/bin/bash
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
test "${SLURMD_NODENAME:-}" = "$(hostname -s)"
SOURCE=$1 PLAN=$2 OUTPUT=$3 MODEL=$4 MODE=$5 ARM=${6:-}
case "$OUTPUT" in /fast/project/HFMI_SynergyUnit/p-doom_shared/salan/*) ;; *) exit 1;; esac
export PYTHONPATH="$SOURCE"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$OUTPUT/tmp_${MODE}_${ARM:-all}_${SLURM_STEP_ID:-batch}"
export XDG_CACHE_HOME="$OUTPUT/cache_${MODE}_${ARM:-all}_${SLURM_STEP_ID:-batch}"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_FLAGS='--xla_gpu_deterministic_ops=true'
export OMEGALAX_DELTANET_KERNEL=pallas
export JAX_COMPILATION_CACHE_DIR="$XDG_CACHE_HOME/jax"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export WANDB_DIR="$OUTPUT"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR"
cd "$SOURCE"
PYTHON='/fast/home/salan.isaqzoi/gh_projects/p(doom)/omegalax/.venv/bin/python'
sha256sum --quiet -c "$SOURCE/SOURCE_SHA256SUMS"
if test "$MODE" = gate; then
    export JAX_LOG_COMPILES=1
    for T in 8192 2048 512; do
        "$PYTHON" scripts/gate_rollout_state_gradient.py --init-hf-model-path "$MODEL" --segment-length "$T" --output "$OUTPUT/gradient_T${T}.json"
        "$PYTHON" scripts/train_rollout_comparison.py --plan "$PLAN" --segment-length "$T" --init-hf-model-path "$MODEL" --output "$OUTPUT/T${T}/resumed" --stop-after-update 1 --resume if_present
        "$PYTHON" scripts/train_rollout_comparison.py --plan "$PLAN" --segment-length "$T" --init-hf-model-path "$MODEL" --output "$OUTPUT/T${T}/resumed" --stop-after-update 2 --resume required
        "$PYTHON" scripts/train_rollout_comparison.py --plan "$PLAN" --segment-length "$T" --init-hf-model-path "$MODEL" --output "$OUTPUT/T${T}/continuous" --stop-after-update 2
        "$PYTHON" scripts/train_rollout_comparison.py --plan "$PLAN" --segment-length "$T" --init-hf-model-path "$MODEL" --output "$OUTPUT/T${T}/resumed" --stop-after-update 3 --resume required
    done
    "$PYTHON" scripts/launch_rollout_comparison.py verify-gate --launch-root "$OUTPUT" --plan "$PLAN"
else
    test "$ARM" = 8192 || test "$ARM" = 2048 || test "$ARM" = 512
    "$PYTHON" scripts/train_rollout_comparison.py --plan "$PLAN" --segment-length "$ARM" --init-hf-model-path "$MODEL" --output "$OUTPUT/T${ARM}" --resume if_present --wandb-project rollout-comparison
fi
