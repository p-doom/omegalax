#!/usr/bin/env bash
# Export the long-action 8B overfit checkpoints to SGLang-loadable HF folders.

#SBATCH --account=envcomp
#SBATCH --partition=booster
#SBATCH --job-name=qwen3vl8b_long_export
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=/p/project1/envcomp/keihan/slurm_logs/%x_%j.out
#SBATCH --error=/p/project1/envcomp/keihan/slurm_logs/%x_%j.err

set -euo pipefail

ROOT_DIR="/p/project1/envcomp/keihan/omegalax-pi0-action-expert-training"
cd "$ROOT_DIR"

mkdir -p /p/project1/envcomp/keihan/.uv_cache /p/project1/envcomp/keihan/slurm_logs
export UV_CACHE_DIR="/p/project1/envcomp/keihan/.uv_cache"
export UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv"
export MODEL_ID="${MODEL_ID:-/p/scratch/envcomp/keihan/models/Qwen/Qwen3-VL-8B-Instruct}"
export EXPORT_TP_SIZE="${EXPORT_TP_SIZE:-4}"
export EXPORT_FSDP_SIZE="${EXPORT_FSDP_SIZE:-1}"
export EXPORT_DP_SIZE="${EXPORT_DP_SIZE:-1}"

NORMAL_ARTIFACT="${NORMAL_ARTIFACT:-/p/project1/envcomp/keihan/action-expert-dummy-data/qwen3vl8b_normal_longactions512_10turn_tp4fsdp2dp1_20260617T161859Z_longactions512}"
PI0_ARTIFACT="${PI0_ARTIFACT:-/p/project1/envcomp/keihan/action-expert-dummy-data/qwen3vl8b_pi0_longactions512_10turn_tp4fsdp2dp1_20260617T161859Z_longactions512}"

export OMEGALAX_JAX_DISTRIBUTED="false"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED="1"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy || true

latest_ckpt() {
  find "$1/checkpoints" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended -regex '.*/[0-9]+' | sort | tail -n 1
}

export_one() {
  local label="$1"
  local artifact="$2"
  local ckpt
  ckpt="$(latest_ckpt "$artifact")"
  if [[ -z "$ckpt" ]]; then
    echo "[export] no checkpoint found for $label under $artifact/checkpoints" >&2
    exit 1
  fi
  echo "[export] $label checkpoint $(basename "$ckpt")"
  rm -rf "$artifact/exported_for_sglang"
  uv run scripts/export_to_hf.py \
    --model_id "$MODEL_ID" \
    --checkpoint_path "$ckpt" \
    --out_dir "$artifact/exported_for_sglang" \
    --tp_size "$EXPORT_TP_SIZE" \
    --fsdp_size "$EXPORT_FSDP_SIZE" \
    --dp_size "$EXPORT_DP_SIZE" \
    --max_grad_norm 1.0 \
    --grad_accum_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_steps 0 \
    --num_steps 500
}

echo "[export] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[export] export_mesh tp=$EXPORT_TP_SIZE fsdp=$EXPORT_FSDP_SIZE dp=$EXPORT_DP_SIZE"
echo "[export] normal_artifact=$NORMAL_ARTIFACT"
echo "[export] pi0_artifact=$PI0_ARTIFACT"

export_one normal "$NORMAL_ARTIFACT"
export_one pi0 "$PI0_ARTIFACT"

echo "[export] outputs"
find "$NORMAL_ARTIFACT/exported_for_sglang" "$PI0_ARTIFACT/exported_for_sglang" \
  -maxdepth 1 -type f \( -name 'model.safetensors' -o -name 'config.json' -o -name 'pi0_action_expert_config.json' -o -name 'qwen3_vl_pi0_action_expert_*.safetensors' \) \
  -printf '%p %s bytes\n' | sort
