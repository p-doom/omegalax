#!/usr/bin/env bash
# 2-node overfit launcher for Qwen3-VL-8B on the long 10 screenshot /
# 10 executable-action-block sequence.
#
# Submit delta 0:
#   RUN_VARIANT=pi0 sbatch scripts/slurm_qwen3vl8b_long_actions_overfit_2node.sbatch
#
# Submit normal:
#   RUN_VARIANT=normal sbatch scripts/slurm_qwen3vl8b_long_actions_overfit_2node.sbatch

#SBATCH --account=envcomp
#SBATCH --partition=booster
#SBATCH --job-name=qwen3vl8b_longact
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=06:00:00
#SBATCH --output=/p/project1/envcomp/keihan/slurm_logs/%x_%j.out
#SBATCH --error=/p/project1/envcomp/keihan/slurm_logs/%x_%j.err

set -euo pipefail

ROOT_DIR="/p/project1/envcomp/keihan/omegalax-pi0-action-expert-training"
cd "$ROOT_DIR"
mkdir -p /p/project1/envcomp/keihan/.uv_cache /p/project1/envcomp/keihan/slurm_logs
export UV_CACHE_DIR="/p/project1/envcomp/keihan/.uv_cache"
export UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv"

RUN_VARIANT="${RUN_VARIANT:-pi0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
TP_SIZE="${TP_SIZE:-4}"
FSDP_SIZE="${FSDP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MESH_TAG="tp${TP_SIZE}fsdp${FSDP_SIZE}dp${DP_SIZE}"

case "$RUN_VARIANT" in
  pi0)
    ENABLE_PI0_ACTION_EXPERT="true"
    FREEZE_VISION_TOWER="false"
    RUN_NAME="qwen3vl8b_pi0_longactions512_10turn_${MESH_TAG}_${RUN_ID}"
    ;;
  normal)
    ENABLE_PI0_ACTION_EXPERT="false"
    FREEZE_VISION_TOWER="true"
    RUN_NAME="qwen3vl8b_normal_longactions512_10turn_${MESH_TAG}_${RUN_ID}"
    ;;
  *)
    echo "RUN_VARIANT must be pi0 or normal, got: $RUN_VARIANT" >&2
    exit 2
    ;;
esac

ARTIFACT_ROOT="/p/project1/envcomp/keihan/action-expert-dummy-data/${RUN_NAME}"
mkdir -p "$ARTIFACT_ROOT"

export RUN_PRESET="custom"
export RUN_ID
export RUN_NAME
export ARTIFACT_ROOT
export MODEL_ID="/p/scratch/envcomp/keihan/models/Qwen/Qwen3-VL-8B-Instruct"
export PROCESSOR="$MODEL_ID"

DATA_ROOT="${DATA_ROOT:-/p/project1/envcomp/keihan/action-expert-dummy-data/qwen3vl8b_long_actions_10turn_20260617T161859Z}"
export TRAIN_JSONL="${TRAIN_JSONL:-$DATA_ROOT/train_one_sequence.jsonl}"
export VAL_JSONL="${VAL_JSONL:-$DATA_ROOT/val_one_sequence.jsonl}"
export PREPROCESSOR_CONFIG="${PREPROCESSOR_CONFIG:-/p/home/jusers/pakseresht1/juwels/Data/action-expert-dummy-data/preprocessor_config.json}"

export TP_SIZE
export FSDP_SIZE
export DP_SIZE
export BATCH_SIZE
export OVERFIT_REPEAT_RECORDS="${OVERFIT_REPEAT_RECORDS:-8}"

export MAX_LENGTH="${MAX_LENGTH:-9281}"
export NUM_LOSS_TILES="${NUM_LOSS_TILES:-16}"
export NUM_STEPS="${NUM_STEPS:-500}"
export LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-1}"
export VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-10}"
export VAL_STEPS="${VAL_STEPS:-1}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-1000}"
export SAVE_STEPS="${SAVE_STEPS:-20,50,100,150,200,300,400,500}"
export EARLY_STOP_LOSS="${EARLY_STOP_LOSS:-0.00005}"
export RESUME_MODE="${RESUME_MODE:-never}"
export RUN_EXPORT="false"

export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

export TEXT_ATTN_BACKEND="${TEXT_ATTN_BACKEND:-triton}"
export VISION_ATTN_BACKEND="${VISION_ATTN_BACKEND:-xla_chunked}"
export GRAIN_WORKERS="${GRAIN_WORKERS:-0}"
export GRAIN_READ_THREADS="${GRAIN_READ_THREADS:-1}"
export GRAIN_READ_BUFFER_SIZE="${GRAIN_READ_BUFFER_SIZE:-1}"
export GRAIN_WORKER_BUFFER_SIZE="${GRAIN_WORKER_BUFFER_SIZE:-1}"
export LOG_MEMORY="${LOG_MEMORY:-true}"
export TOKAMAX_CACHE_DIR="${TOKAMAX_CACHE_DIR:-none}"

export ENABLE_PI0_ACTION_EXPERT
export FREEZE_VISION_TOWER
export PI0_TRAIN_SCOPE="${PI0_TRAIN_SCOPE:-non_vision}"
export PI0_ACTION_WIDTH="${PI0_ACTION_WIDTH:-1024}"
export PI0_ACTION_MLP_SIZE="${PI0_ACTION_MLP_SIZE:-4096}"
export PI0_NUM_LAYERS="${PI0_NUM_LAYERS:-36}"
if [[ "$RUN_VARIANT" == "pi0" ]]; then
  export PI0_ACTION_EXPERT_INIT_PATH="${PI0_ACTION_EXPERT_INIT_PATH:-none}"
else
  export PI0_ACTION_EXPERT_INIT_PATH="${PI0_ACTION_EXPERT_INIT_PATH:-}"
fi

export OMEGALAX_JAX_DISTRIBUTED="${OMEGALAX_JAX_DISTRIBUTED:-auto}"
export PYTHONUNBUFFERED="1"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy || true

{
  echo "[launch] job=$SLURM_JOB_ID variant=$RUN_VARIANT nodes=$SLURM_JOB_NUM_NODES ntasks=$SLURM_NTASKS nodelist=$SLURM_JOB_NODELIST"
  echo "[launch] artifact_root=$ARTIFACT_ROOT"
  echo "[launch] data_root=$DATA_ROOT"
  echo "[launch] mesh tp=$TP_SIZE fsdp=$FSDP_SIZE dp=$DP_SIZE batch=$BATCH_SIZE max_length=$MAX_LENGTH"
  echo "[launch] allocator preallocate=$XLA_PYTHON_CLIENT_PREALLOCATE mem_fraction=$XLA_PYTHON_CLIENT_MEM_FRACTION tf_gpu_allocator=$TF_GPU_ALLOCATOR"
  echo "[launch] train_jsonl=$TRAIN_JSONL"
  echo "[launch] val_jsonl=$VAL_JSONL"
  echo "[launch] model=$MODEL_ID"
  echo "[launch] pi0_init=${PI0_ACTION_EXPERT_INIT_PATH:-<unset>}"
  date -u +"[launch] start_utc=%Y-%m-%dT%H:%M:%SZ"
} | tee "$ARTIFACT_ROOT/launch.log"

srun --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=single:1 \
  bash scripts/run_qwen3vl_pi0_action_expert_overfit.sh 2>&1 | tee "$ARTIFACT_ROOT/run_live.log"

date -u +"[launch] end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "$ARTIFACT_ROOT/launch.log"
