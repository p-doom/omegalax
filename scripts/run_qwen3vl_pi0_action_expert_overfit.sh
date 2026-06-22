#!/usr/bin/env bash
set -euo pipefail

# One-sequence overfit harness for Qwen3-VL pi0 action expert training.
#
# This script is configured with environment variables.  The default preset is a
# 4-GPU coupled run that trains the non-vision Qwen3-VL weights together with
# the action expert, saves one final checkpoint, and exports SGLang weights.
#
# Quick starts:
#   bash scripts/run_qwen3vl_pi0_action_expert_overfit.sh --help
#   RUN_PRESET=smoke_1gpu bash scripts/run_qwen3vl_pi0_action_expert_overfit.sh
#   RUN_PRESET=expert_fsdp4 bash scripts/run_qwen3vl_pi0_action_expert_overfit.sh
#   RUN_PRESET=coupled_fsdp4 NUM_STEPS=200 bash scripts/run_qwen3vl_pi0_action_expert_overfit.sh

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_qwen3vl_pi0_action_expert_overfit.sh

Configure the run with environment variables. Important knobs:

  RUN_PRESET:
    smoke_1gpu      1 GPU, 1 step, expert/adapters only, no export
    expert_1gpu     1 GPU, 20 steps, expert/adapters only, no export
    expert_fsdp4    4 GPU FSDP, 20 steps, expert/adapters only, export
    coupled_fsdp4   4 GPU FSDP, 100 steps, train non-vision backbone + expert, export
    full_fsdp4      4 GPU FSDP, 100 steps, train all weights including vision, export
    custom          No preset defaults; use explicit env vars

  PI0_TRAIN_SCOPE:
    expert_only     action expert + action input/output adapters
    expert_lm_head  expert/adapters + LM head
    non_vision      text backbone + expert/adapters + LM head; vision frozen
    all             all model weights, including vision

Common overrides:
  MODEL_ID=/path/to/Qwen3-VL-2B-Instruct
  ARTIFACT_ROOT=/path/to/output_dir
  NUM_STEPS=200
  LOG_EVERY_STEPS=1
  VAL_EVERY_STEPS=10
  SAVE_EVERY_STEPS=200
  SAVE_STEPS=20,50,100,200
  EARLY_STOP_LOSS=0.00005
  RUN_EXPORT=true
  LOG_MEMORY=true
  TP_SIZE=1 FSDP_SIZE=4 DP_SIZE=1 BATCH_SIZE=4
  PI0_ACTION_WIDTH=1024 PI0_ACTION_MLP_SIZE=4096
  PI0_NUM_LAYERS=28  # only needed to select a specific existing init sidecar
  PI0_ACTION_EXPERT_INIT_PATH=none
                     Disable sidecar auto-discovery and keep random init.
  ENABLE_PI0_ACTION_EXPERT=false FREEZE_VISION_TOWER=true
                     Train the ordinary Qwen3-VL path with vision frozen.

Examples:
  RUN_PRESET=smoke_1gpu bash scripts/run_qwen3vl_pi0_action_expert_overfit.sh

  RUN_PRESET=coupled_fsdp4 NUM_STEPS=200 SAVE_EVERY_STEPS=200 \
    bash scripts/run_qwen3vl_pi0_action_expert_overfit.sh

  RUN_PRESET=custom FSDP_SIZE=4 BATCH_SIZE=4 PI0_TRAIN_SCOPE=all \
    NUM_STEPS=50 SAVE_EVERY_STEPS=50 RUN_EXPORT=true \
    bash scripts/run_qwen3vl_pi0_action_expert_overfit.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -gt 0 ]]; then
  echo "[overfit] unexpected positional arguments: $*" >&2
  usage >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_PRESET="${RUN_PRESET:-coupled_fsdp4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

case "$RUN_PRESET" in
  smoke_1gpu)
    : "${TP_SIZE:=1}"
    : "${FSDP_SIZE:=1}"
    : "${DP_SIZE:=1}"
    : "${BATCH_SIZE:=1}"
    : "${OVERFIT_REPEAT_RECORDS:=1}"
    : "${PI0_TRAIN_SCOPE:=expert_only}"
    : "${NUM_STEPS:=1}"
    : "${LOG_EVERY_STEPS:=1}"
    : "${VAL_EVERY_STEPS:=1}"
    : "${SAVE_EVERY_STEPS:=1}"
    : "${RUN_EXPORT:=false}"
    ;;
  expert_1gpu)
    : "${TP_SIZE:=1}"
    : "${FSDP_SIZE:=1}"
    : "${DP_SIZE:=1}"
    : "${BATCH_SIZE:=1}"
    : "${OVERFIT_REPEAT_RECORDS:=1}"
    : "${PI0_TRAIN_SCOPE:=expert_only}"
    : "${NUM_STEPS:=20}"
    : "${LOG_EVERY_STEPS:=1}"
    : "${VAL_EVERY_STEPS:=5}"
    : "${SAVE_EVERY_STEPS:=20}"
    : "${RUN_EXPORT:=false}"
    ;;
  expert_fsdp4)
    : "${TP_SIZE:=1}"
    : "${FSDP_SIZE:=4}"
    : "${DP_SIZE:=1}"
    : "${BATCH_SIZE:=4}"
    : "${OVERFIT_REPEAT_RECORDS:=4}"
    : "${PI0_TRAIN_SCOPE:=expert_only}"
    : "${NUM_STEPS:=20}"
    : "${LOG_EVERY_STEPS:=1}"
    : "${VAL_EVERY_STEPS:=5}"
    : "${SAVE_EVERY_STEPS:=20}"
    : "${RUN_EXPORT:=true}"
    ;;
  coupled_fsdp4)
    : "${TP_SIZE:=1}"
    : "${FSDP_SIZE:=4}"
    : "${DP_SIZE:=1}"
    : "${BATCH_SIZE:=4}"
    : "${OVERFIT_REPEAT_RECORDS:=4}"
    : "${PI0_TRAIN_SCOPE:=non_vision}"
    : "${NUM_STEPS:=100}"
    : "${LOG_EVERY_STEPS:=1}"
    : "${VAL_EVERY_STEPS:=10}"
    : "${SAVE_EVERY_STEPS:=100}"
    : "${RUN_EXPORT:=true}"
    ;;
  full_fsdp4)
    : "${TP_SIZE:=1}"
    : "${FSDP_SIZE:=4}"
    : "${DP_SIZE:=1}"
    : "${BATCH_SIZE:=4}"
    : "${OVERFIT_REPEAT_RECORDS:=4}"
    : "${PI0_TRAIN_SCOPE:=all}"
    : "${NUM_STEPS:=100}"
    : "${LOG_EVERY_STEPS:=1}"
    : "${VAL_EVERY_STEPS:=10}"
    : "${SAVE_EVERY_STEPS:=100}"
    : "${RUN_EXPORT:=true}"
    ;;
  custom)
    ;;
  *)
    echo "[overfit] unknown RUN_PRESET=$RUN_PRESET" >&2
    usage >&2
    exit 2
    ;;
esac

MODEL_ID="${MODEL_ID:-/p/home/jusers/pakseresht1/juwels/Data/models/Qwen/Qwen3-VL-2B-Instruct}"
PROCESSOR="${PROCESSOR:-$MODEL_ID}"

DATA_ROOT="${DATA_ROOT:-/p/home/jusers/pakseresht1/juwels/Data/action-expert-dummy-data}"
TRAIN_JSONL="${TRAIN_JSONL:-$DATA_ROOT/train_one_sequence.jsonl}"
VAL_JSONL="${VAL_JSONL:-$DATA_ROOT/val_one_sequence.jsonl}"
PREPROCESSOR_CONFIG="${PREPROCESSOR_CONFIG:-$DATA_ROOT/preprocessor_config.json}"

RUN_NAME="${RUN_NAME:-qwen3vl2b_pi0_${RUN_PRESET}_${RUN_ID}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$DATA_ROOT/$RUN_NAME}"
TRAIN_PAYLOAD_DIR="${TRAIN_PAYLOAD_DIR:-$ARTIFACT_ROOT/train_payload}"
VAL_PAYLOAD_DIR="${VAL_PAYLOAD_DIR:-$ARTIFACT_ROOT/val_payload}"
TRAIN_CHUNKS_DIR="${TRAIN_CHUNKS_DIR:-$ARTIFACT_ROOT/train_chunks}"
VAL_CHUNKS_DIR="${VAL_CHUNKS_DIR:-$ARTIFACT_ROOT/val_chunks}"
SAVE_DIR="${SAVE_DIR:-$ARTIFACT_ROOT/checkpoints}"
EXPORT_DIR="${EXPORT_DIR:-$ARTIFACT_ROOT/exported_for_sglang}"
JAX_CACHE_DIR="${JAX_CACHE_DIR:-$ARTIFACT_ROOT/jax_cache}"
TOKAMAX_CACHE_DIR="${TOKAMAX_CACHE_DIR:-$ARTIFACT_ROOT/tokamax_cache}"
if [[ "$TOKAMAX_CACHE_DIR" == "none" || "$TOKAMAX_CACHE_DIR" == "false" ]]; then
  TOKAMAX_CACHE_DIR=""
fi

MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
NUM_LOSS_TILES="${NUM_LOSS_TILES:-16}"
GRAIN_WORKERS="${GRAIN_WORKERS:-0}"
GRAIN_READ_THREADS="${GRAIN_READ_THREADS:-1}"
GRAIN_READ_BUFFER_SIZE="${GRAIN_READ_BUFFER_SIZE:-1}"
GRAIN_WORKER_BUFFER_SIZE="${GRAIN_WORKER_BUFFER_SIZE:-1}"

LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
NUM_LOG_EVENTS="${NUM_LOG_EVENTS:-30}"
NUM_STEPS="${NUM_STEPS:-$((LOG_EVERY_STEPS * NUM_LOG_EVENTS))}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-$LOG_EVERY_STEPS}"
VAL_STEPS="${VAL_STEPS:-1}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-$NUM_STEPS}"
SAVE_STEPS="${SAVE_STEPS:-}"
EARLY_STOP_LOSS="${EARLY_STOP_LOSS:-}"
LOG_MEMORY="${LOG_MEMORY:-true}"
RESUME_MODE="${RESUME_MODE:-if_present}"

TP_SIZE="${TP_SIZE:-1}"
FSDP_SIZE="${FSDP_SIZE:-4}"
DP_SIZE="${DP_SIZE:-1}"
TEXT_ATTN_BACKEND="${TEXT_ATTN_BACKEND:-cudnn}"
VISION_ATTN_BACKEND="${VISION_ATTN_BACKEND:-${OMEGALAX_QWEN3_VL_VISION_ATTN_BACKEND:-cudnn}}"
export OMEGALAX_QWEN3_VL_VISION_ATTN_BACKEND="$VISION_ATTN_BACKEND"

ENABLE_PI0_ACTION_EXPERT="${ENABLE_PI0_ACTION_EXPERT:-true}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-false}"
PI0_ACTION_WIDTH="${PI0_ACTION_WIDTH:-1024}"
PI0_ACTION_MLP_SIZE="${PI0_ACTION_MLP_SIZE:-4096}"
PI0_TRAIN_SCOPE="${PI0_TRAIN_SCOPE:-all}"
PI0_ACTION_EXPERT_INIT_PATH="${PI0_ACTION_EXPERT_INIT_PATH:-}"
PI0_ACTION_EXPERT_INIT_DISABLED="false"
if [[ "$PI0_ACTION_EXPERT_INIT_PATH" == "none" ]]; then
  PI0_ACTION_EXPERT_INIT_DISABLED="true"
fi
PI0_NUM_LAYERS="${PI0_NUM_LAYERS:-}"

if [[ "$ENABLE_PI0_ACTION_EXPERT" == "true" && "$PI0_ACTION_EXPERT_INIT_DISABLED" != "true" && -z "$PI0_ACTION_EXPERT_INIT_PATH" && -n "$PI0_NUM_LAYERS" ]]; then
  DEFAULT_PI0_ACTION_EXPERT_INIT_PATH="$MODEL_ID/qwen3_vl_pi0_action_expert_w${PI0_ACTION_WIDTH}_m${PI0_ACTION_MLP_SIZE}_l${PI0_NUM_LAYERS}.safetensors"
  if [[ -f "$DEFAULT_PI0_ACTION_EXPERT_INIT_PATH" ]]; then
    PI0_ACTION_EXPERT_INIT_PATH="$DEFAULT_PI0_ACTION_EXPERT_INIT_PATH"
  fi
elif [[ "$ENABLE_PI0_ACTION_EXPERT" == "true" && "$PI0_ACTION_EXPERT_INIT_DISABLED" != "true" && -z "$PI0_ACTION_EXPERT_INIT_PATH" && -d "$MODEL_ID" ]]; then
  mapfile -t _pi0_init_candidates < <(
    find "$MODEL_ID" -maxdepth 1 -type f \
      -name "qwen3_vl_pi0_action_expert_w${PI0_ACTION_WIDTH}_m${PI0_ACTION_MLP_SIZE}_l*.safetensors" \
      | sort
  )
  if [[ "${#_pi0_init_candidates[@]}" -eq 1 ]]; then
    PI0_ACTION_EXPERT_INIT_PATH="${_pi0_init_candidates[0]}"
  elif [[ "${#_pi0_init_candidates[@]}" -gt 1 ]]; then
    echo "[overfit] found multiple pi0 init sidecars; set PI0_ACTION_EXPERT_INIT_PATH or PI0_NUM_LAYERS explicitly" >&2
    printf '[overfit] candidate: %s\n' "${_pi0_init_candidates[@]}" >&2
  fi
fi

MAX_VISION_PATCHES_PER_SAMPLE="${MAX_VISION_PATCHES_PER_SAMPLE:-0}"
MAX_VISION_IMAGES_PER_SAMPLE="${MAX_VISION_IMAGES_PER_SAMPLE:-0}"

OVERWRITE_DATASETS="${OVERWRITE_DATASETS:-true}"
RUN_EXPORT="${RUN_EXPORT:-true}"
OVERFIT_REPEAT_RECORDS="${OVERFIT_REPEAT_RECORDS:-$((DP_SIZE * FSDP_SIZE))}"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

mkdir -p "$ARTIFACT_ROOT" "$JAX_CACHE_DIR"
if [[ -n "$TOKAMAX_CACHE_DIR" ]]; then
  mkdir -p "$TOKAMAX_CACHE_DIR"
fi

SLURM_NTASKS_EFFECTIVE="${SLURM_NTASKS:-1}"
SLURM_PROCID_EFFECTIVE="${SLURM_PROCID:-0}"
IS_PRIMARY_PROCESS="false"
if [[ "$SLURM_PROCID_EFFECTIVE" == "0" ]]; then
  IS_PRIMARY_PROCESS="true"
fi

if [[ "$SLURM_NTASKS_EFFECTIVE" -gt 1 && -n "${SLURM_STEP_GPUS:-}" ]]; then
  # JUWELS exports CUDA_VISIBLE_DEVICES as a single physical ordinal per task.
  # JAX distributed expects a dense visible-device list plus a per-process
  # JAX_CUDA_VISIBLE_DEVICES selector.
  export CUDA_VISIBLE_DEVICES="$SLURM_STEP_GPUS"
  export JAX_CUDA_VISIBLE_DEVICES="${JAX_CUDA_VISIBLE_DEVICES:-${SLURM_LOCALID:-$SLURM_PROCID_EFFECTIVE}}"
fi

EXPECTED_JAX_DEVICES="$((TP_SIZE * FSDP_SIZE * DP_SIZE))"
if [[ "${SKIP_JAX_DEVICE_PREFLIGHT:-false}" != "true" && "$SLURM_NTASKS_EFFECTIVE" -eq 1 ]]; then
  set +e
  preflight_output="$(EXPECTED_JAX_DEVICES="$EXPECTED_JAX_DEVICES" uv run -- python - <<'PY'
import os
import sys

import jax

expected = int(os.environ["EXPECTED_JAX_DEVICES"])
devices = jax.devices()
print(
    f"[overfit] JAX preflight: expected_devices={expected} "
    f"device_count={jax.device_count()} local_device_count={jax.local_device_count()} "
    f"process_count={jax.process_count()}"
)
print("[overfit] JAX devices: " + ", ".join(str(d) for d in devices))
for name in (
    "CUDA_VISIBLE_DEVICES",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE",
):
    print(f"[overfit] env {name}={os.environ.get(name, '<unset>')}")

if jax.device_count() != expected:
    print(
        "[overfit] ERROR: JAX cannot build the requested mesh because the "
        f"runtime sees {jax.device_count()} device(s), but tp*fsdp*dp={expected}.",
        file=sys.stderr,
    )
    sys.exit(2)
PY
)"
  preflight_status=$?
  set -e
  echo "$preflight_output"
  if [[ "$preflight_status" -ne 0 ]]; then
    cat >&2 <<EOF
[overfit] Requested mesh: tp=$TP_SIZE fsdp=$FSDP_SIZE dp=$DP_SIZE.
[overfit] For the default 4-way FSDP run, launch one process on a node where all 4 GPUs are visible.
[overfit] On SLURM this usually means one task with four GPUs, e.g.:
[overfit]   srun --nodes=1 --ntasks=1 --gpus-per-node=4 bash scripts/run_qwen3vl_pi0_action_expert_overfit.sh
[overfit] If you intentionally want a 1-GPU expert-only smoke test, use:
[overfit]   FSDP_SIZE=1 BATCH_SIZE=1 PI0_TRAIN_SCOPE=expert_only bash scripts/run_qwen3vl_pi0_action_expert_overfit.sh
EOF
    exit "$preflight_status"
  fi
elif [[ "${SKIP_JAX_DEVICE_PREFLIGHT:-false}" != "true" && "$IS_PRIMARY_PROCESS" == "true" ]]; then
  echo "[overfit] skipping single-process JAX preflight for distributed SLURM_NTASKS=$SLURM_NTASKS_EFFECTIVE"
fi

prepare_repeated_jsonl() {
  local src="$1"
  local dst="$2"
  local repeat="$3"
  uv run -- python - "$src" "$dst" "$repeat" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
repeat = int(sys.argv[3])
lines = [line for line in src.read_text().splitlines() if line.strip()]
if not lines:
    raise SystemExit(f"no records found in {src}")
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w") as f:
    for _ in range(max(1, repeat)):
        for line in lines:
            f.write(line + "\n")
PY
}

TRAIN_JSONL_EFFECTIVE="$ARTIFACT_ROOT/train_one_sequence_repeated.jsonl"
VAL_JSONL_EFFECTIVE="$ARTIFACT_ROOT/val_one_sequence_repeated.jsonl"
DATA_READY_FILE="$ARTIFACT_ROOT/.datasets_ready"
if [[ "$IS_PRIMARY_PROCESS" == "true" ]]; then
  rm -f "$DATA_READY_FILE"
  prepare_repeated_jsonl "$TRAIN_JSONL" "$TRAIN_JSONL_EFFECTIVE" "$OVERFIT_REPEAT_RECORDS"
  prepare_repeated_jsonl "$VAL_JSONL" "$VAL_JSONL_EFFECTIVE" "$OVERFIT_REPEAT_RECORDS"

  echo "[overfit] root=$ROOT_DIR"
  echo "[overfit] preset=$RUN_PRESET run_id=$RUN_ID artifact_root=$ARTIFACT_ROOT"
  echo "[overfit] model=$MODEL_ID"
  echo "[overfit] train_jsonl=$TRAIN_JSONL_EFFECTIVE"
  echo "[overfit] val_jsonl=$VAL_JSONL_EFFECTIVE"
  echo "[overfit] num_steps=$NUM_STEPS log_every=$LOG_EVERY_STEPS val_every=$VAL_EVERY_STEPS save_every=$SAVE_EVERY_STEPS save_steps=${SAVE_STEPS:-<none>} early_stop_loss=${EARLY_STOP_LOSS:-<none>}"
  echo "[overfit] mesh tp=$TP_SIZE fsdp=$FSDP_SIZE dp=$DP_SIZE batch_size=$BATCH_SIZE max_length=$MAX_LENGTH"
  echo "[overfit] attention text=$TEXT_ATTN_BACKEND vision=$VISION_ATTN_BACKEND"
  echo "[overfit] pi0 enabled=$ENABLE_PI0_ACTION_EXPERT width=$PI0_ACTION_WIDTH mlp=$PI0_ACTION_MLP_SIZE train_scope=$PI0_TRAIN_SCOPE"
  echo "[overfit] pi0 init=$PI0_ACTION_EXPERT_INIT_PATH"
  echo "[overfit] normal freeze_vision_tower=$FREEZE_VISION_TOWER"
  echo "[overfit] export=$RUN_EXPORT resume=$RESUME_MODE log_memory=$LOG_MEMORY tokamax_cache=${TOKAMAX_CACHE_DIR:-<disabled>}"

  uv run scripts/compile_sft_dataset.py \
    --data_path "$TRAIN_JSONL_EFFECTIVE" \
    --out_dir "$TRAIN_PAYLOAD_DIR" \
    --messages_per_record 8 \
    --records_per_shard 1 \
    --overwrite="$OVERWRITE_DATASETS"

  uv run scripts/compile_sft_dataset.py \
    --data_path "$VAL_JSONL_EFFECTIVE" \
    --out_dir "$VAL_PAYLOAD_DIR" \
    --messages_per_record 8 \
    --records_per_shard 1 \
    --overwrite="$OVERWRITE_DATASETS"

  uv run scripts/build_sft_chunk_index.py \
    --data_path "$TRAIN_PAYLOAD_DIR" \
    --out_dir "$TRAIN_CHUNKS_DIR" \
    --model_id "$MODEL_ID" \
    --tokenizer "$PROCESSOR" \
    --processor "$PROCESSOR" \
    --preprocessor_config "$PREPROCESSOR_CONFIG" \
    --max_length "$MAX_LENGTH" \
    --records_per_shard 1 \
    --num_workers 2 \
    --overwrite="$OVERWRITE_DATASETS"

  uv run scripts/build_sft_chunk_index.py \
    --data_path "$VAL_PAYLOAD_DIR" \
    --out_dir "$VAL_CHUNKS_DIR" \
    --model_id "$MODEL_ID" \
    --tokenizer "$PROCESSOR" \
    --processor "$PROCESSOR" \
    --preprocessor_config "$PREPROCESSOR_CONFIG" \
    --max_length "$MAX_LENGTH" \
    --records_per_shard 1 \
    --num_workers 2 \
    --overwrite="$OVERWRITE_DATASETS"

  touch "$DATA_READY_FILE"
else
  echo "[overfit] rank=$SLURM_PROCID_EFFECTIVE waiting for dataset preparation: $DATA_READY_FILE"
  until [[ -f "$DATA_READY_FILE" ]]; do
    sleep 5
  done
fi

train_args=(
  scripts/train_vlm_sft.py
  --model_id "$MODEL_ID"
  --processor "$PROCESSOR"
  --preprocessor_config "$PREPROCESSOR_CONFIG"
  --data_path "$TRAIN_CHUNKS_DIR"
  --val_data_path "$VAL_CHUNKS_DIR"
  --val_every "$VAL_EVERY_STEPS"
  --val_steps "$VAL_STEPS"
  --max_length "$MAX_LENGTH"
  --num_steps "$NUM_STEPS"
  --batch_size "$BATCH_SIZE"
  --learning_rate "$LEARNING_RATE"
  --weight_decay "$WEIGHT_DECAY"
  --warmup_steps "$WARMUP_STEPS"
  --max_grad_norm "$MAX_GRAD_NORM"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --save_dir "$SAVE_DIR"
  --save_every "$SAVE_EVERY_STEPS"
  --log_every "$LOG_EVERY_STEPS"
  --jax_cache_dir "$JAX_CACHE_DIR"
  --grain_workers "$GRAIN_WORKERS"
  --grain_read_threads "$GRAIN_READ_THREADS"
  --grain_read_buffer_size "$GRAIN_READ_BUFFER_SIZE"
  --grain_worker_buffer_size "$GRAIN_WORKER_BUFFER_SIZE"
  --tp_size "$TP_SIZE"
  --fsdp_size "$FSDP_SIZE"
  --dp_size "$DP_SIZE"
  --text_attn_backend "$TEXT_ATTN_BACKEND"
  --num_loss_tiles "$NUM_LOSS_TILES"
  --max_vision_patches_per_sample "$MAX_VISION_PATCHES_PER_SAMPLE"
  --max_vision_images_per_sample "$MAX_VISION_IMAGES_PER_SAMPLE"
  --log_memory="$LOG_MEMORY"
  --resume "$RESUME_MODE"
)

if [[ "$ENABLE_PI0_ACTION_EXPERT" == "true" ]]; then
  train_args+=(
    --pi0_action_expert
    --pi0_action_width "$PI0_ACTION_WIDTH"
    --pi0_action_mlp_size "$PI0_ACTION_MLP_SIZE"
    --pi0_train_scope "$PI0_TRAIN_SCOPE"
  )
elif [[ "$FREEZE_VISION_TOWER" == "true" ]]; then
  train_args+=(--freeze_vision_tower)
fi

if [[ -n "$TOKAMAX_CACHE_DIR" ]]; then
  train_args+=(--tokamax_cache_dir "$TOKAMAX_CACHE_DIR")
fi

if [[ "$ENABLE_PI0_ACTION_EXPERT" == "true" && "$PI0_ACTION_EXPERT_INIT_DISABLED" == "true" ]]; then
  train_args+=(--pi0_action_expert_init_path none)
elif [[ "$ENABLE_PI0_ACTION_EXPERT" == "true" && -n "$PI0_ACTION_EXPERT_INIT_PATH" ]]; then
  train_args+=(--pi0_action_expert_init_path "$PI0_ACTION_EXPERT_INIT_PATH")
fi
if [[ -n "$SAVE_STEPS" ]]; then
  train_args+=(--save_steps "$SAVE_STEPS")
fi
if [[ -n "$EARLY_STOP_LOSS" ]]; then
  train_args+=(--early_stop_loss "$EARLY_STOP_LOSS")
fi

uv run "${train_args[@]}"

if [[ "$RUN_EXPORT" == "true" && "$SLURM_NTASKS_EFFECTIVE" -gt 1 ]]; then
  if [[ "$IS_PRIMARY_PROCESS" == "true" ]]; then
    echo "[overfit] skipping export inside distributed training step; run export separately after the checkpoint is written"
  fi
elif [[ "$RUN_EXPORT" == "true" ]]; then
  latest_ckpt="$(find "$SAVE_DIR" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended -regex '.*/[0-9]+' | sort | tail -n 1)"
  if [[ -z "$latest_ckpt" ]]; then
    echo "[overfit] no checkpoint directory found under $SAVE_DIR" >&2
    exit 1
  fi
  echo "[overfit] exporting checkpoint $latest_ckpt to $EXPORT_DIR"
  uv run scripts/export_to_hf.py \
    --model_id "$MODEL_ID" \
    --checkpoint_path "$latest_ckpt" \
    --out_dir "$EXPORT_DIR" \
    --tp_size "$TP_SIZE" \
    --fsdp_size "$FSDP_SIZE" \
    --dp_size "$DP_SIZE"
  if [[ "$ENABLE_PI0_ACTION_EXPERT" == "true" ]]; then
    echo "[overfit] SGLang action expert:"
    exported_sidecar="$(find "$EXPORT_DIR" -maxdepth 1 -type f \
      -name "qwen3_vl_pi0_action_expert_w${PI0_ACTION_WIDTH}_m${PI0_ACTION_MLP_SIZE}_l*.safetensors" \
      | sort | tail -n 1)"
    if [[ -n "$exported_sidecar" ]]; then
      echo "  $exported_sidecar"
    else
      echo "  [overfit] no exported sidecar matching width=$PI0_ACTION_WIDTH mlp=$PI0_ACTION_MLP_SIZE found under $EXPORT_DIR" >&2
    fi
  else
    echo "[overfit] normal SGLang model exported at $EXPORT_DIR"
  fi
fi
