#!/usr/bin/env bash
# Submit a short benchmark of the optimization codebase.
#
#   ./submit_bench.sh <tag> [extra train flags...]
#
# Env:
#   MFU_TRACE=1        capture a profiler trace of steps 5..7
#   BENCH_GPUS=1       GPUs (fsdp_size follows)
#   BENCH_BATCH=24     segments per step  (4 chains x C)
#   BENCH_STEPS=10     steps
#   BENCH_C=6          statepassing chains; picks the matching index
#   BENCH_WANDB=1      log to wandb (default on). Benchmarks go to their own
#                      project so they never mix with real training runs.
#   BENCH_XLA_FLAGS    extra XLA_FLAGS. Setting this also forces a private JAX
#                      compilation cache: the persistent cache key does not cover
#                      the XLA_FLAGS env var, so a hit from an earlier run would
#                      silently reuse an executable compiled without the flags.
#
# Runs against the OPTIMIZATION copy via PYTHONPATH. The venv is a symlink to the
# original's, and omegalax is an editable install pointing at the original, so
# without PYTHONPATH this would silently benchmark the wrong tree.
set -euo pipefail

TAG="${1:?usage: submit_bench.sh <tag> [extra flags]}"
shift || true
EXTRA="$*"

REPO='/fast/home/salan.isaqzoi/gh_projects/p(doom)/omegalax_optimization'
BENCH_ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs/mfu_opt
JOBS_DIR="${BENCH_ROOT}/jobs"
mkdir -p "${JOBS_DIR}"

GPUS="${BENCH_GPUS:-1}"
DEVICE_IDS=$(seq -s, 0 $((GPUS - 1)))
BATCH="${BENCH_BATCH:-24}"
STEPS="${BENCH_STEPS:-10}"
ACCUM="${BENCH_ACCUM:-1}"
C="${BENCH_C:-6}"
INDEX="/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/pretrain_indexes/fineweb_edu_dedup_2048_2kto32k_2048_c${C}_bptt${C}_lr3e4_eos248046"

WANDB_ARGS=""
WANDB_ENV="export WANDB_MODE=disabled"
if [ "${BENCH_WANDB:-1}" = "1" ]; then
  WTAGS="c=${BENCH_C:-6},b=${BENCH_BATCH:-24},gpus=${BENCH_GPUS:-1}"
  WTAGS="${WTAGS},cascade=${MFU_CASCADE:-0},convbtc=${MFU_CONV_BTC:-0}"
  WTAGS="${WTAGS},remat=${MFU_REMAT:-none},mu=${MFU_MU_DTYPE:-fp32}"
  [ -n "${BENCH_XLA_FLAGS:-}" ] && WTAGS="${WTAGS},xlaflags"
  WANDB_ARGS="--wandb_entity=pdoom --wandb_project=omegalax_mfu --wandb_group=mfu_opt --wandb_name=${TAG} --wandb_tags=${WTAGS}"
  WANDB_ENV="export WANDB_MODE=online"
fi

ALLOC_ENV=""
[ -n "${BENCH_ALLOC:-}" ] && ALLOC_ENV="export TF_GPU_ALLOCATOR=${BENCH_ALLOC}"

XLA_ENV=""
JAX_CACHE=/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/jax_cache/mfu_opt
if [ -n "${BENCH_XLA_FLAGS:-}" ]; then
  XLA_ENV="export XLA_FLAGS=\"\${XLA_FLAGS:-} ${BENCH_XLA_FLAGS}\";"
  JAX_CACHE="${JAX_CACHE}_${TAG}"
fi

TRACE_ENV=""
if [ "${MFU_TRACE:-0}" = "1" ]; then
  TRACE_DIR="${BENCH_ROOT}/traces/${TAG}"
  TRACE_ENV="export MFU_TRACE_DIR=${TRACE_DIR}; export MFU_TRACE_START=${MFU_TRACE_START:-5}; export MFU_TRACE_STEPS=${MFU_TRACE_STEPS:-3};"
fi

SBATCH_FILE="${JOBS_DIR}/bench_${TAG}.sbatch"
cat > "${SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=mfuopt_${TAG}
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:${GPUS}
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --output=${BENCH_ROOT}/${TAG}_%j.log
#SBATCH --error=${BENCH_ROOT}/${TAG}_%j.log

set -euo pipefail
cd '${REPO}'
export PYTHONUNBUFFERED=1
export PYTHONPATH='${REPO}'
# jax.debug.print braucht ein CPU-Device fuer den Host-Callback.
export JAX_PLATFORMS=${BENCH_PLATFORMS:-cuda}
export JAX_LOCAL_DEVICE_IDS=${DEVICE_IDS}
# Lowering this leaves room for the driver. Instantiating a CUDA graph with
# a few thousand nodes needs driver memory outside the client pool, and at
# 0.95 that failed with CUDA_ERROR_OUT_OF_MEMORY while device memory was
# still half free.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${BENCH_MEM_FRACTION:-0.95}
${ALLOC_ENV}
export OMEGALAX_DELTANET_KERNEL=pallas
export MFU_RELAX_INDEX=1
export MFU_NUM_LOSS_TILES=${MFU_NUM_LOSS_TILES:-4}
export MFU_CASCADE=${MFU_CASCADE:-0}
export MFU_CONV_BTC=${MFU_CONV_BTC:-0}
export MFU_TRIINV=${MFU_TRIINV:-0}
export MFU_TRIBLOCK=${MFU_TRIBLOCK:-0}
export MFU_TRI_BLOCK=${MFU_TRI_BLOCK:-16}
export MFU_TRIINV_DEBUG=${MFU_TRIINV_DEBUG:-0}
export MFU_BWD_PALLAS=${MFU_BWD_PALLAS:-0}
export MFU_PALLAS_WARPS=${MFU_PALLAS_WARPS:-4}
export MFU_PALLAS_STAGES=${MFU_PALLAS_STAGES:-2}
export MFU_FAST_CE=${MFU_FAST_CE:-0}
export MFU_REMAT=${MFU_REMAT:-none}
export MFU_MEMLOG=${MFU_MEMLOG:-}
export MFU_MU_DTYPE=${MFU_MU_DTYPE:-fp32}
export MFU_BUDGET=${MFU_BUDGET:-0}
${XLA_ENV}
${TRACE_ENV}
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/huggingface
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="\${CUDA_HOME:-/usr/local/cuda}/extras/CUPTI/lib64:\${LD_LIBRARY_PATH:-}"
${WANDB_ENV}

nvidia-smi
echo "=== MFU-BENCH tag=${TAG} gpus=${GPUS} batch=${BATCH} accum=${ACCUM} steps=${STEPS} C=${C}"
echo "=== XLA_FLAGS=\${XLA_FLAGS:-<none>}"
echo "=== source check"
'${REPO}'/.venv/bin/python -c "import omegalax; print('omegalax ->', omegalax.__file__)"

srun --ntasks=1 --ntasks-per-node=1 --gres=gpu:${GPUS} bash -lc '
  set -euo pipefail
  cd "${REPO}"
  export PYTHONPATH="${REPO}"
  "${REPO}"/.venv/bin/python Salan_DONT_PUSH/mfu_opt/run_bench.py \\
    --pretrain_mode=statepassing_bptt \\
    --init_hf_model_path=/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/models/Qwen/Qwen3.5-2B \\
    --train_index_path=${INDEX} \\
    --seq_len=2048 \\
    --batch_size=${BATCH} \\
    --grad_accum_steps=${ACCUM} \\
    --num_steps=${STEPS} \\
    --lr_schedule_steps=1000 \\
    --warmup_tokens=100000000 \\
    --learning_rate=2e-5 \\
    --lr_schedule=wsd \\
    --lr_stable_fraction=0.8 \\
    --lr_end_factor=0.0 \\
    --weight_decay=0.01 \\
    --adam_beta1=0.9 --adam_beta2=0.95 --adam_eps=1e-8 \\
    --max_grad_norm=1.0 \\
    --tp_size=1 --fsdp_size=${GPUS} --dp_size=1 \\
    --iterator_fsdp_size=${GPUS} --iterator_dp_size=1 \\
    --bptt_chunks=${C} \\
    --pass_gdn_state=true \\
    --gdn_state_routing=same_layer \\
    --pass_rope_positions=false \\
    --pass_conv_state=false \\
    --jax_cache_dir=${JAX_CACHE} \\
    --save_every=0 \\
    --log_every=1 \\
    --resume=never \\
    --text_attn_backend=mosaic_gpu \\
    --peak_tflops=h100_sxm \\
    ${WANDB_ARGS} \\
    --seed=0 ${EXTRA}
'
EOF

JOBID=$(sbatch --parsable "${SBATCH_FILE}")
echo "submitted tag=${TAG} job=${JOBID}"
echo "log=${BENCH_ROOT}/${TAG}_${JOBID}.log"
[ "${MFU_TRACE:-0}" = "1" ] && echo "trace=${BENCH_ROOT}/traces/${TAG}"
exit 0
