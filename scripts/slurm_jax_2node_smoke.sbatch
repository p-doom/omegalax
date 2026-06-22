#!/usr/bin/env bash
# Cheap Slurm/JAX distributed smoke test for 2 booster nodes x 4 GPUs.

#SBATCH --account=envcomp
#SBATCH --partition=develbooster
#SBATCH --job-name=ogx_jax2n_smoke
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=/p/project1/envcomp/keihan/slurm_logs/%x_%j.out
#SBATCH --error=/p/project1/envcomp/keihan/slurm_logs/%x_%j.err

set -euo pipefail

ROOT_DIR="/p/project1/envcomp/keihan/omegalax-pi0-action-expert-training"
cd "$ROOT_DIR"
mkdir -p /p/project1/envcomp/keihan/.uv_cache
export UV_CACHE_DIR="/p/project1/envcomp/keihan/.uv_cache"
export UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv"

echo "[smoke] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES ntasks=$SLURM_NTASKS nodelist=$SLURM_JOB_NODELIST"
date -u +"[smoke] start_utc=%Y-%m-%dT%H:%M:%SZ"

srun --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=single:1 bash -lc '
set -euo pipefail
cd /p/project1/envcomp/keihan/omegalax-pi0-action-expert-training
export UV_CACHE_DIR="/p/project1/envcomp/keihan/.uv_cache"
export UV_PROJECT_ENVIRONMENT="/p/project1/envcomp/keihan/omegalax-pi0-action-expert-training/.venv"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy || true

if [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$SLURM_STEP_GPUS"
  export JAX_CUDA_VISIBLE_DEVICES="${JAX_CUDA_VISIBLE_DEVICES:-${SLURM_LOCALID:-${SLURM_PROCID:-0}}}"
fi

uv run python - <<'"'"'PY'"'"'
import os
import socket

import jax

jax.distributed.initialize(initialization_timeout=180)

rank = jax.process_index()
info = {
    "host": socket.gethostname(),
    "rank": rank,
    "process_count": jax.process_count(),
    "device_count": jax.device_count(),
    "local_device_count": jax.local_device_count(),
    "slurm_procid": os.environ.get("SLURM_PROCID"),
    "slurm_localid": os.environ.get("SLURM_LOCALID"),
    "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "jax_cuda_visible_devices": os.environ.get("JAX_CUDA_VISIBLE_DEVICES"),
    "devices": [str(d) for d in jax.devices()],
    "local_devices": [str(d) for d in jax.local_devices()],
}
print("[smoke-rank]", info, flush=True)

if jax.process_count() != 8:
    raise SystemExit(f"expected process_count=8, got {jax.process_count()}")
if jax.device_count() != 8:
    raise SystemExit(f"expected global device_count=8, got {jax.device_count()}")
if jax.local_device_count() != 1:
    raise SystemExit(f"expected local_device_count=1 per rank, got {jax.local_device_count()}")

print("[smoke-rank]", rank, "ok", flush=True)
PY
'

date -u +"[smoke] end_utc=%Y-%m-%dT%H:%M:%SZ"
