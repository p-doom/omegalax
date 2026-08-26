#!/usr/bin/env bash
# Usage: [ARM=gpu_tests|train200|both] [REPO=<checkout>] sbatch scripts/verify_vision_splice.sbatch
#   gpu_tests: interleaved-vs-single-block layout equivalence on 1 GPU
#   train200:  200 steps, 4 processes, per-batch-index loss in the log
#              (pre-fix: loss_bidx_0 far below loss_bidx_1..3)
#SBATCH --job-name=verify_vision_splice
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --exclude=hai001
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=200GB
#SBATCH --time=03:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/logs/%x_%j.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/yll/logs/%x_%j.log

set -uo pipefail
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface
export PATH=/home/yll.kryeziu/.local/bin:/usr/local/cuda-12/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin
export UV_CACHE_DIR=/fast/project/HFMI_SynergyUnit/yll/.cache/uv
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

ARM=${ARM:-both}
REPO=${REPO:-/fast/project/HFMI_SynergyUnit/yll/omegalax}
PY=/fast/project/HFMI_SynergyUnit/yll/omegalax/.venv/bin/python
export PYTHONPATH="$REPO"
DATA=/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/yll.kryeziu/cuagym_p2_stage_06_perstep_records_seqlen_16384_v2
SAVE=/fast/project/HFMI_SynergyUnit/yll/checkpoints/verify_vision_splice_${SLURM_JOB_ID}
CACHE=/fast/project/HFMI_SynergyUnit/yll/.cache/jax_cache/verify_vision_splice

cd "$REPO" || exit 1
echo "REPO=$REPO commit=$(git rev-parse --short HEAD 2>/dev/null)"

rc=0

if [ "$ARM" = "gpu_tests" ] || [ "$ARM" = "both" ]; then
    echo "=== stage 1: interleaved-layout equivalence (1 GPU) ==="
    srun --ntasks=1 --gres=gpu:1 --overlap "$PY" -m unittest \
        tests.test_vision_splice_multiprocess_layout \
        tests.test_fix_slowdown_ports
    rc=$?
    echo "=== stage 1 exit=$rc ==="
    [ "$rc" -ne 0 ] && exit "$rc"
fi

if [ "$ARM" = "train200" ] || [ "$ARM" = "both" ]; then
    echo "=== stage 2: 200 steps, 4 processes, per-batch-index loss ==="
    srun "$PY" "$REPO/scripts/train_vlm_sft.py" \
        --model_id=Qwen/Qwen3.5-9B \
        --data_path="$DATA/train" \
        --save_dir="$SAVE" \
        --resume=never \
        --seed=0 \
        --pad_id=0 \
        --peak_tflops=h100_sxm \
        --log_memory=false \
        --tp_size=1 \
        --fsdp_size=4 \
        --dp_size=1 \
        --max_length=16384 \
        --num_steps=200 \
        --batch_size=4 \
        --grad_accum_steps=1 \
        --save_every=0 \
        --keep_period=0 \
        --keep_latest=1 \
        --log_every=5 \
        --gc_period=100 \
        --text_attn_backend=mosaic_gpu \
        --max_vision_patches_per_sample=41_000 \
        --max_vision_images_per_sample=6 \
        --log_per_sample_loss=true \
        --learning_rate=1e-4 \
        --lr_schedule=wsd \
        --lr_stable_fraction=0.7 \
        --lr_end_factor=0.0 \
        --warmup_steps=20 \
        --weight_decay=0.01 \
        --max_grad_norm=1.0 \
        --num_loss_tiles=32 \
        --grain_read_threads=4 \
        --grain_read_buffer_size=2 \
        --grain_workers=2 \
        --grain_worker_buffer_size=2 \
        --enable_lora=true \
        --freeze_vision_tower=false \
        --lora_rank=128 \
        --lora_alpha=128 \
        --lora_extra_target_modules=in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj \
        --jax_cache_dir="$CACHE"
    rc=$?
    echo "=== stage 2 exit=$rc ==="
fi

exit "$rc"
