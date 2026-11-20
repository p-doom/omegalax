#!/usr/bin/env bash
# Submit the checkpoint-free, ten-step training-equivalence probe.
set -euo pipefail

ROOT='/fast/home/salan.isaqzoi/gh_projects/p(doom)/omegalax_optimization'
OUT='/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs/mfu_opt/equivalence'
mkdir -p "$OUT"

SBATCH_FILE="$OUT/tiny_training_equivalence.sbatch"
cat > "$SBATCH_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=tiny_equiv
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --output=$OUT/tiny_equiv_%j.log
#SBATCH --error=$OUT/tiny_equiv_%j.log

set -euo pipefail
cd '$ROOT'
export PYTHONUNBUFFERED=1
# This must precede every Python process: .venv is shared with the original and
# its editable install otherwise imports omegalax from the wrong tree.
export PYTHONPATH='$ROOT'
export JAX_PLATFORMS=cuda
export JAX_LOCAL_DEVICE_IDS=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# This prevents autotuner timing from selecting different kernels, but it does
# not guarantee bitwise-equal raw gradients. ``--xla_gpu_deterministic_ops``
# is deliberately omitted because it causes an XLA compiler failure for this
# graph; the duplicate original arm remains the determinism gate.
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export WANDB_MODE=disabled

'$ROOT'/.venv/bin/python -c "import omegalax; print('omegalax ->', omegalax.__file__)"
'$ROOT'/.venv/bin/python Salan_DONT_PUSH/mfu_opt/compare_training_equivalence.py \\
  --steps 10 --batch 2 --seed 0 \\
  --platform cuda --kernel pallas --attn-backend cudnn --dtype bf16 \\
  --atol 2e-4 --rtol 2e-4
EOF

job_id=$(sbatch --parsable "$SBATCH_FILE")
echo "submitted job=$job_id"
echo "log=$OUT/tiny_equiv_${job_id}.log"
