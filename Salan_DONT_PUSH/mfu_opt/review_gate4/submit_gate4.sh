#!/usr/bin/env bash
# Gate 4: ten-step A/B training pilot on real FineWeb-Edu data. Checkpoint-free.
# One bounded H100 job. Usage: ./submit_gate4.sh [tag] [seeds]
set -euo pipefail

TAG="${1:-g4}"
SEEDS="${2:-0,1,2}"

ROOT='/fast/home/salan.isaqzoi/gh_projects/p(doom)/omegalax_optimization'
OUT='/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs/mfu_opt/gate4'
mkdir -p "$OUT"

SBATCH_FILE="$OUT/gate4_${TAG}.sbatch"
cat > "$SBATCH_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=gate4_${TAG}
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --output=$OUT/gate4_${TAG}_%j.log
#SBATCH --error=$OUT/gate4_${TAG}_%j.log

set -euo pipefail
cd '$ROOT'
export PYTHONUNBUFFERED=1
# Must precede every Python process in this tree: .venv is a symlink to the
# original's and its editable install otherwise resolves omegalax to the original.
export PYTHONPATH='$ROOT'
export JAX_PLATFORMS=cuda
export JAX_LOCAL_DEVICE_IDS=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# E19/E26: fixed kernel selection, so the orig_a/orig_b control measures trajectory
# variability rather than autotuner variability on top of it.
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export WANDB_MODE=disabled

'$ROOT'/.venv/bin/python -c "import omegalax; print('parent omegalax ->', omegalax.__file__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

'$ROOT'/.venv/bin/python Salan_DONT_PUSH/mfu_opt/review_gate4/gate4_pilot.py \\
  --tag '${TAG}' --seeds '${SEEDS}' --results '$OUT'
EOF

job_id=$(sbatch --parsable "$SBATCH_FILE")
echo "submitted job=$job_id tag=$TAG seeds=$SEEDS"
echo "log=$OUT/gate4_${TAG}_${job_id}.log"
echo "summary=$OUT/gate4_summary_${TAG}.json"
