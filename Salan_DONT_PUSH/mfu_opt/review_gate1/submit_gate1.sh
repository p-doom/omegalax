#!/usr/bin/env bash
# Gate 1: per-flag equivalence of the optimized copy against the original.
# Checkpoint-free. Parameterised so the six seed/shape combinations can be split
# across jobs without reducing coverage.
#
#   ./submit_gate1.sh <tag> <seeds> <shapes> [arms] [steps] [chains] [seq_len] [timelimit]
#
# Examples:
#   ./submit_gate1.sh pilot 0 uniform all 1 2 128 00:40:00
#   ./submit_gate1.sh s0    0 uniform,padtail all 2 4 512 03:00:00
set -euo pipefail

TAG="${1:?tag required}"
SEEDS="${2:-0,1,2}"
SHAPES="${3:-uniform,padtail}"
ARMS="${4:-all}"
STEPS="${5:-2}"
CHAINS="${6:-4}"
SEQ_LEN="${7:-512}"
TIMELIMIT="${8:-03:00:00}"

ROOT='/fast/home/salan.isaqzoi/gh_projects/p(doom)/omegalax_optimization'
OUT='/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs/mfu_opt/gate1'
mkdir -p "$OUT"

SBATCH_FILE="$OUT/gate1_${TAG}.sbatch"
cat > "$SBATCH_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=gate1_${TAG}
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=${TIMELIMIT}
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --output=$OUT/gate1_${TAG}_%j.log
#SBATCH --error=$OUT/gate1_${TAG}_%j.log

set -euo pipefail
cd '$ROOT'
export PYTHONUNBUFFERED=1
# Must precede every Python process in this tree: .venv is a symlink to the
# original's and its editable install otherwise resolves omegalax to the original.
export PYTHONPATH='$ROOT'
export JAX_PLATFORMS=cuda
export JAX_LOCAL_DEVICE_IDS=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# E19/E26: without this XLA picks GEMM kernels by measured time, and two runs of
# identical code differ by ~5e-06. With it the orig-vs-orig arm is the exact noise
# floor. --xla_gpu_deterministic_ops is deliberately omitted: it fails to compile
# this graph, so the duplicate original arm remains the determinism gate.
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export WANDB_MODE=disabled

# Logged, not assumed: each child re-checks and hard-fails if PYTHONPATH selected
# the wrong tree.
'$ROOT'/.venv/bin/python -c "import omegalax; print('parent omegalax ->', omegalax.__file__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

'$ROOT'/.venv/bin/python Salan_DONT_PUSH/mfu_opt/review_gate1/gate1_equivalence.py \\
  --tag '${TAG}' \\
  --steps ${STEPS} \\
  --seeds '${SEEDS}' \\
  --shapes '${SHAPES}' \\
  --arms '${ARMS}' \\
  --chains ${CHAINS} \\
  --seq-len ${SEQ_LEN} \\
  --platform cuda --kernel pallas --attn-backend cudnn --dtype bf16 \\
  --results '$OUT'
EOF

job_id=$(sbatch --parsable "$SBATCH_FILE")
echo "submitted job=$job_id tag=$TAG seeds=$SEEDS shapes=$SHAPES arms=$ARMS steps=$STEPS chains=$CHAINS seq_len=$SEQ_LEN"
echo "log=$OUT/gate1_${TAG}_${job_id}.log"
echo "summary=$OUT/gate1_summary_${TAG}.json"
