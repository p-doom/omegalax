#!/bin/bash
#SBATCH --job-name=rollout-comparison
#SBATCH --partition=standard
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=360G
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
set -euo pipefail
SOURCE=$1 PLAN=$2 OUTPUT=$3 MODEL=$4
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
test "${#NODES[@]}" = 3
PIDS=()
ARMS=(8192 2048 512)
for I in 0 1 2; do
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=360G --gres=gpu:4 -w "${NODES[$I]}" \
        --output="$OUTPUT/T${ARMS[$I]}_${SLURM_JOB_ID}.log" \
        bash "$SOURCE/scripts/rollout_comparison_node.sh" "$SOURCE" "$PLAN" "$OUTPUT" "$MODEL" train "${ARMS[$I]}" &
    PIDS+=("$!")
done
STATUS=0
for P in "${PIDS[@]}"; do wait "$P" || STATUS=1; done
exit "$STATUS"
