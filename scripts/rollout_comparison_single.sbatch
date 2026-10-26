#!/bin/bash
#SBATCH --job-name=rollout-comparison
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=360G
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@600
set -euo pipefail
SOURCE=$1 PLAN=$2 OUTPUT=$3 MODEL=$4 MODE=$5 ARM=${6:-}
# One local JAX controller sees four GPUs. No cross-node JAX initialization.
exec srun --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=360G --gres=gpu:4 \
    bash "$SOURCE/scripts/rollout_comparison_node.sh" "$SOURCE" "$PLAN" "$OUTPUT" "$MODEL" "$MODE" "$ARM"
