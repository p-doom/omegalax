#!/bin/bash
#SBATCH --job-name=rollout-comparison-rmt-conv
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=720G
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@600
set -euo pipefail
SOURCE=$1 PLAN=$2 OUTPUT=$3 MODEL=$4
shift 4
# Both Conv1D + RMT arms in one node allocation, each with the resources of a single-arm job
# (24 CPUs, 360G, 4 GPUs). Remaining arguments go to both trainers.
# A trainer that checkpoints on the deadline signal requeues the job, which would kill the
# other arm while it still saves. The request is recorded instead, and the job is requeued
# only after every arm has exited cleanly. Finished arms are skipped after a requeue.
REQUEUE="$OUTPUT/requeue_requested_$SLURM_JOB_ID"
SHIM="$OUTPUT/requeue_shim_$SLURM_JOB_ID"
SCONTROL=$(command -v scontrol)
mkdir -p "$SHIM"
printf '#!/bin/bash\nif test "${1:-}" = requeue; then touch %q; exit 0; fi\nexec %q "$@"\n' \
    "$REQUEUE" "$SCONTROL" > "$SHIM/scontrol"
chmod +x "$SHIM/scontrol"
export PATH="$SHIM:$PATH"
PYTHON='/fast/home/salan.isaqzoi/gh_projects/p(doom)/omegalax/.venv/bin/python'
UPDATES=$("$PYTHON" -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["updates"]))' "$PLAN")
PIDS=()
for ARM_ARGS in "8192 --pass-conv-state --num-rmt-tokens 128" "512 --pass-conv-state --num-rmt-tokens 24"; do
    ARM=${ARM_ARGS%% *}
    if test -e "$OUTPUT/T$ARM/result_step$UPDATES.json"; then
        continue
    fi
    # shellcheck disable=SC2086
    srun --exact --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=360G --gres=gpu:4 \
        --output="$OUTPUT/T$ARM-%j.log" --open-mode=append \
        bash "$SOURCE/scripts/rollout_comparison_node.sh" "$SOURCE" "$PLAN" "$OUTPUT" "$MODEL" train $ARM_ARGS "$@" &
    PIDS+=("$!")
done
STATUS=0
for P in "${PIDS[@]}"; do wait "$P" || STATUS=1; done
if test "$STATUS" = 0 && test -e "$REQUEUE"; then
    rm "$REQUEUE"
    "$SCONTROL" requeue "$SLURM_JOB_ID"
fi
exit "$STATUS"
