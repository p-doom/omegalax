#!/bin/bash
#SBATCH --job-name=rollout-comparison-rmt-conv
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=720G
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@900
set -euo pipefail
SOURCE=$1 PLAN=$2 OUTPUT=$3 MODEL=$4
shift 4
# Both Conv1D + RMT arms in one node allocation, each with the resources of a single-arm job
# (24 CPUs, 360G, 4 GPUs). Remaining arguments go to both trainers.
# A trainer that checkpoints on the deadline signal requeues the job, which would kill the
# other arm while it still saves. The request is recorded instead and acted on after every
# arm has exited: with two unfinished arms that exited cleanly the pair is requeued; with one,
# that arm continues as a single-arm job (1 node, 4 GPUs) that requeues itself. An arm that
# exits non-zero gets a crashed_T<arm>_<job> marker and is not restarted automatically.
REQUEUE="$OUTPUT/requeue_requested_$SLURM_JOB_ID"
SHIM="$OUTPUT/requeue_shim_$SLURM_JOB_ID"
SCONTROL=$(command -v scontrol)
mkdir -p "$SHIM"
printf '#!/bin/bash\nif test "${1:-}" = requeue; then touch %q; exit 0; fi\nexec %q "$@"\n' \
    "$REQUEUE" "$SCONTROL" > "$SHIM/scontrol"
chmod +x "$SHIM/scontrol"
PYTHON='/fast/home/salan.isaqzoi/gh_projects/p(doom)/omegalax/.venv/bin/python'
UPDATES=$("$PYTHON" -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["updates"]))' "$PLAN")
PIDS=() LAUNCHED=()
for ARM_ARGS in "8192 --pass-conv-state --num-rmt-tokens 128" "512 --pass-conv-state --num-rmt-tokens 24"; do
    ARM=${ARM_ARGS%% *}
    if test -e "$OUTPUT/T$ARM/result_step$UPDATES.json"; then
        continue
    fi
    # Only the trainers see the shim, so a single-arm job submitted below gets the real scontrol.
    # shellcheck disable=SC2086
    PATH="$SHIM:$PATH" srun --exact --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=360G --gres=gpu:4 \
        --output="$OUTPUT/T$ARM-%j.log" --open-mode=append \
        bash "$SOURCE/scripts/rollout_comparison_node.sh" "$SOURCE" "$PLAN" "$OUTPUT" "$MODEL" train $ARM_ARGS "$@" &
    PIDS+=("$!") LAUNCHED+=("$ARM_ARGS")
done
STATUS=0 REMAINING=()
for I in "${!PIDS[@]}"; do
    ARM=${LAUNCHED[$I]%% *}
    if wait "${PIDS[$I]}"; then
        test -e "$OUTPUT/T$ARM/result_step$UPDATES.json" || REMAINING+=("${LAUNCHED[$I]}")
    else
        CODE=$?
        STATUS=1
        echo "T$ARM exited with $CODE in job $SLURM_JOB_ID; not restarted" | tee "$OUTPUT/crashed_T${ARM}_$SLURM_JOB_ID"
    fi
done
if test -e "$REQUEUE"; then
    rm "$REQUEUE"
    if test "${#REMAINING[@]}" = 2; then
        "$SCONTROL" requeue "$SLURM_JOB_ID"
    elif test "${#REMAINING[@]}" = 1; then
        ARM=${REMAINING[0]%% *}
        # shellcheck disable=SC2086
        JOB=$(sbatch --parsable --job-name="rollout-comparison-rmt-conv-T$ARM" --signal=USR1@900 \
            --output="$OUTPUT/T$ARM-%j.log" --open-mode=append \
            "$SOURCE/scripts/rollout_comparison_single.sbatch" "$SOURCE" "$PLAN" "$OUTPUT" "$MODEL" train ${REMAINING[0]} "$@")
        echo "T$ARM continues alone as single-arm job $JOB"
    fi
fi
exit "$STATUS"
