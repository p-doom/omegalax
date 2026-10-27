#!/bin/bash
# Requeue/handoff decisions of the RMT+Conv pair launcher. Only srun, scontrol and sbatch are
# stubbed on PATH; the batch scripts themselves run unchanged.
# Usage: bash tests/test_rollout_comparison_rmt_conv_pair.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir "$WORK/bin"
echo '{"updates": [0, 0, 0]}' > "$WORK/plan.json"
# srun stub: ACTION_<arm> = requeue (deadline checkpoint: trainer calls scontrol requeue and
# exits 0), finish (writes the final result) or crash (exit 3).
cat > "$WORK/bin/srun" <<'EOF'
#!/bin/bash
echo "srun $*" >> "$CALLS"
ARGS=("$@")
for I in "${!ARGS[@]}"; do
    if test "${ARGS[$I]##*/}" = rollout_comparison_node.sh; then
        OUT=${ARGS[$((I + 3))]} ARM=${ARGS[$((I + 6))]}
    fi
done
ACTION="ACTION_$ARM"
case "${!ACTION:-}" in
    requeue) scontrol requeue "$SLURM_JOB_ID";;
    finish) mkdir -p "$OUT/T$ARM" && touch "$OUT/T$ARM/result_step3.json";;
    crash) exit 3;;
esac
EOF
cat > "$WORK/bin/scontrol" <<'EOF'
#!/bin/bash
echo "scontrol $*" >> "$CALLS"
EOF
cat > "$WORK/bin/sbatch" <<'EOF'
#!/bin/bash
case ":$PATH:" in *requeue_shim*) SHIM=yes;; *) SHIM=no;; esac
echo "sbatch $* shim=$SHIM WANDB_ENTITY=${WANDB_ENTITY:-}" >> "$CALLS"
echo 777
EOF
chmod +x "$WORK/bin/"*

run_pair() {  # NAME ACTION_8192 ACTION_512 -> sets OUT, CALLS, LOG, CODE
    OUT="$WORK/$1" CALLS="$WORK/$1.calls" LOG="$WORK/$1.log"
    mkdir -p "$OUT" && touch "$CALLS"
    CODE=0
    PATH="$WORK/bin:$PATH" SLURM_JOB_ID=4242 WANDB_ENTITY=pdoom CALLS="$CALLS" ACTION_8192=$2 ACTION_512=$3 \
        bash "$REPO/scripts/rollout_comparison_rmt_conv_pair.sbatch" "$REPO" "$WORK/plan.json" "$OUT" MODEL \
        --wandb-project omegalax > "$LOG" 2>&1 || CODE=$?
}
fail() { echo "FAIL: $*"; cat "$CALLS" "$LOG"; exit 1; }
count() { grep -c -- "$1" "$CALLS" || true; }

# Both arms checkpoint on the deadline: the pair is requeued with the real scontrol.
run_pair both_requeue requeue requeue
test "$CODE" = 0 || fail "both_requeue exit $CODE"
test "$(count '^scontrol requeue 4242$')" = 1 || fail "both_requeue: pair not requeued once"
test "$(count '^sbatch')" = 0 || fail "both_requeue: unexpected sbatch"
test ! -e "$OUT/requeue_requested_4242" || fail "both_requeue: stale request file"

# One arm crashes: it is reported and not restarted; the healthy arm continues alone.
run_pair crash_8192 crash requeue
test "$CODE" = 1 || fail "crash_8192 exit $CODE"
grep -q "T8192 exited with 3 in job 4242; not restarted" "$OUT/crashed_T8192_4242" || fail "crash marker"
grep -q "T8192 exited with 3" "$LOG" || fail "crash log line"
test "$(count '^scontrol')" = 0 || fail "crash_8192: 8-GPU pair requeued"
test "$(count '^sbatch')" = 1 || fail "crash_8192: no single sbatch"
grep -q -- "--signal=USR1@900 --output=$OUT/T512-%j.log --open-mode=append $REPO/scripts/rollout_comparison_single.sbatch $REPO $WORK/plan.json $OUT MODEL train 512 --pass-conv-state --num-rmt-tokens 24 --wandb-project omegalax shim=no WANDB_ENTITY=pdoom$" "$CALLS" \
    || fail "crash_8192: single sbatch arguments"
grep -q "T512 continues alone as single-arm job 777" "$LOG" || fail "handoff log line"

# One arm finishes: the other does not keep the 8-GPU allocation across the requeue.
run_pair finish_512 requeue finish
test "$CODE" = 0 || fail "finish_512 exit $CODE"
test "$(count '^scontrol')" = 0 || fail "finish_512: pair requeued"
test "$(count 'single.sbatch .* train 8192 --pass-conv-state --num-rmt-tokens 128 --wandb-project omegalax shim=no')" = 1 \
    || fail "finish_512: single sbatch for T8192"

# Pair restarted with T512 already finished: only T8192 runs, then continues alone.
OUT="$WORK/restarted" && mkdir -p "$OUT/T512" && touch "$OUT/T512/result_step3.json"
run_pair restarted requeue crash
test "$CODE" = 0 || fail "restarted exit $CODE"
test "$(count '^srun')" = 1 || fail "restarted: finished arm launched"
test "$(count 'single.sbatch .* train 8192 ')" = 1 || fail "restarted: single sbatch for T8192"

# Both finish: nothing is requeued or submitted.
run_pair both_finish finish finish
test "$CODE" = 0 || fail "both_finish exit $CODE"
test "$(count '^scontrol')$(count '^sbatch')" = 00 || fail "both_finish: requeue or sbatch"

# Both crash: both are reported, nothing is requeued or submitted.
run_pair both_crash crash crash
test "$CODE" = 1 || fail "both_crash exit $CODE"
test -e "$OUT/crashed_T8192_4242" && test -e "$OUT/crashed_T512_4242" || fail "both_crash markers"
test "$(count '^scontrol')$(count '^sbatch')" = 00 || fail "both_crash: requeue or sbatch"

# The single-arm job forwards the arm's trainer arguments with the single-arm resources.
CALLS="$WORK/single.calls" && touch "$CALLS"
PATH="$WORK/bin:$PATH" CALLS="$CALLS" bash "$REPO/scripts/rollout_comparison_single.sbatch" "$REPO" "$WORK/plan.json" \
    "$WORK/single" MODEL train 512 --pass-conv-state --num-rmt-tokens 24 --wandb-project omegalax
grep -q -- "^srun --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=360G --gres=gpu:4 bash $REPO/scripts/rollout_comparison_node.sh $REPO $WORK/plan.json $WORK/single MODEL train 512 --pass-conv-state --num-rmt-tokens 24 --wandb-project omegalax$" "$CALLS" \
    || fail "single.sbatch arguments"
for D in nodes=1 ntasks=1 cpus-per-task=24 mem=360G gres=gpu:4 time=1-00:00:00; do
    grep -qx "#SBATCH --$D" "$REPO/scripts/rollout_comparison_single.sbatch" || fail "single.sbatch --$D"
done
grep -qx "#SBATCH --signal=USR1@900" "$REPO/scripts/rollout_comparison_rmt_conv_pair.sbatch" || fail "pair signal"
echo PASS
