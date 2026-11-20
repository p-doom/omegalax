#!/usr/bin/env bash
# Run the equivalence check ON GPU, against the production kernels.
#
#   ./submit_equiv_gpu.sh [tag]
#
# Why this exists: equiv_run.py defaults to JAX_PLATFORMS=cpu, and the kernel
# dispatcher in omegalax/models/qwen3_5/kernels/__init__.py picks the pure-JAX
# "xla" reference backend whenever no GPU is visible. Every equivalence result so
# far therefore proves the framework path, NOT the Pallas/Triton delta-rule kernel
# and not the mosaic_gpu attention that production actually runs.
#
# That gap matters most for the cascade: it changes how the delta rule is CALLED,
# folding the C segments into the time axis, which turns the kernel's sequential
# chunk walk from T/64 into C*T/64 chunks. On the XLA reference that is
# mathematically identical. On a register-resident Pallas kernel it is a different
# accumulation schedule, and only a GPU run can show whether that stays bit-exact.
#
# Runs three configurations and compares them pairwise:
#   orig      original tree, no cascade   <- ground truth
#   opt_off   optimization tree, cascade off
#   opt_on    optimization tree, cascade on
set -euo pipefail

TAG="${1:-equivgpu}"
ORIG='/fast/home/salan.isaqzoi/gh_projects/p(doom)/omegalax'
OPT='/fast/home/salan.isaqzoi/gh_projects/p(doom)/omegalax_optimization'
TOOLS="${OPT}/Salan_DONT_PUSH/mfu_opt"
OUT=/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/logs/mfu_opt
JOBS="${OUT}/jobs"
mkdir -p "${JOBS}" "${OUT}/equiv"

# One optimizer step only. Past runs showed that after two steps the layernorms
# drift by ~1.7e-3 from ordinary fp accumulation differences, which drowns out the
# signal we are testing for; at one step the forward is directly comparable.
XLA_ENV=""
if [ -n "${EQUIV_XLA_FLAGS:-}" ]; then
  XLA_ENV="export XLA_FLAGS=\"\${XLA_FLAGS:-} ${EQUIV_XLA_FLAGS}\";"
fi

STEPS="${EQUIV_STEPS:-1}"
KERNEL="${EQUIV_KERNEL:-pallas}"
ATTN="${EQUIV_ATTN:-mosaic_gpu}"
DTYPE="${EQUIV_DTYPE:-fp32}"

SB="${JOBS}/equiv_${TAG}.sbatch"
cat > "${SB}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=equiv_${TAG}
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --output=${OUT}/${TAG}_%j.log
#SBATCH --error=${OUT}/${TAG}_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export JAX_LOCAL_DEVICE_IDS=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_HOME=/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/huggingface
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="\${CUDA_HOME:-/usr/local/cuda}/extras/CUPTI/lib64:\${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled
${XLA_ENV}

D=${OUT}/equiv
PY='${OPT}'/.venv/bin/python

run () {  # run <name> <tree> <cascade> <conv_btc>
  echo "=== \$1  tree=\$2  MFU_CASCADE=\$3  MFU_CONV_BTC=\${4:-0}  MFU_TRIINV=\${5:-0}  MFU_BWD_PALLAS=\${6:-0}  kernel=${KERNEL}  attn=${ATTN}"
  ( cd "\$2"
    PYTHONPATH="\$2" MFU_CASCADE="\$3" MFU_CONV_BTC="\${4:-0}" MFU_TRIINV="\${5:-0}" MFU_BWD_PALLAS="\${6:-0}" "\${PY}" '${TOOLS}'/equiv_run.py \\
      --steps ${STEPS} --batch 2 --seed 0 \\
      --kernel ${KERNEL} --attn-backend ${ATTN} --dtype ${DTYPE} \\
      --pass-rope 0 --pass-conv 0 \\
      --out "\${D}/${TAG}_\$1.json" )
}

nvidia-smi --query-gpu=name --format=csv,noheader
# orig twice, in two separate processes. This is the control that makes the other
# numbers readable: XLA picks GEMM algorithms by timing them, so two runs of the
# SAME code can differ in accumulation order. Any deviation at or below this floor
# is run-to-run noise, not an effect of the change under test.
run orig      '${ORIG}' 0
run orig2     '${ORIG}' 0
run opt_off   '${OPT}'  0 0
run opt_on    '${OPT}'  1 0
run opt_conv  '${OPT}'  0 1
run opt_both  '${OPT}'  1 1
run opt_tri   '${OPT}'  1 1 1
run opt_bwd   '${OPT}'  1 1 1 1

echo
echo "############ RAUSCHPEGEL: orig GEGEN orig2 (identischer Code, zwei Prozesse)"
'${OPT}'/.venv/bin/python '${TOOLS}'/equiv_check.py --same-ok "\${D}/${TAG}_orig.json" "\${D}/${TAG}_orig2.json" || true

echo
echo "############ opt_off (Kopie, Default) GEGEN orig"
'${OPT}'/.venv/bin/python '${TOOLS}'/equiv_check.py "\${D}/${TAG}_orig.json" "\${D}/${TAG}_opt_off.json" || true
echo
echo "############ opt_on (Kaskade) GEGEN orig"
'${OPT}'/.venv/bin/python '${TOOLS}'/equiv_check.py "\${D}/${TAG}_orig.json" "\${D}/${TAG}_opt_on.json" || true
echo
echo "############ opt_conv (Conv in BTC) GEGEN orig"
'${OPT}'/.venv/bin/python '${TOOLS}'/equiv_check.py "\${D}/${TAG}_orig.json" "\${D}/${TAG}_opt_conv.json" || true
echo
echo "############ opt_both (Kaskade + Conv BTC) GEGEN orig"
'${OPT}'/.venv/bin/python '${TOOLS}'/equiv_check.py "\${D}/${TAG}_orig.json" "\${D}/${TAG}_opt_both.json" || true
echo
echo "############ opt_tri (Kaskade + Conv BTC + Matmul-Inverse) GEGEN orig"
'${OPT}'/.venv/bin/python '${TOOLS}'/equiv_check.py "\${D}/${TAG}_orig.json" "\${D}/${TAG}_opt_tri.json" || true
echo
echo "############ opt_bwd (zusaetzlich Pallas-Backward) GEGEN orig"
'${OPT}'/.venv/bin/python '${TOOLS}'/equiv_check.py "\${D}/${TAG}_orig.json" "\${D}/${TAG}_opt_bwd.json" || true
EOF

JOB=$(sbatch --parsable "${SB}")
echo "submitted equiv-gpu tag=${TAG} job=${JOB}  kernel=${KERNEL} attn=${ATTN} steps=${STEPS}"
echo "log=${OUT}/${TAG}_${JOB}.log"
