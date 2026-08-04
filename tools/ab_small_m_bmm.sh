#!/bin/bash
# ab_small_m_bmm.sh [rounds] [iters] [extra bench args...]
#
# Paired A/B of the extra tile candidate for the P.V attention BMM
# (pi05_infer/patches/inductor_mm_tiles.py::install_small_m_bmm_configs).
#
#   arm "off" : $AB_VAR=0        arm "on" : $AB_VAR=1
#
# AB_VAR defaults to RLINF_SMALL_M_BMM (stock autotune space vs + BM32/BN64/BK128
# for bmm(8x50x1018, 8x1018x256)). Set AB_VAR=RLINF_SMALL_M_BMM_PIN to A/B the
# Q.K^T tile pin instead -- that one is worth measuring not for its mean but for
# its variance: the "off" arm re-draws a tile per cold compile and the draw spans
# 20.2%, so the two arms differ in spread as much as in centre.
#
# Requires $PI05_MODEL_PATH (or --model-path in the extra args).
#
# Three things this driver does that a naive loop does not, each of which has
# produced a wrong answer on this workload before:
#
#  * ALTERNATING ARM ORDER. Even with the clock pinned, the arm that runs first
#    in a round is systematically faster here (measured; internal record).
#    Comparing same-position pairs is the only way to remove it.
#  * ONE SHARED TORCHINDUCTOR_CACHE_DIR. With a cache dir per arm, autotune
#    re-decides the shapes neither arm touches and the A/B measures those
#    instead -- that inverted the sign of the previous experiment (5.2). The
#    patch bumps cache_key_tag, so the two arms still get separate FXGraphCache
#    entries and cannot replay each other's kernels.
#  * LOCKED SM CLOCK. A 1% boost difference is worth more than the effect
#    (5.3). Restored on exit.
#
# Summarise with:  tools/ab_stage1_summary.py <outdir>
set -u
ROUNDS=${1:-6}
ITERS=${2:-30}
shift 2 2>/dev/null || true
EXTRA=("$@")
REPO=$(cd "$(dirname "$0")/.." && pwd)
D=${AB_OUT:-/workspace/rlinf_pub/pi05_infer_runs/ab_small_m_bmm}
PY=${PY:-/opt/venv/openpi/bin/python}
TI=${TORCHINDUCTOR_CACHE_DIR:-/tmp/ti_ab_small_m_bmm}
LOCK=${GPU_LOCK:-/tmp/pi05_gpu_timing.lock}
CLK=${AB_LOCK_CLOCK:-2100}
AB_VAR=${AB_VAR:-RLINF_SMALL_M_BMM}
mkdir -p "$D" "$TI"
cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

nvidia-smi -i 0 -lgc "$CLK,$CLK" >/dev/null 2>&1 || echo "WARN: SM clock not locked"
trap 'nvidia-smi -i 0 -rgc >/dev/null 2>&1' EXIT

run() { # arm round
  local sw=0
  [ "$1" = "on" ] && sw=1
  env "$AB_VAR=$sw" TORCHINDUCTOR_CACHE_DIR="$TI" \
    $PY -u bench/standalone_infer_bench.py --iters "$ITERS" \
    --clocks-json "$D/r${2}_${1}.json" ${EXTRA[@]+"${EXTRA[@]}"} \
    >"$D/r${2}_${1}.log" 2>&1
  echo "AB_DONE r=$2 arm=$1 rc=$?"
}

# The GPU is shared: hold the timing lock for a whole round so the pair is
# measured back to back under the same conditions.
exec 9>"$LOCK"
for R in $(seq 1 "$ROUNDS"); do
  if [ $((R % 2)) -eq 1 ]; then ORDER="off on"; else ORDER="on off"; fi
  flock 9
  for a in $ORDER; do run "$a" "$R"; done
  flock -u 9
done
echo AB_ALL_DONE
