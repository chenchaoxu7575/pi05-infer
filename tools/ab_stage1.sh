#!/bin/bash
# ab_stage1.sh [rounds] [iters]
#
# Paired A/B of the Stage-1 hand-captured denoise CUDA graph. Alternates
#
#   arm "off" : bench/standalone_infer_bench.py            (default, max-autotune)
#   arm "on"  : bench/standalone_infer_bench.py --stage1   (max-autotune-no-cudagraphs
#                                                           + hand-captured denoise graph)
#
# in separate processes, ROUNDS times, recording SM clock / power per timed window.
# Separate processes are required: the two arms need different torch.compile modes.
#
# Rebuild variance on this box is +-0.7 ms and SM clock tracks runtime ~1:1 under the
# 300 W cap, so a single run of each arm cannot resolve the effect -- hence the pairing.
#
# Summarise with:  tools/ab_stage1_summary.py <outdir>
set -u
ROUNDS=${1:-4}
ITERS=${2:-30}
D=${AB_OUT:-/workspace/rlinf_pub/pi05_infer_runs/ab_stage1}
PY=${PY:-/opt/venv/openpi/bin/python}
mkdir -p "$D"
cd /workspace/rlinf_pub/pi05-infer || exit 1
export CUDA_VISIBLE_DEVICES=0

wait_for_gpu() {
  # The GPU is shared: never run two timing jobs concurrently.
  for _ in $(seq 1 600); do
    pgrep -f "run_var|run_stage1|ab_rlinf" >/dev/null || return 0
    sleep 3
  done
}

for r in $(seq 1 "$ROUNDS"); do
  for arm in off on; do
    EXTRA=""
    [ "$arm" = "on" ] && EXTRA="--stage1"
    wait_for_gpu
    $PY -u bench/standalone_infer_bench.py --iters "$ITERS" \
      --clocks-json "$D/r${r}_${arm}.json" $EXTRA \
      >"$D/r${r}_${arm}.log" 2>&1
    echo "AB_DONE r=$r arm=$arm rc=$?"
  done
done
echo AB_ALL_DONE
