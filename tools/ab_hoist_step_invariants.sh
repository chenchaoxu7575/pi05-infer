#!/bin/bash
# ab_hoist_step_invariants.sh [rounds] [iters] [extra bench args...]
#
# Paired A/B of the step-invariant hoist (pi05_infer/engine.py::_build_step_invariants).
#
#   arm "off" : RLINF_HOIST_STEP_INVARIANTS=0 -- attention mask, position ids and the rotary
#                                                cos/sin table rebuilt on every Euler step
#   arm "on"  : RLINF_HOIST_STEP_INVARIANTS=1 -- built once per predict into persistent
#                                                buffers, read by the captured graph
#
# Separate processes per arm, alternating, ROUNDS times. Pairing is what removes the SM-clock
# drift; lock the clocks as well (tools/README: nvidia-smi -lgc) -- at this effect size a 1 %
# boost-clock difference is worth more than the effect.
#
# ONE SHARED TORCHINDUCTOR_CACHE_DIR for both arms. The two arms genuinely trace different
# graphs (ON feeds cos/sin in as inputs, OFF computes them inside), so they cannot collide on
# a cache key; sharing the directory is what keeps every shape NEITHER arm changed on the same
# autotune winner. Per-arm caches have produced sign-flipped results on this box before.
#
# Summarise with:  tools/ab_stage1_summary.py <outdir>
set -u
ROUNDS=${1:-4}
ITERS=${2:-30}
shift 2 2>/dev/null || true
D=${AB_OUT:-/workspace/rlinf_pub/pi05_infer_runs/ab_hoist}
PY=${PY:-/opt/venv/openpi/bin/python}
TI=${TORCHINDUCTOR_CACHE_DIR:-/tmp/ti_ab_hoist}
mkdir -p "$D"
cd /workspace/rlinf_pub/pi05-infer || exit 1
export CUDA_VISIBLE_DEVICES=0

wait_for_gpu() {
  # The GPU is shared: never run two timing jobs concurrently.
  for _ in $(seq 1 600); do
    pgrep -f "run_var|run_stage1|ab_rlinf|standalone_infer_bench" >/dev/null || return 0
    sleep 3
  done
}

for r in $(seq 1 "$ROUNDS"); do
  # Alternate which arm goes first so a monotone thermal drift cannot masquerade as an effect.
  if [ $((r % 2)) -eq 1 ]; then ARMS="off on"; else ARMS="on off"; fi
  for arm in $ARMS; do
    [ "$arm" = "on" ] && SW=1 || SW=0
    wait_for_gpu
    RLINF_HOIST_STEP_INVARIANTS=$SW TORCHINDUCTOR_CACHE_DIR="$TI" \
      $PY -u bench/standalone_infer_bench.py --stage1 --iters "$ITERS" \
      --clocks-json "$D/r${r}_${arm}.json" "$@" \
      >"$D/r${r}_${arm}.log" 2>&1
    echo "AB_DONE r=$r arm=$arm rc=$?"
  done
done
echo AB_ALL_DONE
