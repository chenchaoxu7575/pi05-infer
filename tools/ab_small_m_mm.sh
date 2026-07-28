#!/bin/bash
# ab_small_m_mm.sh [rounds] [iters] [extra bench args...]
#
# Paired A/B of the small-M mm tile candidates (pi05_infer/inductor_mm_tiles.py).
#
#   arm "off" : RLINF_SMALL_M_MM=0  -- stock inductor mm autotune space
#   arm "on"  : RLINF_SMALL_M_MM=1  -- + BLOCK_M in {16,32} / BLOCK_K=128 candidates
#                                      for the o_proj and down_proj shapes
#
# Separate processes per arm, alternating, ROUNDS times. Pairing is what removes
# the SM-clock drift: the two arms of a round run back to back at nearly the same
# clock (the card is 300 W capped and runtime tracks clock ~1:1).
#
# Each arm gets its own TORCHINDUCTOR_CACHE_DIR. The monkeypatch is invisible to
# inductor's FXGraphCache key (inductor_mm_tiles.py bumps cache_key_tag for the
# same reason), and a shared cache would let one arm replay the other's kernels.
#
# Summarise with:  tools/ab_stage1_summary.py <outdir>
set -u
ROUNDS=${1:-4}
ITERS=${2:-30}
shift 2 2>/dev/null || true
D=${AB_OUT:-/workspace/rlinf_pub/pi05_infer_runs/ab_small_m_mm}
PY=${PY:-/opt/venv/openpi/bin/python}
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
  for arm in off on; do
    [ "$arm" = "on" ] && SW=1 || SW=0
    wait_for_gpu
    RLINF_SMALL_M_MM=$SW \
    TORCHINDUCTOR_CACHE_DIR=${TI_BASE:-/tmp/ti_ab_small_m_mm}/$arm \
      $PY -u bench/standalone_infer_bench.py --iters "$ITERS" \
      --clocks-json "$D/r${r}_${arm}.json" "$@" \
      >"$D/r${r}_${arm}.log" 2>&1
    echo "AB_DONE r=$r arm=$arm rc=$?"
  done
done
echo AB_ALL_DONE
