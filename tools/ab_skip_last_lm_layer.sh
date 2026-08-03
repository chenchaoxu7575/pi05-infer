#!/bin/bash
# ab_skip_last_lm_layer.sh [rounds] [iters] [extra bench args...]
#
# Paired A/B of the prefix-LM last-layer skip (pi05_infer/patches/prefix_last_layer.py).
#
#   arm "off" : RLINF_SKIP_LAST_LM_LAYER=0  -- full 18th decoder layer
#   arm "on"  : RLINF_SKIP_LAST_LM_LAYER=1  -- layer 17 reduced to input_layernorm
#                                              + k_proj/v_proj + RoPE(k) + cache update
#
# Separate processes per arm, alternating, ROUNDS times, arm order flipped on odd
# rounds so a monotone thermal drift cannot masquerade as an effect. Pairing is what
# removes the SM-clock drift (a 1 % boost-clock difference is worth ~0.5 ms here, i.e.
# more than the expected effect -- lock the clocks before running this).
#
# Both arms share ONE TORCHINDUCTOR_CACHE_DIR: the skip removes ops from the traced
# graph but does not change the shape of any surviving GEMM, so a shared autotune
# result cache pins every untouched decision. Per-arm caches have produced
# sign-flipped results on this box before.
#
# Summarise with:  tools/ab_stage1_summary.py <outdir>
set -u
ROUNDS=${1:-4}
ITERS=${2:-30}
shift 2 2>/dev/null || true
D=${AB_OUT:-/workspace/rlinf_pub/pi05_infer_runs/ab_skip_last_lm_layer}
PY=${PY:-/opt/venv/openpi/bin/python}
TI=${TI_DIR:-/tmp/ti_ab_skip_last_lm_layer}
# Note the `-` (not `:-`): GPU_LOCK= must mean "no per-run lock", otherwise wrapping the
# whole campaign in one flock deadlocks against the per-run flock on the same file.
LOCK=${GPU_LOCK-/tmp/pi05_gpu_timing.lock}
mkdir -p "$D" "$TI"
cd /workspace/rlinf_pub/pi05-infer || exit 1
export CUDA_VISIBLE_DEVICES=0

# Optional clock lock. At this effect size (~1 ms out of ~44 ms) boost jitter alone is
# worth more than the effect: a 29 MHz (1.2 %) difference between arms measured 0.52 ms.
# LOCK_MHZ=2100 tools/ab_skip_last_lm_layer.sh ...
LOCK_MHZ=${LOCK_MHZ:-}
if [ -n "$LOCK_MHZ" ]; then
  nvidia-smi -i 0 -lgc "$LOCK_MHZ,$LOCK_MHZ" || echo "WARN: clock lock failed"
  trap 'nvidia-smi -i 0 -rgc' EXIT
fi

for r in $(seq 1 "$ROUNDS"); do
  # Alternate the arm order between rounds.
  if [ $((r % 2)) -eq 1 ]; then ARMS="off on"; else ARMS="on off"; fi
  for arm in $ARMS; do
    [ "$arm" = "on" ] && SW=1 || SW=0
    # The GPU is shared with another agent: never run two timing jobs concurrently.
    # GPU_LOCK= (empty) skips the per-run lock -- use that when the whole campaign is
    # already wrapped in one `flock`, which is stronger: it also stops a foreign job
    # from slipping in *between* the two arms of a round and breaking the pairing.
    ${LOCK:+flock "$LOCK"} env \
      RLINF_SKIP_LAST_LM_LAYER=$SW \
      TORCHINDUCTOR_CACHE_DIR="$TI" \
      CUDA_VISIBLE_DEVICES=0 \
      "$PY" -u bench/standalone_infer_bench.py --iters "$ITERS" \
        --clocks-json "$D/r${r}_${arm}.json" "$@" \
        >"$D/r${r}_${arm}.log" 2>&1
    echo "AB_DONE r=$r arm=$arm rc=$?"
  done
done
echo AB_ALL_DONE
