#!/bin/bash
# ab_prefix_qkv.sh [rounds] [iters] [extra bench args...]
#
# Paired A/B of the fused prefix-LM QKV GEMM (pi05_infer/patches/prefix_qkv_fused.py).
#
#   arm "off" : RLINF_FUSE_PREFIX_QKV=0  -- three GEMMs per layer, q(2048)/k(256)/v(256)
#   arm "on"  : RLINF_FUSE_PREFIX_QKV=1  -- one [2560, 2048] GEMM per layer, 17 of 18
#                                           layers (the KV-only last one is excluded)
#
# Separate processes per arm, alternating, ROUNDS times, arm order flipped on odd
# rounds so a monotone thermal drift cannot masquerade as an effect. Pairing is what
# removes the SM-clock drift: on this box a 30 MHz boost difference between arms was
# worth 0.7 ms, i.e. more than the expected effect. ALWAYS pass LOCK_MHZ.
#
# Both arms share ONE TORCHINDUCTOR_CACHE_DIR. The fusion changes the traced graph, so
# the two arms get different FXGraphCache entries anyway; what the shared dir buys is
# that every shape NEITHER arm touched keeps one autotune winner. Per-arm caches have
# produced sign-flipped results on this box before.
#
# Summarise with:  tools/ab_stage1_summary.py <outdir>
set -u
ROUNDS=${1:-4}
ITERS=${2:-30}
shift 2 2>/dev/null || true
D=${AB_OUT:-/workspace/rlinf_pub/pi05_infer_runs/ab_prefix_qkv}
PY=${PY:-/opt/venv/openpi/bin/python}
TI=${TI_DIR:-/tmp/ti_ab_prefix_qkv}
# Note the `-` (not `:-`): GPU_LOCK= must mean "no per-run lock", otherwise wrapping the
# whole campaign in one flock deadlocks against the per-run flock on the same file.
LOCK=${GPU_LOCK-/tmp/pi05_gpu_timing.lock}
mkdir -p "$D" "$TI"
cd /workspace/rlinf_pub/pi05-infer || exit 1
export CUDA_VISIBLE_DEVICES=0

# Optional clock lock. At this effect size (~1 ms out of ~44 ms) boost jitter alone is
# worth more than the effect: a 29 MHz (1.2 %) difference between arms measured 0.52 ms.
# LOCK_MHZ=2100 tools/ab_prefix_qkv.sh ...
LOCK_MHZ=${LOCK_MHZ:-}
if [ -n "$LOCK_MHZ" ]; then
  # Persistence mode FIRST. Without it the driver unloads when the last CUDA process
  # exits, and -lgc is lost with it: measured here, the arm that ran second came back
  # at 2400 MHz against the locked arm's 2092 MHz and "won" by 6.1 ms, all of it clock.
  nvidia-smi -i 0 -pm 1 >/dev/null 2>&1 || echo "WARN: persistence mode not enabled"
  nvidia-smi -i 0 -lgc "$LOCK_MHZ,$LOCK_MHZ" >/dev/null || echo "WARN: clock lock failed"
  trap 'nvidia-smi -i 0 -rgc >/dev/null' EXIT
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
    # Re-assert the lock per run: belt and braces on top of persistence mode, and it
    # also recovers if a foreign job called -rgc between the arms.
    [ -n "$LOCK_MHZ" ] && nvidia-smi -i 0 -lgc "$LOCK_MHZ,$LOCK_MHZ" >/dev/null 2>&1
    ${LOCK:+flock "$LOCK"} env \
      RLINF_FUSE_PREFIX_QKV=$SW \
      TORCHINDUCTOR_CACHE_DIR="$TI" \
      CUDA_VISIBLE_DEVICES=0 \
      "$PY" -u bench/standalone_infer_bench.py --iters "$ITERS" \
        --clocks-json "$D/r${r}_${arm}.json" "$@" \
        >"$D/r${r}_${arm}.log" 2>&1
    echo "AB_DONE r=$r arm=$arm rc=$?"
  done
done
echo AB_ALL_DONE
