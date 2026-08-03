#!/bin/bash
# Backfill driver: the three evidence gaps left open by RESULTS_dump_actions_determinism.md section 7.
#
#   1  SigLIP three-view batching        -> tools/bitexact_siglip_batch.py      (in-process A/B)
#   2  repo extraction equivalence       -> tools/bitexact_extraction.py        (in-process A/B)
#   3  the four eager-only optimizations -> tools/bitexact_compiled_toggles.py  (4-run gate)
#
# Every stage runs its OFF-vs-OFF control first and reports INCONCLUSIVE rather than PASS
# when the control fails, per RESULTS_dump_actions_determinism.md section 5.
#
#   docker exec pi05bench bash -lc "/path/to/pi05-infer/tools/run_bitexact_backfill.sh <stage>"
#
# GPU 1 by default (another agent owns GPU 0 and is doing precise timing there): no clock
# locking, no /tmp/pi05_gpu_timing.lock -- none of these checks measures time.
set -u
cd "$(dirname "$0")/.." || exit 1

PY=${PY:-/opt/venv/openpi/bin/python}
OUT=${OUT:-/tmp/pi05_bitexact_backfill}
TI=${TI:-/tmp/ti_bitexact_backfill}
PREFIX=${PREFIX:-$OUT/frozen_prefix.pt}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export TORCHINDUCTOR_CACHE_DIR=$TI
mkdir -p "$OUT" "$TI"

STAGE=${1:-all}
shift 2>/dev/null || true

# One toggle, four runs (off_a off_b on_a on_b), all sharing $TI so that every decision
# neither arm touches keeps the same autotune winner.
gate() {
  local name=$1 toggle=$2; shift 2
  local d="$OUT/$name"
  mkdir -p "$d"
  for arm in off_a off_b on_a on_b; do
    case $arm in off_*) DIS=$toggle ;; on_*) DIS=none ;; esac
    echo "=== $name/$arm (disable=$DIS) ==="
    $PY -u tools/bitexact_compiled_toggles.py --disable "$DIS" \
        --freeze-prefix "$PREFIX" --out "$d/$arm.json" --save-actions "$d/$arm.pt" "$@" \
        >"$d/$arm.log" 2>&1
    echo "  rc=$? $(grep -h '^arm signature' "$d/$arm.log" | tail -1)"
  done
  echo "--- $name verdict ---"
  $PY -u tools/bitexact_compiled_toggles.py --verdict \
      "$d/off_a.json" "$d/off_b.json" "$d/on_a.json" "$d/on_b.json" \
      2>&1 | tee "$d/verdict.txt"
}

case $STAGE in
  siglip)
    $PY -u tools/bitexact_siglip_batch.py --no-compile --out "$OUT/siglip_eager.json" \
        2>&1 | tee "$OUT/siglip_eager.log" | grep -vE 'AUTOTUNE|triton_|SingleProcess|select_algorithm'
    $PY -u tools/bitexact_siglip_batch.py --out "$OUT/siglip_maxautotune.json" \
        2>&1 | tee "$OUT/siglip_maxautotune.log" | grep -vE 'AUTOTUNE|triton_|SingleProcess|select_algorithm'
    ;;
  extraction)
    $PY -u tools/bitexact_extraction.py --out "$OUT/extraction.json" "$@" \
        2>&1 | tee "$OUT/extraction.log" | grep -vE 'AUTOTUNE|triton_|SingleProcess|select_algorithm'
    ;;
  prefix)
    # Write the frozen prefix ONCE, from the fully-optimized arm. Every later run replays
    # it, so the SigLIP tower -- the one stage that moves between processes -- is out of
    # the comparison and the off-vs-off control has a chance of being clean.
    rm -f "$PREFIX"
    $PY -u tools/bitexact_compiled_toggles.py --disable none --freeze-prefix "$PREFIX" \
        --out "$OUT/prefix_seed.json" "$@" >"$OUT/prefix_seed.log" 2>&1
    echo "rc=$?  $(ls -la "$PREFIX")"
    ;;
  attmask)
    # Row 5 is upstream of the prefix, so it cannot be frozen; settle it at the tensor.
    $PY -u tools/bitexact_compiled_toggles.py --attmask-tensor-check \
        --out "$OUT/attmask_tensor.json" "$@" \
        2>&1 | tee "$OUT/attmask_tensor.log" | grep -vE 'AUTOTUNE|triton_|SingleProcess|select_algorithm'
    ;;
  adarms)   gate adarms   adarms   "$@" ;;
  adarms_eager) gate adarms_eager adarms --no-compile "$@" ;;
  qkv)      gate qkv      qkv      "$@" ;;
  kvstatic) gate kvstatic kvstatic "$@" ;;
  *)
    echo "usage: $0 {siglip|extraction|prefix|adarms|adarms_eager|qkv|kvstatic|attmask} [extra bench args]"
    exit 64
    ;;
esac
