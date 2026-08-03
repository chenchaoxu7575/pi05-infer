#!/bin/bash
# End-to-end bit-exactness gate WITH the mandatory empty control.
#
#   GATE_OFF="RLINF_SMALL_M_MM=0" GATE_ON="RLINF_SMALL_M_MM=1" \
#     tools/bitexact_gate.sh /tmp/gate_small_m --stage1 --iters 1 --warmup 4
#
# Why the control is mandatory: `--dump-actions` is a *seeded* end-to-end call, but the
# process it runs in is not fully determined by the code. Under `max-autotune` inductor
# picks each generated kernel's launch config by benchmarking at first launch, and
# coordinate-descent tuning re-benchmarks even on a warm TORCHINDUCTOR_CACHE_DIR. When a
# reduction's R0_BLOCK/num_warps moves, the accumulation split moves with it, and two
# processes running byte-identical code disagree in the last bits -- measured at up to
# 3.6e-3 on the [1,50,6] actions (1.5% of their range). See
# the internal dump-actions determinism study.
#
# So this driver runs FOUR processes: each arm twice. The cross-arm comparison is only
# reported once both same-arm controls are clean; otherwise the verdict is INCONCLUSIVE,
# never PASS. All four share one TORCHINDUCTOR_CACHE_DIR so that the shapes neither arm
# touches keep the same autotune winner (see RESULTS_small_m_mm_tiles.md section 5.2).
set -u
D=${1:?usage: bitexact_gate.sh OUTDIR [bench args...]}
shift
PY=${PY:-/opt/venv/openpi/bin/python}
GATE_OFF=${GATE_OFF:-}
GATE_ON=${GATE_ON:-}
TI=${TORCHINDUCTOR_CACHE_DIR:-/tmp/ti_bitexact_gate}
mkdir -p "$D"
cd "$(dirname "$0")/.." || exit 1

# Preflight: an autotune entry saved with "found_by_coordesc": false is NOT a pin.
# torch/_inductor/runtime/autotune_cache.py::_load_cached_autotuning only restores the
# found_by_coordesc attribute for entries that carry it, and CachingAutotuner.run
# re-runs coordinate-descent tuning for any launcher that lacks it -- so those kernels
# get re-benchmarked, and re-chosen, in every process.
if [ -d "$TI" ]; then
  BAD=$(grep -rl '"found_by_coordesc": false' "$TI" --include='*.best_config' 2>/dev/null | wc -l)
  echo "preflight: $BAD un-pinned (found_by_coordesc=false) autotune entries in $TI"
  [ "$BAD" -gt 0 ] && echo "  -> those kernels will be re-tuned per process; expect the control to fail"
fi

for run in off_a off_b on_a on_b; do
  case $run in off_*) EV=$GATE_OFF ;; on_*) EV=$GATE_ON ;; esac
  # shellcheck disable=SC2086
  env $EV TORCHINDUCTOR_CACHE_DIR="$TI" \
    $PY -u bench/standalone_infer_bench.py --dump-actions "$D/$run.pt" \
    "$@" >"$D/$run.log" 2>&1
  echo "GATE_RUN $run rc=$?"
done

$PY - "$D" <<'EOF'
import itertools, json, os, sys
import torch

d = sys.argv[1]
names = ["off_a", "off_b", "on_a", "on_b"]
t = {n: torch.load(os.path.join(d, n + ".pt")).double() for n in names}
meta = {}
for n in names:
    p = os.path.join(d, n + ".pt.meta.json")
    if os.path.exists(p):
        with open(p) as fh:
            meta[n] = json.load(fh)

print("\npairwise max|delta|")
print("        " + " ".join(f"{n:>10s}" for n in names))
for a in names:
    print(f"{a:>7s} " + " ".join(f"{(t[a]-t[b]).abs().max().item():10.2e}" for b in names))

ctrl_off = torch.equal(t["off_a"], t["off_b"])
ctrl_on = torch.equal(t["on_a"], t["on_b"])
cross = torch.equal(t["off_a"], t["on_a"]) and torch.equal(t["off_b"], t["on_b"])
print(f"\ncontrol off_a vs off_b : {ctrl_off}")
print(f"control on_a  vs on_b  : {ctrl_on}")

if meta:
    w = {n: meta[n].get("autotune_winners", {}) for n in meta}
    keys = sorted(set().union(*[set(v) for v in w.values()]))
    moved = [k for k in keys if len({json.dumps(v.get(k), sort_keys=True) for v in w.values()}) > 1]
    print(f"autotune winners differing across the four runs: {len(moved)}/{len(keys)}")
    for k in moved[:10]:
        print(f"  {k}: " + " | ".join(f"{n}={json.dumps(w[n].get(k))}" for n in names))

if not (ctrl_off and ctrl_on):
    print("\nVERDICT: INCONCLUSIVE -- the same-arm control is not bit-exact, so the "
          "cross-arm comparison carries no information. Pin the autotune winners "
          "(see RESULTS_dump_actions_determinism.md) or gate at the kernel level "
          "with tools/bitexact_denoise_gemms.py / tools/bitgate.py.")
    sys.exit(2)
print(f"\nVERDICT: {'PASS' if cross else 'FAIL -- the arms really do differ'}")
sys.exit(0 if cross else 1)
EOF
echo "GATE_DONE rc=$?"
