#!/bin/bash
# prof.sh <tag> <arm> [iters]
#
#   arm = pi05infer -> bench/standalone_infer_bench.py     (this package)
#   arm = rlinf     -> tools/ab_rlinf_reference.py         (the RLinf reference)
#
# nsys 2026 profile + sqlite export, both with the SAME binary: nsys 2025.x cannot
# read its own output on this GPU. Adapted from
# claude_mem/pi05_rollout_forward/kernel_fusion/scripts/prof.sh.
set -x
TAG=$1
ARM=${2:-pi05infer}
ITERS=${3:-12}
D=${PROF_OUT:-/workspace/rlinf_pub/pi05_infer_runs}
NSYS=/opt/nsys2026/target-linux-x64/nsys
mkdir -p "$D"
cd /workspace/rlinf_pub/pi05-infer || exit 1
export CUDA_VISIBLE_DEVICES=0

if [ "$ARM" = "rlinf" ]; then
  SCRIPT="tools/ab_rlinf_reference.py"
else
  SCRIPT="bench/standalone_infer_bench.py"
fi

# The GPU is shared: never run two timing jobs concurrently.
for _ in $(seq 1 300); do
  pgrep -f "run_var|run_stage1|standalone_infer|ab_rlinf" >/dev/null || break
  sleep 3
done

$NSYS profile -t cuda,nvtx --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
  --gpu-metrics-devices=cuda-visible \
  --force-overwrite=true -o "$D/$TAG" \
  /opt/venv/openpi/bin/python -u "$SCRIPT" --cuda-profiler --iters "$ITERS"
echo "PROF_RC=$?"

$NSYS export --type sqlite --force-overwrite=true \
  -o "$D/$TAG.sqlite" "$D/$TAG.nsys-rep" 2>&1 | tail -2
echo "PROF_DONE_$TAG"
