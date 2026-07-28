# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-denoise-step GPU idle, measured on the kernel timeline.

Method (kept identical across builds so the numbers stay comparable):

  * **Anchor** on a kernel that fires exactly once per expert layer per denoise
    step. ``_qkv_rope_kernel`` (the fused QKV+RoPE Triton kernel) fires 18x/step,
    one per gemma-expert layer. Anchoring by *name* rather than by stream is what
    makes the measurement survive Stage 1: the hand-captured graph moves the
    denoise kernels onto a different stream, but the kernel names do not change.
  * **Step boundary** = consecutive step starts, i.e. every 18th anchor kernel.
    A step whose successor is more than 5 ms away is the last step of a predict;
    its "wall" would swallow the next prefix phase, so it is dropped.
  * **Idle** = step wall clock minus the union of *all* kernel intervals (any
    stream) clipped to that step's window. Union, not sum, so concurrent kernels
    are not double counted.

Usage:
    step_idle.py <sqlite> [<sqlite> ...] [--anchor NAME] [--per-step N]

Discriminator note: this tool also reports the number of ``denoise/expert_forward``
NVTX ranges per predict, which is the reliable Stage-1 on/off signal. 10 per predict
means the step body is dispatched eagerly once per step; 0 means the body is inside
the hand-captured graph and only ran during capture. Do *not* use a non-null
``graphNodeId`` on the denoise kernels as the signal -- inductor's ``max-autotune``
emits its own cudagraphs, so those kernels are graph nodes even with Stage 1 off.
"""

import bisect
import sqlite3
import statistics
import sys

ANCHOR = "_qkv_rope_kernel"
PER_STEP = 18
PREDICT_GAP_NS = 5_000_000


def _nvtx_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Count NVTX push/pop ranges by text, for the Stage-1 discriminator."""
    try:
        rows = conn.execute(
            "select s.value, count(*) from NVTX_EVENTS n "
            "join StringIds s on s.id = n.textId "
            "where s.value like 'denoise/%' or s.value like 'bench/iter%' "
            "group by s.value"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {v: n for v, n in rows}


def analyse(
    db: str, anchor: str = ANCHOR, per_step: int = PER_STEP
) -> tuple[float, float]:
    """Print and return (mean step wall us, mean step idle us) for one profile."""
    conn = sqlite3.connect(db)
    ids = [
        r[0] for r in conn.execute("select id from StringIds where value=?", (anchor,))
    ]
    assert ids, f"{db}: anchor kernel {anchor!r} not found in StringIds"
    placeholders = ",".join("?" * len(ids))
    anchors = [
        r[0]
        for r in conn.execute(
            "select start from CUPTI_ACTIVITY_KIND_KERNEL "
            f"where shortName in ({placeholders}) order by start",
            ids,
        )
    ]
    assert anchors and len(anchors) % per_step == 0, (
        f"{db}: {len(anchors)} {anchor!r} kernels is not a multiple of {per_step}/step; "
        "the anchor no longer fires once per layer per step."
    )
    step_starts = anchors[::per_step]

    kernels = list(
        conn.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start")
    )
    kstarts = [k[0] for k in kernels]

    walls: list[float] = []
    idles: list[float] = []
    for lo_start, hi_start in zip(step_starts, step_starts[1:]):
        wall = hi_start - lo_start
        if wall > PREDICT_GAP_NS:  # last step of a predict: next start is post-prefix
            continue
        # Union of kernel intervals clipped to [lo_start, hi_start).
        j = max(0, bisect.bisect_left(kstarts, lo_start) - 64)
        busy = 0
        cur_s = cur_e = None
        while j < len(kernels) and kernels[j][0] < hi_start:
            s = max(kernels[j][0], lo_start)
            e = min(kernels[j][1], hi_start)
            j += 1
            if e <= s:
                continue
            if cur_e is None:
                cur_s, cur_e = s, e
            elif s > cur_e:
                busy += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            busy += cur_e - cur_s
        walls.append(wall / 1e3)
        idles.append((wall - busy) / 1e3)

    mean_wall = statistics.mean(walls)
    mean_idle = statistics.mean(idles)
    nvtx = _nvtx_counts(conn)
    npredict = nvtx.get("denoise/loop") or (len(anchors) // per_step // 10) or 1
    print(f"{db}")
    print(f"  anchor {anchor!r}: {len(anchors)} kernels -> {len(step_starts)} steps")
    print(f"  steps measured (predict-internal): {len(walls)}")
    print(f"  step wall  : {mean_wall:8.1f} us  (p50 {statistics.median(walls):.1f})")
    print(f"  step idle  : {mean_idle:8.1f} us  (p50 {statistics.median(idles):.1f})")
    print(f"  idle       : {100 * mean_idle / mean_wall:8.2f} %")
    for name in ("denoise/loop", "denoise/step", "denoise/expert_forward"):
        if name in nvtx:
            print(
                f"  nvtx {name:<24} {nvtx[name]:5d} total  "
                f"{nvtx[name] / npredict:6.2f}/predict"
            )
    print(
        "  stage1 verdict: "
        + (
            "OFF (eager per-step dispatch)"
            if nvtx.get("denoise/expert_forward", 0) >= npredict * 10
            else "ON (step body inside the captured graph)"
        )
    )
    return mean_wall, mean_idle


def main() -> None:
    argv = sys.argv[1:]
    anchor, per_step, dbs = ANCHOR, PER_STEP, []
    i = 0
    while i < len(argv):
        if argv[i] == "--anchor":
            anchor = argv[i + 1]
            i += 2
        elif argv[i] == "--per-step":
            per_step = int(argv[i + 1])
            i += 2
        else:
            dbs.append(argv[i])
            i += 1
    assert dbs, "usage: step_idle.py <sqlite> [<sqlite> ...]"
    for db in dbs:
        analyse(db, anchor, per_step)


if __name__ == "__main__":
    main()
