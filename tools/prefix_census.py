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
r"""NVTX-attributed, per-stream kernel census for an nsys-2026 sqlite export.

``stream_summary.py`` splits by stream, which cannot say whether a kernel belongs
to the SigLIP tower or the Gemma-2B prefill -- both run on the same one. This
attributes each kernel to the innermost enclosing NVTX range via its launch site
(``correlationId`` -> the ``cudaLaunchKernel`` -> the range open on that thread),
which is exact; comparing GPU timestamps against CPU ranges is not, because
kernels routinely finish after their range has closed.

WARNING: kernels launched from a captured graph replay carry the correlation of the
replay call, so everything inside the Stage-1 denoise graph is attributed to
whatever range wraps ``graph.replay()``. Correct for the prefix, which is never
captured; the denoise rows are graph-granular.

Usage::

    prefix_census.py <sqlite> <n_predicts> [--phase prefix/vlm_forward] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from collections import defaultdict


def load_nvtx(conn: sqlite3.Connection) -> dict[int, list[tuple[int, int, str]]]:
    """Return {globalTid: [(start, end, text), ...]} for every push/pop range."""
    cur = conn.execute(
        """select e.start, e.end, coalesce(e.text, s.value), e.globalTid
             from NVTX_EVENTS e left join StringIds s on s.id = e.textId
            where e.end is not null"""
    )
    out: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for start, end, text, tid in cur:
        if text is None:
            continue
        out[tid].append((start, end, text))
    for tid in out:
        out[tid].sort()
    return out


def innermost(ranges: list[tuple[int, int, str]], t: int) -> str:
    """Innermost NVTX range containing timestamp ``t`` (linear scan, sorted input)."""
    best = None
    best_len = None
    for s, e, txt in ranges:
        if s > t:
            break
        if e >= t:
            ln = e - s
            if best_len is None or ln < best_len:
                best, best_len = txt, ln
    return best or "(none)"


def kernel_rows(conn: sqlite3.Connection):
    return conn.execute(
        """select k.correlationId, k.streamId, k.start, k.end,
                  st.value,
                  k.gridX * k.gridY * k.gridZ,
                  k.blockX * k.blockY * k.blockZ,
                  k.registersPerThread,
                  k.staticSharedMemory + k.dynamicSharedMemory
             from CUPTI_ACTIVITY_KIND_KERNEL k
             join StringIds st on st.id = k.demangledName"""
    )


def runtime_map(conn: sqlite3.Connection) -> dict[int, tuple[int, int]]:
    """{correlationId: (cpu_start, globalTid)} for every CUDA runtime call."""
    return {
        cid: (start, tid)
        for cid, start, tid in conn.execute(
            "select correlationId, start, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    }


def short(name: str) -> str:
    n = name
    for pat in ("void ", "std::enable_if<!T7, void>::type "):
        n = n.replace(pat, "")
    # cuBLAS dispatches everything through the same `cutlass::Kernel2<...>`
    # wrapper, so collapsing at the first '<' would merge every GEMM shape into
    # one row and hide the fact that they are different (Ampere-tuned) tiles.
    m = re.match(r"cutlass::Kernel2<([^,>]+)", n)
    if m:
        return m.group(1).strip()
    n = n.split("<")[0].split("(")[0]
    return n.strip()


def collapse(name: str) -> str:
    """Collapse inductor's per-call-site numeric suffix so 18 layers roll up to 1."""
    return re.sub(r"_\d+$", "", name) if name.startswith("triton_") else name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sqlite")
    p.add_argument("n_predicts", type=float)
    p.add_argument(
        "--phase", default="", help="restrict the per-kernel table to this NVTX range"
    )
    p.add_argument(
        "--prefix-only", action="store_true", help="phase table over all prefix/* ranges"
    )
    p.add_argument("--csv", default="")
    p.add_argument("--collapse", action="store_true", help="merge inductor _NN suffixes")
    args = p.parse_args()

    conn = sqlite3.connect(args.sqlite)
    nvtx = load_nvtx(conn)
    rt = runtime_map(conn)

    # phase totals: (phase, stream) -> [n, us]
    phase_tot: dict[tuple[str, int], list] = defaultdict(lambda: [0, 0.0])
    # per-kernel rows for the selected phase
    kern: dict[tuple, list] = {}
    grand_n = 0
    grand_us = 0.0

    for cid, stream, ks, ke, name, ctas, block, regs, smem in kernel_rows(conn):
        us = (ke - ks) / 1e3
        grand_n += 1
        grand_us += us
        launch = rt.get(cid)
        phase = innermost(nvtx.get(launch[1], []), launch[0]) if launch else "(nolaunch)"
        phase_tot[(phase, stream)][0] += 1
        phase_tot[(phase, stream)][1] += us

        sel = False
        if args.phase:
            sel = phase == args.phase
        elif args.prefix_only:
            sel = phase.startswith("prefix/")
        if sel:
            nm = short(name)
            if args.collapse:
                nm = collapse(nm)
            key = (nm, ctas, block, regs, smem)
            r = kern.setdefault(key, [0, 0.0])
            r[0] += 1
            r[1] += us

    np_ = args.n_predicts
    print(f"=== NVTX phase x stream totals ({np_:.0f} predicts) ===")
    print(f"{'phase':<34s} {'stream':>7s} {'k/pred':>8s} {'us/pred':>10s} {'%GPU':>6s}")
    for (phase, stream), (n, us) in sorted(phase_tot.items(), key=lambda kv: -kv[1][1]):
        print(
            f"{phase[:34]:<34s} {stream:7d} {n / np_:8.2f} {us / np_:10.1f} "
            f"{100 * us / grand_us:6.2f}"
        )
    print(f"{'TOTAL':<34s} {'':>7s} {grand_n / np_:8.2f} {grand_us / np_:10.1f} 100.00")

    if kern:
        label = args.phase or "prefix/*"
        sub_us = sum(v[1] for v in kern.values())
        print(f"\n=== per-kernel census: {label} ({sub_us / np_:.1f} us/predict) ===")
        print(
            f"{'kernel':<62s} {'n/pred':>7s} {'us/pred':>9s} {'us/inst':>8s} "
            f"{'%phase':>7s} {'CTAs':>7s} {'blk':>5s} {'regs':>5s} {'smem':>7s}"
        )
        rows = []
        for (nm, ctas, block, regs, smem), (n, us) in sorted(
            kern.items(), key=lambda kv: -kv[1][1]
        ):
            rows.append(
                {
                    "kernel": nm,
                    "n_per_predict": n / np_,
                    "us_per_predict": us / np_,
                    "us_per_inst": us / n,
                    "pct_phase": 100 * us / sub_us,
                    "CTAs": ctas,
                    "block": block,
                    "regs": regs,
                    "smem_B": smem,
                }
            )
            print(
                f"{nm[:62]:<62s} {n / np_:7.2f} {us / np_:9.2f} {us / n:8.3f} "
                f"{100 * us / sub_us:7.2f} {ctas:7d} {block:5d} {regs:5d} {smem:7d}"
            )
        if args.csv:
            with open(args.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
