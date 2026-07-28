"""Kernels per denoise step, counted on the GPU timeline.

The denoise phase of each predict is delimited on the GPU by the first and last
kernel on the inductor denoise-cudagraph stream (157). Everything executing in
that window on any stream belongs to the denoise loop.

NOTE: the window ends at the LAST stream-157 kernel, so the trailing eager glue of
the final denoise step is excluded. That accounts for the ~3 kernels/step gap between
what this reports (234.90) and the historically recorded 238. Widening the window past
that point spills into the next predict's prefix, so this tight definition is used
consistently for both arms of an A/B.

Usage: denoise_kernels.py <sqlite> [<sqlite> ...]   (default: the two A/B profiles)
"""

import sqlite3
import sys

_DEFAULT = [
    "/workspace/rlinf_pub/pi05_infer_runs/nsys_rlinf.sqlite",
    "/workspace/rlinf_pub/pi05_infer_runs/nsys_pi05infer.sqlite",
]

for tag in sys.argv[1:] or _DEFAULT:
    c = sqlite3.connect(tag)
    k157 = list(c.execute(
        "select start,end from CUPTI_ACTIVITY_KIND_KERNEL where streamId=157 order by start"))
    # split into predicts: a gap > 5 ms on stream 157 = a prefix phase in between
    windows, cur = [], [k157[0][0], k157[0][1]]
    for s, e in k157[1:]:
        if s - cur[1] > 5_000_000:      # 5 ms in ns
            windows.append(tuple(cur))
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    windows.append(tuple(cur))
    nsteps = 10
    per = {}
    for s, e in windows:
        for stream, n in c.execute(
                "select streamId,count(*) from CUPTI_ACTIVITY_KIND_KERNEL "
                "where start>=? and start<=? group by streamId", (s, e)):
            per[stream] = per.get(stream, 0) + n
    npred = len(windows)
    tot = sum(per.values())
    print(f"{tag}: {npred} predicts detected")
    for k in sorted(per):
        print(f"    stream {k:4d}: {per[k]/npred/nsteps:8.2f} kernels/step")
    print(f"    TOTAL     : {tot/npred/nsteps:8.2f} kernels/step")
