"""Kernel-count summary for a denoise stream: total/step plus a category rollup.

Usage: ksum.py <sqlite> [stream] [nsteps]
Categories collapse inductor's per-layer name suffixes (…_31, …_35) so that the
36 distinct RMSNorm kernels show up as one row with n/step = 36.
"""
import re
import sqlite3
import sys

f = sys.argv[1]
stream = int(sys.argv[2]) if len(sys.argv) > 2 else 157
nsteps = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0

c = sqlite3.connect(f)
rows = list(
    c.execute(
        """select s.value, count(*), sum(k.end-k.start)/1e3
           from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
           where k.streamId=? group by s.value""",
        (stream,),
    )
)
tot_n = sum(r[1] for r in rows)
tot_us = sum(r[2] for r in rows)


def cat(name):
    if name.startswith("triton_"):
        # drop the trailing numeric id inductor appends
        return re.sub(r"_\d+$", "", name)
    n = name
    for pat in ("void ", "std::enable_if<!T7, void>::type "):
        n = n.replace(pat, "")
    n = n.split("<")[0].split("(")[0]
    return n.strip()


agg = {}
for name, n, us in rows:
    k = cat(name)
    a = agg.setdefault(k, [0, 0.0, 0])
    a[0] += n
    a[1] += us
    a[2] += 1

print(f"{'us/step':>8s} {'n/step':>7s} {'distinct':>8s}  kernel category")
for k, (n, us, d) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print(f"{us / nsteps:8.1f} {n / nsteps:7.2f} {d:8d}  {k[:78]}")
print(
    f"\nTOTAL stream{stream}: {tot_n / nsteps:.2f} kernels/step, "
    f"{tot_us / nsteps:.1f} us/step, {tot_us / nsteps * 10 / 1e3:.3f} ms/predict"
)
print(f"implied graph-node dispatch @1.3us: {tot_n / nsteps * 10 * 1.3 / 1e3:.2f} ms/predict")
