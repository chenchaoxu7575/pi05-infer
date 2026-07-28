"""Per-stream kernel rollup for an nsys-2026 sqlite export.

Splits the profile by CUDA stream so a prefix-side regression cannot hide behind a
healthy denoise-side metric -- a fusion regression once cost +4 ms entirely in the
prefix (stream 7) with every denoise (stream 157) number unchanged.

Usage: stream_summary.py <sqlite> <n_predicts> [denoise_steps_per_predict]
"""

import sqlite3
import sys

f = sys.argv[1]
n_predicts = float(sys.argv[2])
n_steps = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

c = sqlite3.connect(f)
rows = list(
    c.execute(
        """select k.streamId, count(*), sum(k.end - k.start) / 1e3
           from CUPTI_ACTIVITY_KIND_KERNEL k
           group by k.streamId order by 3 desc"""
    )
)

print(f"predicts={n_predicts:.0f}  denoise_steps/predict={n_steps:.0f}")
print(f"{'stream':>7s} {'kernels':>10s} {'k/predict':>10s} {'us/predict':>11s} {'k/step':>8s}")
tot_n = tot_us = 0
for stream, n, us in rows:
    tot_n += n
    tot_us += us
    print(
        f"{stream:7d} {n:10d} {n / n_predicts:10.2f} {us / n_predicts:11.1f} "
        f"{n / n_predicts / n_steps:8.2f}"
    )
print(f"{'TOTAL':>7s} {tot_n:10d} {tot_n / n_predicts:10.2f} {tot_us / n_predicts:11.1f}")
