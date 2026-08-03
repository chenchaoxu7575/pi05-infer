# Per-kernel snapshots

| date | commit | launches/step | us/step | notes |
|---|---|--:|--:|---|
| [2026-08-03](denoise_20260803.md) | `eccaeb6` | 190 | 1239.15 | retiled `down_proj`/`o_proj`, `Q*K^T` tile pinned |

Locked-clock measurements, dated rather than overwritten: a snapshot is comparable to
another only under the same commit, clock and profiler. The README's end-to-end number is
unlocked; the two rulers differ by around 20 %.
