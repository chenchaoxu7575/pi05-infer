# Per-kernel snapshots

Where a denoise step's GPU time goes, one file per measured build. Each file carries its
own commit, clock and profiler version -- **snapshots are not comparable to each other
unless those match**, which is why they are dated rather than overwritten.

| date | commit | launches/step | us/step | notes |
|---|---|--:|--:|---|
| [2026-08-03](denoise_20260803.md) | `eccaeb6` | 190 | 1239.15 | retiled `down_proj`/`o_proj`, `Q*K^T` tile pinned |

These are **locked-clock** measurements, on purpose: a per-kernel breakdown is analysis, and
without a fixed clock it measures drift instead of the kernels. The README's end-to-end
number is measured **unlocked**, because that is the speed the card actually runs at. The
two rulers differ by around 20 % and must not be mixed.

The chart in `docs/denoise_light.png` is a different, earlier snapshot and is stamped with
its own commit.
