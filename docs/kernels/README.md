# Per-kernel snapshots

Where a denoise step's GPU time goes on one build -- an end-state cost breakdown, not a
speedup. The README's charts answer "how much faster"; these answer "what is left".

| date | build | commit | launches/step | us/step | notes |
|---|---|---|--:|--:|---|
| [2026-08-03](denoise_20260803.md) | `main` | `eccaeb6` | 190 | 1239.15 | retiled `down_proj`/`o_proj`, `Q*K^T` tile pinned |

Locked-clock measurements, dated rather than overwritten: a snapshot is comparable to
another only under the same commit, clock and profiler. The README's end-to-end number is
unlocked; the two rulers differ by 14 % on the shipping build (see
[`../locked_clock.md`](../locked_clock.md)).

The 1239.15 us/step here and the 1237.83 us/step on the README's denoise chart are separate
profiling sessions of the same build, 0.1 % apart. Neither is derived from the other.

Shape:

| | |
|---|---|
| one profiling pass | one `denoise_<date>.md` + one `.csv` |
| several builds in one pass | one `us/step` column each in the `.md`, one row each here |
| `.csv` | long form, keyed by `build`; a build adds rows, never columns |
