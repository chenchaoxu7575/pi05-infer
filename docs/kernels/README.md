# Per-kernel snapshots

Where a phase's GPU time goes on one build -- an end-state cost breakdown, not a speedup.
The README's charts answer "how much faster"; these answer "what is left".

| date | phase | build | commit | launches | GPU time | notes |
|---|---|---|---|--:|--:|---|
| [2026-08-03](denoise_20260803.md) | denoise step | `main` | `eccaeb6` | 190 /step | 1239.15 us/step | retiled `down_proj`/`o_proj`, `Q*K^T` tile pinned |
| [2026-08-03](prefix_20260803.md) | prefix: LM prefill | `main` | `eccaeb6` | 245 /predict | 24 443.2 us/predict | 74.5 % MFU; 93.7 % of the time is GEMM |
| [2026-08-03](prefix_20260803.md) | prefix: SigLIP | `main` | `eccaeb6` | 385 /predict | 5 933.7 us/predict | 54.0 % MFU; unfused q/k/v, head_dim 72 |

The prefix is 70.8 % of a predict's GPU time at bs=1 and the denoise step is 28.9 %; a
predict runs ten steps. The two pages carry **different columns on purpose** -- denoise is
memory-bound so it is keyed on `FLOP/byte` and roofline side, the prefix is compute-bound so
it is keyed on achieved TFLOP/s and % of peak. Neither page has ncu-measured DRAM bytes for
the prefix, which is why the prefix page has no `FLOP/byte` column at all.

Locked-clock measurements, dated rather than overwritten: a snapshot is comparable to
another only under the same commit, clock and profiler. The README's end-to-end number is
unlocked; the two rulers differ by 14 % on the shipping build (see
[`../locked_clock.md`](../locked_clock.md)).

The 1239.15 us/step here and the 1237.83 us/step on the README's denoise chart are separate
profiling sessions of the same build, 0.1 % apart. Neither is derived from the other.

Shape:

| | |
|---|---|
| one profiling pass | one `<phase>_<date>.md` + one `.csv` |
| several builds in one pass | one time column each in the `.md`, one row each here |
| `.csv` | long form, keyed by `build`; a build adds rows, never columns |

The prefix `.csv` carries a `phase` key as well, because one pass produces two phases
(`prefix/vlm_forward` and `prefix/vision_siglip`) that must not be summed into one row.
