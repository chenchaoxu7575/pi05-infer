# The same chain at a locked clock

The README chart runs the card as shipped: unlocked, 300 W cap. That is what a reader has,
and it is the right thing to publish. It is a poor ruler for attribution, because the card is
power-limited and the arms do not all get the same clock.

This page is the same five arms measured again with the SM clock pinned. It removes the power
cap from the comparison. It does not turn the chain into ground truth -- see the second
section. Nothing here may be added to a number in the README; the two rulers differ by 14 %
on `c3`.

## Both chains

Five arms, one session each, six rounds of rotating order, each delta paired within a round.
Span is `sample_actions` (`--model-only`), bs=1, 224^2, 968 prefix tokens, chunk 50, bf16.

| arm | | unlocked ms | locked ms |
|---|---|--:|--:|
| `eager` | `--no-compile` | 124.71 | 130.97 |
| `base` | `torch.compile max-autotune`, everything off | 51.57 | 65.01 |
| `c1` | + CPU overhead | 48.16 | 58.03 |
| `c2` | + denoise-step work removed | 42.90 | 51.83 |
| `c3` | + kernel fusion & optimization (ships) | **39.64** | 45.22 |

| step | unlocked | locked |
|---|--:|--:|
| `eager -> base` | +73.14 | +65.96 |
| `base -> c1` | +3.41 | **+6.98** |
| `c1 -> c2` | +5.26 | +6.20 |
| `c2 -> c3` | +3.26 | **+6.61** |

| | unlocked | locked |
|---|---|---|
| null control (`c3` vs `c3`) | -0.01 +/- 0.02, max abs 0.04 | +0.02, max abs 0.07 |
| SM clock | 2220-2437 MHz, falling along the chain | 1882-1890 MHz, spread 8 MHz |
| memory clock | 13365 MHz | 13365 MHz -- `-lgc` does not touch it |
| power | 295-301 W against a 300 W cap | 226.9 W peak |
| chain closes | yes | yes, residual -0.0000 |

## Why two of the deltas roughly double

Unlocked, each successive arm keeps the GPU busy a larger fraction of the time, so against a
fixed 300 W budget the governor takes more clock from each: 2437 -> 2362 -> 2317 -> 2220 MHz.
Every arm is compared against a predecessor running faster, which compresses the deltas.
`base` at 294.9 W is only partly capped, which is why it holds the highest clock. `eager` is
not part of that decay at all -- at 172 W it never approaches the cap.

WARNING: **the clock is only about half of it.** Taking `base -> c1`, whose delta moves from
+3.41 unlocked to +6.98 locked -- a change of **+3.57 ms**. How much of that change does a
pure SM-clock model predict?

| clock statistic used | predicts | actual change | unexplained |
|---|--:|--:|--:|
| mean SM | +1.91 | +3.57 | **+1.66** |
| pooled median SM | +2.64 | +3.57 | **+0.93** |

**`-lgc` locks the SM clock and nothing else. The memory clock is 13365 MHz on both rulers**
-- identical, zero spread, all 2160 samples. So DRAM bandwidth never changed between them,
and memory-bandwidth-bound work costs the same on both. Which is exactly what the
optimizations leave behind: on the shipping build the four largest denoise kernels
(`_geglu_mm_kernel`, `down_proj`, `_qkv_rope_kernel`, `o_proj`) are all memory-side on
[the roofline](kernels/denoise_20260803.md) and are **70.6 %** of the step; only `Q*K^T`
(7.4 %) is compute-side.

More optimized arm -> larger memory-bound share -> less response to SM clock. The check that
this is boundedness and not just occupancy: `c1`/`c2`/`c3` are all 95 % GPU-busy yet have
SM-clock sensitivities of 0.80 / 0.89 / 0.74, while `base` is 90 % busy at 0.94.

NEVER: **do not describe the locked chain as a corrected version of the unlocked one.** They are
two machines, not two views of one. Part of the difference is the power cap; part is work
that never responded to SM clock in the first place. Print each chain with its own clock and
power and let them stand separately.

## It is the power cap, not heat

20 Hz `nvidia-smi` sampling inside the model-execution window, 4659 samples:

| | `c3` | `eager` |
|---|---|---|
| power under load | **300.0 W** = the limit | 173.7 W = 58 % of it |
| SM clock, capped vs not | **2212 vs 2370 MHz (-158)** | 2325 vs 2325 (0) |
| `hw_slowdown`, `hw_thermal`, `sw_thermal`, `hw_power_brake` | 0 % | 0 % |
| temperature p50 / max | 52 / 60 C | 57 / 63 C, 32 C of margin |

WARNING: read the power, not the bit. `sw_power_cap` is also asserted on `eager`, where it
means only that the clock is below maximum boost (3090 MHz); at 173 W the board cap is doing
nothing. The evidence is power pinned at the limit *and* the clock 158 MHz below its
unconstrained value in the same run.

`Max Power Limit` on this card is 350 W, 50 W above the default. Raising it is a third
configuration, not a correction to either of these two.

## Reproduce

```bash
flock /tmp/pi05_gpu_timing.lock -c '
  nvidia-smi -i 0 -pm 1; nvidia-smi -i 0 -lgc 1900
  PI05_MODEL_PATH=/path/to/ckpt CUDA_VISIBLE_DEVICES=0 \
    python bench/standalone_infer_bench.py --config-name pi05_turtle \
      --warmup 20 --iters 30 --model-only --stage1 --clocks-json /tmp/clocks.json
  nvidia-smi -i 0 -rgc'
```

WARNING: the driver drops `-lgc` mid-run, and ignores it entirely without `-pm 1`. Re-apply
every 0.5 s and sample `clocks.sm` for the whole run; a spread above ~30 MHz voids the data.
Drop `-lgc`/`-rgc` for the unlocked arm and record the clock next to the number -- unlocked
absolutes are not comparable across sessions.
