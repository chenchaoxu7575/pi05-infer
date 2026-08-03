**English** | [简体中文](README.zh-CN.md)

# pi05-infer

A standalone **bs=1 inference engine for the π0.5 action expert**, extracted from
[RLinf](https://github.com/RLinf/RLinf) and optimized for the **RTX PRO 5000
(GB202 / sm_120, Blackwell)**. Every item is an algebraically equivalent transform —
**no quantization, no change of sampler, no reduction in denoising steps**.

## Results

End-to-end `predict_action_batch`, on the card named above:

```
Ledger (paired A/B chain, 9 steps, plain wall clock):  52.60 -> 42.90 ms  (-18.4%)
Since the ledger, each on its own locked baseline:     two tile changes, -0.73 / -0.11 ms
Current main, unlocked plain wall clock, n=30:         40.63 ms  (p50; mean 40.56,
                                                       39.40 .. 41.47, SM clock
                                                       sampled 2235-2265 MHz)
```

**Those three lines use different rulers and must not be chained.** The ledger is a paired
chain: each row is its own A/B against the row above it, so the rows add up. The two tile
changes landed after the ledger closed and were each measured against their own baseline
with the SM clock locked, so `-0.73` and `-0.11` are *not* subtractable from `42.90`. The
last line is the only number that describes main as you will run it: one process, clocks
left alone, plain wall clock.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger_dark.png">
  <img alt="Three-panel optimization ledger: the prehistory before this repository (a different ruler), the paired end-to-end waterfall from 52.60 to 42.90 ms, and the same optimizations accounted per denoising step" src="docs/ledger_light.png">
</picture>

Three panels, **three different rulers**: the prehistory before this repository (a different
measurement protocol), this repository's paired end-to-end ledger, and the same optimizations
counted as GPU busy per denoising step.

The two dashed lines mark where reference implementations sit — neither is a paired
measurement and no win/loss is claimed.

<details>
<summary>Where a denoise step goes, and why the prefix is the ceiling</summary>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/denoise_dark.png">
  <img alt="Per-kernel breakdown of one denoise step" src="docs/denoise_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/phases_dark.png">
  <img alt="Phase split: the 968-token prefix is 71.7% of GPU busy, the denoise loop 28.3%" src="docs/phases_light.png">
</picture>

Both charts are stamped with the commit and date they were profiled at, and neither is
re-derived against current main — see the note in `docs/make_charts.py` for why not.
The second one is the reason this project has an upper bound: **the 968-token prefix is
71.7 % of GPU busy, so even a free denoise loop caps the whole-predict speedup at 1.39×.**

</details>

## Hardware

Everything here was measured, tuned and verified on **one card**: RTX PRO 5000 Blackwell
(GB202 / sm_120), 110 SMs, 96 MB L2, 300 W cap.

**On any other GPU, both the performance claims and the bit-exactness claims stop
applying** — and the second one is the surprising half. The inductor tile choices were
tuned against this card's roofline knee and SM count, which is the expected kind of
non-portability. But "bit-identical" here means *identical to what an unpatched build
produces on this card*, and an unpatched build picks a different kernel on a different
card. The reference itself moves, so the claim has no subject there.

The code does not stop you. It warns once, the hardware-specific tile pin declines to
install off sm_120 by itself, and every optimization has a `RLINF_*=0` kill switch. Re-run
the gates in [`tools/`](tools/README.md) on your card before quoting any number from here.

## Install and run

You need a checkpoint. The published SFT checkpoints work directly — for example
[`RLinf/RLinf-Pi05-LIBERO-SFT`](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT):

```bash
huggingface-cli download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir /path/to/RLinf-Pi05-LIBERO-SFT
export PI05_MODEL_PATH=/path/to/RLinf-Pi05-LIBERO-SFT
```

The loader expects a directory holding the safetensors shards plus openpi's normalization
statistics under an asset-id subdirectory:

```
RLinf-Pi05-LIBERO-SFT/
  model.safetensors
  physical-intelligence/libero/norm_stats.json
```

The second path is openpi's `<asset_id>/norm_stats.json`, and the asset id comes
from the `--config-name` TrainConfig -- `pi05_turtle` and the LIBERO configs both
resolve to `physical-intelligence/libero`. If you point `--model-path` at a
checkpoint whose asset id differs, the weights load and the run fails later on the
missing stats.

Use the existing RLinf benchmark container image — **no Docker rebuild required**. Install
editable and with `--no-deps`, so the torch / transformers / openpi versions pinned inside
the container are left alone:

```bash
docker exec -w /path/to/pi05-infer pi05bench \
    /opt/venv/openpi/bin/pip install -e . --no-deps --no-build-isolation

# benchmark
/opt/venv/openpi/bin/python bench/standalone_infer_bench.py \
    --model-path $PI05_MODEL_PATH --config-name pi05_turtle --iters 30
... --stage1        # enable the hand-captured denoise CUDA graph (opt-in; every number
                    # from ledger row 7 onwards was measured with it on)
... --phases        # per-phase timing
... --dump-actions /tmp/a.pt --clocks-json /tmp/clocks.json   # numerical A/B + SM clock/power
```

`pi05-infer` does not touch `site-packages` (it only adds one path entry), so the container
stays pristine and can serve as the reference arm of an A/B.

`--stage1` rewrites `max-autotune` into `max-autotune-no-cudagraphs` and then **asserts**
after warmup that the graph really was captured, failing the run if it did not. That check
exists because the failure it catches is invisible: a shape-signature mismatch degrades to
the eager denoise loop with no symptom other than the runtime.

<a id="r-verify"></a>

## Verification: numerical agreement

```bash
export PI05_MODEL_PATH=/path/to/RLinf-Pi05-LIBERO-SFT

python tools/isolation_check.py          # expert = pi05_infer.gemma, prefix = transformers

# kernel / GEMM / KV-level bit-exactness gates
python tools/bitgate.py                  # the two Triton fusion kernels
python tools/bitexact_denoise_gemms.py   # small-M mm retile
python tools/bitexact_denoise_bmms.py    # attention bmm retile + the Q·Kᵀ tile pin
python tools/bitexact_prefix_kv.py       # prefix last-layer skip
python tools/bitexact_prefix_qkv.py      # fused prefix QKV

# structural optimizations on the compiled path (frozen prefix + four-process control gate)
bash tools/run_bitexact_backfill.sh <stage>   # siglip|extraction|prefix|adarms|adarms_eager|qkv|kvstatic|attmask

# end-to-end numerical A/B -- ⚠️ four processes, always with an empty control; declares
# INCONCLUSIVE (never PASS) unless both same-arm controls come back clean
GATE_OFF="RLINF_SMALL_M_MM=0" GATE_ON="RLINF_SMALL_M_MM=1" \
  tools/bitexact_gate.sh /tmp/gate_small_m --stage1 --iters 1 --warmup 4
```

Each gate runs both arms and prints a digest; the two must match. Every optimization has a
kill switch, and the OFF arm exercises a verified fallback path.

⚠️ **Bit-identity here is tiered by compile path, and is not claimed uniformly.** Some items
are bit-identical under eager but not under the shipping `max-autotune`, whose own kernel
choice is not stable across cold autotunes — on one shape, 1 of 4 cold caches picked cuBLAS
over the Triton template, which has a different digest. What holds for every item without
qualification is that the transform is algebraically equivalent. See the bit-exactness note
in `pi05_infer/inductor_mm_tiles.py`, which documents a rule this project believed, shipped,
and then measured to be false.

<a id="r-layout"></a>

## Repository layout

```
pi05_infer/    the engine (engine.py, the vendored action-expert Gemma + Triton fusion
               kernels, prefix_last_layer.py, prefix_qkv_fused.py, inductor_mm_tiles.py)
bench/         standalone_infer_bench.py -- latency bench
tools/         verification gates and measurement drivers -- see tools/README.md
docs/          the charts above, and make_charts.py which regenerates them
_extract_src/  the original RLinf files, before extraction
```

`import pi05_infer` routes **only the action expert** through the vendored Gemma; the
PaliGemma **prefix** keeps stock transformers. That seam is deliberate: it is what stops a
denoise kernel change from reaching the 968-token prefix.

**`_extract_src/` is not part of the package** — nothing imports it, and it is excluded from
the build and from linting. It is the unmodified RLinf source that `pi05_infer/` was
extracted from, kept in the tree so that the extraction can be audited by diff rather than
taken on trust. It is not refactored and is not meant to be; `EXTRACTION_NOTES.md` is the
per-file account of what changed.

## Further reading

> **The detailed optimization notes are not published yet.** The per-item
> derivations, the correctness argument, the measurement methodology and the raw
> A/B archive are kept internally until this work converges, and will be released
> then. Where the source cites one of those documents by name, the name is provenance,
> not a link.

* **[`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md)** — the extraction boundary against RLinf.
* **[`tools/README.md`](tools/README.md)** — which scripts are portable, and which are not.

## License and provenance

Apache-2.0 ([`LICENSE`](LICENSE)). Vendors code from HuggingFace Transformers,
[openpi](https://github.com/Physical-Intelligence/openpi) (via the
[RLinf/openpi](https://github.com/RLinf/openpi) fork) and
[RLinf](https://github.com/RLinf/RLinf); per-file modifications are listed in
[`NOTICE`](NOTICE). `dexmal/realtime-vla` is referenced as a peer; none of its code is
reused.
