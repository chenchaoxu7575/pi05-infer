**English** | [简体中文](README.zh-CN.md)

# pi05-infer

A standalone **bs=1 inference engine for the pi0.5 action expert**, extracted from
[RLinf](https://github.com/RLinf/RLinf) and optimized for the **RTX PRO 5000
(GB202 / sm_120, Blackwell)**. Every item is an algebraically equivalent transform --
**no quantization, no change of sampler, no reduction in denoising steps**.

## Results

End-to-end `predict_action_batch`, on the card named above:

```
Ledger (paired A/B chain, 8 optimizations, plain wall clock):
    52.60 -> 42.90 ms  (-18.4%)

Since the ledger, two tile changes, each on its own baseline:
    down_proj / o_proj retile   -0.52 +/- 0.28 ms   locked paired A/B, 4/4 rounds same sign
    Q*K^T tile pin              -0.106 ms           in expectation, not a paired A/B

Current main, unlocked plain wall clock, n=30:
    40.63 ms  (p50; mean 40.56, 39.40 .. 41.47, SM clock sampled 2235-2265 MHz)
```

**Those numbers use different rulers and must not be chained.** The ledger is a paired
chain: each row is its own A/B against the row above it, so the rows add up. The two tile
changes landed after the ledger closed, so neither is subtractable from `42.90` -- and the
two are not established the same way. The retile is a locked paired A/B. The tile pin is
not a paired A/B at all: `-0.106 ms` is an expectation over the tiles autotune was drawing
for that shape, and the reason to pin is variance rather than mean -- it takes the
draw-to-draw spread on this shape to zero, which is what makes any later A/B on it readable.
The last line is the only number that describes main as you will run it: one process, clocks
left alone, plain wall clock.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger_dark.png">
  <img alt="Three-panel optimization ledger: the prehistory before this repository (a different ruler), the paired end-to-end waterfall from 52.60 to 42.90 ms, and the same optimizations accounted per denoising step" src="docs/ledger_light.png">
</picture>

Three panels, **three different rulers**: the prehistory before this repository (a different
measurement protocol), this repository's paired end-to-end ledger, and the same optimizations
counted as GPU busy per denoising step.

The two dashed lines mark where reference implementations sit -- neither is a paired
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
re-derived against current main -- see the note in `docs/make_charts.py` for why not.
The second one is the reason this project has an upper bound: **the 968-token prefix is
71.7 % of GPU busy, so even a free denoise loop caps the whole-predict speedup at 1.39x.**

</details>

## Hardware

Everything here was measured, tuned and verified on **one card**: RTX PRO 5000 Blackwell
(GB202 / sm_120), 110 SMs, 96 MB L2, 300 W cap.

**On any other GPU, both the performance claims and the bit-exactness claims stop
applying** -- and the second one is the surprising half. The inductor tile choices were
tuned against this card's roofline knee and SM count, which is the expected kind of
non-portability. But "bit-identical" here means *identical to what an unpatched build
produces on this card*, and an unpatched build picks a different kernel on a different
card. The reference itself moves, so the claim has no subject there.

The code does not stop you. It warns once, the hardware-specific tile pin declines to
install off sm_120 by itself, and most optimizations carry an `RLINF_*=0` kill switch
(four structural ones do not -- they are listed under Verification below). Re-run the
gates in [`tools/`](tools/README.md) on your card before quoting any number from here.

## Install and run

You need a checkpoint. The published SFT checkpoints work directly -- for example
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

Use the existing RLinf benchmark container image -- **no Docker rebuild required**. Install
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
export CUDA_VISIBLE_DEVICES=0     # run_bitexact_backfill.sh otherwise defaults to GPU 1

python tools/isolation_check.py          # expert = pi05_infer.gemma, prefix = transformers

# Same-process gates: one command runs both arms and prints both digests.
python tools/bitgate.py                  # the two Triton fusion kernels
python tools/bitexact_prefix_qkv.py      # fused prefix QKV

# One process per arm, sharing one inductor cache dir -- separate caches let autotune
# re-pick untouched shapes and have produced a sign-flipped result. Digests must match.
TORCHINDUCTOR_CACHE_DIR=/tmp/ti_be RLINF_SMALL_M_MM=0 python tools/bitexact_denoise_gemms.py
TORCHINDUCTOR_CACHE_DIR=/tmp/ti_be RLINF_SMALL_M_MM=1 python tools/bitexact_denoise_gemms.py

TORCHINDUCTOR_CACHE_DIR=/tmp/ti_be RLINF_SMALL_M_BMM=0 python tools/bitexact_denoise_bmms.py
TORCHINDUCTOR_CACHE_DIR=/tmp/ti_be RLINF_SMALL_M_BMM=1 python tools/bitexact_denoise_bmms.py

# Prefix last-layer skip: two arms write JSON, a third call compares them.
TORCHINDUCTOR_CACHE_DIR=/tmp/ti_kv RLINF_SKIP_LAST_LM_LAYER=0 \
  python tools/bitexact_prefix_kv.py --out off.json
TORCHINDUCTOR_CACHE_DIR=/tmp/ti_kv RLINF_SKIP_LAST_LM_LAYER=1 \
  python tools/bitexact_prefix_kv.py --out on.json
python tools/bitexact_prefix_kv.py --compare off.json on.json

# Structural optimizations on the compiled path (frozen prefix + four-process control
# gate). One stage per invocation, and `prefix` must run first: it writes the frozen
# prefix that adarms / adarms_eager / qkv / kvstatic replay. Running one of those on
# its own silently produces a weaker gate rather than an error.
bash tools/run_bitexact_backfill.sh prefix
bash tools/run_bitexact_backfill.sh adarms      # then adarms_eager | qkv | kvstatic
bash tools/run_bitexact_backfill.sh siglip      # independent of the frozen prefix
bash tools/run_bitexact_backfill.sh attmask     # ditto -- it lives inside the prefix
RLINF_ROOT=/path/to/RLinf \
  bash tools/run_bitexact_backfill.sh extraction   # needs a second, RLinf checkout

# end-to-end numerical A/B -- WARNING: four processes, always with an empty control; declares
# INCONCLUSIVE (never PASS) unless both same-arm controls come back clean
GATE_OFF="RLINF_SMALL_M_MM=0" GATE_ON="RLINF_SMALL_M_MM=1" \
  tools/bitexact_gate.sh /tmp/gate_small_m --stage1 --iters 1 --warmup 4
```

`bitgate.py` and `bitexact_prefix_qkv.py` run both arms inside one process and print both
digests. The rest need one process per arm, which is what the pinned cache dir is for.
Either way the two digests must match.

Most optimizations carry an `RLINF_*=0` kill switch, and a gate's OFF arm exercises that
fallback path. Four do not: the adaRMS precompute, the action expert's fused QKV, the
static prefix-KV buffer and the device-side attention mask are structural and have no env
var. They can only be turned off by `tools/bitexact_compiled_toggles.py --disable`, which
monkey-patches the seam on a live model -- that is what `run_bitexact_backfill.sh` drives.

WARNING: **Bit-identity here is tiered by compile path, and is not claimed uniformly.** Some items
are bit-identical under eager but not under the shipping `max-autotune`, whose own kernel
choice is not stable across cold autotunes -- on one shape, 1 of 4 cold caches picked cuBLAS
over the Triton template, which has a different digest. What holds for every item without
qualification is that the transform is algebraically equivalent. See the bit-exactness note
in `pi05_infer/patches/inductor_mm_tiles.py`, which documents a rule this project believed, shipped,
and then measured to be false.

<a id="r-layout"></a>

## Repository layout

```
pi05_infer/    the engine
  gemma/       the vendored, modified action-expert Gemma + Triton fusion kernels
  patches/     optimizations applied to code we do not own (the stock-transformers
               prefix, and inductor's tile candidates) -- all opt-out
bench/         standalone_infer_bench.py -- latency bench
tools/         verification gates and measurement drivers -- see tools/README.md
docs/          the charts above, and make_charts.py which regenerates them
_extract_src/  the original RLinf files, before extraction
```

`import pi05_infer` routes **only the action expert** through the vendored Gemma; the
PaliGemma **prefix** keeps stock transformers. That seam is deliberate: it is what stops a
denoise kernel change from reaching the 968-token prefix.

**`_extract_src/` is not part of the package** -- nothing imports it, and it is excluded from
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

* **[`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md)** -- the extraction boundary against RLinf.
* **[`tools/README.md`](tools/README.md)** -- which scripts are portable, and which are not.

## License and provenance

Apache-2.0 ([`LICENSE`](LICENSE)). Vendors code from HuggingFace Transformers,
[openpi](https://github.com/Physical-Intelligence/openpi) (via the
[RLinf/openpi](https://github.com/RLinf/openpi) fork) and
[RLinf](https://github.com/RLinf/RLinf); per-file modifications are listed in
[`NOTICE`](NOTICE). `dexmal/realtime-vla` is referenced as a peer; none of its code is
reused.
