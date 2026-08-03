**English** | [简体中文](README.zh-CN.md)

# pi05-infer

A standalone **bs=1 inference engine for the pi0.5 action expert**, extracted from
[RLinf](https://github.com/RLinf/RLinf) and optimized for the **RTX PRO 5000
(GB202 / sm_120)**. Every change is an algebraically equivalent transform: no
quantization, no change of sampler, no reduction in denoising steps.

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger_dark.png">
  <img alt="52.60 to 42.90 ms, split into three blocks: eager to CUDA graph -3.44 ms, kernel fusion -3.17 ms, denoise-step work removed -3.09 ms" src="docs/ledger_light.png">
</picture>

Current `main`, unlocked plain wall clock, n=30: **40.63 ms** (p50; mean 40.56,
39.40 .. 41.47, SM clock sampled 2235-2265 MHz).

> The ledger, the two post-ledger tile changes and the number above use three
> different rulers and are not addable. Only the last line describes `main` as
> you will run it.

<details>
<summary>Where a denoise step goes</summary>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/denoise_dark.png">
  <img alt="Per-kernel breakdown of one denoise step" src="docs/denoise_light.png">
</picture>

Stamped with the commit it was profiled at; not re-derived against current `main`.
The 968-token prefix is 71.7 % of GPU busy, so even a free denoise loop caps the
whole-predict speedup at 1.39x.

</details>

## Hardware

Measured, tuned and verified on **one card**: RTX PRO 5000 Blackwell (sm_120),
110 SMs, 96 MB L2, 300 W cap. On any other GPU neither the performance numbers
nor the bit-exactness digests carry over -- "bit-identical" is defined against
*this* card's unpatched build, and elsewhere that reference is a different kernel.

Nothing is blocked: the engine warns once, the hardware-specific tile pin declines
to install off sm_120, and most optimizations carry an `RLINF_*=0` kill switch.

## Install

`pyproject.toml` declares `dependencies = []` on purpose -- this installs with
`--no-deps` into an existing **openpi** environment, whose torch build must not be
touched. That environment must provide:

| | |
|---|---|
| Python | `>=3.10,<3.12` |
| PyTorch | **2.7.1+cu128**, the build the numbers above were measured on. Any build works if it targets sm_120; a `+cu124` wheel compiles for sm_50..sm_90 and cannot run on a GB202. |
| openpi | installed **with its `transformers_replace` patch** (openpi's `install.sh` does this). The prefix runs on stock transformers and is called with `adarms_cond=`, which vanilla transformers rejects. |
| transformers | 4.53.2, as patched by openpi |
| also imported | `numpy`, `einops`, `nvtx` |

The RLinf benchmark container already satisfies all of it.

```bash
pip install -e . --no-deps --no-build-isolation

huggingface-cli download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir /path/to/ckpt
export PI05_MODEL_PATH=/path/to/ckpt
```

The checkpoint directory needs the weights plus openpi's norm stats under their
asset id:

```
ckpt/
  model.safetensors
  physical-intelligence/libero/norm_stats.json
```

## Run

```bash
python bench/standalone_infer_bench.py --config-name pi05_turtle --iters 30
... --stage1        # hand-captured denoise CUDA graph (opt-in)
... --phases        # per-phase timing
... --dump-actions /tmp/a.pt --clocks-json /tmp/clocks.json
```

## Verify

Every optimization has a gate that runs both arms and compares a byte-level digest.

```bash
export PI05_MODEL_PATH=/path/to/ckpt
export CUDA_VISIBLE_DEVICES=0

python tools/isolation_check.py     # expert = pi05_infer.gemma, prefix = transformers
python tools/bitgate.py             # the two Triton fusion kernels
python tools/bitexact_prefix_qkv.py # fused prefix QKV
```

The remaining gates need one process per arm, or a fixed stage order.
**[`tools/README.md`](tools/README.md) has the exact invocations** -- run them as
written; run as one-liners they print a digest against nothing.

> Bit-identity is tiered by compile path and is not claimed uniformly: some items
> are bit-identical under eager but not under the shipping `max-autotune`, whose
> own kernel choice is not stable across cold autotunes. What holds for every item
> is that the transform is algebraically equivalent. The note in
> `pi05_infer/patches/inductor_mm_tiles.py` documents a rule this project believed,
> shipped, and then measured to be false.

## Layout

```
pi05_infer/    the engine
  gemma/       vendored, modified action-expert Gemma + Triton fusion kernels
  patches/     optimizations applied to code we do not own -- all opt-out
bench/         latency bench
tools/         verification gates and measurement drivers
docs/          the charts, and make_charts.py which regenerates them
  kernels/     per-kernel snapshots of a denoise step, one file per measured build
_extract_src/  the original RLinf files, kept so the extraction can be diffed
```

`import pi05_infer` routes **only the action expert** through the vendored Gemma;
the PaliGemma **prefix** keeps stock transformers. That seam is what stops a
denoise kernel change from reaching the 968-token prefix.

## Further reading

The detailed optimization record is not published yet. Where the source cites one
of those documents by name, the name is provenance, not a link.

* [`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md) -- the extraction boundary against
  RLinf, and what it leaves open.
* [`tools/README.md`](tools/README.md) -- which scripts are portable, and which are not.

## License

Apache-2.0 ([`LICENSE`](LICENSE)). Vendors code from HuggingFace Transformers,
[openpi](https://github.com/Physical-Intelligence/openpi) (via the
[RLinf/openpi](https://github.com/RLinf/openpi) fork) and
[RLinf](https://github.com/RLinf/RLinf); per-file provenance in [`NOTICE`](NOTICE).
