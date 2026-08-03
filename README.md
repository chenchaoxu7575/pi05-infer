**English** | [简体中文](README.zh-CN.md)

# pi05-infer

A standalone **bs=1 inference engine for the pi0.5 action expert**, extracted from
[RLinf](https://github.com/RLinf/RLinf) and optimized for the **RTX PRO 5000
(GB202 / sm_120)**. Every change is algebraically equivalent: no quantization, no
change of sampler, no reduction in denoising steps.

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger_dark.png">
  <img alt="Paired A/B speedup, split into three categories: CPU overhead, denoise-step work removed, kernel fusion and optimization" src="docs/ledger_light.png">
</picture>

Current `main`: **40.50 ms**, unlocked plain wall clock, n=30, p50.

<details>
<summary>Where a denoise step goes</summary>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/denoise_dark.png">
  <img alt="Per-kernel breakdown of one denoise step" src="docs/denoise_light.png">
</picture>

Per-kernel snapshots of the current build: [`docs/kernels/`](docs/kernels/).
The 968-token prefix is 71.7 % of GPU busy, so a free denoise loop would still
cap the whole-predict speedup at 1.39x.

</details>

## Hardware

```
GPU            RTX PRO 5000 Blackwell (GB202 / sm_120), 110 SM, 96 MB L2, 300 W
PyTorch        2.7.1+cu128
transformers   4.53.2, as patched by openpi
Python         3.11
```

Timings and bit-exactness digests are specific to this card.

## Install

```bash
# 1. an openpi environment with its transformers_replace patch applied
#    (openpi's own install.sh does this)

# 2. this package, --no-deps so the environment's torch is left alone
pip install -e . --no-deps --no-build-isolation

# 3. a checkpoint
huggingface-cli download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir /path/to/ckpt
export PI05_MODEL_PATH=/path/to/ckpt
```

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

Each optimization has a gate that runs both arms and compares a byte-level digest.

```bash
export PI05_MODEL_PATH=/path/to/ckpt
export CUDA_VISIBLE_DEVICES=0

python tools/isolation_check.py     # expert = pi05_infer.gemma, prefix = transformers
python tools/bitgate.py             # the two Triton fusion kernels
python tools/bitexact_prefix_qkv.py # fused prefix QKV
```

The rest are in [`tools/README.md`](tools/README.md).

## Layout

```
pi05_infer/    the engine
  gemma/       vendored, modified action-expert Gemma + Triton fusion kernels
  patches/     optimizations applied to code we do not own -- all opt-out
bench/         latency bench
tools/         verification gates and measurement drivers
docs/          the charts, and make_charts.py which regenerates them
  kernels/     per-kernel snapshots of a denoise step
_extract_src/  the original RLinf files, so the extraction can be diffed
```

## Further reading

* [`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md) -- the extraction boundary against RLinf.
* [`tools/README.md`](tools/README.md) -- which scripts are portable, and which are not.

The detailed optimization record is not published yet.

## License

Apache-2.0 ([`LICENSE`](LICENSE)). Vendors code from HuggingFace Transformers,
[openpi](https://github.com/Physical-Intelligence/openpi) and
[RLinf](https://github.com/RLinf/RLinf); per-file provenance in [`NOTICE`](NOTICE).
