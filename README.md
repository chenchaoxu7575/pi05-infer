**English** | [简体中文](README.zh-CN.md)

# pi05-infer

A standalone **bs=1 inference engine for the pi0.5 action expert**, extracted from
[RLinf](https://github.com/RLinf/RLinf) and optimized for the **NVIDIA RTX PRO 5000**.
Every change is algebraically equivalent: no quantization, no
change of sampler, no reduction in denoising steps. Equivalence is against
[openpi](https://github.com/RLinf/openpi)'s PyTorch pi0.5, which is the baseline this
was extracted from -- not against the JAX reference implementation.

## Results

**39.64 ms** of model inference per predict on the shipping build.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger_dark.png">
  <img alt="Waterfall of model-inference time per predict on a stock, unlocked card. The torch.compile max-autotune baseline of 51.57 ms drops 3.41 ms of CPU overhead, then 5.26 ms of denoise-step work removed, then 3.26 ms of kernel fusion and optimization, ending at 39.64 ms. Above the waterfall, eager 124.71 ms to torch.compile max-autotune 51.57 ms. A footer gives each arm its own SM clock and power: base 2437 MHz 295 W, c1 2362 MHz 301 W, c2 2317 MHz 301 W, c3 2220 MHz 301 W, with eager at 2325 MHz 172 W, never near the cap." src="docs/ledger_light.png">
</picture>

```
bs=1, 3 cameras -> 224^2 model input, 968 prefix tokens, action chunk 50, 10 denoise steps, bf16
NVIDIA RTX PRO 5000 Blackwell (sm_120), stock 300 W cap, unlocked clock
```

* Timed span is `sample_actions` -- prefix + denoise. Preprocessing and the output
  transform are outside it (`--model-only`).
* The prefix last-layer skip declines to install when a VLM value head is present;
  measured without one.
* Same chain with the clock pinned, so every arm runs at one speed:
  [`docs/locked_clock.md`](docs/locked_clock.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/phases_dark.png">
  <img alt="One stacked bar splitting 42.88 ms of model-inference GPU time into SigLIP vision 5.93 ms (13.8%, compute-bound, 54-57% of peak FLOP/s), the PaliGemma LM prefix over 968 tokens 24.44 ms (57.0%, compute-bound, 75-79% of peak FLOP/s), and ten denoise steps 12.39 ms (28.9%, memory-bound, 46% of peak DRAM bandwidth)." src="docs/phases_light.png">
</picture>

```
nsys kernel time, clock locked 1897 MHz -- GPU time, not the wall clock above
roofline knee 169 FLOP/byte (206.2 TFLOP/s bf16, 1222 GB/s DRAM read, both at 1897 MHz)
```

<details>
<summary>The same three categories, on one denoise step</summary>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/denoise_dark.png">
  <img alt="Waterfall of GPU time per denoise step in the same three categories, 2477.56 us down to 1237.83 us: CPU overhead -226 us, denoise-step work removed -460 us, kernel fusion and optimization -554 us." src="docs/denoise_light.png">
</picture>

Per-kernel cost breakdown of the shipping build: [`docs/kernels/`](docs/kernels/).

</details>

## Hardware

```
GPU            NVIDIA RTX PRO 5000 Blackwell (GB202 / sm_120), 110 SMs, 96 MB L2, 300 W
PyTorch        2.7.1+cu128
transformers   4.53.2
Python         3.11
```

## Install

```bash
# 1. openpi (the RLinf fork) and its transformers_replace patch
uv pip install git+https://github.com/RLinf/openpi
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -r "$SITE/openpi/models_pytorch/transformers_replace/"* "$SITE/transformers/"

# 2. this package
git clone https://github.com/chenchaoxu7575/pi05-infer
cd pi05-infer
pip install -e . --no-deps --no-build-isolation

# 3. a checkpoint
hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir /path/to/ckpt
export PI05_MODEL_PATH=/path/to/ckpt
```

```
ckpt/
  model.safetensors
  physical-intelligence/libero/norm_stats.json
```

## Run

```bash
python bench/standalone_infer_bench.py --config-name pi05_turtle \
    --warmup 20 --iters 30 --model-only --stage1

... --no-compile
... --phases
... --dump-actions /tmp/a.pt --clocks-json /tmp/clocks.json
```

## Verify

```bash
export PI05_MODEL_PATH=/path/to/ckpt
export CUDA_VISIBLE_DEVICES=0

python tools/isolation_check.py
python tools/bitgate.py
python tools/bitexact_prefix_qkv.py
```

The rest are in [`tools/README.md`](tools/README.md).

## Layout

```
pi05_infer/
  _vendored/
  dataconfig/
  gemma/
  openpi_patched/
  patches/
  builder.py
  engine.py
bench/
tools/
docs/
  kernels/
  make_charts.py
_extract_src/
```

## Further Reading

* [`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md): the extraction boundary against RLinf.

## License

Apache-2.0 ([`LICENSE`](LICENSE)). Vendored third-party code, and its per-file
provenance, are listed in [`NOTICE`](NOTICE).
