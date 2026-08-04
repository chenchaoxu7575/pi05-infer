[English](README.md) | **简体中文**

# pi05-infer

**pi0.5 动作专家的独立 bs=1 推理引擎**,从 [RLinf](https://github.com/RLinf/RLinf)
抽出,针对 **NVIDIA RTX PRO 5000** 优化。每一项改动都代数等价:
不量化、不换采样器、不减去噪步数。等价性的参照是
[openpi](https://github.com/RLinf/openpi) 的 PyTorch pi0.5,即本仓库抽取自的基线,
而非 JAX 参考实现。

## 成果

出厂构建每次 predict 的纯模型推理耗时 **39.64 ms**。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger_dark.png">
  <img alt="出厂配置(非锁频)下每次 predict 的纯模型推理时间瀑布图。torch.compile max-autotune 基线 51.57 ms,依次减去 CPU 开销 3.41 ms、去噪步冗余削减 5.26 ms、kernel 融合与调优 3.26 ms,终点 39.64 ms。上方为 eager 124.71 ms 到 torch.compile max-autotune 51.57 ms。底部一行给出每个 arm 自己的 SM 时钟与功耗:base 2437 MHz 295 W、c1 2362 MHz 301 W、c2 2317 MHz 301 W、c3 2220 MHz 301 W,eager 2325 MHz 172 W 从未接近功耗上限。" src="docs/ledger_light.png">
</picture>

```
bs=1, 3 cameras -> 224^2 model input, 968 prefix tokens, action chunk 50, 10 denoise steps, bf16
NVIDIA RTX PRO 5000 Blackwell (sm_120), stock 300 W cap, unlocked clock
```

* 计时区间是 `sample_actions` —— prefix + 去噪。预处理与 output transform 在区间之外
  (`--model-only`)。
* prefix 末层跳过在存在 VLM value head 时拒绝安装;本次测量不带 value head。
* 同一条链锁频重测(每个 arm 跑在同一时钟):[`docs/locked_clock.md`](docs/locked_clock.md)。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/phases_dark.png">
  <img alt="一条堆叠条形图,把 42.88 ms 的纯模型推理 GPU 时间拆为 SigLIP 视觉 5.93 ms(13.8%,算力受限,54-57% 峰值 FLOP/s)、968 token 的 PaliGemma LM prefix 24.44 ms(57.0%,算力受限,75-79% 峰值 FLOP/s)、十步去噪 12.39 ms(28.9%,内存受限,46% 峰值 DRAM 带宽)。" src="docs/phases_light.png">
</picture>

```
nsys kernel time, clock locked 1897 MHz -- GPU time, not the wall clock above
roofline knee 169 FLOP/byte (206.2 TFLOP/s bf16, 1222 GB/s DRAM read, both at 1897 MHz)
```

<details>
<summary>同样三类,落在单个去噪步上</summary>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/denoise_dark.png">
  <img alt="单个去噪步 GPU 时间的瀑布图,同样分三类,2477.56 us 降到 1237.83 us:CPU 开销 -226 us、去噪步冗余削减 -460 us、kernel 融合与调优 -554 us。" src="docs/denoise_light.png">
</picture>

出厂构建的逐 kernel 耗时分解:[`docs/kernels/`](docs/kernels/)。

</details>

## 硬件

```
GPU            NVIDIA RTX PRO 5000 Blackwell (GB202 / sm_120), 110 SMs, 96 MB L2, 300 W
PyTorch        2.7.1+cu128
transformers   4.53.2
Python         3.11
```

## 安装

```bash
# 1. openpi(RLinf fork)及其 transformers_replace 补丁
uv pip install git+https://github.com/RLinf/openpi
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -r "$SITE/openpi/models_pytorch/transformers_replace/"* "$SITE/transformers/"

# 2. 装本包
git clone https://github.com/chenchaoxu7575/pi05-infer
cd pi05-infer
pip install -e . --no-deps --no-build-isolation

# 3. 一个 checkpoint
hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir /path/to/ckpt
export PI05_MODEL_PATH=/path/to/ckpt
```

```
ckpt/
  model.safetensors
  physical-intelligence/libero/norm_stats.json
```

## 运行

```bash
python bench/standalone_infer_bench.py --config-name pi05_turtle \
    --warmup 20 --iters 30 --model-only --stage1

... --no-compile
... --phases
... --dump-actions /tmp/a.pt --clocks-json /tmp/clocks.json
```

## 验证

```bash
export PI05_MODEL_PATH=/path/to/ckpt
export CUDA_VISIBLE_DEVICES=0

python tools/isolation_check.py
python tools/bitgate.py
python tools/bitexact_prefix_qkv.py
```

其余的在 [`tools/README.md`](tools/README.md)。

## 仓库结构

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

## 延伸阅读

* [`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md):从 RLinf 抽取的边界。

## 许可证

Apache-2.0([`LICENSE`](LICENSE))。vendored 的第三方代码及其逐文件来源,见
[`NOTICE`](NOTICE)。
