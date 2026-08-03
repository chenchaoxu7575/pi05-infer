[English](README.md) | **简体中文**

# pi05-infer

**pi0.5 动作专家的独立 bs=1 推理引擎**,从 [RLinf](https://github.com/RLinf/RLinf)
抽出,针对 **NVIDIA RTX PRO 5000(GB202 / sm_120)** 优化。每一项改动都代数等价:
不量化、不换采样器、不减去噪步数。

## 成果

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger_dark.png">
  <img alt="52.60 到 42.90 ms,分三类:CPU 开销 -3.44、去噪步冗余削减 -3.09、kernel 融合与调优 -3.17" src="docs/ledger_light.png">
</picture>

```
bs=1, 3 x 224^2 cameras, 968 prefix tokens, action chunk 50, 10 denoise steps, bf16
NVIDIA RTX PRO 5000 Blackwell (sm_120), 300 W, unlocked clock
```

<details>
<summary>一个去噪步花在哪里</summary>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/denoise_dark.png">
  <img alt="单个去噪步的逐核分解" src="docs/denoise_light.png">
</picture>

逐核快照:[`docs/kernels/`](docs/kernels/)。

</details>

## 硬件

```
GPU            NVIDIA RTX PRO 5000 Blackwell (GB202 / sm_120), 110 SMs, 96 MB L2, 300 W
PyTorch        2.7.1+cu128
transformers   4.53.2, openpi 打过补丁的那份
Python         3.11
```

## 安装

```bash
# 1. 一个打了 transformers_replace 补丁的 openpi 环境
#    (openpi 自带的 install.sh 会做)

# 2. 装本包
pip install -e . --no-deps --no-build-isolation

# 3. 一个 checkpoint
huggingface-cli download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir /path/to/ckpt
export PI05_MODEL_PATH=/path/to/ckpt
```

```
ckpt/
  model.safetensors
  physical-intelligence/libero/norm_stats.json
```

## 运行

```bash
python bench/standalone_infer_bench.py --config-name pi05_turtle --iters 30
... --stage1        # 手写的去噪 CUDA 图(opt-in)
... --phases        # 分阶段耗时
... --dump-actions /tmp/a.pt --clocks-json /tmp/clocks.json
```

## 验证

```bash
export PI05_MODEL_PATH=/path/to/ckpt
export CUDA_VISIBLE_DEVICES=0

python tools/isolation_check.py     # expert = pi05_infer.gemma,prefix = transformers
python tools/bitgate.py             # 两个 Triton 融合核
python tools/bitexact_prefix_qkv.py # prefix QKV 融合
```

其余的在 [`tools/README.md`](tools/README.md)。

## 仓库结构

```
pi05_infer/    引擎本体
  gemma/       vendoring 并改过的动作专家 Gemma + Triton 融合核
  patches/     运行时打在第三方代码上的优化
bench/         延迟基准
tools/         验证 gate 与测量驱动
docs/          那两张图,以及重新生成它们的 make_charts.py
  kernels/     去噪步的逐核快照
_extract_src/  抽取前的 RLinf 原始文件
```

## 延伸阅读

* [`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md):从 RLinf 抽取的边界。
* [`tools/README.md`](tools/README.md):哪些脚本可移植,哪些不可。

## 许可证

Apache-2.0([`LICENSE`](LICENSE))。本仓库 vendored 了 HuggingFace Transformers、
[openpi](https://github.com/Physical-Intelligence/openpi) 与
[RLinf](https://github.com/RLinf/RLinf) 的代码,逐文件来源见 [`NOTICE`](NOTICE)。
