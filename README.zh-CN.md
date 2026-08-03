[English](README.md) | **简体中文**

# pi05-infer

**pi0.5 动作专家的独立 bs=1 推理引擎**,从 [RLinf](https://github.com/RLinf/RLinf)
抽出,针对 **RTX PRO 5000(GB202 / sm_120)** 优化。每一项改动都代数等价:
不量化、不换采样器、不减去噪步数。

## 成果

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger_dark.png">
  <img alt="配对 A/B 测出的加速,按三类拆开:CPU 开销、去噪步冗余削减、kernel 融合与调优" src="docs/ledger_light.png">
</picture>

当前 `main`:**40.50 ms**,不锁频 plain wall clock,n=30,p50。

<details>
<summary>一个去噪步花在哪里</summary>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/denoise_dark.png">
  <img alt="单个去噪步的逐核分解" src="docs/denoise_light.png">
</picture>

当前构建的逐核快照:[`docs/kernels/`](docs/kernels/)。
968 token 的 prefix 占 GPU busy 的 71.7%,所以哪怕去噪循环免费,
整个 predict 也只能快 1.39x。

</details>

## 硬件

```
GPU            RTX PRO 5000 Blackwell (GB202 / sm_120), 110 SM, 96 MB L2, 300 W
PyTorch        2.7.1+cu128
transformers   4.53.2, openpi 打过补丁的那份
Python         3.11
```

时延数字与 bit-exactness digest 都只对这张卡成立。

## 安装

```bash
# 1. 一个打了 transformers_replace 补丁的 openpi 环境
#    (openpi 自带的 install.sh 会做)

# 2. 装本包,--no-deps 以免动到环境里的 torch
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

每一项优化都有一个 gate,跑两个臂并比对字节级 digest。

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
  patches/     打在"我们不拥有的代码"上的优化 -- 全部可关
bench/         延迟基准
tools/         验证 gate 与测量驱动
docs/          那两张图,以及重新生成它们的 make_charts.py
  kernels/     去噪步的逐核快照
_extract_src/  抽取前的 RLinf 原始文件,留着让整个抽取可以用 diff 审计
```

## 延伸阅读

* [`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md) -- 从 RLinf 抽取的边界与遗留项。
* [`tools/README.md`](tools/README.md) -- 哪些脚本可移植,哪些不可。

详细的优化记录尚未发布。

## 许可证

Apache-2.0([`LICENSE`](LICENSE))。本仓库 vendored 了 HuggingFace Transformers、
[openpi](https://github.com/Physical-Intelligence/openpi) 与
[RLinf](https://github.com/RLinf/RLinf) 的代码,逐文件来源见 [`NOTICE`](NOTICE)。
