[English](README.md) | **简体中文**

# pi05-infer

**pi0.5 动作专家的独立 bs=1 推理引擎**,从 [RLinf](https://github.com/RLinf/RLinf)
抽出,针对 **RTX PRO 5000(GB202 / sm_120)** 优化。每一项改动都是代数等价变换:
不量化、不换采样器、不减去噪步数。

## 成果

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger_dark.png">
  <img alt="配对 A/B 测出的加速,按三类拆开:CPU 开销、去噪步冗余削减、kernel 融合与调优" src="docs/ledger_light.png">
</picture>

当前 `main`,不锁频 plain wall clock,n=30:**40.63 ms**(p50;mean 40.56,
39.40 .. 41.47,SM 时钟采样 2235-2265 MHz)。

> 台账、台账之后的两项 tile 改动、以及上面这个数,用的是三把不同的尺,不能相加。
> 只有最后一行描述的是你实际跑起来的 `main`。

<details>
<summary>一个去噪步花在哪里</summary>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/denoise_dark.png">
  <img alt="单个去噪步的逐核分解" src="docs/denoise_light.png">
</picture>

图上标了它代表的 commit,没有对当前 `main` 重新推导。逐核的当前快照在
[`docs/kernels/`](docs/kernels/)。968 token 的 prefix 占 GPU busy 的 71.7%,
所以哪怕去噪循环免费,整个 predict 也只能快 1.39x。

</details>

## 硬件

所有数字、调优与验证都只在**一张卡**上做过:RTX PRO 5000 Blackwell(sm_120),
110 SM,96 MB L2,300 W 功耗墙。换任何别的 GPU,性能数字和 bit-exactness digest
都不成立 -- "bit-identical" 是相对**这张卡**未打补丁的构建定义的,而在别的卡上
那个参照本身就是另一个 kernel。

不拦你:引擎会警告一次,硬件相关的 tile pin 在非 sm_120 上自己拒绝安装,
大部分优化带 `RLINF_*=0` kill switch。

## 安装

`pyproject.toml` 故意写 `dependencies = []` -- 本包用 `--no-deps` 装进一个已有的
**openpi** 环境,那里的 torch 构建不能被动。该环境需要提供:

| | |
|---|---|
| Python | `>=3.10,<3.12` |
| PyTorch | **2.7.1+cu128**,上面的数字就是在这个构建上测的。任何构建都行,只要面向 sm_120;`+cu124` 的 wheel 只编到 sm_50..sm_90,在 GB202 上跑不起来。 |
| openpi | 必须**带 `transformers_replace` 补丁**安装(openpi 自己的 `install.sh` 会做)。prefix 跑在原厂 transformers 上并会被传 `adarms_cond=`,而未打补丁的 transformers 不接受这个参数。 |
| transformers | 4.53.2,openpi 打过补丁的那份 |
| 还会 import | `numpy`、`einops`、`nvtx` |

RLinf 的 benchmark 容器里这些已经齐了。

```bash
pip install -e . --no-deps --no-build-isolation

huggingface-cli download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir /path/to/ckpt
export PI05_MODEL_PATH=/path/to/ckpt
```

checkpoint 目录需要权重,加上放在 asset id 下的 openpi 归一化统计量:

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

其余 gate 需要一个臂一个进程,或者固定的 stage 顺序。
**准确的调用方式在 [`tools/README.md`](tools/README.md) 里** -- 照着跑;
当成一行命令跑,它们只会打出一个和谁都不比的 digest,然后正常退出。

> 位一致性是按编译路径分层的,并没有被无条件声明:有些项在 eager 下逐位相同,
> 在实际发布用的 `max-autotune` 下并不 -- 而它自己的 kernel 选择在冷 autotune
> 之间就不稳定。对每一项都无条件成立的说法是:变换是代数等价的。
> `pi05_infer/patches/inductor_mm_tiles.py` 里那条说明记录了一条本项目
> **曾经相信、已经发布、后来被实测推翻**的规则。

## 仓库结构

```
pi05_infer/    引擎本体
  gemma/       vendoring 并改过的动作专家 Gemma + Triton 融合核
  patches/     打在"我们不拥有的代码"上的优化 -- 全部可关
bench/         延迟基准
tools/         验证 gate 与测量驱动
docs/          那两张图,以及重新生成它们的 make_charts.py
  kernels/     去噪步的逐核快照,一次测量一个文件
_extract_src/  抽取前的 RLinf 原始文件,留着是为了整个抽取可以用 diff 审计
```

`import pi05_infer` 只让**动作专家**走 vendoring 的 Gemma,PaliGemma 的 **prefix**
仍用原厂 transformers。这条缝就是"去噪核的改动够不着 968 token 的 prefix"的保证。

## 延伸阅读

详细的优化记录尚未发布。源码里凡是按名字引用那些文档的地方,那个名字是出处标注,不是链接。

* [`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md) -- 从 RLinf 抽取的边界与遗留项。
* [`tools/README.md`](tools/README.md) -- 哪些脚本可移植,哪些不可。

## 许可证

Apache-2.0([`LICENSE`](LICENSE))。本仓库 vendored 了 HuggingFace Transformers、
[openpi](https://github.com/Physical-Intelligence/openpi)(经
[RLinf/openpi](https://github.com/RLinf/openpi) fork)与
[RLinf](https://github.com/RLinf/RLinf) 的代码,逐文件来源见 [`NOTICE`](NOTICE)。
