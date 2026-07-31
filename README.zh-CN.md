[English](README.md) | **简体中文**

# pi05-infer

**π0.5 动作专家(action expert)的独立 bs=1 推理引擎**,从
[RLinf](https://github.com/RLinf/RLinf) 里抽出来,针对
**RTX PRO 5000(GB202 / sm_120,Blackwell)** 做过一轮系统性优化。每一项都是代数等价变换 ——
**不量化、不换采样器、不减去噪步数**。

## 成果

端到端 `predict_action_batch`:**52.60 ms → 42.90 ms(−9.70 ms,−18.4 %)**,
基线是 `torch.compile max-autotune`。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger_dark.png">
  <img alt="优化台账三栏:仓库之前的前史(另一把尺)、52.60 到 42.90 ms 的端到端配对瀑布图、同样优化按去噪单步记的账" src="docs/ledger_light.png">
</picture>

三栏用的是**三把不同的尺,不能首尾相接**:本仓库开始之前的前史(另一套测量口径)、
本仓库的端到端配对台账(52.60 → 42.90 ms,就是上面那个数)、以及同样这些优化按一个去噪步
记的账(GPU busy 2025.6 → 1185.0 µs/step,347 → 217 kernels/step)。推导见
[opt.md § 台账](opt.md#s-ledger)、[§ 去噪单步](opt.md#s-per-step)。

台账收尾之后又落地五项。⚠️ **另一把尺,不要接到 42.90 上**(它们各自的绝对基线来自不同
session,有的锁频、有的不锁频,所以没有拼进那条配对链,也不在图里):

| 优化 | 收益 | commit | 详见 |
|---|--:|---|---|
| 小 `M` 的 mm tile 候选(`down_proj` / `o_proj`) | **−0.88 ms/predict** | `ca4ae39` | [opt.md §3.1](opt.md#s-3-1) |
| 跳过 prefix LM 最后一层的死算 —— ⚠️ **条件安装,见限定 3** | **−1.11 ms/predict** | `72af442` | [opt.md](opt.md#s-after-ledger) |
| P·V 的 attention `bmm` 换 tile 长宽比 | **−0.18 ms/predict** | `ff237bf` | [opt.md §3.2b](opt.md#s-3-2b) |
| 把步不变量从去噪循环里外提 | **−0.32 ± 0.05 ms/predict** | `0ed3ca2` | [opt.md](opt.md#s-hoist) |
| prefix LM 的 Q/K/V 三投影并成一个 GEMM | **−0.61 ± 0.22 ms/predict** | `d7cf3c2` | [opt.md](opt.md#s-prefix-qkv) |

<a id="r-config"></a>
**测量配置**:π0.5,batch 1,**K = 10** 步 Euler 去噪,**968 个 prefix token**,
action chunk 50,**全程 bf16**;RTX PRO 5000 72 GB(GB202,sm_120,110 SM,**300 W 功耗墙**),
checkpoint `RLinf-Pi05-LIBERO-SFT`,torch 2.7.1+cu128,nsys 2026.1.2
([完整配置](opt.md#s-roadmap))。

## 限定

1. **数值**:所有变换都是代数等价的,但**逐位一致性按编译路径分档** —— 有几项只在 eager 下
   逐位相同,在出货的 `max-autotune` 路径上不成立。逐项分档见
   [opt.md § 正确性](opt.md#s-correctness)。
2. **对标**:图里那两条虚线是参考实现的位置;**均非配对测量,不作胜负判断**
   ([opt.md § 与参考实现的对比](opt.md#s-baselines))。
3. **prefix 跳最后一层是条件安装的** —— 检测到 VLM value head 就不装,**已发布的 19 份
   pi0.5 PPO 配置里有 15 份命中这个条件**(拿不到这 −1.11 ms)。Kill switch
   `RLINF_SKIP_LAST_LM_LAYER=0`。

## 安装与运行

用现成的 RLinf benchmark 容器镜像,**不需要重建 Docker**。editable 安装,并且带
`--no-deps`,以免动到容器里钉死的 torch / transformers / openpi:

```bash
docker exec -w /path/to/pi05-infer pi05bench \
    /opt/venv/openpi/bin/pip install -e . --no-deps --no-build-isolation

# 基准测试
/opt/venv/openpi/bin/python bench/standalone_infer_bench.py \
    --model-path /path/to/RLinf-Pi05-LIBERO-SFT --config-name pi05_turtle --iters 30
... --stage1        # 开启手写的去噪 CUDA 图(opt-in;台账第 7 行之后的数字都开着它测)
... --phases        # 分阶段耗时
... --dump-actions /tmp/a.pt --clocks-json /tmp/clocks.json   # 数值 A/B + SM 时钟/功耗
```

`pi05-infer` 不碰 `site-packages`(只加一条 path entry),所以容器保持原状,可以作为 A/B
的参考臂。`--stage1` 会把 `max-autotune` 改写成 `max-autotune-no-cudagraphs`,并在 warmup
之后断言图真的捕获成功 —— 否则会**静默**退回 eager loop([opt.md](opt.md#s-stage1))。

<a id="r-verify"></a>

## 验证:数值一致性

```bash
python tools/isolation_check.py          # expert = pi05_infer.gemma,prefix = transformers

# 核级 / GEMM 级 / KV 级 bit-exact gate
python tools/bitgate.py                        # 两个 Triton 融合核
python tools/bitexact_denoise_gemms.py         # small-M mm 重 tile
python tools/bitexact_denoise_bmms.py          # P·V bmm 重 tile
python tools/bitexact_prefix_kv.py             # prefix 跳最后一层
python tools/bitexact_prefix_qkv.py            # prefix QKV 融合

# 编译路径上的结构性优化(冻结 prefix + 四进程空对照门),一个 stage 一条命令
bash tools/run_bitexact_backfill.sh <stage>    # siglip|extraction|prefix|adarms|adarms_eager|qkv|kvstatic|attmask

# 端到端数值 A/B,固定 seed —— ⚠️ 必须带空对照
GATE_OFF="RLINF_SMALL_M_MM=0" GATE_ON="RLINF_SMALL_M_MM=1" \
  tools/bitexact_gate.sh /tmp/gate_small_m --stage1 --iters 1 --warmup 4
```

`bitexact_gate.sh` 跑四个进程(每臂两次),只有两个同臂空对照都干净时才报告跨臂比较,
否则判 INCONCLUSIVE 而**绝不判 PASS**。每一项优化都带 kill switch,OFF 臂走的是被验证过的
降级路径([opt.md](opt.md#s-fallback))。

<a id="r-layout"></a>

## 仓库结构

```
pi05_infer/    引擎本体(engine.py、vendoring 的动作专家 Gemma + Triton 融合核、
               prefix_last_layer.py、prefix_qkv_fused.py、inductor_mm_tiles.py)
bench/         standalone_infer_bench.py —— 延迟基准
tools/         隔离检查、bit-exact gate、配对 A/B 驱动、profile 分析
docs/          make_charts.py(重新生成图)+ MEASUREMENTS.md
_extract_src/  抽取前的 RLinf 原始文件(未重构)
```

`import pi05_infer` 只让**动作专家**走我们 vendoring 的 Gemma,PaliGemma 的 **prefix** 仍用
原厂 transformers([opt.md § prefix / expert 的隔离](opt.md#s-isolation))。
逐文件清单见 [opt.md § 仓库文件清单](opt.md#s-inventory)。

## 延伸阅读

* **[`opt.md`](opt.md)** —— 完整优化记录:每项的为什么/怎么做/量了多少、正确性论证、
  测量方法学、踩过的坑与明确排除的做法。
* **[`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md)** —— 逐次 A/B 的原始测量数据存档。
* **[`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md)** —— 从 RLinf 抽取的边界与遗留项。

## 许可证与来源

Apache-2.0([`LICENSE`](LICENSE))。本仓库 vendored 了 HuggingFace Transformers、
[openpi](https://github.com/Physical-Intelligence/openpi)(经
[RLinf/openpi](https://github.com/RLinf/openpi) fork)与
[RLinf](https://github.com/RLinf/RLinf) 的代码,逐文件的修改清单见 [`NOTICE`](NOTICE)。
`dexmal/realtime-vla` 作为 peer 被引用,未复用其代码。
