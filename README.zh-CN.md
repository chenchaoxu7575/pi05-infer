[English](README.md) | **简体中文**

# pi05-infer

**π0.5 动作专家(action expert)的独立 bs=1 推理引擎**,从
[RLinf](https://github.com/RLinf/RLinf) 里抽出来,针对 **RTX PRO 5000(GB202 / sm_120,
Blackwell)** 做过一轮系统性优化。每一项都是代数等价变换 —— **不量化、不换采样器、不减去噪步数**。

## 成果

端到端 `predict_action_batch`:**52.60 ms → 42.90 ms(−9.70 ms,−18.4 %)**,
基线是 `torch.compile max-autotune`。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger_dark.png">
  <img alt="优化台账三栏:仓库之前的前史(另一把尺)、52.60 到 42.90 ms 的端到端配对瀑布图、同样优化按去噪单步记的账" src="docs/ledger_light.png">
</picture>

三栏用的是**三把不同的尺,不能首尾相接**:本仓库开始之前的前史(另一套测量口径)、
本仓库的端到端配对台账(52.60 → 42.90 ms,就是上面那个数)、以及同样这些优化按一个去噪步
记的账(GPU busy 2025.6 → 1185.0 µs/step,347 → 217 kernels/step)。

图里那两条虚线是参考实现的位置 —— **均非配对测量,不作胜负判断**。

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
之后断言图真的捕获成功 —— 否则会**静默**退回 eager loop。

<a id="r-verify"></a>

## 验证:数值一致性

```bash
python tools/isolation_check.py          # expert = pi05_infer.gemma,prefix = transformers

# 核级 / GEMM 级 / KV 级 bit-exact gate
python tools/bitgate.py                  # 两个 Triton 融合核
python tools/bitexact_denoise_gemms.py   # small-M mm 重 tile
python tools/bitexact_denoise_bmms.py    # P·V bmm 重 tile
python tools/bitexact_prefix_kv.py       # prefix 跳最后一层
python tools/bitexact_prefix_qkv.py      # prefix QKV 融合

# 编译路径上的结构性优化(冻结 prefix + 四进程空对照门),一个 stage 一条命令
bash tools/run_bitexact_backfill.sh <stage>   # siglip|extraction|prefix|adarms|adarms_eager|qkv|kvstatic|attmask

# 端到端数值 A/B —— ⚠️ 四进程,必须带空对照;两个同臂对照不干净就判 INCONCLUSIVE,绝不判 PASS
GATE_OFF="RLINF_SMALL_M_MM=0" GATE_ON="RLINF_SMALL_M_MM=1" \
  tools/bitexact_gate.sh /tmp/gate_small_m --stage1 --iters 1 --warmup 4
```

每一项优化都带 kill switch,OFF 臂走的是被验证过的降级路径。

<a id="r-layout"></a>

## 仓库结构

```
pi05_infer/    引擎本体(engine.py、vendoring 的动作专家 Gemma + Triton 融合核、
               prefix_last_layer.py、prefix_qkv_fused.py、inductor_mm_tiles.py)
bench/         standalone_infer_bench.py —— 延迟基准
tools/         隔离检查、bit-exact gate、配对 A/B 驱动、profile 分析
_extract_src/  抽取前的 RLinf 原始文件(未重构)
```

`import pi05_infer` 只让**动作专家**走我们 vendoring 的 Gemma,PaliGemma 的 **prefix** 仍用
原厂 transformers。

## 延伸阅读

> **详细的优化记录尚未发布。** 逐项推导、正确性论证、测量方法学与原始 A/B 存档
> 暂存内部,等这部分工作收敛后再一并发布。

* **[`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md)** —— 从 RLinf 抽取的边界与遗留项。

## 许可证与来源

Apache-2.0([`LICENSE`](LICENSE))。本仓库 vendored 了 HuggingFace Transformers、
[openpi](https://github.com/Physical-Intelligence/openpi)(经
[RLinf/openpi](https://github.com/RLinf/openpi) fork)与
[RLinf](https://github.com/RLinf/RLinf) 的代码,逐文件的修改清单见 [`NOTICE`](NOTICE)。
`dexmal/realtime-vla` 作为 peer 被引用,未复用其代码。
