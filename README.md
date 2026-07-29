# pi05-infer

**π0.5 动作专家(action expert)的独立 bs=1 推理引擎**,从 [RLinf](https://github.com/RLinf/RLinf)
里抽出来,针对 **RTX PRO 5000(GB202 / sm_120,Blackwell)** 做过一轮系统性优化。
**没有降精度、没有近似** —— 每一项都是代数等价变换,不量化、不换采样器、不减去噪步数。

> **延伸阅读** —— **[`opt.md`](opt.md)**:完整优化记录(三段脉络、逐项台账与脚注、
> ①/②/③ 各小节的为什么/怎么做/量了多少、正确性论证、测量方法学、踩过的坑与明确排除的做法);
> **[`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md)**:逐次 A/B 的原始测量数据存档;
> **[`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md)**:从 RLinf 抽取的边界与遗留项。

## 成果

端到端 `predict_action_batch`:**52.60 ms → 42.90 ms(−9.70 ms,−18.4 %)**,
基线是 `torch.compile max-autotune`。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger_dark.png">
  <img alt="优化台账三栏:仓库之前的前史(另一把尺)、52.60 到 42.90 ms 的端到端配对瀑布图、同样优化按去噪单步记的账" src="docs/ledger_light.png">
</picture>

图分三栏,而且**三栏用的是三把不同的尺,不能首尾相接**:

* **栏 1:这个仓库开始之前。** 测于**完整 RLinf worker + nsys** 的另一套口径。
  `torch.compile` 本身就值 4.1×(270.6 → 65.8 ms),是全程最大的单一杠杆 —— 但
  **58.9 → 52.60 的那 6 ms 是换尺子,不是优化**。
* **栏 2:本仓库的端到端台账。** 独立 bench 的 plain wall clock,每一行一次配对 A/B。
  52.60 → 42.90 ms,就是上面那个数。
* **栏 3:同样这些优化,按一个去噪步来记账。** GPU busy 2025.6 → 1185.0 µs/step,
  kernel 数 347 → 217 —— 这一栏说明栏 2 的那些毫秒是从哪里省出来的。

图里那两条虚线是参考实现的位置;均非配对测量,不作胜负判断,拆解见
[opt.md](opt.md#s-baselines)。

台账收尾之后又落地三项,**它们各自的绝对基线来自不同 session(有的锁频、有的不锁频),
所以没有拼进 52.60 → 42.90 那条配对链,也不在图里 —— 不要接到 42.90 上**:

| 优化 | 收益 | commit | 详见 |
|---|--:|---|---|
| 小 M 的 mm tile 候选(`down_proj` / `o_proj`) | **−0.88 ms/predict** | `ca4ae39` | [opt.md §3.1](opt.md#s-3-1) |
| 跳过 prefix LM 最后一层的死算 ⚠️ 条件安装 | **−1.11 ms/predict**(保守口径) | `72af442` | [opt.md](opt.md#s-after-ledger) |
| P·V 的 attention `bmm` 换 tile 长宽比 | **−0.18 ms/predict** | `ff237bf` | [opt.md §3.2b](opt.md#s-3-2b) |

## 一页纸总览

近期对 π0.5 单次推理做了两方面优化——**消除**(去掉 GPU 空转和白算的工作)和**融合**
(合并算子以减少访存往返)。在 RTX PRO 5000(GB202)上,bs=1、K=10 步、968 prefix token、
bf16,端到端延迟从 **52.60 ms 降到 42.90 ms,加速 18.4%**,数值对齐,不含任何量化或降精度,
也未改动采样步数等算法层设置。

**消除,共 −5.99 ms**

* **预计算 adaRMS 调制量**:因为去噪的时间表是固定的,那 37 个条件投影每步吃到的输入完全
  相同,所以可以整体预先算成一张表、按步索引取用,对应的核实例由 300 个降到 0。**−2.83 ms**
* **整个去噪步捕获成一张 CUDA 图**:因为 `torch.compile` 只包住被编译的子图,循环里还夹着
  70 个核/步的 eager 胶水由 Python 逐个发射,所以把整步(编译区加胶水)一次性录成一张图、
  每步只做一次 replay,GPU 空闲从 142 µs 降到 60 µs 而 GPU busy 不变。**−2.04 ms**
* **去掉每步重复的数据搬运**:因为 prefix KV 和 attention mask 在 10 步之间根本不变,却每步
  都被重新拼接、从 host 重新传入,所以改成常驻静态缓冲区并直接在 GPU 上构造。**−0.82 ms**
* **删除死代码**:因为那段 timestep 条件计算的结果下游从不消费,所以整段连同正弦编码和两个
  time-MLP GEMM 一并删掉。**−0.30 ms**

**融合,共 −3.17 ms**

* **Q/K/V 三个投影合并成一个 GEMM**:因为三者读的是同一份输入,分开做就要把它从显存读三遍,
  所以把权重并成一张 `[2560,1024]`,一次读取代替三次。**−2.12 ms**
* **SwiGLU 与 RoPE 分别融进 GEMM 尾部**:因为 gate 和 up 同样共用一份激活、且它们 4096 宽的
  中间结果算完立刻就被消费,不必落显存再读回;而 RoPE 需要在同一个 tile 内跨 `d` 与 `d+128`
  取数,是编译器的 tile 模型表达不了的——两者都只能手写。单步核数由 305 降到 238。
  **合计 −1.05 ms(作为一对测得)**

第三段「压 kernel」正在进行中,已落地部分见 [opt.md](opt.md)。

(上面这段里的"数值对齐"指**不降精度、不做近似**,和下面「数值一致性分两级」里的
"逐位一致"是两件不同的事,见 [§ 数值一致性](#r-numerics)。)

<a id="r-config"></a>

## 配置(所有数字都在这个配置下测得)

π0.5,batch 1,**K = 10** 步 Euler 去噪,**968 个 prefix token**
(3 路相机 × 256 patch + 200 个语言 token),action chunk 50,**全程 bf16**。
动作专家是 gemma_300m:18 层,d = 1024,mlp 4096,8 个 query head / 1 个 KV head(MQA),
head_dim 256,50 个 action token。
机器:RTX PRO 5000 72 GB Blackwell,GB202,sm_120,110 SM,1344 GB/s,**300 W 功耗墙**。
checkpoint `RLinf-Pi05-LIBERO-SFT`,torch 2.7.1+cu128,nsys 2026.1.2。

---

## 两条必须跟着数字一起读的限定

<a id="r-numerics"></a>

### 数值一致性分两级

**代数等价**和**在出货的编译路径上逐位相同**是两件事,本仓库分开写:

* **代数等价 —— 全部优化成立**,没有一项是"算错了"。
* **在 `max-autotune` 编译路径上逐位相同、且有强判据**(核级 / 张量级 / GEMM 级 / 同进程)
  —— 只有五项:设备端 att_masks、GEMM 尾部融合、small-M 重 tile、P·V 的 bmm 重 tile、
  以及"从 RLinf 剥离"这件事本身。另有三项判据偏弱但结论成立。
* **只在 eager 下逐位相同,编译路径上不成立** —— 预计算 adaRMS 调制量、Q/K/V 并成一个
  GEMM、prefix KV 静态缓冲区,三项在 `max-autotune` 下各自给出 **2.4–2.9e-3** 的动作差
  (≈ 动作幅度 1 %)。机制是**同一个代数式的两种写法被 inductor 编成了不同的核**
  (形状一变就换 tile、换 K 方向的 fp32 累加分块),即浮点舍入顺序不同,
  **不是数值错误,也不是精度损失**。

完整的证据表、FAIL 清单、判据分级与机制解释见 [opt.md § 正确性](opt.md#s-correctness)。

### prefix 跳最后一层是**条件安装**的

那 −1.11 ms 不是所有部署都能拿到。RLinf 的 `get_value_from_vlm(prefix_output)` 读的正是
被跳掉的那个 hidden state,所以 `install_skip_last_lm_layer()` 在检测到 VLM value head
(`value_after_vlm and add_value_head`)时**直接不安装**。实测已发布的 **19 份 pi0.5 PPO
配置里有 15 份命中这个条件**(→ 拿不到这 1.11 ms);另外 4 份以及 DSRL / SAC 那几个会安装。
`pi05-infer` 是纯推理包、没有 value head,所以这里默认开。
Kill switch:`RLINF_SKIP_LAST_LM_LAYER=0`。

---

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

`pi05-infer` 不碰 `site-packages`,只加一条 path entry,所以容器保持原状,可以作为 A/B
的参考臂。自定义算子注册在 `pi05_infer::` 命名空间下而不是 `rlinf::`,原因见
[opt.md](opt.md#s-opnamespace)。
`--stage1` 会把 `max-autotune` 改写成 `max-autotune-no-cudagraphs`,并在 warmup 之后断言图
真的捕获成功(否则会**静默**退回 eager loop)—— 原因见
[opt.md](opt.md#s-stage1)。

<a id="r-verify"></a>

## 验证:数值一致性

```bash
# 0. 隔离:expert 必须是 pi05_infer.gemma,PaliGemma prefix 必须是 transformers
python tools/isolation_check.py          # 打印 ISOLATION_OK

# 1. 核级:两个 Triton 融合核 / 两个 small-M GEMM / 两个 attention bmm / prefix KV
python tools/bitgate.py
python tools/bitexact_denoise_gemms.py
python tools/bitexact_denoise_bmms.py
python tools/bitexact_prefix_kv.py

# 2. 编译路径上的四项结构性优化(冻结 prefix + 四进程空对照门),一个 stage 一条命令
bash tools/run_bitexact_backfill.sh <stage>   # siglip|extraction|prefix|adarms|adarms_eager|qkv|kvstatic|attmask

# 3. 端到端的数值 A/B,固定 seed —— ⚠️ 必须带空对照,单独一次 dump 不作数
GATE_OFF="RLINF_SMALL_M_MM=0" GATE_ON="RLINF_SMALL_M_MM=1" \
  tools/bitexact_gate.sh /tmp/gate_small_m --stage1 --iters 1 --warmup 4
```

`bitexact_gate.sh` **跑四个进程**(每臂两次),只有两个同臂空对照都干净时才报告跨臂比较,
否则判 INCONCLUSIVE 而**绝不判 PASS**;四个进程共享一个 `TORCHINDUCTOR_CACHE_DIR`,
让两臂都没碰过的 shape 保持同一个 autotune 冠军。
参考臂对拍是 `tools/ab_rlinf_reference.py --dump-actions /tmp/ref.pt`。
每一项优化都带 kill switch,OFF 臂走的是被验证过的降级路径([opt.md](opt.md#s-fallback))。
`tools/` 下每个文件的作用见 [opt.md 附录](opt.md#s-inventory)。

---

<a id="r-layout"></a>

## 仓库结构

```
pi05_infer/       引擎本体:engine.py(纯推理编排 + 手写去噪 CUDA 图 + adaRMS 调制表)、
                  builder.py、dataconfig/、_vendored/、gemma/(动作专家的 Gemma fork
                  + 两个 Triton 融合核)、openpi_patched/、inductor_mm_tiles.py、
                  prefix_last_layer.py
bench/            standalone_infer_bench.py —— 延迟基准(e2e、分阶段、nsys、actions dump)
tools/            隔离检查、核级/GEMM 级/KV 级 bit-exact gate、四进程端到端 gate、
                  配对 A/B 驱动(部分支持锁频)、profile 分析
docs/             make_charts.py(重新生成本 README 的三张图)+ MEASUREMENTS.md
_extract_src/     抽取前的 RLinf 原始文件(未重构)
```

逐文件的清单见 [opt.md § 仓库文件清单](opt.md#s-inventory)。

`import pi05_infer` 只让**动作专家**走我们 vendoring 的 Gemma,PaliGemma 的 **prefix** 仍用
原厂 transformers;`tools/isolation_check.py` 逐模块断言这条边界。这不是形式主义 ——
一次 +4 ms 的回归就是因为两者共享同一份模型代码,见
[opt.md § prefix / expert 的隔离](opt.md#s-isolation)。
边界的完整说明见 [`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md) §8。

---

## 致谢与来源

本仓库 vendored 了以下 Apache-2.0 代码并做了标注过的修改:

| 组件 | 来源 |
|---|---|
| `pi05_infer/gemma/modeling_gemma.py` | HuggingFace Transformers(Copyright 2024 Google Inc. & HuggingFace Inc.) |
| `pi05_infer/openpi_patched/` | [openpi](https://github.com/Physical-Intelligence/openpi),经 [RLinf/openpi](https://github.com/RLinf/openpi) fork |
| `engine.py` / `builder.py` / `_vendored/` / `dataconfig/` / `bench/` | [RLinf](https://github.com/RLinf/RLinf) |
| `pi05_infer/gemma/rlinf_fused_denoise.py` | 本项目新写 |

逐文件的修改清单见 [`NOTICE`](NOTICE),许可证见 [`LICENSE`](LICENSE)(Apache-2.0)。
`dexmal/realtime-vla` 作为 peer 被引用与对比,未复用其代码。
