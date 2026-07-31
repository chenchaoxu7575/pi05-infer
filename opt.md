# opt.md —— π0.5 单次推理优化的完整记录

这份文档是 [`README.md`](README.md) 的**细节篇**:README 只放结论和成果,这里放
**每一项优化为什么这么做、怎么做、量了多少、踩了什么坑、哪些做法被证伪**。

> **和 [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md) 的分工**:那份是**原始测量数据存档**
> (每次 A/B 的逐轮读数、kernel census、profile 窗口),这份是**叙事**。该引用的地方直接
> 引用,不复制。

**导航**

| 段落 | 内容 |
|---|---|
| [优化脉络:三段](#s-roadmap) | 消除 / 融合 / 压 kernel 的分工与优先级 |
| [台账之前:前史](#s-prehistory) | E-series、`torch.compile` 的 4.1×、换尺子的那 6 ms |
| [优化台账](#s-ledger) | 52.60 → 42.90 ms 的配对链、逐行脚注 |
| [台账之后](#s-after-ledger) | small-M mm tile、prefix 跳层、P·V bmm、[步不变量外提](#s-hoist)、[prefix QKV 融合](#s-prefix-qkv) |
| [命名对照](#s-naming) | 台账里的说法 ↔ 代码/开关/profile 里的简写 |
| [去噪单步的台账](#s-per-step) | 换纵轴:µs/step 与 kernels/step |
| [① 消除](#s-eliminate) | §1.1–1.5 |
| [② 融合](#s-fuse) | §2.1–2.5 |
| [③ 压 kernel](#s-kernel) | §3.1–3.4 |
| [天花板在哪](#s-ceiling) | prefix 占 GPU busy 的 71.7 % |
| [与参考实现的对比](#s-baselines) | realtime-vla / FluxVLA 的完整拆解 |
| [正确性](#s-correctness) | 判据分级、PASS/FAIL 完整清单、机制 |
| [测量方法学](#s-methodology) | 7 条规矩 + `torch.empty` 零权重陷阱 |
| [已知限制 / 没做的事](#s-limits) | 剩下的坑 |

**测试配置(所有数字都在这个配置下测得)**,另见
[README 的「配置」一节](README.md#r-config):π0.5,bs=1,K=10 步 Euler 去噪,
**968 个 prefix token**(3 路相机 × 256 patch + 200 个语言 token),action chunk 50,
全程 bf16。动作专家是 gemma_300m:**18 层,d = 1024,mlp 4096,8 个 query head /
1 个 KV head(MQA),head_dim 256,50 个 action token**。机器 RTX PRO 5000 72 GB
(GB202 / sm_120,110 SM,1344 GB/s,300 W 功耗墙);checkpoint `RLinf-Pi05-LIBERO-SFT`,
torch 2.7.1+cu128,nsys 2026.1.2。

---

<a id="s-roadmap"></a>

## 优化脉络:三段

一个去噪步的墙钟时间可以拆成三块:**GPU 空转的时间**、**多余的 kernel 数**、
**每个 kernel 自己跑多久**。这三块的性价比差了一个数量级,所以工作是按这个顺序推进的:

| | 这一段做什么 | 主要手段 | 兑现 |
|---|---|---|--:|
| **① 消除** | 让 GPU 别空转,也别做本来就不该做的事 | 预计算 adaRMS 调制量、把整个去噪步捕获成一张 CUDA 图、prefix KV 常驻静态缓冲区、在 GPU 上构造 attention mask、删掉没人读的 timestep 条件计算、**跳过 prefix LM 最后一层的死算** | 台账内 **−5.99 ms**,另加 **−1.11 ms**(prefix 跳层,另一把尺) |
| **② 融合** | 该做的事,用更少的 kernel 做完 | Q/K/V 三个投影并成一个 GEMM、把 SwiGLU 与 RoPE 融进 GEMM 尾部 | **−3.17 ms** |
| **③ 压 kernel** | 剩下的 kernel,让每一个自己跑得更快 | 给饿死的 GEMM 补小 `BLOCK_M` 候选、给 P·V 的 `bmm` 换 tile 长宽比 | **−0.88 ms** + **−0.18 ms** |

(台账内前两段相加 −9.16 ms;余下的 −0.58 ms 是「剥离成独立包 + 打开去噪图」那一步,
见台账第 7 行。第 ③ 段的 −0.88 / −0.18 与 prefix 跳层的 −1.11 在台账之后,基线各自不同,
见「台账之后」一节。)

顺序不是随手排的:**"消除"优于"融合"优于"调优"**。前两段的三个主力项(预计算调制量、
合并 GEMM、消拷贝)没有一行手写 Triton;第三段则更贵 —— 不过 2026-07-28 发现它的**前两笔
收益都不需要手写 kernel**:只要把 inductor 的候选集补上它自己没有的 tile,
让它自己 benchmark 选,就拿到了 −0.88 ms(§3.1)和 −0.18 ms(§3.2b)。
真正要手写 Triton 的那部分仍然没开始。

<a id="s-prehistory"></a>

## 台账之前:前史(2026-07-03,**另一把尺**)

52.60 ms 不是"完全没优化"的起点。它之前还有一段,但那一段是在**完整 RLinf Ray + EnvWorker
链路 + nsys** 下测的,机器也是另一台(`10.172.160.142`,4× RTX PRO 5000),e2e 的定义是
"predict 各段 CPU 墙钟之和"。**它和下面的台账不是同一把尺,不能相减、不能接成一条链。**

| 配置 | e2e | vs 只编译 | 笔记里的代号 |
|---|--:|--:|:--:|
| 完全不编译(eager 逐算子执行) | 270.6 ms | — | `opt_eager` |
| **编译整图:`torch.compile` max-autotune** | **65.8 ms** | 基线 | `opt_E0` |
| 关掉推理期的类型检查(单独开) | 63.0 | −2.8 | `opt_E1` |
| 三路相机合成一次视觉编码(单独开) | 61.7 | −4.1 | `opt_E2` |
| 上面两项同时开 | **58.9** | **−6.9(−10.5 %)** | `opt_E3` |

**编译器本身值 4.1×**(270.6 → 65.8),是全程最大的单一杠杆,而且它不是这个仓库做的 ——
所以本仓库把「只开 `torch.compile`」当成真正的起点来看待,而不是拿 270.6 去算加速比。

⚠️ **中间那两行是各自单独开的,不是累加链。**「三路相机合成一次视觉编码」那一行的类型
检查仍然是**开**的,所以它的 −4.1 是"相对只编译",不是"在 63.0 上再减 4.1"。两项正交可叠,
合起来才是最后一行的 −6.9。

⚠️ **58.9 → 52.60 的那 ~6 ms 不是优化,是换尺子。** 上游记录写得很直白:"纯推理基线(同上
配置)53.1 ms —— 换测法:去掉 Ray/worker 口径约 6ms",并且专门警告"**两种测法别相减**"。
两边跑的是**同一份 compile 配置**(max-autotune / #968);独立推理脚本本来就默认内置了
关类型检查与相机合批,所以 **52.60 就是最后一行那个配置在另一把尺上重测**的结果。
同期在 pro5k 上留下的读数是 52.1(plain)/ 53.3(nsys)/ 53.12 / 52.60,彼此的散布落在
本机 ±0.7 ms 的 rebuild variance 之内。这条换尺从来没有做过配对 A/B —— 那台 `.142` 后来
就没再连上,所以那 6 ms 是**归因的,不是测出来的**。

⚠️ **「三路相机合成一次视觉编码」不是逐位一致的优化 —— 这条以前记错了。** 它把 3 个视角
`torch.cat` 成 batch=3 过一次 ViT。**数学上是恒等变换**(SigLIP 里 LayerNorm / GELU /
residual / 每个 Linear 都在 batch 维上独立,attention 只在 token 维内做,**不跨 batch**),
但**比特上不是**:合批把每个 GEMM 的 M 从 256 抬到 768、把每个 LayerNorm 归约的 `xnumel`
从 256 抬到 768,cuBLAS 选核和 inductor 的 tile / split-k / `R0_BLOCK` 都是形状的函数。
2026-07-28 的**同进程**三级 A/B(ViT 输出 / 18 层 prefix KV / 动作,每臂各跑两次做空对照,
6 个空对照全部逐位相同)测到:ViT 输出平均差 **~15 bf16 ULP**,动作
`max|Δ|` **4.582e-03**(eager)/ **2.528e-03**(编译)。历史记录里那个 4.9e-3 **是真的,
不是门的噪声** —— 当时"归因为 cuBLAS 选核噪声"这一步方向没错(确实是选核 / 累加顺序),
但由此推出"所以可以忽略、仍算 bit-exact"这一步不成立。
两个排除接线 bug 的对照:故意错配 view 的对照差 **186 ULP**(大一个数量级);
逐层剖面从 layer0 的 **2.04 ULP** 单调涨到 layer26 的 **18.60 ULP**,是典型的 bf16 舍入
放大形态(接线 / 别名 bug 会在第 0 层就满量程)。
这项改动**在后续所有 A/B 的两臂里都在**,所以台账里每一行的 Δ 都不受它影响。

<a id="s-ledger"></a>

## 优化台账

e2e = `predict_action_batch`,plain wall clock(**不是** nsys 的 wall time),
30 iterations after 8 warmup,串行,单任务独占 GPU。

| # | 优化 | 段 / 详见 | e2e | Δ | 逐位一致(判据强度) |
|---|---|:--:|--:|--:|:--:|
| 0 | baseline(`torch.compile max-autotune`) | — | 52.60 ms | — | — |
| 1 | 预计算 adaRMS 调制量 | ① §1.1 | 49.77 | **−2.83** | **eager ✅ / 编译 ✗**(2.568e-3) |
| 2 | 把整个去噪步捕获成一张 CUDA 图 | ① §1.2 | 47.73 | −2.04 | ✅ 端到端 ◇ |
| 3 | Q/K/V 三个投影并成一个 GEMM | ② §2.1 | 45.61 | **−2.12** | **eager ✅ / 编译 ✗**(2.431e-3) |
| 4 | prefix KV 常驻静态缓冲区 | ① §1.3 | 45.10 | −0.51 | **eager ✅ / 编译 ✗**(2.858e-3) |
| 5 | 在 GPU 上构造 attention mask | ① §1.4 | 44.79 | −0.31 | ✅ 编译(张量级,bs=1) |
| 6 | 把 SwiGLU 与 RoPE 融进 GEMM 尾部 | ② §2.2–2.3 | 43.74 | **−1.05** † | ✅ 编译(核级) |
| 7 | 剥离成独立包,并打开去噪图(`--stage1`) | — | 43.16 | −0.58 ‡ | ✅ 编译(同进程 24/24 digest) |
| 8 | 删掉没人读的 timestep 条件计算 | ① §1.5 | **42.90** | **−0.30** ★ | ✅ 端到端 ◇ |
| | **小计** | | | **−9.70 ms(−18.4 %)** | |

**"逐位一致"这一列怎么读**(判据分级见[§ 正确性](#s-correctness)):

* **✅ 编译** = 在实际出货的 `max-autotune` 编译路径上验过,判据是核级 / 张量级 / 同进程的,可复现。
* **eager ✅ / 编译 ✗** = 历史上的 `0.00e+00` 是 `--no-compile` 下测的;2026-07-28 用
  "冻结 prefix + 四进程空对照"的门在编译路径上重验,**三项都是 FAIL**,括号里是动作的
  `max|Δ|`(300/300 元素,≈ 动作幅度 1.0–1.2 %)。**这不是 bug,也不是精度损失** ——
  三项都是代数恒等变换,差异来自 inductor 为改动前后的两种写法选了不同的 tile / 不同的
  K 方向 fp32 累加分块。
  ⚠️ **限制:这三条 FAIL 只在 base `max-autotune` 下测过,没有在 `--stage1` 下重跑**,
  而第 7 行之后的台账数字用的都是 `--stage1`。Stage-1 把这些算子包进手抓图,
  **不保证结论相同**,严格说这三个 ✗ 目前只对 base `max-autotune` 成立。
* **✅ 端到端 ◇** = 只有端到端 `--dump-actions` 的 `0.00e+00` 记录(第 2、8 行,当时确实测到,
  见 `docs/MEASUREMENTS.md`)。这个门**分辨力有限**:base 模式下它跨进程的噪声底就有 ~4–5e-3
  (根因是 SigLIP 的 LayerNorm 归约核跨进程改 launch config),所以它**证不伪**,只能算弱证据。
  这两项本次**没有**用强判据重验。

† 这一行有两个都正确、口径不同的数字,一并列出:在这条**累积链**上它是 **−1.05 ms**
(44.79 → 43.74);而在**它自己那次 4 轮配对 A/B** 里,基线是同 session 的 44.87 ms,
测得 **−1.14 ms ± 0.11**(sd 0.21,n = 4,4/4 轮同号)。两者之差是 session 间的基线漂移,
不是收益本身;链上口径用 −1.05,单项因果用 −1.14。

‡ 第 7 行是**跨 session** 的差值(43.74 来自融合那次的 A/B,43.16 来自今天),不是配对测量。
这一行里唯一做过配对的部分是独立包内的 `--stage1`:同一 session 4 轮交替,
**44.08 → 43.16 ms,Δ = −0.93 ms**(sd 0.36,n = 4),每轮两臂 SM 时钟相差 20 MHz 以内。

★ 第 8 行是同一 session、同一份代码、只翻 `RLINF_SKIP_DEAD_ADARMS_COND` 的 4 轮交替配对:
**43.20 → 42.90 ms,Δ = −0.30 ms**(sd 0.07,n = 4,4/4 轮同号,散布 −0.20…−0.37)。
关掉那一臂测到 43.20,与第 7 行跨 session 的 43.16 相差 0.04 ms,两条链因此对得上。
这一轮还顺手量到了本机**配对 A/B 的噪声地板**:先前误把两臂配成同一个 build 跑了一次
(见 §1.5 的坑),得到 Δ = −0.02 ms、sd 0.04 —— 所以 −0.30 ms 是噪声地板的 ~15 倍,
是可测的,不是把噪声讲成收益。

<a id="s-after-ledger"></a>

### 台账之后:又落地的五项(**第三把尺,不要接到 42.90 上**)

这五项的绝对基线来自**各自的 session**,而且有的锁频、有的不锁频,所以它们
**不能**和上面那条 52.60 → 42.90 的配对链首尾相接 —— 和"前史"那一栏是同样的处理。
每项都列出全部口径,不挑好看的:

| 优化 | 结论取值 | commit | 逐位一致 |
|---|--:|---|---|
| 小 M 的 mm tile 候选(`down_proj` / `o_proj`) | **−0.88 ms/predict** | `ca4ae39` | ✅ 编译(GEMM 级 sha256) |
| 跳过 prefix LM 最后一层的死算 | **−1.11 ms/predict**(保守口径) | `72af442` | ✅ eager(KV 级 36/36);编译态见下 |
| P·V 的 attention `bmm` 换 tile 长宽比 | **−0.18 ms/predict** | `ff237bf` | ✅ 编译(核级 digest,18 层) |
| 把步不变量从去噪循环里外提 | **−0.32 ± 0.05 ms/predict** | `0ed3ca2` | ✅ 编译(核级 `0.00e+00`) |
| prefix LM 的 Q/K/V 三投影并成一个 GEMM | **−0.61 ± 0.22 ms/predict** | `d7cf3c2` | ✅ 编译(KV 级 36/36 + digest) |

**小 M 的 mm tile 候选(−0.88 ms)。** M=50 时 inductor 的库存候选里 `BLOCK_M` 只有
`{32, 64}`,冠军 `BM64 BN32` 让 `down_proj`/`o_proj` 各只铺 **32 个 CTA**(110 个 SM)。
`pi05_infer/inductor_mm_tiles.py` 只对 `(m ≤ 64, n = 1024, k ∈ {2048, 4096})` 这两个 shape
追加 5 个小 `BLOCK_M`、深流水的候选(`BLOCK_K` 钉死 128,有 assert 挡着),其余 shape 原样
透传,然后让 inductor 自己 benchmark 选。CTA 32 → 64,`down_proj` 15.06 → 11.71 µs/call
(591 → 760 GB/s)、`o_proj` 8.47 → 6.94 µs/call(530 → 647 GB/s)。三个口径互相印证:

| 口径 | 折算到自然频率 ~2420 MHz |
|---|--:|
| nsys 核时间(只有这两个核变了,kernel 数完全不变) | **−0.879 ms** |
| 不锁频 6 轮配对 A/B,按 SM 时钟归一 | −0.874 ms |
| 锁频 2065 MHz 6 轮配对 A/B,取同臂序位置对比 | −0.88 ms |

(不锁频的 raw 读数只有 −0.35 ms —— on 臂的 boost 时钟系统性低 29 MHz,光时钟就值
+0.52 ms,比效应本身还大。**这一档 <1 ms 的效应必须锁频**。)

**跳过 prefix LM 最后一层的死算(−1.11 ms)。** `sample_actions` 只要 prefix 的 KV cache,
LM 的输出 embedding 绑完就被丢弃 —— 所以第 17 层(共 18 层)里除
`input_layernorm → k/v_proj → RoPE(k) → cache.update` 之外全是死算(按 FLOP 是该层的 99.1 %)。
`pi05_infer/prefix_last_layer.py` 只替换 `paligemma.model.language_model.layers[-1]` 这**一个实例**
的 `forward`(`types.MethodType`),模块树 / 参数名 / `state_dict` 全不变,所以 **RL 的权重同步
不受影响**;训练的 joint 分支从不调 `GemmaDecoderLayer.forward`,结构上摸不到这个 patch。
stream 7 少了 **12 个 kernel/predict**,stream 157 的 kernel 数**一个不差**(1710 = 1710)。
三个口径都列出来:

| 口径 | 值 |
|---|--:|
| **nsys 核时间(stream 7),自然频率** | **−1.11 ms/predict** ← 结论取这个,最保守 |
| e2e 配对 A/B,锁频 2072 MHz,12 轮交替臂序,12/12 同号 | −1.60 ms ± 0.18(SE) |
| 同上,折算到自然频率 2420 MHz | −1.37 ms |

e2e 比核时间多出来的 0.26 ms(约 1.5 SE)**没有**被算作收益:少掉的 12 次 launch 按
1.3 µs/次只有 16 µs,多出来的部分更可能是 e2e 口径里连带少掉的 GPU 间隙。

⚠️ **这一项是条件性的,不能无条件开。** RLinf 的 `get_value_from_vlm(prefix_output)` 读的
正是这个被丢弃的 hidden state,门是 `use_vlm_value = value_after_vlm and add_value_head`。
所以 `install_skip_last_lm_layer()` 在检测到 VLM value head 时**直接不安装**
(判据是照 RLinf 的 `use_vlm_value` 写的,搬回 RLinf 也成立)。实测
`examples/embodiment/config/*_ppo_openpi_pi05*.yaml` 共 19 份,其中 **15 份**两个开关都是
`True`(→ 不安装,拿不到这 1.11 ms);另外 4 份(`behavior_*`、`robotwin_*`)只设了
`add_value_head`,`value_after_vlm` 用默认的 `False`(→ 会安装)。DSRL / SAC 那几个
`add_value_head: False`,同样会安装。`pi05-infer` 是纯推理包、没有 value head,所以这里默认开。
Kill switch:`RLINF_SKIP_LAST_LM_LAYER=0`。

⚠️ **编译态的逐位结论只到"不比重编一次更糟"。** eager 下 18 层 36 个 KV 张量
**36/36 逐位相同**(sha256),这是这条改动的代数论证。但 `max-autotune` 下 prefix 的输出
**本来就跨进程不可复现**(SigLIP 那个 LayerNorm 归约核的 launch config 会跳变);
把 SigLIP 输出冻住回放之后,跨臂的 KV 差(2.000 / 2.250)与**把同一份代码重编一次**的
空对照差(1.625 / 1.750)是同一量级 —— 也就是说这条改动落在既有噪声底里,
**不是它引入的新问题**,但也**没有**在编译态拿到 `0.00e+00`。

**P·V 的 attention `bmm` 换 tile 长宽比(−0.18 ms)。** 完整叙述见 §3.2b。

<a id="s-hoist"></a>

#### 把步不变量从去噪循环里外提(−0.32 ms)

**为什么。** 4 维 attention mask、position ids
(`sum(prefix_pad_masks) + cumsum(suffix_pad_masks) − 1`)和 RoPE 的 cos/sin 表,在 10 个
Euler 步上**逐字节相同** —— `suffix_pad_masks` 是全 1 常量,`prefix_pad_masks` 在整个
predict 内固定 —— 可它们每步都被重建一次;而且 inductor 每步会把 `[1, 50, 256]` 的 cos/sin
表**物化 32 份**(每个消费它的层一份,因为融合的 QKV+RoPE 是个不透明的自定义算子)。
这正是 §3.3 点名的那一类"逐步重复的 glue"。

**怎么做。** 改成**每 predict 建一次**,写进常驻缓冲区,位置和理由都与 §1.1 的 adaRMS 调制表
相同:构造是 eager 的、必须留在 CUDA 图捕获之外,而它的输出缓冲区必须在捕获之前就存在,
好让图把地址记进去。缓冲区用 `copy_` 重填,**绝不重新分配**。

**量了多少。** nsys 2026.1.2,4 轮交替配对,SM 时钟锁 2302 MHz:

```
denoise  stream 157   217.00 -> 190.00 kernels/step,  1202.68 -> 1165.32 us/step
prefix   stream 7     +27 kernels/predict, +36.2 us/predict   (每 predict 一次的重建开销)
e2e                   43.93 -> 43.61 ms,  -0.32 +- 0.05 ms,  4/4 轮同号
```

注意 prefix 侧是**净增**的:把每步一次的重建换成每 predict 一次,代价记在 stream 7 上
(+36.2 µs),收益记在 stream 157 上(−37.4 µs/step × 10 步)。e2e 的 −0.32 ms 是两者之差,
已经把这笔代价扣掉了。

**数值。** **核级逐位相同**:生产 shape 下 eager 与 inductor 编译出的 cos/sin 差
`0.00e+00`(12800 个元素 0 个不同),mask 和 position ids 同样如此。
Kill switch:`RLINF_HOIST_STEP_INVARIANTS=0`。

<a id="s-prefix-qkv"></a>

#### prefix LM 的 Q/K/V 合成一个 GEMM(−0.61 ms)

**为什么 —— 而且理由和去噪侧不是同一个,别把论证照搬。** expert(M = 50)那边的机理是
**占用率崩塌**(§2.1):k/v 只产出 50×256,Triton 落在 grid = 8,110 个 SM 里只有 8 个在干活。
prefix 的 M = 968,**根本不缺并行度** —— 它的 k/v GEMM 跑 **248 个 CTA**,机器是满的。
它慢是因为 **N = 256 太窄,摊不动 A 的流量**:inductor 的冠军 tile **要走 2048 步 K 才产出
1024 个数**,只跑到 **41 TFLOP/s,对比 MLP 的 188**。k+v **只占 LM 的 0.9 % FLOP,却吃掉
3.3 % 的核时间**。把 N 从 256 拓宽到 2560,k 和 v 就落进"A 的流量已经付过钱"的 tile 里。

**怎么做。** 三个投影读的是同一份激活,所以并成一个 `[2560, 2048]` 的 GEMM 再切开 ——
在 **18 层里的 17 层**上生效(`pi05_infer/prefix_qkv_fused.py`)。只有 KV-only 的末层
(归 `prefix_last_layer.py` 管)保留两个独立 GEMM,因为那里的 `cat[k, v]` **不是**
逐位相同的。

**量了多少。** 实测在 vendoring 边界恢复之后(见[§ prefix / expert 的隔离](#s-isolation)),
nsys 2026.1.2,12 个 predict,SM 时钟锁 2092 MHz:

```
prefix   stream 7     23762.2 -> 23091.2 us/predict   -671.0 us   (633 -> 616 kernels)
denoise  stream 157   两臂都是 1630 个核,launch delta 0,耗时 -0.02%(噪声)
SigLIP   stream 158   两臂都是 383 个核,未变
e2e 配对 A/B,12 轮,锁频:   -0.61 +- 0.22 ms   (t = -2.75, 9/12 同号)
```

结论取 nsys 核时间的 **−671 µs**;e2e 那个数是量级旁证(噪声底 sd ≈ 0.8 ms)。

**数值。** **在编译路径上逐位相同**,而且判据够强:两臂分别是 0 层融合与 17 层融合,
**36 个 prefix KV 张量全部 `0.00e+00`**,combined digest 完全一致 —— 与"跳过 prefix LM
最后一层"同档(gate:`tools/bitexact_prefix_qkv.py`)。
Kill switch:`RLINF_FUSE_PREFIX_QKV=0`。

#### 实测否定:prefix 上的 GeGLU epilogue 融合(不交付)

把 `gelu_tanh(gate) * up` 融进 GEMM 尾部,在去噪侧成立(§2.2),**在 prefix 上不成立**。
prefix 的 gate/up GEMM 已经跑在 **188 TFLOP/s ≈ 该形状 cuBLAS 可达峰值的 92 %**;要融就得
把它从 cutlass 换成 Triton,而 Triton 在这个形状上**慢 6.3 %(+44 µs/层)**,
被融掉的 pointwise 只值 **28 µs/层**。**入场费大于奖品** —— 实测两次,两次都亏。
(对比去噪侧:那边 Triton 本来就比 cuBLAS 快 9 %,入场费是负的 —— 所以
"epilogue 融合总是赚"这条经验**不能跨 shape 迁移**,要先量 Triton 与 cutlass 在该形状上的
差距,那才是入场费。)

<a id="s-naming"></a>

### 命名对照:每个名字对应代码里的什么

台账里用的是"这项优化做了什么"的说法。代码、命令行开关和 profile 文件用的是另一套简写,
下表把两者对上,免得在仓库里搜不到。

| 台账里的名字 | 代码 / 开关 / 笔记里的叫法 |
|---|---|
| 预计算 adaRMS 调制量 | `engine.py` 里的调制表 `adarms_mod`;笔记里叫 "adaRMS cache" |
| 把整个去噪步捕获成一张 CUDA 图 | `--stage1`、`_denoise_graph_captured`;笔记里叫 "Stage-1" |
| Q/K/V 三个投影并成一个 GEMM | `qkv_fused_weight` |
| prefix KV 常驻静态缓冲区 | static KV buffer、`_copy_kv_into_static` |
| 在 GPU 上构造 attention mask | `embed_prefix` 里的 `att_masks` |
| 把 SwiGLU 与 RoPE 融进 GEMM 尾部 | `_swiglu_mm_kernel` / `_qkv_rope_kernel`;开关 `RLINF_FUSE_SWIGLU` / `RLINF_FUSE_QKV_ROPE` |
| 删掉没人读的 timestep 条件计算 | `skip_adarms_cond`;开关 `RLINF_SKIP_DEAD_ADARMS_COND` |
| 小 M 的 mm tile 候选 | `pi05_infer/inductor_mm_tiles.py`;开关 `RLINF_SMALL_M_MM`(默认开) |
| P·V 的 attention `bmm` 换 tile | 同一个文件的 `install_small_m_bmm_configs()`;开关 `RLINF_SMALL_M_BMM`(默认开) |
| 跳过 prefix LM 最后一层的死算 | `pi05_infer/prefix_last_layer.py`;开关 `RLINF_SKIP_LAST_LM_LAYER`(默认开,检测到 VLM value head 自动不装) |
| 把步不变量从去噪循环里外提 | `engine.py` 里的常驻 mask / position id / cos-sin 缓冲区;开关 `RLINF_HOIST_STEP_INVARIANTS`(默认开) |
| prefix LM 的 Q/K/V 并成一个 GEMM | `pi05_infer/prefix_qkv_fused.py`;开关 `RLINF_FUSE_PREFIX_QKV`(默认开,末层不融) |
| 关掉推理期的类型检查(前史) | `RLINF_DISABLE_OPENPI_TYPECHECK=1` |
| 三路相机合成一次视觉编码(前史) | `openpi_patched/pi0_pytorch.py::embed_prefix` 里写死的 `torch.cat(images, dim=0)` —— ⚠️ **没有开关**(曾经写成 `RLINF_SIGLIP_BATCHED=1`,那个环境变量全树不存在),而且它**不是**逐位一致的,见「台账之前」 |

内核层面的配套数字:

| 指标 | before | after | 归属 |
|---|--:|--:|:--:|
| 每 step GPU idle | 142.2 µs(10.2 %) | **56.5 µs(4.5 %)** | ① |
| adaRMS 投影 `triton_per_fused_addmm_0` | 300 instances,395 µs/step,DRAM-read 87 % | **0 instances** | ① |
| prefix KV 的 `cat` kernel | 88 µs/step(物理下限 27) | **0** | ① |
| timestep 条件计算(正弦嵌入 + time MLP) | 21 kernel/step,47.3 µs/step | **0** | ① |
| denoise kernels/step | 347 | **217**(−37 %) | ①② |
| denoise µs/step(GPU busy) | 2025.6 | **1185.0**(−41.5 %) | ①② |
| k/v_proj 的 launch grid | **8**(110 个 SM 里只有 8 个在忙) | 80(融合后的 QKV GEMM) | ② |
| `_swiglu_mm_kernel` 达成带宽 | — | **973 GB/s**(实测字节 ÷ nsys µs;可达上限 1222,见 §3.2) | ② |
| `down_proj` 达成带宽 | 557 GB/s | **760 GB/s**(small-M 重 tile 之后,15.06 → 11.71 µs/call) | ③ |
| `o_proj` 达成带宽 | 530 GB/s | **647 GB/s**(同上,8.47 → 6.94 µs/call) | ③ |
| `triton_tem_fused_bmm_7`(P·V) | 6.214 µs/call | **5.208 µs/call**(−16.2 %,见 §3.2b) | ③ |
| prefix LM 第 17 层的死算(stream 7) | 12 kernel/predict,1112 µs/predict | **0** | ① |

<a id="s-per-step"></a>

## 换一个纵轴:去噪单步的台账(文首那张图的栏 3)

同一批优化,按**一个去噪步**来记账。因为整个 predict 里能动的就是这 10 步,
所以这张台账比 e2e 更能说明"改了什么"。

口径:**GPU busy per denoise step** = 去噪循环里 kernel 区间的并集 ÷ 10 步。bs=1 下没有任何
重叠,所以它等于逐核时长求和 —— 这正是后面几行 nsys stream-157 的算法,两套工具因此同尺。

| # | 优化 | µs/step | Δ | kernels/step |
|---|---|--:|--:|--:|
| 0 | 基线(预计算 adaRMS 之前) | 2025.6 | — | 347 |
| 1 | 预计算 adaRMS 调制量 | 1620.9 | **−404.7** | 346 |
| 2 | 并 Q/K/V ＋ 静态 KV ＋ GPU 上建 mask(＋ CUDA 图) | 1368.0 | −252.9 ※ | 305 |
| 3 | 把 SwiGLU 与 RoPE 融进 GEMM 尾部 | 1236.0 | **−132.0** | 238 |
| 4 | 删掉没人读的 timestep 条件计算 | **1185.0** | −51.0 ◆ | **217** |
| | **小计** | | **−840.6(−41.5 %)** | **−130** |

⚠️ **这张表停在 small-M 重 tile 之前。** 那一项之后 `down_proj` / `o_proj` 各快了
3.35 / 1.53 µs/call × 18 层 = **−87.9 µs/step**,kernel 数一个不变 —— 但它自己那对 profile
的窗口读数是 stream 157 的 1148.2 → 1055.5 µs/step,和这一列的 1185.0 **不是同一个窗口**
(这一列含 stream 157 之外的 per-step glue)。所以**不要**把 1185.0 − 87.9 当成新台阶,
要用就用同一对 profile 里的那两个数。**P·V 的 bmm 重 tile(§3.2b)同理**:它自己那对
profile 的窗口读数是 denoise stream busy 10.556 → 10.401 ms/predict,kernel 数一个不变。

※ 第 2 行是**跨 session 的四项合并**(对应 e2e 台账的第 2–5 行),不是配对 A/B。而且要
注意:**去噪 CUDA 图在 GPU busy 口径下并不省时间** —— 它把 idle 变成 busy(每 predict
idle 3.74 → 0.93 ms),收益体现在墙钟而不是这一列。第 1 行(同一张表、同一 session)和
第 3 行(`RESULTS_fusion.md` 明确记 `−132.0(−9.6 %)`)才是干净的配对。

◆ 第 4 行自己的配对基线是**同 session** 的 1232.3 µs/step(`prof_skip0`),即 **−47.3**;
链上口径是 −51.0。同一份 238 核 build 在 07-28 重测还给出 1232.6(逐核普查)与 1233.3
(union),三个数是同一个 build 的不同工具/窗口,**不是一级台阶**。

更早的一段用的是**墙钟**口径(NVTX `denoise/loop` span ÷ 10,含 launch gap 与每步 CPU):
2026-07-09 的基线是 **2255 µs/step**(同一份 profile 的 busy 是 2189),预计算 adaRMS 那次
配对的 sync-timed 墙钟是 **2286 → 1881**。这把尺和上表不能相接;今天两把尺的读数是
**1242.7(wall)vs 1185.0(busy)**,每步还剩 57.7 µs 的 GPU idle。

> **有一个数我没有画进图**:最早记录的 "2115 µs/step @ 417 kernels"。它出处的 sqlite 已经
> 不在笔记树里,而且 2.115 ms 在同期另一份 profile 里恰好是 `denoise/expert_forward` 的
> span(那份的 `denoise/loop` 是 2.255),所以它很可能是 **expert-only 的更窄口径**,与其它
> 行不同尺。与其猜一个口径,不如不画。

参考实现在**同一把尺**上的位置(详见「[与参考实现的对比](#s-baselines)」一节的限定):

| | denoise µs/step | kernels/step |
|---|--:|--:|
| `dexmal/realtime-vla` @`b86a942` | **1191.0**(sd 13.2) | **165** |
| `limxdynamics/FluxVLA` @`7f9f774` | 1419.0 | ~205 |
| 我们(上表第 4 行,small-M 之前) | **1185.0** | 217 |

⚠️ 1185.0 与 1191.0 **不是配对测量**(相隔一天、不同 build)。realtime-vla 那次做过配对的
对手是我们当天的 **1368.7 µs/step @ 306 核**,他们领先 1.15×。**不能**据此宣称反超。
之后 small-M 重 tile 又拿掉了 ~88 µs/step、P·V 重 tile 再拿掉 ~18 µs/step,但那**同样不是**
对着他们做的配对测量 ——
**"我们反超了 realtime-vla"这个判词到现在都没有证据支持,不要写。**
FluxVLA 那一行的 chunk 是 10 而不是 50 —— 后缀 action token 少 5 倍,也不是同一个 workload。

---

<a id="s-eliminate"></a>

## ① 消除:让 GPU 别空转,也别做白工 —— 台账内 −5.99 ms(另加 prefix 跳层 −1.11 ms)

> **观测**:一个去噪步墙钟 1390 µs,其中 **142.2 µs(10.2 %)GPU 完全空闲**;
> 而在忙的那 1247.8 µs 里,最大的单个 kernel 做的事情**根本不需要每步重做**。

这一段的五项没有一项在"让 kernel 变快",全部是**让它不发生**。
(同一段的第六项「跳过 prefix LM 最后一层的死算」在台账之后才落地,基线是另一把尺,
写在[「台账之后」](#s-after-ledger)那一节。)

### 1.1 预计算 adaRMS 调制量(−2.83 ms,单项最大)

**为什么。** 每个 adaRMS norm 前面挂着一个 `dense(cond)` 投影,一共 37 个。它们只依赖扩散
timestep,而 timestep 走的是**固定 schedule** —— 也就是说**与输入无关**,却每步重算一遍。
inductor 把这 37 个投影**横向融合**成了一个 memory-bound 的巨核
(`triton_per_fused_addmm_0`):**300 instances、395 µs/step、DRAM-read 占比 87 %**,
搬运 483 MB 而实际只需要 233 MB —— **2.08× 的冗余流量**。

**怎么做。** 一次性预计算成一张 `[num_steps, 37, 3072]` 的表,每步只做一次 device gather。
表在 CUDA 图**捕获之前**建好(建表用的是 capture-illegal 的 eager 算子),每步取表是
device gather,所以两者都能活过捕获。

**量了多少。** e2e **−2.83 ms**;那个核 **300 → 0 instances**。

> **对照实验(这条决定了整段的方法论)**:先前试过"把 37 个投影 batch 成一个大 GEMM"
> (Stage A),结果是**持平** —— M = 1 的大 GEMV 仍然要读 ~477 MB、本来就跑在 92 % 带宽上,
> 时间一样。**只有消除投影才有用。所以这里的取舍依据是性能,不是数值。**
>
> ⚠️ **这段以前写的是"预算表位级一致(`0.00e+00`),Stage A 因为改了归约顺序是 `2.71e-3`,
> 本来就不合格" —— 那个论证已经垮了。** 预算表的 `0.00e+00` 是 eager 下测的;在实际出货的
> `max-autotune` 路径上,它相对 per-dense 基线是 **`2.568e-3`**,和它当初用来淘汰 Stage A 的
> `2.71e-3` **同一个量级**。以"位级一致"为由二选一的说法不成立;成立的只有性能:表消掉了
> 整个投影(−2.83 ms),Stage A 持平。
>
> 机制在核级直接量到了,不是推的:表是**在所有编译区之外**用 eager 的 `n.dense(cond)` 建的,
> 而被它替换掉的基线是**编译区内**由 inductor 生成的那个投影核。拿真实的 37 个 `dense` 权重
> 和真实的 `cond` 直接对拍 eager `F.linear` 与 `torch.compile(F.linear, "max-autotune-no-cudagraphs")`:
> **逐位相同的 norm 数 0 / 37**,112640 / 113664 个元素不同,`max|Δ| = 4.394e-3`。
> 两个核算的是同一个 GEMM,没有义务给出同样的比特。

### 1.2 把整个去噪步捕获成一张 CUDA 图(链上 −2.04 ms;独立包内配对 −0.93 ms)

**为什么。** 每步 142.2 µs 的 GPU 空闲(10.2 %)是纯粹的 launch gap:inductor 只把
expert block 包进了它自己的 cudagraph,step 里其余的 eager glue 仍然一个一个 launch。

**怎么做。** 把**一整个 flow_ode 去噪步**(expert forward + Euler 更新 + log-prob)捕获成一个
`torch.cuda.CUDAGraph` 并逐步 replay,一次 replay 取代"inductor 的 expert-only cudagraph
＋ 中间所有 eager glue 的 launch"。

**量了多少。**

| build | step wall | GPU busy | idle | idle % |
|---|--:|--:|--:|--:|
| `--stage1` **off** | 1390.0 µs | 1247.8 | 142.2 | 10.2 % |
| `--stage1` **on** | **1294.2 µs** | 1233.7 | **60.5 µs** | **4.7 %** |

每 predict 的**总 GPU busy 基本不变**(40.32 → 40.26 ms),说明收益是纯粹的 launch-gap 消除;
−95.8 µs/step × 10 步 = **−0.96 ms/predict**,正好解释掉配对 A/B 的 −0.93 ms。

它是**无损**的:`flow_ode` 的 `x_t_std == 0`,所以图里的 `x_t_next = x_t_mean` 在代数上等于
eager 的 `x_t_mean + noise * 0`;而 `sample_noise` 的抽样留在图之外,全局 RNG 消耗不变。

> **怎么确认图真的生效**:`graphNodeId` **不是**判据 —— 开关关掉时那 2160 个 denoise
> kernel 也全都带 `graphNodeId`(inductor 自己就发了一个 cudagraph),我据此误判过一次。
> 可靠信号:`denoise/expert_forward` 的 NVTX range 数(10/predict = eager,**0 = 在图内**)、
> distinct `graphId` 数(2 → 1)、stream 157 上的 kernels/step(171 → 238)。
> (238 是当时的数;§1.5 之后是 217,判据本身不变。)

### 1.3 prefix KV 常驻静态缓冲区(−0.51 ms)

**为什么。** denoise 路径跑的是 `use_cache=False`,于是 attention 走了
`torch.cat([prefix_kv, new_kv])` 分支 —— **每步、每层都在重新物化整个 968 token 的 prefix
KV**(18 层 × 2 × 10 步)。实测这个 `cat` kernel **88 µs/step,而物理下限是 27 µs/step**。

**怎么做。** 改成 vLLM / SGLang 那套:每层预分配一个
`[B, kv_heads, prefix+suffix, head_dim]` 的静态 buffer,每个 predict 写一次 prefix,
每步只写 50 个 token 的尾巴。buffer 只在 shape 变化时重分配 → **地址稳定 → CUDA 图
replay 安全**。

**量了多少。** e2e **−0.51 ms**,`cat` kernel 归零。这一步还是 ② 里 QKV+RoPE 融合能"直写
KV 尾巴"的前提。

**数值。** eager 下 `0.00e+00`;`max-autotune` 下 **`2.858e-03`**(300/300,≈ 动作幅度
1.19 %),**FAIL**,和 §1.1 / §2.1 是同一类现象。
⚠️ **机制没证实,不编**:off 臂走 `torch.cat([prefix_kv, suffix_kv], dim=2)`、on 臂走
"预分配 buffer + `copy_` 写尾部",喂给 SDPA 的张量**应当**是同 shape、同 layout、同数值的,
但那个张量在编译图内部取不到,**没有直接对拍过**。已测的事实只有:两臂在第一步去噪的输出
(`step0/mean`)上就分叉,而 off/off、on/on 两个空对照 24/24 干净。

### 1.4 在 GPU 上构造 attention mask(−0.31 ms,但价值远不止)

**为什么。** `embed_prefix` 里的 `att_masks = torch.tensor(<python list>, device=cuda)`
是热路径上的一次**同步** host→device 拷贝。

**怎么做。** 读代码可以证明这个 mask **恒为全零**(整个 prefix —— 所有图像 view 加语言
token —— 是一个 full-attention block,两处 append 都是 `[0]*n`),长度也恒等于
`pad_masks.shape[1]`,于是直接用 `torch.zeros(..., device=...)` 构造。

**量了多少。** e2e **−0.31 ms**。但它真正的价值是:这是**唯一**一条阻止 prefix 阶段被
CUDA 图捕获的语句 —— 改之前 `torch.cuda.CUDAGraph()` 捕获 `_build_prefix_cache` 会在这一行抛
`operation not permitted when stream is capturing`,改之后 capture / replay 都成功。
prefix 是 GPU busy 的 **71.7 %**,所以这条的长期价值远大于它自己的 0.31 ms。

**数值:这一项不需要端到端判据,而且拿到了完备证明。** `att_masks` 只被
`make_att_2d_masks(pad_masks, att_masks)` 和(经 `pad_masks`)`position_ids` 消费,
只要这几个张量逐位相同且 layout 相同,后面就是"同一段代码吃同一批实参"。同进程直接对拍
`att_masks` / `att_2d` / `att_4d` / `position_ids` / `pad_masks` 五个张量:
**全部逐位相等,shape / stride / dtype 全同**,eager 与 `max-autotune` 两次跑结果一致。
⚠️ **边界:只验了 bs=1。** 原实现是 `torch.tensor(list)[None,:].expand(B, L)`,bs=1 时
`expand` 是恒等、stride 恰好也一样;**bs>1 时 `expand` 出来的是 stride-0 视图**,值仍全等
但 layout 不同,inductor 可能因此编出不同的核,**未测**。(出货配置就是 bs=1。)

### 1.5 删掉没人读的 timestep 条件计算(−0.30 ms)

**为什么。** §1.1 的预算表落地之后留下了一段**没人删的死代码**。`get_suffix_out` 每步仍然调
`embed_suffix` 算 `cond = time_mlp(sinusoidal(timestep))` —— 一个 fp64 的正弦嵌入加两个
`Linear(1024→1024)` 加两次 silu。但下游 `modeling_gemma.py` 里是这样挑的:

```python
if adarms_mod is not None:        # 我们永远传(就是那张预算表)
    adarms_mod_all = adarms_mod
elif adarms_cond is not None and self.adarms_Wstacked is not None:   # 永远进不来
    adarms_mod_all = F.linear(adarms_cond, self.adarms_Wstacked, ...)
```

预算表上线那天起 `adarms_mod` 就一直有值,`elif` 再也没执行过,`adarms_cond` 算完直接被丢掉。
π0.5 上它也确实没有别的去处 —— `action_time_emb` 就是 `action_emb` 本身,时间嵌入不进 token。

**怎么做。** 加一个 `skip_adarms_cond` 开关,**不删那条 `elif`**:它是 fallback,而且非 π0.5
路径要把时间嵌入 concat 进 action token,那里仍然必须算。`get_suffix_out` 只在自己确实拿着
`adarms_mod` 时才传这个开关,默认值保持今天的行为。开关是**编译期的 Python 常量**,不是对
device tensor 的判断,所以 `embed_suffix` 仍然可以待在去噪图里(图只会 trace 到
其中一条分支)。`GemmaRMSNorm` 的 guard 同步放宽:拿着 `mod` 的调用方现在可以合法地传
`cond=None`。

**量了多少。**

| | before | after |
|---|--:|--:|
| denoise kernels/step | 238 | **217**(−21) |
| denoise GPU busy | 1232.3 µs/step | **1185.0**(−47.3) |
| step wall(nsys 时间线) | 1294.0 µs | **1242.7**(−51.3 → −0.51 ms/predict) |
| e2e(4 轮配对) | 43.20 ms | **42.90**(**−0.30 ms**,sd 0.07) |

消掉的 21 个核正好是那两段计算:`internal::gemvx` ×2(两个 time-MLP,18.4 µs/step)
＋ 它们的 `cublasLt` splitK/epilogue 辅助核 ×4(4.3 µs)＋ `cos`/`sin`/`silu` 与 fp64 正弦
嵌入的一堆 elementwise ×15(24.6 µs)。**没有任何 GEMM / attention / Triton 核的数量或时间
发生变化**,prefix(stream 7)的 673 kernels/predict 也一个没动。

e2e 的 −0.30 ms 小于时间线上的 −0.51 ms:差额落在 prefix 的 run-to-run 漂移里
(28.00 → 28.13 ms/predict),不是这项改动造成的。

> **坑:配对 A/B 的两臂必须验证它们真的不一样。** 第一轮测量得到 Δ = −0.02 ms(sd 0.04),
> 看着像"收益在噪声里"。实际是打到远端机器上的 patch 是**加 kill switch 之前**的版本,
> 两臂跑的是同一个 build —— 那次测量其实是一次**空对照**。用 nsys 分别数两臂的
> kernels/step(238 vs 217)才把它抓出来。空对照本身是有价值的副产品:它把本机配对 A/B 的
> 噪声地板钉在了 **±0.04 ms**,正是靠它才能说 −0.30 ms 是真的。

---

<a id="s-fuse"></a>

## ② 融合:同样的事,用更少的 kernel 做完 —— −3.17 ms,305 → 238 kernels/step

> **观测,而且先做的是一个否定性探针**:强制
> `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON` 对 kernel 数**毫无影响**(305 → 305),
> 而且在 prefix 里倒亏 ~4 ms;`TRITON,ATEN` 更糟(323 —— autotune 重新挑了 ATEN,
> 多出 18 个 `splitKreduce`)。

原因读生成代码就知道,不用猜:

1. **denoise 的每一个 GEMM 本来就是 Triton template**(`_mm_4` 是 QKV、`_mm_13` gate、
   `_mm_15` down、`_clone_mm_11` o_proj、`_gelu_mm_mul_14` up)。profile 里的 cutlass
   extern kernel 是 Euler/logprob glue 里的 1–2 个残余,不在 expert 里 —— 所以一开始就
   不存在"外部 GEMM 挡住融合"这回事。
2. **inductor 结构上能融的 epilogue 已经融完了**:`gelu_mm_mul_14` 本身就是
   up_proj + `gelu(gate)·up` 的 epilogue 融合;`clone_mm_11` 是 prologue 融合;
   gated residual 早已被**向前**折进消费端的 RMSNorm 归约里(所以"把 gated residual 塞进
   GEMM epilogue"是个零收益提案 —— 归约不可能当 epilogue,kernel 数一个都不会少)。
3. **剩下这两个是 inductor 表达不出来的**,换任何 backend 设置都一样。

所以这一段只有三项,每一项都属于"inductor 结构上做不到"的那类。

### 2.1 Q/K/V 三个投影并成一个 GEMM(−2.12 ms)

**为什么。** MQA(`num_kv_heads = 1`)让 k_proj / v_proj 的输出只有 50 × 256,triton 把它切成
**grid = 8** —— 110 个 SM 里只有 8 个在干活,比它自己的内存下限慢 **8.3×**,纯粹的
launch / occupancy 浪费。

**怎么做。** 把 q(2048)+ k(256)+ v(256)沿 N 拼成一个 `[2560, 1024]` 的权重,
变成**一个更宽的 GEMM**。数学上完全等价(每个输出列都是独立的点积)。

**量了多少。** grid 8 → **80**;e2e **−2.12 ms**;18/18 层融合成功。

**数值。** 未融合 vs 融合 `max|Δ| = 0.00e+00` —— 但那是 **eager**(`test_qkv.py`)测的。
在出货的 `max-autotune` 路径上重验(冻结 prefix + 四进程空对照,空对照 24/24 干净)是
**`2.431e-03`**(300/300 元素,≈ 动作幅度 1.02 %),**FAIL**。
把 q(2048)+ k(256)+ v(256)沿 N 拼成 2560 **数学上确实等价**(每个输出列都是独立点积),
但 N 一变 inductor 就换了 tile、K 方向的 fp32 累加分块随之改变。eager 下 cuBLAS 对两种 N
用了同样的 K-loop,所以那里确实是 `0.00e+00`;**编译路径上不成立**。

### 2.2 把 SwiGLU 的 gate/up 并成一个 GEMM(inductor 不做的**横向**合并)

**为什么。** gate 和 up 是两个共享同一个输入的矩阵乘,inductor **根本不做横向融合**
(两个 matmul 共享一个操作数不在它的融合规则里),于是每层要把 gate 激活写出去 400 KB、
再读回来 400 KB。

**怎么做。** 一个 kernel、两个 fp32 累加器、**共享同一个 A tile**:`acc_g += dot(a, Wg)`、
`acc_u += dot(a, Wu)`,然后 `gelu_tanh(bf16(acc_g)) * acc_u` → 一次 bf16 store。
配置 `BLOCK_M=64, BLOCK_N=32, BLOCK_K=64`,4 warps,4 stages,grid 128。

**量了多少。**

| | kernels/step | µs/step |
|---|--:|--:|
| 前:`triton_tem_fused_mm_13`(gate)+ `triton_tem_fused_gelu_mm_mul_14`(up+act) | 36 | 405.2 |
| 后:`_swiglu_mm_kernel`(17.35 µs × 18) | **18** | **312.4** |

收益**不在 FLOPs**(一样多),而在于省掉每层那 800 KB 的往返和每层一次 launch。
达成带宽 16.78 MB / 17.35 µs = **967 GB/s**(按 ncu 实测的 DRAM 字节 16.899 MB 算是
**973 GB/s**;⚠️ 天花板是独立测出来的 **1222 GB/s**,不是 996 —— 见 §3.2)。

### 2.3 把 RoPE 做在 QKV GEMM 的累加器上

**为什么。** RoPE 要求输出列 `d` 和 `d + head_dim/2` 在**同一个 program** 里,而 Triton
template 的 epilogue 只能看到自己刚算完的那块 tile(`rotate_half` 是两段不同列区间的
`torch.cat`),所以 inductor **必然**把它拆成独立的 pointwise kernel。

**怎么做。** 融合后每个 program 拥有同一个 head 的**一对**列 tile,旋转直接作用在累加器上;
rotated k / raw v **直接写进 static KV cache 的尾巴**(§1.3 的产物),顺带干掉两个拷贝核。
q 写进的 `[B, M, Hq, D]` buffer,其 `transpose(1,2)` 与旧的 `view().transpose()`
**逐字节同 stride**(已验证),所以下游没有任何 kernel 需要改布局。
配置 `BLOCK_M=64, BLOCK_N=16, BLOCK_K=128`,4 warps,4 stages,grid 88 —— 刻意贴近被替换
GEMM 的 80,占用率不变。

**量了多少。**

| | kernels/step | µs/step |
|---|--:|--:|
| 前:`triton_tem_fused_mm_4` 112.7 + RoPE-q 28.3 + RoPE-k&store 15.0 + store-v 9.9 | 72 | 165.9 |
| 后:`_qkv_rope_kernel`(7.33 µs × 18) | **18** | **132.0** |

### 2.4 GEMM 尾部融合怎么做到位级一致

epilogue 融合通常**不**位级一致:融合版跑在 fp32 累加器上,未融合版跑在"存回内存又读出来"
的 bf16 上。这两个核**故意**把每一个"原本会以 bf16 落盘"的累加器做一次
`acc.to(bf16).to(fp32)` 往返;再把 `BLOCK_K` 钉死在 inductor autotuner 选中的值上
(`BLOCK_K` 是唯一会改变 K 方向 fp32 归约顺序的 tile 参数,`BLOCK_M`/`BLOCK_N` 可证明不会),
结果就是**精确相等**。

对拍的参照系是 **inductor 在 `max-autotune-no-cudagraphs` 下为同一批算子编出来的 kernel**
—— 即被替换掉的那些本体,不是 cuBLAS:

| tensor | bitwise | max\|Δ\| | 不同元素 |
|---|---|--:|--:|
| SwiGLU 输出 | **True** | **0.00e+00** | 0 / 204800 |
| RoPE 后的 q | **True** | **0.00e+00** | 0 / 102400 |
| RoPE 后的 k(在 KV cache 里) | **True** | **0.00e+00** | 0 / 12800 |
| v(在 KV cache 里) | **True** | **0.00e+00** | 0 / 12800 |

独立的 tile 不变性检查扫过 `BLOCK_N ∈ {16, 32, 64}` 与 `num_warps`/`num_stages`,
输出全部逐比特相同 —— 确认只有 `BLOCK_K` 是承重的。

### 2.5 这一段踩到的两个坑(都值钱)

**坑 1:同一份代码在另一个阶段是负收益。** SwiGLU 融合第一版没加 M 护栏,于是它也捕获了
PaliGemma **prefix** 那个 968 token 的语言模型 —— 因为 `modeling_gemma.py` 是共享的,
PaliGemma 的 prefix LM 也是 Gemma。一个为 M=50 手调的 tile config 对 M=968 是错的
(那边是 compute-bound,省下的 gate 往返早被摊薄了):

| variant | prefix(stream 7)ms/predict | denoise(stream 157)µs/step |
|---|--:|--:|
| baseline | **27.2** | 1368.0 |
| SwiGLU 融合(未加护栏) | **33.7(+6.5)** | 1273.6 |

**prefix 的损失是整个 denoise 侧收益的 ~5 倍**,而只看 denoise 指标完全看不见 ——
这就是"**必须按 stream 分开看**"这条规矩的由来。现在 `RLINF_FUSE_SWIGLU_MAX_M`(默认 64)
把融合限制在能装进一个 M-tile 的 token 数以内;本仓库后来做的 vendoring(expert 用
`pi05_infer.gemma`,prefix 仍用原厂 transformers)则从**结构上**消除了这一类 bug。

**坑 2:dispatch 理论被自己的数据证伪。** 减掉 67 kernel/step,按 1.3 µs/node 预测能省
0.87 ms 的 dispatch:

| 分量 | 预测 | 实测 |
|---|--:|--:|
| denoise kernel 执行时间(nsys 逐核求和) | −1.32 ms | −1.32 ms |
| graph-node dispatch,67 nodes/step × 10 × 1.3 µs | −0.87 ms | **≈ 0** |
| **e2e** | −1.3 ~ −2.2 ms | **−1.14 ms** |

**e2e 收益几乎完全等于 kernel 执行时间的减少。** 一旦图里的节点足够密集、让 GPU 前端跑在
前面(去噪图已经做到了),per-node dispatch 常数就不再适用。
**kernel 数仍然是选靶子的好指标**(便宜、方差小),但**换算成时间必须用实测的逐核时长**,
不能乘一个 dispatch 常数。

---

<a id="s-kernel"></a>

## ③ 压 kernel:让剩下的每个核自己跑得更快 —— **已兑现 −0.88 + −0.18 ms**

前两段把"不该做的事"和"多余的 launch"清完之后,剩下的时间**全在 kernel 自己的执行**里。
这一段 2026-07-28 开张了,而且前两笔收益**都没有写一行 Triton**。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/denoise_dark.png">
  <img alt="当前 build 的 denoise 每核耗时分解" src="docs/denoise_light.png">
</picture>

**先说这一段的诊断依据变了。** 之前只有 nsys 的逐核时长 + 一个字节模型;2026-07-28 补了
Nsight Compute(`ncu` 2025.1.1,在 sm_120 上完全可用,occupancy / stall / 内存管线全部出真
数据),结论**改写了这一段的靶子排序**:

* **这张卡的可达 DRAM 读带宽是 1222 GB/s,不是 996。** 之前把 996 当天花板是循环论证 ——
  996 就是 `_swiglu_mm_kernel` 自己的数。独立的 STREAM 式基准(2 GB / 4 GB,`torch.randn`
  造数)测到纯读 **1222 GB/s**(= spec 1344 的 91 %),读写混合 1106–1144。
* **denoise 的六个主力核没有一个是被带宽卡住的,全部是并行度不足。** 六个里有**五个**存在
  "一个 cycle 都没跑过"的 SM(`sm__cycles_active.min = 0`):64 CTA 的四个核有 **46 / 110 个 SM
  全程空转**,`_qkv_rope_kernel`(88 CTA)空转 20 %。`eligible warps per scheduler` 全部
  ≤ 0.21(硬件上限 12),`not_selected` 有四个核**精确为 0.0 %** —— 连"两个 warp 抢发射槽"
  都从没发生过。这是"warp 不够",不是"等 DRAM"。
* MFU 可以精确分解成 `每 SM 张量管线利用率 × 开工 SM 比例`,六个核全部在 1–3 个百分点内
  复现 roofline 的 MFU —— ncu 计数器与 roofline 互证。

⚠️ **但 ncu 的占用率诊断不能直接当处方**:§3.2b 里 ncu 明确指向 `BLOCK_M=16`(把 grid 从
64 抬到 256 CTA),实测下来它比现状**慢 30 %**,真正赢的是同 grid、同 1 CTA/SM 的
**tile 长宽比**。占用率读数用来**排序靶子**是可靠的,用来**挑参数**不可靠 ——
最后仍然要 benchmark。

<a id="s-3-1"></a>

### 3.1 已落地:给饿死的 GEMM 补小 `BLOCK_M` 候选(−0.88 ms)

**观测。** `down_proj` 是剩下最大的单个 kernel(271 µs/step),而它和 `o_proj` 在 M=50 下
各只铺 **32 个 CTA**。原因不在硬件而在 inductor 的候选表:`mm_kernel_configs` 里**没有任何
`BLOCK_M=16` 条目**,`filtered_configs()` 只把 `BLOCK_M` 往下夹到 `next_power_of_2(50)=64`、
**从不往上抬**,唯一那个 `BLOCK_M=32 & BLOCK_K=128` 的库存候选输在流水线深度(stages=2)。

**怎么做。** 不写 custom op,**扩 inductor 自己的候选集**:`pi05_infer/inductor_mm_tiles.py`
只对 `(m ≤ 64, n = 1024, k ∈ {2048, 4096})` 追加 5 个小 `BLOCK_M`、深流水的候选,其余 shape
原样透传,然后让 inductor 自己 benchmark 选。`BLOCK_K` 全部钉死 128(有 assert 挡着)。

**量了多少。** CTA 32 → 64;`down_proj` 15.06 → 11.71 µs/call(591 → **760 GB/s**)、
`o_proj` 8.47 → 6.94 µs/call(530 → **647 GB/s**);**−0.88 ms/predict**,三个口径互相印证
(见「台账之后」)。**GEMM 级 sha256 逐字节相同**,包括两臂各自独立 autotune、确实编出了
不同 tile(BM64 BN32 vs BM16 BN64)的那一组。

**还没做到位。** ncu 显示 re-tile 后 grid = ceil(50/16) × (1024/64) = **64 CTA,仍然少于
110 个 SM**,`sm__cycles_active.min = 0`,46 个 SM 空转;内存侧只到可达带宽的 52 % / 62 %,
离带宽墙还远。下一步是 `_DEFAULT_CFGS` 里补 `(16,32,128,5,4)` / `(16,32,128,4,4)`
(现有列表在 BN=32 上只有 `warps=2` 的版本,很可能是它落选的原因):BN 32 ⇒ 128 CTA,
smem 48 KB ⇒ 2 CTA/SM,双赢,且只动 `BLOCK_M`/`BLOCK_N`,仍然 bit-exact。
⚠️ 这条"下一步"是 §3.2b 之前写的推断;§3.2b 的实测已经说明**"多造 CTA"不是普适规律**,
所以它仍然只是候选,要以 benchmark 为准。

split-K 那条路(微基准 `mb_splitk.py` 测到 11.55 µs,−18 %)**降级为备选**:它改 K 方向的
fp32 归约顺序,**不是 bit-exact 的**,而"多造 CTA"这条路在同一个核上还没走完。

### 3.2 已证伪:SwiGLU 的剩余带宽**靠调 tile 拿不到**

这里以前写的是"给 SwiGLU 融合核扫 tile,离流式地板还有 25 %,~0.8 ms"(后来 ncu 把口径修
成:973 GB/s = 可达 1222 的 **79.6 %**,理论空间 ~3.5 µs/层 ≈ **0.63 ms/predict**)。
**扫完了,结论是拿不到。**

8 个配置(只动 `BLOCK_M`/`BLOCK_N`/`warps`/`stages`,`BLOCK_K` 钉死 64):

| 变化方向 | grid(CTA) | 核时间 |
|---|--:|--:|
| **现行 `(64,32,64,4,4)`** | 128 | **18.89 µs** ← 最快 |
| 缩 tile 造更多 CTA | 256 | 20.63 / 21.75 µs |
| 再缩 | 512 | 22.30 µs |
| `warps` 4 → 8 | — | 慢 **17 %** |

**CTA 变多是单调变慢的**,和 §3.1 里"多造 CTA 就变快"的方向**相反** —— 因为这个核已经是
六个里**唯一 110 个 SM 全部开工**的(SM active 84.4 %),再切碎只会摊薄每个 CTA 的访存效率。
**8 个配置的输出逐位相同**,证实 `BLOCK_M`/`BLOCK_N`/`warps`/`stages` 对数值惰性,
所以这轮扫描本身是零风险的。

> ⚠️ 这张表的 18.89 µs 和上文的 17.35 µs 是**两把尺**(扫描用的是隔离微基准,17.35 来自
> nsys 稳态),只能组内比较,不能跨表相减。

**结论:SwiGLU 那 ~20 % 的带宽空间不在 tile 里,要改结构。** roofline 侧的读数是
AI ≈ 47–50 FLOP/byte,而这张卡的 machine balance 是 179 FLOP/byte(按 spec 1344 GB/s)、
~196(按实测可达的 1222 GB/s)—— 它离拐点差 ~4×,确实是访存受限;剩下的 20 %
有两个可见来源:(i) 128 CTA 填 220 个槽位,18 个 SM 拿 2 个、92 个拿 1 个,
`cyc_active` min/max = 0.85 的不均衡;(ii) L2/DRAM = 2.03、L2 hit 50.1 %,即 A 矩阵被
128 个 N-tile 重读 ~13 MB,吃掉的是 L2 带宽。两条都不是 tile 参数能解的。
**状态:关闭,不再扫 tile。**

<a id="s-3-2b"></a>

### 3.2b 已落地:P·V 的 attention `bmm` 换 tile 长宽比(−0.18 ms)

**观测(先是一个实现遗漏)。** §3.1 的 patch 只换了 `torch._inductor.kernel.mm.mm_configs`,
而 `torch._inductor.kernel.bmm` **在 import 时就按值绑定了 `mm_configs`** —— 所以两个
attention `bmm` **一次都没被碰过**。这是已确认的实现遗漏,不是取舍。修法是直接 patch
`bmm_configs` 本身(`install_small_m_bmm_configs()`)。

ncu 当时的读数:`bmm_7`(P·V)全 GPU 有效占用 **3.9 %**,DRAM 只有 13.1 %,主 stall 是
`wait`(定长依赖)29.6 % + `long_scoreboard` 23.6 %,而 L2 hit 87.4 % —— **等的是 L2 往返
延迟,不是 DRAM 带宽**,而 L2 延迟正是"多几个 warp 就能掩盖"的那一类。
`bmm_7` 的 96 KB smem 把它压到 1 CTA/SM(**寄存器不是限制因子**:
254 regs 允许 2 个,所以"降寄存器压力"是无效动作),而 64 CTA 只够 110 个 SM 里的 64 个开工。

**怎么做,以及 ncu 的处方被实测推翻。** 上面那份 occupancy 剖析给出的处方是
`BLOCK_M=16`:M 方向 ceil(50/16)=4 个 tile ⇒ grid 4×8×8 = **256 CTA**,smem 降到 48 KB
⇒ 2 CTA/SM,而 M 方向覆盖的行数仍是 64,**零额外浪费**;只动 `BLOCK_M`、`BLOCK_K` 保持
128 ⇒ 仍然 bit-exact。纸面上完全说得通。**实测 `BLOCK_M=16` 比现状慢 30 %。** 真正赢的候选是
**`BM32/BN64/BK128/stages 4/warps 4`**,它**保持同一个 64-CTA grid、同样 1 CTA/SM**,
赢在 **tile 长宽比**:固定 `num_stages` 时 `BM64×BN32 → BM32×BN64` 是 **−18.8 %**,
而随之而来的 `num_stages` 5 → 4 只值 **−5.5 %**。

**量了多少。** P·V = `bmm(8×50×1018, 8×1018×256)`,是去噪循环里 SwiGLU 之后**最贵的单个核**:

| 口径 | before | after |
|---|--:|--:|
| `triton_tem_fused_bmm_7` | 6.214 µs/call | **5.208 µs/call(−16.2 %)** |
| 折算(1.006 µs × 18 层 × 10 步) | — | **−0.18 ms/predict** |
| denoise stream busy | 10.556 ms/predict | **10.401**(kernel 数一个不变) |

**Q·Kᵀ 什么都拿不到,而且是故意不给的。** 它的库存冠军已经在扫过的 28 个配置的最优值
**1.6 % 以内**,而且 inductor **根本分不开这个 shape 的头部** —— 9 个候选挤在
0.0056–0.0062 ms 之内,里面 `BLOCK_K` 有 32、64、128 三种。也就是说**这个 shape 即使不打
patch 也不是跨冷 autotune 比特稳定的**,再加候选只会把它重新掷一次骰子。

**这一条修正了 `BLOCK_K` 的规矩。** 它不是全模块钉死 128,而是**按 shape 钉在库存冠军
用的那个值上** —— 因为那才是未打 patch 的 build 会产生的 fp32 归约顺序。
`_DEFAULT_BMM_SHAPES` 带着这个值,每个候选都对它 assert。

**数值。** 核级 gate 是新增的 `tools/bitexact_denoise_bmms.py`:三个进程(off / on / off,
共享 inductor cache),覆盖全部 18 层、生产 shape 与 `repeat_kv` 的广播 layout,
**产出同一个 digest**,并且确认两臂真的编出了不同 tile
(0.0102 ms 的 BM64/BN32 vs 0.0082 ms 的 BM32/BN64)。
端到端 `--dump-actions` 门维持 **INCONCLUSIVE**,理由和别处一样:它自己的 off-vs-off
空对照就差 **2.46e-3**。
Kill switch:`RLINF_SMALL_M_BMM=0`。
调参口子:`RLINF_SMALL_M_BMM_SHAPES` / `RLINF_SMALL_M_BMM_CFGS` / `RLINF_SMALL_M_BMM_MAX_M`。

### 3.3 顺带:把每步都一样的 glue 也预计算掉(~0.5 ms 还在)

图里那根橙色的 **eager per-step glue**,严格说属于第 ① 段而不是第 ③ 段。
它原本是 99.7 µs/step、~70 kernel/step,其中**纯死代码**的那一半已经由 §1.5 摘掉,
现在是 **50.6 µs/step、49 kernel/step**。

剩下的那一半不是死代码,是**逐步重复**:position id 的 `cumsum`/`DeviceScan`、
rotary 表的 `cos`/`sin`、`_get_timesteps` 的 `linspace` —— 这些在 10 个 Euler 步上
**完全相同**,和已经预计算掉的 adaRMS 调制量是同一类问题,同一套解法(预计算 + 每步 gather)。
另外还有真正每步都不同的 Euler 更新与 log-prob,那部分动不了。

**状态:部分已做。** 其中 attention mask、position ids 和 RoPE 的 cos/sin 表已经在
[「把步不变量从去噪循环里外提」](#s-hoist)里外提掉了(kernels/step 217 → 190,
e2e −0.32 ms);`_get_timesteps` 的 `linspace` 与其余零碎还留着 —— 原估的 ~0.5 ms 里已兑现
0.32 ms,**剩下那部分没有单独量过**。

### 3.4 明确排除的做法

- ⛔ **算法层一律不做**:不减去噪步数、不蒸馏、不换采样器、不引入 staleness 的重叠。
- ⛔ **不做任何降精度**:fp8 / int8 / 任何量化都不在选项内 —— 本项目的前提是
  **不降精度、不做近似**(所有改动都是代数等价变换)。⚠️ 这个前提以前写成
  "全部前提就是 `max|Δ| = 0.00e+00`",那个更强的说法**事实上已经不成立**了
  (见[§ 正确性](#s-correctness));`0.00e+00` 现在只留给真的做到的那几项。
- ⛔ 已经排除、别再试的:整模型单图(prefix 是 compute-bound,收益 ≈ 0,且被 compile 嵌套
  挡住)、强制 Triton-only 后端(§② 开头那个已结的负结果)、把 gated residual 塞进 GEMM
  epilogue(本来就已融进消费端 RMSNorm,kernel 数不变)、给 SwiGLU 融合核扫 tile
  (§3.2)、给 P·V 的 bmm 用 `BLOCK_M=16`(§3.2b,慢 30 %)、给 Q·Kᵀ 加候选
  (§3.2b,库存冠军已在最优 1.6 % 内,加候选只会重掷骰子)、**在 prefix 上做 GeGLU
  epilogue 融合**(见「台账之后」,入场费 +44 µs/层 > 奖品 28 µs/层,实测两次都亏)。

---

<a id="s-ceiling"></a>

## 天花板在哪

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/phases_dark.png">
  <img alt="GPU busy 的阶段拆分:prefix 71.7%,denoise 28.3%" src="docs/phases_light.png">
</picture>

去噪循环只占 GPU busy 的 **28.3 %**。即使把它优化到 0,整个 predict 的加速上限也只有
**1.39×**。真正的大头是那个 968 token 的 prefix(PaliGemma 语言塔 24.10 ms + SigLIP
视觉塔 4.82 ms)。这就是为什么 §1.4(att_masks)虽然自己只值 0.31 ms 却仍然重要 ——
它解锁了 prefix 阶段的 CUDA 图捕获。

⚠️ 这张图和上面这两个数拍摄于 prefix 跳层之前。跳掉第 17 层之后 stream 7 是
23.95 → 22.83 ms/predict,denoise 侧一个核都没变,所以 prefix 的占比只是从 ~71.7 %
略降,结论方向不变。**"大头在 prefix"这条正是「跳过 prefix LM 最后一层」的由来** ——
它是本仓库第一项打在 prefix 上的优化,第二项是[prefix QKV 融合](#s-prefix-qkv)(−671 µs)。
同类审计还没做完:
`model.norm`(968×2048 的 RMSNorm,同样只喂给被丢弃的 `last_hidden_state`)还留着
~20–30 µs,没摘是因为训练的 joint 分支会直接调 `models[i].norm(...)`,改它会误伤训练路径;
SigLIP 最后一层的同类审计**没做过**。

顺带解释上一张图里的另一个数:expert block 本身现在是 **163 kernels/step**,参考实现是
165 —— 在 transformer 内部,kernel 数的差距已经抹平,两个融合块的执行时间还反超了。
剩下的 217 − 163 = 54 个 kernel 全是 expert 之外的 eager per-step glue(§3.3)。

图里的数字可以从 profile 重新推导:
`python docs/make_charts.py --sqlite <stage1_on.sqlite> --sqlite-off <stage1_off.sqlite>`
会把推导结果和图里写死的常量并排打印出来。

---

<a id="s-baselines"></a>

## 与参考实现的对比

我们做过正面对比的是 **[`dexmal/realtime-vla`](https://github.com/dexmal/realtime-vla)**
(`@ b86a942`),同一天、同一张卡、逐项对齐的配置、同一个计时 scope、双方都用非退化权重:

| | theirs | ours(当天 build) |
|---|--:|--:|
| e2e(scope B,n=30) | **43.41 ms** | 44.55 ms |
| prefix(时钟归一化) | 64.5 Gcycles | **63.6**(我们少 1.4 %) |
| denoise(时钟归一化) | **30.1 Gcycles** | 35.1(他们少 14.3 %) |
| kernels / denoise step | 165 | 306(当时) |

**当天他们快 1.14 ms(2.6 %)。** 之后 §② 的 kernel fusion 把 expert 拉到 163 kernels/step,
两个融合块的执行时间反超(SwiGLU 312 µs vs 380,QKV+RoPE 132 µs vs 144)。

⚠️ 本仓库当前的 42.90 ms 与他们的 43.41 ms **不是配对测量** —— 相隔一天、不同 session、
不同 build,差值(0.51 ms)小于本机记录在案的 ±0.7 ms rebuild variance。**不能**据此
宣称超越。把 realtime-vla 当作同水位的 peer 看待即可。
**唯一做过配对的那次正面对比,是他们 43.41 快过我们 44.55,领先 1.14 ms。**

### 另一个参考实现:`limxdynamics/FluxVLA`

[`limxdynamics/FluxVLA`](https://github.com/limxdynamics/FluxVLA)(@`7f9f774`)是**另一个
仓库**,不是 realtime-vla —— kernel 同名只是因为 FluxVLA 把 realtime-vla 的 Triton 核重新
实现了一遍,**行为并不相同**。在同一张卡、他们真实的 LIBERO-10 π0.5 权重下实测:

| | FluxVLA @ 968 token | FluxVLA @ 560 token(他们的默认) |
|---|--:|--:|
| e2e | **44.9 ms/predict** | 31.1 ms |
| denoise | 1419.0 µs/step(union-busy;marker 口径 1404) | — |
| kernels / denoise step | ~205 | — |

⚠️ **那个 31.1 ms 绝对不能拿来和我们的数比。** 它是 2 视角 + 48 语言 token 的**更轻**配置,
和 968 token 差的那 ~14 ms 纯粹是 workload,不是优化。

⚠️ **44.9 ms 这个数,曾经有一条结论被撤回,必须说清撤的是哪一部分。** 被撤回的是它的
**归属和判词**,不是这次测量本身:当时它被记成"**他们(= realtime-vla)的代码跑在我们的
配置下 = 44.89 ms,打平**",而实际上跑的是 **FluxVLA 这个不同的仓库**,并且那份 LIBERO
配置的 `n_action_steps=10`,即 **chunk 10 而不是 50**。所以"dead heat"这个判词、以及由它
派生的"他们每步重算 adaRMS 所以我们有优势"一条,都已作废 —— realtime-vla 在
`Pi05Inference.__init__` 里就把 37 × 10 个 adaRMS 投影预计算完了,和我们一样,是**平手**
而不是优势。作为"FluxVLA 在 968 token 下的一次实测",44.9 ms 仍然成立。

所以本仓库**不把 FluxVLA 当作 e2e 的正面对手**:它和我们的配置在两个方向上都没对齐 ——
chunk 10(比我们轻)、他们的计时器跳过我们含在里面的 ~2–3 ms CPU 预处理(也比我们轻),
而他们的实现每步重算 adaRMS 与 time MLP、每次调用还有一次 `.item()` device sync(比我们重)。
图里的 FluxVLA 线因此画成灰色虚线并标注 "NOT config-matched",只当参考点看,不做胜负判断。

---

<a id="s-correctness"></a>

## 正确性:代数等价 ≠ 逐位一致,是两件事

这一节 2026-07-28 重写过。以前它论证的是"每一项都 `0.00e+00`";补验之后那个总括说法
**站不住**,现在分两级说。

### 第 0 条:判据是分级的,别用一个弱门去证强结论

| 判据 | 例子 | 可复现? |
|---|---|---|
| **核级 / 张量级 / GEMM 级** | `tools/bitgate.py`、`tools/bitexact_denoise_gemms.py`、`tools/bitexact_denoise_bmms.py`、`tools/bitexact_prefix_kv.py`、`--attmask-tensor-check` | ✅ 是最强的,与端到端噪声无关 |
| **同进程双臂** | `tools/bitexact_siglip_batch.py`、`tools/bitexact_extraction.py` | ✅ 同一套 autotune 状态、同一个 cuBLAS handle,跨进程漂移进不来 |
| **冻结 prefix + 四进程门** | `tools/bitexact_compiled_toggles.py --freeze-prefix` | ✅ 把唯一会跨进程漂的那一级(SigLIP)整个摘掉之后,base `max-autotune` 的端到端门就可复现了 |
| **裸的端到端 `--dump-actions`** | `bench/standalone_infer_bench.py --dump-actions` | ⚠️ **base 模式下跨进程噪声底 ~4–5e-3**,证不伪 |

那个噪声底的根因已经定位:`max-autotune` 会在**首次启动时 benchmark** SigLIP 视觉塔的
LayerNorm 归约核并挑 launch config,这个选择**跨进程会跳变**;`R0_BLOCK`/`num_warps` 一变,
Welford 累加的切分就变,bf16 输出的最后几位就变,经 prefix KV → 10 步 denoise 传到动作上。
"选中的 config 集合相同 ⟺ 输出逐位相同",在记录了 winner 的 10 个 run 里 **0 例外**。
**所以历史上那些端到端 `0.00e+00` 不是假的,只是它们那次恰好 winner 没动;这个门通不过
"能证伪"的检验,不能拿来支撑强结论。**

### 第 1 条:两级结论

* **代数等价 —— 全部成立。** 每一项改动都是恒等变换,没有一项算错。
* **在出货的 `max-autotune` 路径上逐位相同 —— 只有下面这些**(每行的判据强弱按上表口径
  如实标注,带 ⚠️ 的两行判据偏弱):

  | 项 | 判据 | 结果 |
  |---|---|---|
  | 在 GPU 上构造 attention mask | 同进程张量级(5 个张量的值 + shape + stride + dtype) | ✅ PASS(bs=1,完备证明) |
  | 把 SwiGLU 与 RoPE 融进 GEMM 尾部 | `bitgate.py` 核级,参照系是 inductor 自己编出来的那批 kernel | ✅ PASS,`0.00e+00` |
  | 小 M 的 mm tile 候选 | `bitexact_denoise_gemms.py`,18 层 × 2 个 GEMM 的 sha256(真实权重、生产 stride) | ✅ PASS,含"两臂确实编出不同 tile"那一组 |
  | P·V 的 bmm 换 tile | `bitexact_denoise_bmms.py`,三进程 off/on/off、18 层、生产 shape + `repeat_kv` 广播 layout | ✅ PASS,同一 digest,两臂确实编出不同 tile |
  | 跳过 prefix LM 最后一层 | `bitexact_prefix_kv.py`,18 层 36 个 KV 张量的 sha256 | ✅ PASS **eager 36/36**;⚠️ 编译态只到"不比重编一次更糟",见 §「台账之后」 |
  | 把步不变量从去噪循环里外提 | 核级:生产 shape 下 eager vs inductor 编出的 cos/sin、mask、position ids | ✅ PASS,`0.00e+00`(cos/sin 0 / 12800) |
  | prefix LM 的 Q/K/V 并成一个 GEMM | `bitexact_prefix_qkv.py`,两臂 0 层 vs 17 层融合,36 个 KV 张量 + combined digest | ✅ PASS,36/36 `0.00e+00`,digest 一致 |
  | 删掉没人读的 timestep 条件计算 | 端到端 dump,两种 compile mode × 开关 on/off + 参考臂共 10 对 | ⚠️ 弱门,但 10/10 全过、且删的是确凿的死代码(`elif` 分支永不进入) |
  | 把整个去噪步捕获成一张 CUDA 图 | 端到端 dump(on vs off、on vs RLinf、off vs RLinf) | ⚠️ 弱门,`0.00e+00`;代数上无损(`flow_ode` 的 `x_t_std ≡ 0`,`sample_noise` 留在图外) |
  | **从 RLinf 剥离这件事本身** | 同进程双代码树逐级 digest(`noise0` → `prefix/kv` → 10 步 → `actions`) | ✅ PASS,24/24 digest 相同,`actions` 的 `max\|Δ\| = 0.00e+00`,0/300 |

* **只在 eager 下逐位相同,编译路径上 FAIL 的三项**:预计算 adaRMS 调制量(`2.568e-3`)、
  Q/K/V 并成一个 GEMM(`2.431e-3`)、prefix KV 静态缓冲区(`2.858e-3`)。
  三条 FAIL 的共同机制是**同一类**:改动动了某个 GEMM/归约的形状(N: 2048/256/256 → 2560)、
  或把计算**挪出了编译区**(eager 建表 vs 图内投影)、或换了张量的来源(`cat` vs 常驻 buffer),
  inductor 因此换了 tile / 换了累加分块 —— **数值等价,比特不等,没有一条是算错**。
  其中 adaRMS 那一条把**完全同一个门**用 `--no-compile` 又跑了一遍:同一份代码、同一个
  toggle、同一个判据,`0.00e+00`(0/300)—— **只换编译模式,结论就翻转**。
  另两条的 eager `0.00e+00` 是历史记录(`test_qkv.py` 等),本次**没有**用同一个 harness 重跑。
* **前史里的 SigLIP 三路合批也是 FAIL**(4.582e-3 eager / 2.528e-3 编译),见「台账之前」。

⚠️ **两条必须说清的限制**:

1. **三条 FAIL 没有在 `--stage1` 下重跑。** 补验全部跑在 base `max-autotune`,而台账第 7 行
   之后的数字用的是 `--stage1`。Stage-1 把这些算子包进手抓图,**不保证结论相同** ——
   严格说这三个 ✗ 目前只对 base `max-autotune` 成立。
2. **`att_masks` 只验了 bs=1**(出货配置就是 bs=1);`bs>1` 时原实现的 `expand` 是 stride-0
   视图,**未测**。

另外记一条相关的事实(§3.2b):**Q·Kᵀ 这个 shape 即使完全不打 patch,也不是跨冷 autotune
比特稳定的** —— inductor 分不开它头部的 9 个候选(0.0056–0.0062 ms,`BLOCK_K` 横跨
32/64/128)。这也是不给它加候选的理由之一。

### 第 2 条:这些门自己有没有被验过

有。补验一共跑了 **26 个同臂空对照,26/26 全部逐位相同**(6 个四进程门各 off/off + on/on
共 12、SigLIP 同进程 A/B 的 12、剥离等价性的 2)。所以每一条 FAIL 都是"两臂真的不同",
不是门在放空炮;每一条 PASS 也都是在**证明过有分辨力的门**上拿到的。
配对 A/B 的两臂也必须先验证"确实不同"(arm signature / kernel census)—— §1.5 的坑就是
两臂跑了同一个 build,那次其实是一次空对照。

具体怎么跑见 [README 的「验证」一节](README.md#r-verify)。

---

<a id="s-methodology"></a>

## 测量方法学

这部分和优化本身同等重要 —— 上面所有数字的可信度都建立在它上面。

1. **只信 plain wall clock,永远不信 nsys 的 wall time。** `--cuda-graph-trace=node` 会
   膨胀 `cudaGraphLaunch` 的 CPU 时间,GPU-metrics 采集再叠 3–4 ms。这已经造成过至少两个
   被撤回的结论。nsys 只用来读**确定性**指标:launch 计数、memcpy 计数与字节、kernel 计数、
   per-kernel GPU 时间。
2. **配对 A/B,同一份源码,串行,至少 4 轮交替。** 单次运行的 e2e 抖动是 ±1 ms,大于绝大多数
   bubble 级效应;**rebuild variance 是 ±0.7 ms**(同一份代码重新编译一次就能差这么多)。
   一次运行分辨不出小回归。
3. **记录 SM 时钟。** 这张卡卡在 300 W,运行时间和时钟近似 1:1 反比。冷启动的 30 次迭代和
   持续负载能差 ~8 %。只比较时钟相当的两臂 —— 上面每一轮配对里两臂都在 ~20 MHz 以内。
   **<1 ms 的效应必须锁频**(见「台账之后」里 small-M 那条:不锁频的 raw 读数被 29 MHz 的
   时钟差污染成 −0.35 ms,真值是 −0.88)。
4. **GPU 必须独占。** predict 里有 ~1.5 ms 的串行 CPU 段,一个并发任务曾经吃掉 1.5 ms 的
   CPU 竞争并让我们得出错误结论。
5. **`idle` 只在同一 session 内可比。** 同样的代码、同样的 GPU busy(1246.7 vs 1247.8
   µs/step),两次采集的 idle 能差 72 µs/step —— 那是 host 侧 stall,不是代码。
6. **必须按 stream 分开看**(7 = prefix,157 = denoise,158 = vision)。§2.5 里 +4 ms 的
   回归全在 prefix,只看 denoise 指标完全看不见。
7. **denoise 的逐核 GPU metrics 不可用**:`--gpu-metrics-frequency=20000` 的采样周期是
   50 µs,而 denoise 的核是 0.5–21 µs,一个采样点是 5–30 个相邻核的平均值。只能引用相位级
   数字(prefix ≈ 32 ms / denoise ≈ 12 ms,都远大于 50 µs)。

### ⚠️ `torch.empty` 零权重陷阱

这条值得单独拎出来,因为它会让任何人的 benchmark 数字凭空好看一大截:

**driver 返回的 `torch.empty` CUDA 页是全零的。** 零操作数几乎不产生晶体管翻转 →
动态功耗极低 → GPU 撞不到功耗墙 → 时钟一路 boost → 延迟数字凭空变好。

我们在**自己的** stack 上做了对照实验(6 轮拉丁方交替以排除热漂移、每种填充 1000 次计时
迭代、kernel census 完全相同:145 个 distinct kernel、3903 次 launch、零差异):

| 权重 | e2e(中位) | SM 时钟 | 功耗 | 功耗墙 throttle |
|---|--:|--:|--:|--:|
| 真实 checkpoint | 48.47 ms | 2251 MHz | 299.9 W | 99.0 % |
| 随机 N(0, 0.02) | 48.26 ms | 2259 MHz | 299.8 W | 100 % |
| `torch.zeros` | **42.78 ms** | **2559 MHz** | 273.9 W | 32.8 % |

时间比 **1.1330** vs 时钟比 **1.1368** —— 吻合到 **0.33 %**,即 100 % 是时钟效应,
零性能含量;随机权重与真实 checkpoint 无法区分,所以问题是**零**,不是"不是真权重"。
效应的 **95.6 % 落在 compute-dense 的 prefix**(32.33 → 26.24 ms);
denoise 则是干净的阴性对照 —— 零权重让它省了 37.5 W,却只快 0.28 ms(1.4 %),
因为它**本来就没被功耗墙压着**。

同样的对照在 `dexmal/realtime-vla` 公开的 `benchmark.py` 上也成立:它把权重留成
`torch.empty(...)`(实测 2,826,721,040 个元素,**0 个非零**),同一份代码在我们的配置下
零权重 37.0 ms、随机权重 43.1 ms,**虚高约 18 %**。

**在有功耗墙的 GPU 上,永远不要用 `torch.empty` / `torch.zeros` 的权重测延迟**,
并且永远把 `clocks.sm` / `power.draw` / `clocks_event_reasons.sw_power_cap` 和延迟一起发表:
如果同一份代码的两次运行报出不同的平均 SM 时钟,那么在被证伪之前,延迟差就是时钟差,
跨越它们比较 ms 是没有意义的 —— 要比就比 cycle。

---

<a id="s-isolation"></a>

## prefix / expert 的隔离

`import pi05_infer` 拉进来的是**它自己**的模型代码:

```
engine.OpenPi0Inference
  └─ pi05_infer.openpi_patched.pi0_pytorch.PI0Pytorch
       └─ pi05_infer.openpi_patched.gemma_pytorch.PaliGemmaWithExpertModel
            ├─ paligemma    = transformers.PaliGemmaForConditionalGeneration   ← 原版
            └─ gemma_expert = pi05_infer.gemma.modeling_gemma.GemmaForCausalLM ← 我们的
```

这一个构造点就是全部的强制机制,`tools/isolation_check.py` 逐模块断言它。这个隔离不是
形式主义:§2.5 那次 +4 ms 的回归,原因正是 PaliGemma 的 **prefix** 语言模型也是 Gemma,
全局覆盖 `transformers/models/gemma/` 会让一个为 50 token denoise suffix 调优的 kernel
打到 968 token 的 prefix 上。现在 prefix 和 expert 是来自不同文件的不同类,这类 bug
结构上不可能发生。

只有真正改过的符号才被 vendor;`modeling_gemma.py` 其余全部从安装好的 transformers
import。边界的完整说明(以及这**没有**带来独立性的那一处)见
[`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md) §8。

### 这条边界被硬碰硬地验证过一次

GPU 机容器里的 `site-packages/.../modeling_gemma.py` 曾在某个时点被替换成我们 fork 的
旧版本,现已恢复成原厂文件(只保留 openpi 自己的 `use_adarms` 补丁)。恢复前后的 profile
显示:去噪侧(stream 157)**44 个不同核的集合完全相同、每个核的 launch 数逐一相同、
总 launch delta = 0** —— expert 侧从未依赖容器那份 fork,vendoring 边界是真实的。
(这同时印证了 `pi05_infer::` 这个算子命名空间从没被 `rlinf::` 抢过。)prefix 侧的计时
同样未变;**变了的是 prefix 的数值,而且是变好** —— [prefix QKV 融合](#s-prefix-qkv)
的那次配对测量就是在恢复之后做的,这也是为什么它的绝对基线和别的 session 对不上。

<a id="s-opnamespace"></a>

### 自定义算子的命名空间

因为 `torch.library` 的命名空间是进程全局的,而容器里那份旧的 `rlinf_fused_denoise.py`
也会被 import,本包把自定义算子注册成 `pi05_infer::gate_up_swiglu` /
`pi05_infer::qkv_rope_kv` 而不是 `rlinf::*` —— 否则第二次注册会抛错并**静默关掉**融合。

<a id="s-stage1"></a>

### `--stage1` 替你处理的两件要命的事

* 把 `--compile-mode max-autotune` 改写成 `max-autotune-no-cudagraphs`。inductor 自己的
  cudagraph 不能嵌套在手写图里,失败模式是
  `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten`。
* warmup 之后**断言** `is_cuda_graph_enabled()` 和 `_denoise_graph_captured`。图是在第一次
  eval-shape 的 `sample_actions` 上懒捕获的,shape signature 不匹配就会**静默**退回 eager
  loop,除了跑得慢一点没有任何症状 —— 这个包最初漏掉这个 flag 就是这么发生的。

<a id="s-fallback"></a>

### 融合核的降级路径

融合核都带**降级路径**:kill switch(`RLINF_FUSE_SWIGLU=0` / `RLINF_FUSE_QKV_ROPE=0`)、
triton 不可用、激活不是 tanh-GELU、dtype 不是 bf16/fp16、权重非行主序、shape 除不尽 tile、
static KV 未 prime(训练 / `use_cache=True`)、需要 autograd —— 任一条件成立就静默走回原始
PyTorch 路径。A/B 的每一个 OFF 臂都走的是这条降级路径,所以它是被验证过的,不是被假设的。

§1.5 的消除也带 kill switch:`RLINF_SKIP_DEAD_ADARMS_COND=0` 恢复"每步都算 `adarms_cond`"
的旧行为(import 时读一次,所以它是编译期常量,不影响 CUDA 图捕获)。

---

<a id="s-limits"></a>

## 已知限制 / 没做的事

* **第 ③ 段只兑现了一部分**:small-M 重 tile 落地了 −0.88 ms、P·V 的 bmm 重 tile 落地了
  −0.18 ms,但 `down_proj` / `o_proj` 仍然只有 64 CTA、46 个 SM 全程空转;`bmm_7` 也仍是
  64 CTA / 1 CTA-per-SM。
  ⛔ **SwiGLU 的 tile 扫参已证伪,不要再试**(§3.2):8 个配置里现行的最优,
  CTA 变多单调变慢;那 ~20 % 的带宽空间要改结构才拿得到。
  ⛔ **P·V 的 `BLOCK_M=16` 已证伪**(§3.2b,慢 30 %);**Q·Kᵀ 不加候选**(库存冠军已在
  28 个配置的最优 1.6 % 内,且这个 shape 本来就不跨冷 autotune 比特稳定)。
* **per-step-invariant glue 只外提了一部分**(§3.3):mask / position ids / RoPE cos-sin 已经
  外提(−0.32 ms,kernels/step 217 → 190),`_get_timesteps` 的 `linspace` 等零碎还留着。
* **去噪图路径上还留着一次死拷贝** `_copy_kv_into_static`(17.8 MB/predict,~0.03 ms)。
* **不做**任何算法层改动、不做任何降精度:全部是代数等价的推理侧改动。⚠️ 但**不能**
  再说"逐比特保持模型输出" —— 三项优化在出货的编译路径上会产生 ~1 % 动作幅度的 bf16 级
  重排差异,见[§ 正确性](#s-correctness)。
* **`--dump-actions` 在 base `max-autotune` 下跨进程不可复现**(噪声底 ~4–5e-3,根因在
  SigLIP 侧的 LayerNorm 归约核选 launch config)。这是个**独立 bug**,修好之前任何编译态的
  端到端 bit-exact 验收都立不住 —— 所以现在一律走 `tools/bitexact_gate.sh` 的四进程门。
* **prefix 跳层是条件性的**:检测到 RLinf 的 VLM value head(`value_after_vlm and
  add_value_head`)就不安装,已发布的 19 份 pi0.5 PPO 配置里有 15 份命中这个条件。
  另外 `model.norm`(~20–30 µs)和 SigLIP 最后一层的同类审计都还没做。
* openpi 侧文件的三方合并、把 `openpi_action_model.py` 接回本引擎、把容器
  `site-packages` 还原成 openpi 原版 —— 见 `EXTRACTION_NOTES.md` §7–§8。
* **RL 集成**:CUDA 图在权重原地同步后**不需要重捕**(它记的是地址),但
  `qkv_fused_weight` / adaRMS 表这类**权重派生量会静默过期**;刷新必须原地 `copy_`
  不能重分配。`invalidate_weight_derived_caches()` 就是这个钩子。

---

<a id="s-inventory"></a>

## 附录:仓库文件清单

[README 的仓库结构](README.md#r-layout)只到目录级,这里是逐文件的对照(常用命令见
[README 的「验证」一节](README.md#r-verify))。

| 文件 | 作用 |
|---|---|
| `pi05_infer/engine.py` | `OpenPi0Inference`:纯推理编排。`predict_action_batch` → `sample_actions` → `sample_mean_var_val` → `get_velocity` → `get_suffix_out`,以及 `_build_prefix_cache`、`enable_torch_compile`、手写去噪 CUDA 图(代码里叫 Stage-1)、adaRMS 调制表和 `invalidate_weight_derived_caches` |
| `pi05_infer/builder.py` | `build_model()`:checkpoint + norm-stats + transform 组装 |
| `pi05_infer/dataconfig/` | RLinf openpi dataconfig 的最小子集(turtle + libero) |
| `pi05_infer/_vendored/` | 无 RLinf 依赖的 helper 逐字副本:`base_policy` / `cuda_graph` / `nvtx` |
| `pi05_infer/gemma/` | 动作专家跑的那份 Gemma fork:`modeling_gemma.py`(相对 transformers +245 行)+ `rlinf_fused_denoise.py`(两个 Triton 融合核) |
| `pi05_infer/openpi_patched/` | 我们改过的两个 openpi 文件:`pi0_pytorch.py` + `gemma_pytorch.py` |
| `pi05_infer/inductor_mm_tiles.py` | 给 M≤64 的两个 denoise GEMM 补小 `BLOCK_M` 候选(`RLINF_SMALL_M_MM`),以及给 P·V 的 attention bmm 补候选(`RLINF_SMALL_M_BMM`) |
| `pi05_infer/prefix_last_layer.py` | 跳过 prefix LM 最后一层的死算(`RLINF_SKIP_LAST_LM_LAYER`;检测到 VLM value head 时自动不安装) |
| `pi05_infer/prefix_qkv_fused.py` | prefix LM 前 17 层的 Q/K/V 并成一个 GEMM(`RLINF_FUSE_PREFIX_QKV`) |
| `bench/standalone_infer_bench.py` | 延迟基准(e2e、分阶段、nsys、actions dump) |
| `_extract_src/` | 抽取前的 RLinf 原始文件(未重构) |
| `tools/isolation_check.py` | 证明 expert = `pi05_infer.gemma`,prefix = transformers |
| `tools/bitgate.py` | 两个 Triton 融合核的位级一致 gate(核级) |
| `tools/bitexact_denoise_gemms.py` | small-M mm 重 tile 的 GEMM 级 sha256 gate(§3.1) |
| `tools/bitexact_denoise_bmms.py` | P·V bmm 重 tile 的核级 digest gate,三进程 off/on/off(§3.2b) |
| `tools/bitexact_prefix_kv.py` | prefix 跳层的 KV 级 sha256 gate(18 层 36 张量) |
| `tools/bitexact_prefix_qkv.py` | prefix QKV 融合的 KV 级 gate(0 层 vs 17 层融合,36 张量 + combined digest) |
| `tools/bitexact_gate.sh` | 端到端 gate,四进程、强制空对照,不干净就判 INCONCLUSIVE |
| `tools/bitexact_siglip_batch.py` | SigLIP 三路合批的同进程三级 A/B(「台账之前」) |
| `tools/bitexact_extraction.py` | 「从 RLinf 剥离」的同进程双代码树逐级 digest |
| `tools/bitexact_compiled_toggles.py` | 冻结 prefix + 四进程门,编译路径上的结构性优化补验 |
| `tools/bitexact_adarms_dense.py` | adaRMS 预算表 vs 编译区内投影核的直接对拍(§1.1) |
| `tools/run_bitexact_backfill.sh` | 上面几个补验的驱动,一个 stage 一条命令 |
| `tools/determinism_probe.py` | 逐级 digest + 记录 inductor autotune 冠军,查 dump 不可复现 |
| `tools/ab_rlinf_reference.py` | 与 RLinf 参考臂的 `--dump-actions` 对拍 |
| `tools/ab_stage1.sh` / `ab_stage1_summary.py` | `--stage1` 的配对 A/B 驱动与汇总 |
| `tools/ab_small_m_mm.sh` / `ab_small_m_bmm.sh` / `ab_skip_last_lm_layer.sh` | 三项台账之后优化的配对 A/B 驱动(支持锁频) |
| `tools/step_idle.py` / `stream_summary.py` / `denoise_kernels.py` / `ksum.py` / `prof.sh` | profile 分析:每步 idle、按 stream 汇总、逐核普查、核时间求和、采集脚本 |
| `docs/make_charts.py` | 重新生成 README 的三张图(含从 sqlite 重新推导) |
| `docs/MEASUREMENTS.md` | 完整测量记录 |

---

完整的原始测量记录见 [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md);
成果概览回到 [`README.md`](README.md)。
