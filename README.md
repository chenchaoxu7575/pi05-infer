# pi05-infer

**π0.5 动作专家(action expert)的独立 bs=1 推理引擎**,从 [RLinf](https://github.com/RLinf/RLinf)
里抽出来,针对 **RTX PRO 5000(GB202 / sm_120,Blackwell)** 做过一轮系统性优化。

端到端 `predict_action_batch`:**52.60 ms → 42.90 ms(−9.70 ms,−18.4 %)**。

**每一项优化都是位级一致的**(bit-exact,`max|Δ| = 0.00e+00`,固定 seed 与未优化路径对拍)。
没有量化、没有降精度、没有改采样器、没有减少去噪步数 —— 模型输出逐比特不变。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger_dark.png">
  <img alt="优化台账:52.60 ms 到 42.90 ms 的瀑布图" src="docs/ledger_light.png">
</picture>

---

## 配置(下面所有数字都在这个配置下测得)

π0.5,batch 1,**K = 10** 步 Euler 去噪,**968 个 prefix token**
(3 路相机 × 256 patch + 200 个语言 token),action chunk 50,**全程 bf16**。
动作专家是 gemma_300m:18 层,d = 1024,mlp 4096,8 个 query head / 1 个 KV head(MQA),
head_dim 256,50 个 action token。
机器:RTX PRO 5000 72 GB Blackwell,GB202,sm_120,110 SM,1344 GB/s,**300 W 功耗墙**。
checkpoint `RLinf-Pi05-LIBERO-SFT`,torch 2.7.1+cu128,nsys 2026.1.2。

---

## 优化脉络:三段

一个去噪步的墙钟时间可以拆成三块:**GPU 空转的时间**、**多余的 kernel 数**、
**每个 kernel 自己跑多久**。这三块的性价比差了一个数量级,所以工作是按这个顺序推进的:

| | 这一段做什么 | 主要手段 | 兑现 |
|---|---|---|--:|
| **① 消除** | 让 GPU 别空转,也别做本来就不该做的事 | adaRMS 调制预算表、Stage-1 denoise CUDA 图、static KV buffer、att_masks 上设备、去掉死的 timestep conditioning | **−5.99 ms** |
| **② 融合** | 该做的事,用更少的 kernel 做完 | fused QKV 权重、SwiGLU 与 QKV+RoPE 两个 epilogue 融合 | **−3.17 ms** |
| **③ 压 kernel** | 剩下的 kernel,让每一个自己跑得更快 | split-K、tile 扫参 | **0 ms —— 才刚开始** |

(前两段相加 −9.16 ms;余下的 −0.58 ms 是剥离成独立包并接上 `--stage1` 那一步,见台账第 7 行。)

顺序不是随手排的:**"消除"优于"融合"优于"调优"**。前两段的三个主力项(预算表、
合并 GEMM、消拷贝)没有一行手写 Triton;而第三段每 0.1 ms 都要付出一个手写 kernel 的代价。
同样重要的是,前两段已经把便宜的收益吃干净了 —— 所以第三段现在是真正的前沿,
也确实**还没有产出**。

### 优化台账

e2e = `predict_action_batch`,plain wall clock(**不是** nsys 的 wall time),
30 iterations after 8 warmup,串行,单任务独占 GPU。

| # | 优化 | 段 | e2e | Δ | 位级一致 |
|---|---|:--:|--:|--:|:--:|
| 0 | baseline(`torch.compile max-autotune`) | — | 52.60 ms | — | — |
| 1 | ＋ adaRMS 调制预算表 | ① | 49.77 | **−2.83** | ✅ |
| 2 | ＋ Stage-1 denoise CUDA 图 | ① | 47.73 | −2.04 | ✅ |
| 3 | ＋ fused QKV 权重 | ② | 45.61 | **−2.12** | ✅ |
| 4 | ＋ static KV buffer | ① | 45.10 | −0.51 | ✅ |
| 5 | ＋ att_masks 上设备(去 host 同步) | ① | 44.79 | −0.31 | ✅ |
| 6 | ＋ SwiGLU / QKV+RoPE 两个 epilogue 融合 | ② | 43.74 | **−1.05** † | ✅ |
| 7 | 独立包 ＋ `--stage1` | — | 43.16 | −0.58 ‡ | ✅ |
| 8 | ＋ 去掉死的 timestep conditioning(当前) | ① | **42.90** | **−0.30** ★ | ✅ |
| | **总计** | | | **−9.70 ms(−18.4 %)** | |

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

内核层面的配套数字:

| 指标 | before | after | 归属 |
|---|--:|--:|:--:|
| 每 step GPU idle | 142.2 µs(10.2 %) | **56.5 µs(4.5 %)** | ① |
| adaRMS 投影 `triton_per_fused_addmm_0` | 300 instances,395 µs/step,DRAM-read 87 % | **0 instances** | ① |
| prefix KV 的 `cat` kernel | 88 µs/step(物理下限 27) | **0** | ① |
| timestep conditioning(正弦 + time MLP) | 21 kernel/step,47.3 µs/step | **0** | ① |
| denoise kernels/step | 305 | **217**(−29 %) | ①② |
| denoise µs/step | 1368.0 | **1185.0**(−13.4 %) | ①② |
| k/v_proj 的 launch grid | **8**(110 个 SM 里只有 8 个在忙) | 80(融合后的 QKV GEMM) | ② |
| `_swiglu_mm_kernel` 达成带宽 | — | **967 GB/s** | ② |
| `down_proj` 达成带宽 | 557 GB/s | **557 GB/s(未动)** | ③ |

---

## ① 消除 GPU bubble —— −5.69 ms

> **观测**:一个去噪步墙钟 1390 µs,其中 **142.2 µs(10.2 %)GPU 完全空闲**;
> 而在忙的那 1247.8 µs 里,最大的单个 kernel 做的事情**根本不需要每步重做**。

这一段的四项没有一项在"让 kernel 变快",全部是**让它不发生**。

### 1.1 adaRMS 调制预算表(−2.83 ms,单项最大)

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
> 时间一样。**只有消除投影才有用。** 而且预算表是位级一致的(`0.00e+00`),
> batched-GEMM 因为改了归约顺序是 `2.71e-3`,本来就不合格。

### 1.2 Stage-1 denoise CUDA 图(链上 −2.04 ms;独立包内配对 −0.93 ms)

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

> **怎么确认图真的生效**:`graphNodeId` **不是**判据 —— Stage-1 关闭时那 2160 个 denoise
> kernel 也全都带 `graphNodeId`(inductor 自己就发了一个 cudagraph),我据此误判过一次。
> 可靠信号:`denoise/expert_forward` 的 NVTX range 数(10/predict = eager,**0 = 在图内**)、
> distinct `graphId` 数(2 → 1)、stream 157 上的 kernels/step(171 → 238)。
> (238 是当时的数;§1.5 之后是 217,判据本身不变。)

### 1.3 static KV buffer(−0.51 ms)

**为什么。** denoise 路径跑的是 `use_cache=False`,于是 attention 走了
`torch.cat([prefix_kv, new_kv])` 分支 —— **每步、每层都在重新物化整个 968 token 的 prefix
KV**(18 层 × 2 × 10 步)。实测这个 `cat` kernel **88 µs/step,而物理下限是 27 µs/step**。

**怎么做。** 改成 vLLM / SGLang 那套:每层预分配一个
`[B, kv_heads, prefix+suffix, head_dim]` 的静态 buffer,每个 predict 写一次 prefix,
每步只写 50 个 token 的尾巴。buffer 只在 shape 变化时重分配 → **地址稳定 → CUDA 图
replay 安全**。

**量了多少。** e2e **−0.51 ms**,`cat` kernel 归零。这一步还是 ② 里 QKV+RoPE 融合能"直写
KV 尾巴"的前提。

### 1.4 att_masks 上设备(−0.31 ms,但价值远不止)

**为什么。** `embed_prefix` 里的 `att_masks = torch.tensor(<python list>, device=cuda)`
是热路径上的一次**同步** host→device 拷贝。

**怎么做。** 读代码可以证明这个 mask **恒为全零**(整个 prefix —— 所有图像 view 加语言
token —— 是一个 full-attention block,两处 append 都是 `[0]*n`),长度也恒等于
`pad_masks.shape[1]`,于是直接用 `torch.zeros(..., device=...)` 构造。

**量了多少。** e2e **−0.31 ms**。但它真正的价值是:这是**唯一**一条阻止 prefix 阶段被
CUDA 图捕获的语句 —— 改之前 `torch.cuda.CUDAGraph()` 捕获 `_build_prefix_cache` 会在这一行抛
`operation not permitted when stream is capturing`,改之后 capture / replay 都成功。
prefix 是 GPU busy 的 **71.7 %**,所以这条的长期价值远大于它自己的 0.31 ms。

### 1.5 去掉死的 timestep conditioning(−0.30 ms)

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
device tensor 的判断,所以 `embed_suffix` 仍然可以待在 Stage-1 捕获的图里(图只会 trace 到
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

## ② 融合 kernel —— −3.17 ms,305 → 238 kernels/step

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

### 2.1 fused QKV 权重(−2.12 ms)

**为什么。** MQA(`num_kv_heads = 1`)让 k_proj / v_proj 的输出只有 50 × 256,triton 把它切成
**grid = 8** —— 110 个 SM 里只有 8 个在干活,比它自己的内存下限慢 **8.3×**,纯粹的
launch / occupancy 浪费。

**怎么做。** 把 q(2048)+ k(256)+ v(256)沿 N 拼成一个 `[2560, 1024]` 的权重,
变成**一个更宽的 GEMM**。数学上完全等价(每个输出列都是独立的点积)。

**量了多少。** grid 8 → **80**;e2e **−2.12 ms**;18/18 层融合成功,未融合 vs 融合
`max|Δ| = 0.00e+00`。

### 2.2 SwiGLU:gate/up 的**横向** GEMM 合并

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
达成带宽 16.78 MB / 17.35 µs = **967 GB/s**。

### 2.3 QKV + RoPE:把旋转做在累加器上

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

### 2.4 epilogue 融合怎么做到位级一致

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
前面(Stage-1 图已经做到了),per-node dispatch 常数就不再适用。
**kernel 数仍然是选靶子的好指标**(便宜、方差小),但**换算成时间必须用实测的逐核时长**,
不能乘一个 dispatch 常数。

---

## ③ 优化 kernel 本身 —— **进行中,目前兑现 0 ms**

前两段把"不该做的事"和"多余的 launch"清完之后,剩下的时间**全在 kernel 自己的执行**里。
这一段才刚开始,**上面那个 42.90 ms 里没有它的任何贡献**。下面写的是靶子和证据,不是成果。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/denoise_dark.png">
  <img alt="当前 build 的 denoise 每核耗时分解" src="docs/denoise_light.png">
</picture>

### 3.1 靶子一:`down_proj` 只跑 557 GB/s(~1.1 ms)

**观测。** 它是**剩下最大的单个 kernel**:270.6 µs/step。而紧挨着它、流式读同类权重的
`_swiglu_mm_kernel` 已经到 **967 GB/s** —— 同一个 stream、同一层、同一种访存模式,差 1.7×。
差距来自 K 方向没有拆分,并行度不足。

**打算怎么做。** split-K。一个独立的 split-K 微基准(`mb_splitk.py`)测到
**11.55 µs,−18 %**。障碍是 inductor 2.7 的 mm 模板**没有 split-K**,要自己写 custom op;
而且归约必须做成**确定性**的,否则保不住 `0.00e+00`。

**量级 ~1.1 ms/predict。状态:未开始。**

### 3.2 靶子二:`_swiglu_mm_kernel` 离流式地板还有 25 %(~0.8 ms)

**观测。** 它是 17.35 µs,而 16.78 MB / ~1.3 TB/s ≈ **13 µs** 才是流式下限。它的 tile
config 是手钉的、从没 autotune 过 —— 因为 custom op 对 inductor 不透明。

**打算怎么做。** 扫 tile。好消息是这个扫描**免重编译**(`RLINF_FUSE_SWIGLU_CFG` /
`RLINF_FUSE_QKV_CFG`,traced graph 不变、warm cache 复用,每次 ~2 min 而不是 ~20 min),
而且 tile shape 已被证明对数值零影响,所以是零风险扫描。
⚠️ 但 **`BLOCK_K` 不能动** —— 它是唯一影响 fp32 归约顺序的参数,动了就失去 `0.00e+00`。

**已经扫过一轮,结果不确定,如实报告而不是挖掘**:n = 1/配置,散布 43.1–44.4 ms,
和本机单次运行的方差同量级(出货配置在那一轮测到 43.43,在 4 轮 A/B 里是 43.74);
表面领先 0.3 ms 的那个配置还与隔离的 CUDA 图微基准**相反**(那边它慢 4 µs/call)。
**没有证据支持改动**,所以保留了经过 4 轮配对验证的出货值。

**量级 ~0.8 ms。状态:需要一次正经的配对(交替、≥4 轮)扫描。**

### 3.3 顺带:那项"消除"做了一半(~0.5 ms 还在)

图里那根橙色的 **eager per-step glue**,严格说属于第 ① 段而不是第 ③ 段。
它原本是 99.7 µs/step、~70 kernel/step,其中**纯死代码**的那一半已经由 §1.5 摘掉,
现在是 **50.6 µs/step、49 kernel/step**。

剩下的那一半不是死代码,是**逐步重复**:position id 的 `cumsum`/`DeviceScan`、
rotary 表的 `cos`/`sin`、`_get_timesteps` 的 `linspace` —— 这些在 10 个 Euler 步上
**完全相同**,和已经外提的 adaRMS 表是同一类问题,同一套解法(预计算 + 每步 gather)。
另外还有真正每步都不同的 Euler 更新与 log-prob,那部分动不了。
**量级 ~0.5 ms。状态:未做。**

### 3.4 明确排除的做法

- ⛔ **算法层一律不做**:不减去噪步数、不蒸馏、不换采样器、不引入 staleness 的重叠。
- ⛔ **不做任何降精度**:fp8 / int8 / 任何量化都不在选项内 —— 本项目的全部前提就是
  `max|Δ| = 0.00e+00`。
- ⛔ 已经排除、别再试的:整模型单图(prefix 是 compute-bound,收益 ≈ 0,且被 compile 嵌套
  挡住)、强制 Triton-only 后端(§② 开头那个已结的负结果)、把 gated residual 塞进 GEMM
  epilogue(本来就已融进消费端 RMSNorm,kernel 数不变)。

---

## 天花板在哪

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/phases_dark.png">
  <img alt="GPU busy 的阶段拆分:prefix 71.7%,denoise 28.3%" src="docs/phases_light.png">
</picture>

去噪循环只占 GPU busy 的 **28.3 %**。即使把它优化到 0,整个 predict 的加速上限也只有
**1.39×**。真正的大头是那个 968 token 的 prefix(PaliGemma 语言塔 24.10 ms + SigLIP
视觉塔 4.82 ms)。这就是为什么 §1.4(att_masks)虽然自己只值 0.31 ms 却仍然重要 ——
它解锁了 prefix 阶段的 CUDA 图捕获。

顺带解释上一张图里的另一个数:expert block 本身现在是 **163 kernels/step**,参考实现是
165 —— 在 transformer 内部,kernel 数的差距已经抹平,两个融合块的执行时间还反超了。
剩下的 217 − 163 = 54 个 kernel 全是 expert 之外的 eager per-step glue(§3.3)。

图里的数字可以从 profile 重新推导:
`python docs/make_charts.py --sqlite <stage1_on.sqlite> --sqlite-off <stage1_off.sqlite>`
会把推导结果和图里写死的常量并排打印出来。

---

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

⚠️ [`limxdynamics/FluxVLA`](https://github.com/limxdynamics/FluxVLA) 是**另一个**仓库
(kernel 同名是因为 FluxVLA 重新实现了一遍)。早期笔记把两者混为一谈,由此得出的
"他们每步重算 adaRMS"、"44.89 ms 打平"两条结论**不适用于** realtime-vla —— 后者在
`Pi05Inference.__init__` 里就把 37 × 10 个 adaRMS 投影预计算完了,和我们一样。

---

## 安装

用现成的 RLinf benchmark 容器镜像,**不需要重建 Docker**。editable 安装,并且带
`--no-deps`,以免动到容器里钉死的 torch / transformers / openpi:

```bash
docker exec -w /path/to/pi05-infer pi05bench \
    /opt/venv/openpi/bin/pip install -e . --no-deps --no-build-isolation
```

`pi05-infer` 不碰 `site-packages`,只加一条 path entry,所以容器保持原状,可以作为 A/B
的参考臂。

> 因为 `torch.library` 的命名空间是进程全局的,而容器里那份旧的 `rlinf_fused_denoise.py`
> 也会被 import,本包把自定义算子注册成 `pi05_infer::gate_up_swiglu` /
> `pi05_infer::qkv_rope_kv` 而不是 `rlinf::*` —— 否则第二次注册会抛错并**静默关掉**融合。

## 用法

```bash
# 基准测试
/opt/venv/openpi/bin/python bench/standalone_infer_bench.py \
    --model-path /path/to/RLinf-Pi05-LIBERO-SFT \
    --config-name pi05_turtle --iters 30

# 开启 Stage-1 手写 denoise CUDA 图(opt-in,默认路径不变)
... --stage1

# 分阶段耗时 / 导出 actions 做数值 A/B / 记录计时窗口内的 SM 时钟与功耗
... --phases
... --dump-actions /tmp/a.pt --clocks-json /tmp/clocks.json
```

`--stage1` 替你处理两件都很要命的事:

* 把 `--compile-mode max-autotune` 改写成 `max-autotune-no-cudagraphs`。inductor 自己的
  cudagraph 不能嵌套在手写图里,失败模式是
  `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten`。
* warmup 之后**断言** `is_cuda_graph_enabled()` 和 `_denoise_graph_captured`。图是在第一次
  eval-shape 的 `sample_actions` 上懒捕获的,shape signature 不匹配就会**静默**退回 eager
  loop,除了跑得慢一点没有任何症状 —— 这个包最初漏掉这个 flag 就是这么发生的。

---

## 正确性:位级一致怎么验的

```bash
# 0. 隔离:expert 必须是 pi05_infer.gemma,PaliGemma prefix 必须是 transformers
python tools/isolation_check.py          # 打印 ISOLATION_OK

# 1. 两个 Triton 融合核 vs inductor 自己编出来的输出,逐比特
python tools/bitgate.py

# 2. 整条路径的数值 A/B,固定 seed
python tools/ab_rlinf_reference.py --dump-actions /tmp/ref.pt
python bench/standalone_infer_bench.py --dump-actions /tmp/new.pt
python -c "import torch;a=torch.load('/tmp/ref.pt');b=torch.load('/tmp/new.pt');print((a-b).abs().max())"
```

两级检查是有意为之:

* **`tools/bitgate.py`** 拿融合核和 **inductor 在 `max-autotune-no-cudagraphs` 下为同一批
  算子编出来的 kernel** 对拍(SwiGLU 的输出、QKV+RoPE 的 q / k / v 各自比),这是最严格的
  参照系 —— 被替换掉的正是它。全部 `max|Δ| = 0.00e+00`,零个不同元素。
* **端到端**用固定 seed 把 `[1, 50, 6]` 的动作在 float64 下和 RLinf 参考路径对拍。
  `--stage1` on vs off、on vs RLinf、off vs RLinf,三组全是 `0.00e+00`;
  §1.5 之后又补了一组两种 compile mode × 开关 on/off 加参考臂共 5 份 dump 的两两对拍
  (10 对),同样全部 `0.00e+00`。

§1.5 的消除也带 kill switch:`RLINF_SKIP_DEAD_ADARMS_COND=0` 恢复"每步都算 `adarms_cond`"
的旧行为(import 时读一次,所以它是编译期常量,不影响 CUDA 图捕获)。

融合核还都带**降级路径**:kill switch(`RLINF_FUSE_SWIGLU=0` / `RLINF_FUSE_QKV_ROPE=0`)、
triton 不可用、激活不是 tanh-GELU、dtype 不是 bf16/fp16、权重非行主序、shape 除不尽 tile、
static KV 未 prime(训练 / `use_cache=True`)、需要 autograd —— 任一条件成立就静默走回原始
PyTorch 路径。A/B 的每一个 OFF 臂都走的是这条降级路径,所以它是被验证过的,不是被假设的。

完整测量记录见 **[`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md)**。

---

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

## 仓库结构

```
pi05_infer/
  engine.py            OpenPi0Inference:纯推理编排。
                       predict_action_batch -> sample_actions -> sample_mean_var_val
                       -> get_velocity -> get_suffix_out,以及 _build_prefix_cache、
                       enable_torch_compile、Stage-1 denoise CUDA graph、
                       adaRMS 调制表和 invalidate_weight_derived_caches。
  builder.py           build_model():checkpoint + norm-stats + transform 组装。
  dataconfig/          RLinf openpi dataconfig 的最小子集(turtle + libero)。
  _vendored/           无 RLinf 依赖的 helper 逐字副本:base_policy / cuda_graph / nvtx。
  gemma/               动作专家跑的那份 Gemma fork:modeling_gemma.py(相对 transformers
                       +245 行)+ rlinf_fused_denoise.py(两个 Triton 融合核)。
  openpi_patched/      我们改过的两个 openpi 文件:pi0_pytorch.py + gemma_pytorch.py。
bench/
  standalone_infer_bench.py   延迟基准(e2e、分阶段、nsys、actions dump)
tools/
  isolation_check.py   证明 expert = pi05_infer.gemma,prefix = transformers
  bitgate.py           两个 Triton 融合核的位级一致 gate
  ab_rlinf_reference.py / ab_stage1.sh / ab_stage1_summary.py   配对 A/B 驱动
  step_idle.py / stream_summary.py / denoise_kernels.py / ksum.py / prof.sh   profile 分析
docs/
  make_charts.py       重新生成本 README 的三张图(含从 sqlite 重新推导)
  MEASUREMENTS.md      完整测量记录
_extract_src/          抽取前的 RLinf 原始文件(未重构)
```

### prefix / expert 的隔离

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

---

## 已知限制 / 没做的事

* **第 ③ 段整体还没有产出**:`down_proj` 的 split-K(~1.1 ms)、SwiGLU 的 tile 扫参
  (~0.8 ms)都只有靶子和微基准,没有落地收益。
* **~1.0 ms/predict 的 per-step-invariant glue 还没外提**(§3.3)—— 这是目前剩下最大的一项。
* **Stage-1 路径上还留着一次死拷贝** `_copy_kv_into_static`(17.8 MB/predict,~0.03 ms)。
* **不做**任何算法层改动、不做任何降精度。本项目全部是推理侧改动,且逐比特保持模型输出。
* openpi 侧文件的三方合并、把 `openpi_action_model.py` 接回本引擎、把容器
  `site-packages` 还原成 openpi 原版 —— 见 `EXTRACTION_NOTES.md` §7–§8。
* **RL 集成**:CUDA 图在权重原地同步后**不需要重捕**(它记的是地址),但
  `qkv_fused_weight` / adaRMS 表这类**权重派生量会静默过期**;刷新必须原地 `copy_`
  不能重分配。`invalidate_weight_derived_caches()` 就是这个钩子。

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
