# pi05-infer

**π0.5 动作专家(action expert)的独立 bs=1 推理引擎**,从 [RLinf](https://github.com/RLinf/RLinf)
里抽出来,针对 **RTX PRO 5000(GB202 / sm_120,Blackwell)** 做过一轮系统性优化。

端到端 `predict_action_batch`:**52.60 ms → 43.16 ms(−9.44 ms,−17.9 %)**。

**每一项优化都是位级一致的**(bit-exact,`max|Δ| = 0.00e+00`,固定 seed 对拍未优化路径)。
没有量化、没有降精度、没有改采样器、没有减少去噪步数 —— 模型输出逐比特不变。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger_dark.png">
  <img alt="优化台账:52.60 ms 到 43.16 ms 的瀑布图" src="docs/ledger_light.png">
</picture>

---

## 配置(下面所有数字都在这个配置下测得)

π0.5,batch 1,**K = 10** 步 Euler 去噪,**968 个 prefix token**
(3 路相机 × 256 patch + 200 个语言 token),action chunk 50,**全程 bf16**。
动作专家是 gemma_300m:18 层,d = 1024,mlp 4096,8 个 query head / 1 个 KV head(MQA),
head_dim 256,50 个 action token。
机器:RTX PRO 5000 72 GB Blackwell,GB202,sm_120,110 SM,1344 GB/s,**300 W 功耗墙**。
checkpoint `RLinf-Pi05-LIBERO-SFT`,torch 2.7.1+cu128。

---

## 优化台账

e2e = `predict_action_batch`,plain wall clock(不是 nsys 的 wall time),
30 iterations after 8 warmup,串行,单任务独占。

| 优化 | e2e | Δ | 位级一致 |
|---|--:|--:|:--:|
| baseline(`torch.compile max-autotune`) | 52.60 ms | — | — |
| ＋ adaRMS 调制预算表 | 49.77 | **−2.83** | ✅ |
| ＋ Stage-1 denoise CUDA graph(与上一项叠加) | 47.73 | −2.04 | ✅ |
| ＋ fused QKV 权重 | 45.61 | **−2.12** | ✅ |
| ＋ static KV buffer | 45.10 | −0.51 | ✅ |
| ＋ att_masks 上设备(去 host 同步) | 44.79 | −0.31 | ✅ |
| ＋ SwiGLU epilogue 融合 & QKV+RoPE epilogue 融合 | 43.74 | −1.05 † | ✅ |
| 独立包 ＋ `--stage1`(当前) | **43.16** | −0.58 | ✅ |
| **总计** | | **−9.44 ms(−17.9 %)** | |

† 这一行有两个数字,都列出来:在这条累积链上它是 −1.05 ms(44.79 → 43.74);而在
它自己那次 4 轮配对 A/B 里,基线是同 session 的 44.87 ms,测得 **−1.14 ms**。
两者的差是 session 间的基线漂移,不是收益本身。

最后一行是独立包开 `--stage1` 的 4 轮配对 A/B:**44.08 → 43.16 ms,Δ = −0.93 ms**
(sd 0.36,n = 4),每轮两臂 SM 时钟相差 20 MHz 以内。

内核层面的配套数字:

| 指标 | before | after |
|---|--:|--:|
| denoise kernels/step | 305 | **238**(−22 %) |
| denoise µs/step | 1368.0 | **1236.0** |
| 每 step GPU idle(Stage-1) | 142.2 µs(10.2 %) | **60.5 µs(4.7 %)** |
| adaRMS 投影 `triton_per_fused_addmm_0` | 300 instances,395 µs/step,DRAM-read 87 % | **0 instances** |
| k/v_proj 的 launch grid | **8**(110 个 SM 里只有 8 个在忙) | 80(融合后的 QKV GEMM);当前融合核 88 |
| `_swiglu_mm_kernel` 达成带宽 | — | **967 GB/s** |
| `_qkv_rope_kernel` | — | 132 µs/step(参考实现 144) |

---

## 每一项优化做了什么,以及为什么有效

### 1. adaRMS 调制预算表 —— 单项最大收益(−2.83 ms)

每个 adaRMS norm 前面挂着一个 `dense(cond)` 投影,一共 37 个。它们只依赖扩散
timestep,而 timestep 走的是固定 schedule —— 也就是说**与输入无关**,可以一次性预计算
成一张 `[num_steps, 37, 3072]` 的表,每步只做一次 device gather。
inductor 原本把这 37 个投影**横向融合**成了一个 memory-bound 的巨核
(`triton_per_fused_addmm_0`,395 µs/step,DRAM-read 占比 87 %),搬运 483 MB 而实际
只需要 233 MB —— **2.08× 的冗余流量**。表一建,这个核直接从 300 instances 变成 0。

> 对照:先前试过"把 37 个投影 batch 成一个大 GEMM"(Stage A),结果是**持平**:
> M=1 的大 GEMV 仍然要读 ~477 MB、跑在 92 % 带宽上,时间一样。只有**消除**投影才有用。
> 而且预算表是位级一致的(0.00e+00),batched-GEMM 因为改了归约顺序是 2.71e-3。

### 2. fused QKV 权重(−2.12 ms)

MQA(`num_kv_heads = 1`)让 k_proj / v_proj 的输出只有 50 × 256,triton 把它切成
**grid = 8** —— 110 个 SM 里只有 8 个在干活,比它的内存下限慢 **8.3×**,纯粹的
launch / occupancy 浪费。把 q(2048)+ k(256)+ v(256)拼成一个
`[2560, 1024]` 的权重,变成**一个更宽的 GEMM**,grid 回到 80。
数学上完全等价(沿 N 拼接,每个输出列都是独立的点积);实测 18/18 层融合成功,
未融合 vs 融合 `max|Δ| = 0.00e+00`。

### 3. static KV buffer(−0.51 ms)

denoise 路径跑的是 `use_cache=False`,于是 attention 走了
`torch.cat([prefix_kv, new_kv])` 分支 —— **每步、每层都在重新物化整个 968 token 的
prefix KV**(18 层 × 2 × 10 步)。实测 `cat` kernel 88 µs/step,而物理下限是 27 µs/step。
改成 vLLM / SGLang 那套:每层预分配一个
`[B, kv_heads, prefix+suffix, head_dim]` 的静态 buffer,每个 predict 写一次 prefix,
每步只写 50 个 token 的尾巴。buffer 只在 shape 变化时重分配 → 地址稳定 → CUDA graph
replay 安全。

### 4. att_masks 上设备(−0.31 ms)

`embed_prefix` 里的 `att_masks = torch.tensor(<python list>, device=cuda)` 是热路径上
的一次**同步** host→device 拷贝。读代码可以证明这个 mask **恒为全零**(整个 prefix ——
所有图像 view 加语言 token —— 是一个 full-attention block,两处 append 都是 `[0]*n`),
长度也恒等于 `pad_masks.shape[1]`,于是直接 `torch.zeros(..., device=...)` 构造。

它还是**唯一**一条阻止 prefix 阶段被 CUDA graph 捕获的语句:改之前
`torch.cuda.CUDAGraph()` 捕获 `_build_prefix_cache` 会在这一行抛
`operation not permitted when stream is capturing`,改之后 capture / replay 都成功。
prefix 是 GPU 时间的 ~72 %,所以这条的长期价值远大于它自己的 0.31 ms。

### 5. 两个 epilogue 融合(−1.05 ms,kernels/step 305 → 238)

先做过一次**否定性探针**:强制 `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON`
对 kernel 数**毫无影响**(305 → 305),而且在 prefix 里倒亏 ~4 ms。原因读生成代码就知道:
denoise 的每一个 GEMM 本来就是 Triton template,inductor 结构上能融的 epilogue 已经融完了。
剩下这两个是 inductor **表达不出来**的:

* **SwiGLU**:gate/up 两个矩阵乘 + `gelu(g)·u` 合成一个 kernel,两个 fp32 累加器共享同一个
  A tile。这是**横向** GEMM 合并(两个 matmul 共享一个操作数),inductor 根本不做横向融合。
  收益不在 FLOPs(一样多),而在于省掉每层 400 KB 的 gate 激活存 + 400 KB 读回,以及每层一次
  launch。36 kernel / 405.2 µs → 18 kernel / 312.4 µs。
* **QKV + RoPE**:RoPE 要求输出列 `d` 和 `d + head_dim/2` 在同一个 program 里,而 Triton
  template 的 epilogue 只能看到自己刚算完的那块 tile(`rotate_half` 是两段不同列区间的
  `torch.cat`),inductor 必然把它拆成独立的 pointwise kernel。融合后每个 program 拥有同一个
  head 的**一对**列 tile,旋转直接作用在累加器上,rotated k / raw v **直接写进 static KV
  cache 的尾巴**。72 kernel / 165.9 µs → 18 kernel / 132.0 µs。

**两个都是构造上位级一致的**:epilogue 融合通常*不*位级一致(融合版跑在 fp32 累加器上,
未融合版跑在存回内存又读出来的 bf16 上)。这两个核**故意**把每一个"原本会以 bf16 落盘"的
累加器做一次 `acc.to(bf16).to(fp32)` 往返;再把 `BLOCK_K` 钉死在 inductor autotuner 选中
的值上(`BLOCK_K` 是唯一会改变 K 方向 fp32 归约顺序的 tile 参数,`BLOCK_M`/`BLOCK_N` 可证明
不会),结果就是精确相等。

> 这里踩过一个真坑:SwiGLU 融合第一版没加 M 护栏,于是它也捕获了 PaliGemma **prefix**
> 那个 968 token 的语言模型,prefix 直接 +6.5 ms。现在 `RLINF_FUSE_SWIGLU_MAX_M`(默认 64)
> 把它限制在只对能装进一个 M-tile 的 token 数生效。

### 6. Stage-1 denoise CUDA graph(−2.04 ms)

把**一整个 flow_ode 去噪步**(expert forward + Euler 更新 + log-prob)捕获成一个
`torch.cuda.CUDAGraph` 并逐步 replay,一次 replay 取代"inductor 的 expert-only cudagraph
＋ 中间所有 eager glue 的 launch"。效果是纯粹的 launch-gap 消除:每 predict 的总 GPU busy
基本不变(40.32 → 40.26 ms),而每步 idle 从 142.2 µs(10.2 %)降到 **60.5 µs(4.7 %)**,
−95.8 µs/step × 10 = −0.96 ms/predict —— 正好解释掉配对 A/B 的 −0.93 ms。

它是**无损**的:`flow_ode` 的 `x_t_std == 0`,所以 graph 里的 `x_t_next = x_t_mean`
在代数上等于 eager 的 `x_t_mean + noise * 0`;而 `sample_noise` 的抽样留在 graph 之外,
全局 RNG 消耗不变。

---

## 一个去噪步现在花在哪

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/denoise_dark.png">
  <img alt="当前 build 的 denoise 每核耗时分解" src="docs/denoise_light.png">
</picture>

数据直接从 nsys sqlite(stream 157,12 predicts × 10 steps)重新推导,
`python docs/make_charts.py --sqlite <stage1_on.sqlite>` 会把推导结果和图里的常量并排打印出来。

expert block 本身现在是 **163 kernels/step**,参考实现是 165 —— 在 transformer 内部,
kernel 数的差距已经抹平,而两个融合块的执行时间反超。剩下的 238 − 163 = 75 个 kernel
全是 expert 之外的 **eager per-step glue**,那是另一个问题(见"已知限制")。

---

## 天花板在哪

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/phases_dark.png">
  <img alt="GPU busy 的阶段拆分:prefix 71.7%,denoise 28.3%" src="docs/phases_light.png">
</picture>

去噪循环只占 GPU busy 的 **28.3 %**。即使把它优化到 0,整个 predict 的加速上限也只有
**1.39×**。真正的大头是那个 968 token 的 prefix(PaliGemma 语言塔 24.10 ms + SigLIP
视觉塔 4.82 ms)。这就是为什么第 4 项(att_masks)虽然自己只值 0.31 ms 却仍然重要 ——
它解锁了 prefix 阶段的 CUDA graph 捕获。

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

**当天他们快 1.14 ms(2.6 %)。** 之后的 kernel fusion 把 expert 拉到 163 kernels/step,
两个融合块的时间反超(SwiGLU 312 µs vs 380,QKV+RoPE 132 µs vs 144)。

⚠️ 本仓库当前的 43.16 ms 与他们的 43.41 ms **不是配对测量** —— 相隔一天、不同 session、
不同 build,差值(0.25 ms)小于本机记录在案的 ±0.7 ms rebuild variance。**不能**据此
宣称超越。把 realtime-vla 当作同水位的 peer 看待即可。

⚠️ [`limxdynamics/FluxVLA`](https://github.com/limxdynamics/FluxVLA) 是**另一个**仓库。
早期笔记把两者混为一谈,由此得出的"他们每步重算 adaRMS"、"44.89 ms 打平"两条结论
**不适用于** realtime-vla —— 后者在 `__init__` 里就把 37 × 10 个 adaRMS 投影预计算完了,
和我们一样。

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

# 开启 Stage-1 手写 denoise CUDA graph(opt-in,默认路径不变)
... --stage1

# 分阶段耗时(注:与 --stage1 不兼容,它直接调用模型内部,活不过捕获的 graph)
... --phases

# 导出 actions 做数值 A/B / 记录计时窗口内的 SM 时钟与功耗
... --dump-actions /tmp/a.pt --clocks-json /tmp/clocks.json
```

`--stage1` 替你处理两件都很要命的事:

* 把 `--compile-mode max-autotune` 改写成 `max-autotune-no-cudagraphs`。inductor 自己的
  cudagraph 不能嵌套在手写 graph 里,失败模式是
  `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten`。
* warmup 之后**断言** `is_cuda_graph_enabled()` 和 `_denoise_graph_captured`。graph 是
  在第一次 eval-shape 的 `sample_actions` 上懒捕获的,shape signature 不匹配就会**静默**
  退回 eager loop,除了跑得慢一点没有任何症状 —— 这个包最初漏掉这个 flag 就是这么发生的。

调制表在捕获**之前**建好(建表用的是 capture-illegal 的 eager 算子),每步的取表是
device gather,所以两者都能活过捕获。

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
* **端到端** 用固定 seed 把 `[1, 50, 6]` 的动作在 float64 下和 RLinf 参考路径对拍。
  `--stage1` on vs off、on vs RLinf、off vs RLinf,三组全是 `0.00e+00`。

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
4. **GPU 必须独占。** 一个并发任务曾经吃掉 1.5 ms 的 CPU 竞争。
5. **`idle` 只在同一 session 内可比。** 同样的代码、同样的 GPU busy(1246.7 vs 1247.8
   µs/step),两次采集的 idle 能差 72 µs/step —— 那是 host 侧 stall,不是代码。

### ⚠️ `torch.empty` 零权重陷阱

这条值得单独拎出来,因为它会让任何人的 benchmark 数字凭空好看一大截:

**driver 返回的 `torch.empty` CUDA 页是全零的。** 零操作数几乎不产生晶体管翻转 →
动态功耗极低 → GPU 撞不到功耗墙 → 时钟一路 boost → 延迟数字凭空变好。

实测(我们自己的 stack,kernel census 完全相同,3903 次 launch 零差异):

| 权重 | e2e | SM 时钟 |
|---|--:|--:|
| 真实 checkpoint | 48.47 ms | 2251 MHz |
| `torch.zeros` | **42.78 ms** | **2559 MHz** |

时间比 1.1330 vs 时钟比 1.1368 —— 吻合到 **0.33 %**,即 100 % 是时钟效应,零性能含量。
同样的对照在 `dexmal/realtime-vla` 公开的 `benchmark.py` 上也成立:它把权重留成
`torch.empty(...)`(实测 2,826,721,040 个元素,**0 个非零**),同一份代码在我们的配置下
零权重 37.0 ms、随机权重 43.1 ms,**虚高约 18 %**。

**在有功耗墙的 GPU 上,永远不要用 `torch.empty` / `torch.zeros` 的权重测延迟。**

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
形式主义:融合期间那次 +4 ms 的回归,原因正是 PaliGemma 的 **prefix** 语言模型也是
Gemma,全局覆盖 `transformers/models/gemma/` 会让一个为 50 token denoise suffix 调优的
kernel 打到 968 token 的 prefix 上。现在 prefix 和 expert 是来自不同文件的不同类,这类
bug 结构上不可能发生。

只有真正改过的符号才被 vendor;`modeling_gemma.py` 其余全部从安装好的 transformers
import。边界的完整说明(以及这**没有**带来独立性的那一处)见
[`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md) §8。

---

## 已知限制 / 没做的事

* **`down_proj` 还跑在 557 GB/s**,而紧挨着它、流式读同类权重的 SwiGLU 核已经到 967 GB/s。
  它是剩下最大的单个 kernel(270.6 µs/step)。要补上需要确定性的 split-K,值 ~1.1 ms/predict。
* **~1.0 ms/predict 的 per-step-invariant glue 还没外提。** 每步 ~70 个 eager kernel
  (`embed_suffix`、正弦时间嵌入、position id 的 `cumsum`/`DeviceScan`、Euler 更新、
  log-prob)。其中时间嵌入、position id、rotary 表的 `cos`/`sin` 在 10 个 Euler 步上**完全相同**
  —— 和已经外提的 adaRMS 表是同一类问题。这是目前剩下最大的一项。
* **SwiGLU 的 tile config 没有 autotune 过。** 因为 custom op 对 inductor 不透明,
  `RLINF_FUSE_SWIGLU_CFG` / `RLINF_FUSE_QKV_CFG` 可以**免重编**扫描。扫过一轮,结果
  **不确定**(n=1,散布 43.1–44.4,和单次运行方差同量级),所以按"没有证据支持改动"处理,
  保留了经过 4 轮配对验证的出货值。tile shape 已证明不影响位级一致,所以这是个零风险的扫描。
* **`_swiglu_mm_kernel` 自己是 17.35 µs,对着 16.78 MB / ~1.3 TB/s ≈ 13 µs 的流式下限。**
* **不做**任何算法层改动:不减去噪步数、不蒸馏、不换采样器、不量化。本项目全部是推理侧改动,
  且逐比特保持模型输出。
* openpi 侧文件的三方合并、把 `openpi_action_model.py` 接回本引擎、把容器
  `site-packages` 还原成 openpi 原版 —— 见 `EXTRACTION_NOTES.md` §7–§8。

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
