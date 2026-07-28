# 测量记录 / Measurement log

本文件是 `README.md` 里那些数字的原始记录。每一节都注明测量日期、机器、配置和
产出的 artifact,便于逐条复核。

统一环境:RTX PRO 5000 72 GB Blackwell(GB202,sm_120,110 SM,1344 GB/s,
**300 W 功耗墙**),容器 `pi05bench`,torch 2.7.1+cu128,nsys 2026.1.2。
配置:π0.5,bs=1,`pi05_turtle`(action_horizon 50),K=10 denoise steps,
968 prefix token,3 × 128² 相机(transform 内 resize 到 224),bf16,
checkpoint `RLinf-Pi05-LIBERO-SFT`。

> ⚠️ **读下面每一条"位级一致"之前,先读这一条(2026-07-28 补记)。**
> 本文件里凡是**端到端 `--dump-actions` 对拍**得到的 `0.00e+00`,当时确实测到了,
> 但那个门的**分辨力有限**:在 base `max-autotune` 下它跨进程的噪声底就有 ~4–5e-3
> (根因是 inductor 会 benchmark SigLIP 视觉塔的 LayerNorm 归约核并挑 launch config,
> 这个选择跨进程会跳变)。所以它**证不伪**;"winner 集合相同 ⟺ 输出逐位相同"在记录了
> winner 的 10 个 run 里 0 例外。**核级 / 张量级 / 同进程 / 冻结 prefix 的判据才是强的。**
> 用编译路径上的强判据重验之后,四项结构性优化里有三项 FAIL(见 §5),README 的
> 「正确性」一节已按两级(代数等价 / 编译路径逐位相同)重写。

---

## 1. 2026-07-28 · 独立包 vs RLinf 参考路径(去噪 CUDA 图关闭)

30 iterations after 8 warmup calls,串行,plain wall clock。

| | RLinf path(arm A,参考) | `pi05_infer`(arm B) | Δ |
|---|--:|--:|--:|
| e2e `predict_action_batch`,CPU wall clock,mean | 44.53 ms | **44.01 ms** | −0.52 ms |
| … p50 | 44.50 | 43.95 | |
| … min / max | 44.01 / 45.35 | 43.39 / 45.08 | |
| GPU event span,mean | 44.51 ms | 43.99 ms | −0.52 ms |
| SM clock during window | 未采样 | 2445 MHz(2430–2452),227.6 W | |

两臂都是同一 session 里的 cold 30-iteration run,比历史 43.74 ms 高 0.3–0.8 ms —— 落在
±0.7 ms 的 rebuild variance 和 ~8 % 的 cold/sustained 时钟差里。arm B 快 0.52 ms,
nsys 数据把它精确归因到被删掉的 6 个 per-predict RL bookkeeping kernel。

### 位级一致

| check | result |
|---|---|
| `fused_gate_up_swiglu` vs inductor `max-autotune-no-cudagraphs` | bitwise equal,`max\|Δ\| = 0.00e+00` |
| `fused_qkv_rope_kv` q / k / v vs inductor | bitwise equal,`max\|Δ\| = 0.00e+00` |
| 同两个 gate 跑在**抢救出来的** `pi05_infer/gemma` 副本上 | bitwise equal,`max\|Δ\| = 0.00e+00` |
| 端到端 actions,`pi05_infer` vs RLinf path,固定 seed,`[1,50,6]` float64 | **bitwise equal,`max\|Δ\| = 0.00e+00`** ✅ 2026-07-28 用**同进程双代码树**的强判据重验通过:24/24 个 stage digest 相同(`noise0` → `prefix/pad` → `prefix/kv` → 10 步 → `actions`),`actions max\|Δ\| = 0.00e+00`,0/300,双臂空对照均干净 |

### 切换到自带 `gemma/` 之后重测

同机同 harness,一个 session 内 B A B A 配对,各 30 iterations:

| run | arm | mean | p50 | min / max | clocks |
|---|---|--:|--:|--:|---|
| B1 | `pi05_infer`(vendored expert) | **44.29 ms** | 44.27 | 43.52 / 45.21 | 2445 MHz,240.1 W |
| A1 | RLinf reference | 44.37 | 44.21 | 43.75 / 45.32 | — |
| B2 | `pi05_infer`(vendored expert) | **44.05 ms** | 44.03 | 43.56 / 44.42 | 2438 MHz,208.8 W |
| A2 | RLinf reference | 44.97 | 44.91 | 44.34 / 45.70 | — |

B mean 44.17 vs A mean 44.67 → Δ = −0.50 ms,在可比 SM 时钟下把改动前的 −0.52 ms
复现到 0.02 ms。B 的绝对值相对上面的 44.01 ms 移动了 +0.16 ms,在 ±0.7 ms 之内。

| check(改动后) | result |
|---|---|
| isolation,`tools/isolation_check.py` | `ISOLATION_OK` —— expert `pi05_infer.gemma.modeling_gemma`,prefix `transformers.models.gemma.modeling_gemma` |
| `tools/bitgate.py` on `pi05_infer/gemma`(两个 fusion,q/k/v) | bitwise equal,`max\|Δ\| = 0.00e+00` |
| 端到端 actions vs RLinf path,固定 seed | **bitwise equal,`max\|Δ\| = 0.00e+00`** |
| denoise kernels/step | 234.90,不变 |
| prefix stream-7 kernels/predict | 1018.00,不变 |
| streams 7 / 157 / 158 的每一类 kernel 计数 | 改动前后、两臂全部相同 |

### Kernel 计数

nsys 2026.1.2,`-t cuda,nvtx --cuda-graph-trace=node --gpu-metrics-devices=cuda-visible`,
`cudaProfilerApi` 内 12 个 predict:

| | arm A(RLinf) | arm B(`pi05_infer`) |
|---|--:|--:|
| denoise,kernels/step(stream 157 + 其窗口内全部) | 234.90 | **234.90** |
| … 其中 stream 157(inductor denoise cudagraph) | 171.00 | 171.00 |
| **prefix,stream 7,kernels/predict** | 1024.00 | **1018.00** |
| prefix,stream 7,GPU busy µs/predict | 23771.8 | 23863.9(+0.39 %) |
| stream 158,kernels/step | 35.60 | 35.60 |
| total kernels/predict | 3090 | 3084 |

stream-157 的 kernel 分类表两臂完全一致(同 22 类、同计数,时间差 <0.1 %)。prefix 的
差是 6 个 kernel/predict,全在被删掉的 RL bookkeeping —— `index_elementwise_kernel` −1
(`log_probs[arange, denoise_inds]` gather)、`reduce_kernel` −1(values mean)、
`torch.stack` 的 `chains` / `log_probs` / `values` 拷贝 −4。**没有任何 GEMM / attention /
Triton kernel 的计数发生变化**(`cutlass::Kernel2` 75.00,`triton_tem_fused_mm` 54.00,
`triton_tem_fused_bmm` 36.00 … 全同),所以 prefix 结构上未被触动;+0.39 % 的 busy time
是一个严格更小 kernel 集合上的 run-to-run 噪声。

关于历史上的 **238 kernels/step**:`tools/denoise_kernels.py` 在**两臂**上都测到
234.90。它的窗口在最后一个 stream-157 kernel 处结束,因此排除了最后一个 denoise step
的尾部 eager glue;把窗口放宽就会溢进下一个 predict 的 prefix。3 个 kernel 的差是定义
差异而非回归 —— 产生 238 这个数字的那份代码(arm A)用这个工具测同样是 234.90。

---

## 2. 2026-07-28 · 把整个去噪步捕获成一张 CUDA 图(开关 `--stage1`)

这个图的机制随抽取一起搬了过来,但没有任何代码调用它,所以上面每一个数字都是
**eager denoise loop** 测出来的。

**配对 A/B**,4 轮交替、独立进程(两臂需要不同的 compile mode),各 30 iterations
after 8 warmup,串行,plain wall clock
(`tools/ab_stage1.sh 4 30` → `tools/ab_stage1_summary.py`):

| round | off(max-autotune) | on(`--stage1`) | Δ | SM clock off / on | W off / on |
|---|--:|--:|--:|---|---|
| 1 | 44.16 ms | **43.07 ms** | −1.09 | 2428 / 2430 MHz | 216.3 / 215.9 |
| 2 | 44.28 | **43.30** | −0.98 | 2438 / 2432 | 211.9 / 209.8 |
| 3 | 43.50 | **43.09** | −0.41 | 2433 / 2423 | 219.1 / 217.5 |
| 4 | 44.41 | **43.16** | −1.25 | 2448 / 2430 | 208.7 / 210.0 |
| **grand mean** | **44.08 ms** | **43.16 ms** | **−0.93 ms**(sd 0.36,n=4) | | |

每一轮都是负的,且同一轮两臂的 SM 时钟相差 ~20 MHz 以内,所以效应大于配对本来要
抵消的时钟漂移。

**每 denoise step 的 GPU idle**(`tools/step_idle.py`,两份 profile 用相同 flag 背靠背
拍摄;anchor `_qkv_rope_kernel`,18×/step,108 个 predict 内部 step):

| build | step wall | busy | idle | idle % |
|---|--:|--:|--:|--:|
| `--stage1` **off** | 1390.0 µs | 1247.8 | 142.2 | 10.2 % |
| `--stage1` **on** | **1294.2 µs** | 1233.7 | **60.5 µs** | **4.7 %** |

−95.8 µs/step × 10 steps = **−0.96 ms/predict**,单靠这一项就解释了 −0.93 ms 的配对
wall-clock 差。每 predict 的总 GPU busy 基本持平(40.32 → 40.26 ms),即收益是纯粹的
launch-gap 消除;丢掉 `vision_tower` 的 inductor cudagraph 的连带代价(不可避免:
`--stage1` 把整个 build 切到 `-no-cudagraphs`)约 +0.08 ms 的 prefix busy,在噪声内。

**怎么确认 graph 真的生效。** `graphNodeId` **不是**判据:开关关掉时,2160 个
denoise kernel 也全都带 `graphNodeId`,因为 inductor 自己就发了一个 cudagraph。可靠的
信号是:

| signal | off | on |
|---|---|---|
| `denoise/expert_forward` NVTX ranges | 10 / predict | **0**(body 在 graph 里) |
| distinct `graphId`s | 2(expert 20520 kern + vision 4272) | **1**(28560 kern) |
| kernels/step on stream 157 | 171 | **238** |
| `_denoise_graph_captured` asserted by the bench | n/a | True |

238 kernels/step 正是历史出货 build 的数字 —— 现在整个 step body(不只是 expert
block)是一组 graph node。

**位级一致**,固定 seed,`[1,50,6]` 在 float64 下比较:

| check | result |
|---|---|
| `--stage1` on vs off | **bitwise equal,`max\|Δ\| = 0.00e+00`** |
| `--stage1` on vs RLinf reference path | **bitwise equal,`max\|Δ\| = 0.00e+00`** |
| `--stage1` off vs RLinf reference path | bitwise equal,`max\|Δ\| = 0.00e+00` |

⚠️ 这三行是**端到端 dump**,属于开头那条警告里说的弱门(当时确实测到 `0.00e+00`,
但它证不伪)。本项**没有**用编译路径上的强判据重验过。设计上它是无损的(下一段的
代数论证),这一点没有被推翻。

设计上就是无损的:`flow_ode` 的 `x_t_std == 0`,所以 graph 里捕获的 Euler 更新
(`x_t_next = x_t_mean`)在代数上就是 eager 的 `x_t_mean + noise * 0`;而 eager 的
`sample_noise` 抽样留在 graph 外,全局 RNG 消耗不变。

Profiles:`claude_mem/pi05_rollout_forward/20260728_stage1_pi05infer_pro5k/`
(`stage1_on.nsys-rep`、`stage1_off.nsys-rep`,加 sqlite 导出、A/B 日志和两份工具输出)。

⚠️ 与旧 idle 数字对比的一个坑:2026-07-28 这张图之前的那份 profile
(`pi05_infer_runs/nsys_pi05infer.sqlite`)用同一工具测到 1461.4 µs wall / 214.7 µs
idle,而今天的 off 臂是 1390.0 / 142.2 —— 但两者 GPU busy **完全一致**(1246.7 vs
1247.8 µs/step)。72 µs/step 的差是那次采集里的 host-side stall,不是代码变化。
**只在同一 session 拍的 profile 之间比较 idle。**

---

## 2b. 2026-07-28 · 删掉没人读的 timestep 条件计算(开关 `RLINF_SKIP_DEAD_ADARMS_COND`)

`get_suffix_out` 每步都算 `cond = time_mlp(sinusoidal(timestep))`,但 adaRMS 调制预算表
上线之后 `adarms_mod` 恒有值,`modeling_gemma` 的 `elif adarms_cond is not None` 分支再也
没进过 —— 算出来的 `adarms_cond` 直接被丢弃。改动加了一个 `skip_adarms_cond` 开关
(默认沿用旧行为;`get_suffix_out` 只在自己确实拿着 `adarms_mod` 时才传),**保留** fallback
分支。kill switch:`RLINF_SKIP_DEAD_ADARMS_COND=0` 恢复旧行为。

两臂**同一份代码、同一 session**,只翻这个环境变量。

**位级一致**,固定 seed,`[1,50,6]` 在 float64 下比较,两种 compile mode 各一组
(`max-autotune` 与 `--stage1` 的 `max-autotune-no-cudagraphs`),再加 RLinf 参考臂,
共 5 份 dump 两两对拍(10 对):

| check | result |
|---|---|
| skip1 vs skip0,`--stage1` | **bitwise equal,`max\|Δ\| = 0.00e+00`**,0/300 元素不同 |
| skip1 vs skip0,`max-autotune` | **bitwise equal,`max\|Δ\| = 0.00e+00`**,0/300 |
| skip1 vs RLinf 参考路径 | **bitwise equal,`max\|Δ\| = 0.00e+00`**,0/300 |
| 其余 7 对 | 全部 bitwise equal,`0.00e+00` |

⚠️ 同样是端到端 dump 的弱门。不过这一项的代数论证很硬:被删掉的 `adarms_cond` 在
π0.5 路径上**没有任何消费者**(`elif` 分支永不进入,时间嵌入也不进 action token),
删的是死代码而不是改算法;10 对全过、且**两种 compile mode 各一组**都过,
比单组一致更难用"winner 恰好没动"解释。

**Kernel**,nsys 2026.1.2,`--gpu-metrics-devices=cuda-visible --cuda-graph-trace=node`,
12 predicts,两臂前后脚拍于同一 session,stream 157(捕获后的去噪图):

| 类别 | skip0 n/step | skip1 n/step | Δn | Δµs/step |
|---|--:|--:|--:|--:|
| `internal::gemvx::kernel`(两个 time-MLP `Linear`) | 2 | **0** | −2 | **−18.38** |
| `vectorized_elementwise`(`cos`/`sin`/`silu`/fp64 正弦嵌入) | 28 | 18 | −10 | −14.57 |
| `elementwise_kernel` | 10 | 8 | −2 | −8.28 |
| `cublasLt::splitKreduce` | 4 | 2 | −2 | −2.19 |
| `cublasLt::epilogue::globalKernel` | 2 | **0** | −2 | −2.07 |
| `unrolled_elementwise` + 其余两类零碎 | 9 | 6 | −3 | −3.10 |
| **stream 157 全量** | **238.00** | **217.00** | **−21** | **−47.3** |

**没有任何 GEMM / attention / Triton kernel 的数量或时间改变**
(`triton_tem_fused_mm` 18、`_swiglu_mm_kernel` 18、`_qkv_rope_kernel` 18、
`triton_tem_fused_bmm` 36 全部不变,时间差 < 0.8 µs/step);prefix(stream 7)
673 kernels/predict 两臂完全一致。

**每 step GPU idle / wall**(`tools/step_idle.py`,同上两份 profile):

| arm | step wall | GPU busy | idle | idle % |
|---|--:|--:|--:|--:|
| skip0(旧行为) | 1294.0 µs | 1233.3 | 60.7 | 4.69 % |
| skip1(新,默认) | **1242.7 µs** | 1186.2 | **56.5** | **4.54 %** |

−51.3 µs/step × 10 = **−0.51 ms/predict**(时间线口径)。

**去噪图仍然正常捕获**(两臂都是):`denoise/expert_forward` NVTX = **0/predict**、
distinct `graphId` = **1**(skip0 28560 kern,skip1 26040 kern)、
bench 的 `_denoise_graph_captured` 断言通过。

**e2e 配对 A/B**,4 轮交替,每轮 30 iterations after 8 warmup,串行:

| round | skip0(旧) | skip1(新) | Δ | SM clock 旧/新 |
|---|--:|--:|--:|---|
| 1 | 43.20 ms | **42.90 ms** | −0.30 | 2416 / 2423 MHz |
| 2 | 43.25 | **42.91** | −0.33 | 2425 / 2420 |
| 3 | 43.26 | **42.90** | −0.37 | 2430 / 2417 |
| 4 | 43.07 | **42.88** | −0.20 | 2412 / 2416 |
| **grand mean** | **43.20 ms** | **42.90 ms** | **−0.30 ms**(sd 0.07,n=4) | |

4/4 轮同号,散布 −0.20…−0.37。e2e 的 −0.30 ms 小于时间线的 −0.51 ms,差额落在 prefix 的
run-to-run 漂移(28.00 → 28.13 ms/predict),不是这项改动造成的。

⚠️ **坑,值得单独记一笔:配对 A/B 的两臂必须先验证它们确实不同。** 第一轮测量给出
Δ = −0.02 ms(sd 0.04),看上去像"收益淹没在噪声里"。真实原因是同步到远端机器的 patch 是
**加 kill switch 之前**的版本,两臂跑的是同一个 build —— 那次其实是一次**空对照**。
是 nsys 分别数两臂的 kernels/step(都是 217,而不是 238 vs 217)把它抓出来的。
这次事故的副产品很有用:它把本机**配对 A/B 的噪声地板**钉在 **Δ = −0.02 ms、sd 0.04**,
正是有了这个地板,才能说 −0.30 ms(≈ 15× 地板)是可测的收益而不是噪声。

Artifacts:`claude_mem/pi05_rollout_forward/20260728_adarms_cond/`
(`prof_skip0/1.nsys-rep` + sqlite、`ab/` 下 8 份 clocks json + log、`ab_summary.txt`、
`step_idle.txt`、`v2_driver.log`)。

---

## 2c. 2026-07-28 · 小 M 的 mm tile 候选(`RLINF_SMALL_M_MM`,commit `ca4ae39`)

只对 `(m ≤ 64, n = 1024, k ∈ {2048, 4096})` 追加 5 个小 `BLOCK_M` 候选,`BLOCK_K` 钉死 128。
inductor 自己 benchmark 之后 `down_proj`/`o_proj` 从 `BM64 BN32`(32 CTA)换到
`BM16 BN64`(64 CTA)。

**核时间**(nsys 2026.1.2,12 predict / 120 denoise step,两臂 kernel 数完全相同):

| kernel(stream 157) | off µs/call | on µs/call | Δ | 达成带宽 off → on |
|---|--:|--:|--:|---|
| `triton_tem_fused_mm`(down_proj,8.90 MB/call) | 15.06 | **11.71** | −22.2 % | 591 → **760 GB/s** |
| `triton_tem_fused_clone_mm`(o_proj,4.49 MB/call) | 8.47 | **6.94** | −18.1 % | 530 → **647 GB/s** |

stream 157 合计 11482.3 → 10554.6 µs/predict(−8.1 %);stream 7 / 158 只有噪声级变化。
**两个 GEMM 自己的贡献 = −0.879 ms/predict。**

**e2e**,三个口径统一折算到自然频率 ~2420 MHz:

| 口径 | 值 |
|---|--:|
| nsys 核时间 | **−0.879 ms** |
| 不锁频 6 轮配对 A/B(6/6 同号),按 SM 时钟归一 | −0.874 ms |
| 锁频 2065 MHz 6 轮配对 A/B,取同臂序位置对比(48.52 → 47.49 @2065) | **−0.88 ms** |
| (锁频全 6 轮配对均值,含 off 臂位置效应,偏乐观) | −1.31 ms |
| (不锁频 raw,时钟污染,偏悲观) | −0.35 ms |

**结论取 −0.88 ms/predict。** ⚠️ 不锁频时 on 臂 boost 时钟系统性低 29 MHz(1.2 %),
光时钟就值 +0.52 ms,比效应本身还大 —— 功耗只有 ~215 W / 300 W,**不是功耗墙,是 boost 抖动**。
这一档 <1 ms 的效应**必须锁频**。⚠️ 锁频后 off 臂仍有臂序效应(排第二时 +1.2 ms,原因未查),
所以要交替臂序 + 同位置对比。

**位级一致(强判据)**:`tools/bitexact_denoise_gemms.py` 用真实 checkpoint 权重跑全部 18 层
的 `down_proj` + `o_proj`,生产 shape / dtype / **stride**,对 36 个输出张量取 sha256 ——
共享 cache 与**各自独立 fresh cache**(日志确认两臂确实编出了 `BM64 BN32` vs `BM16 BN64`)
两组的 digest **全部相同**,`max|Δ| = 0.00e+00`。独立 tile 扫描 15 个 config × 2 shape
(`BLOCK_K` 全 128)输出 hash 全同,印证 `BLOCK_M`/`BLOCK_N`/`warps`/`stages` 不改结果。

⚠️ **坑**:两臂**必须共用**一个 `TORCHINDUCTOR_CACHE_DIR`。第一次实验给每臂一个,
autotune 把**没被改动碰过**的 shape 也重新裁决了(SigLIP 的 `mm(768x1152, 1152x4304)`
一臂选 cuBLAS、另一臂选 Triton),结果是 **+0.25 ms 反号**、bit-exact 检查 300/300 不同。
`inductor_mm_tiles.py` 里 bump `cache_key_tag` 保证两臂的 FXGraphCache 条目仍然分开。

原始数据:`claude_mem/pi05_rollout_forward/20260728_small_m_mm_tiles/`。

---

## 2d. 2026-07-28 · 跳过 prefix LM 最后一层的死算(`RLINF_SKIP_LAST_LM_LAYER`,commit `72af442`)

`sample_actions` 只消费 prefix 的 KV cache,LM 的输出 embedding 绑完就丢。所以第 17 层
(共 18 层)里除 `input_layernorm → k/v_proj → RoPE(k) → cache.update` 之外全是死算
(按 FLOP 是该层的 99.1 %,是 18 层 LM 的 5.5 %)。

**核时间**(nsys,12 predict,自然频率):

| stream | off kernel/predict | on kernel/predict | off µs/predict | on µs/predict | Δ |
|---|--:|--:|--:|--:|--:|
| **7**(prefix LM) | **808.00** | **796.00** | 23946.3 | 22834.3 | **−1112.0** |
| 157(denoise) | **1710.00** | **1710.00** | 10561.4 | 10574.3 | +12.9(噪声) |
| 158(SigLIP) | 383.00 | 383.00 | 4713.3 | 4775.0 | +61.7(噪声) |

**stream 157 的 kernel 数一个不差** —— 这是最强的隔离证据:这条改动只动了 prefix。
消失的 12 个核正好是一层里的死算:5 个 GEMM(q、o、gate、up、down)+ 2 个 attention bmm
+ gelu + softmax + 第二个 residual-cat + `post_attention_layernorm`。
FLOP 占比 99.1 % 对应实测时间占比 **85.7 %**(1.112 / 1.297 ms/层)—— 差在留下来的 k/v 投影
效率低(占该层 0.9 % 的 FLOP 却吃掉 14 % 的时间),**FLOP 占比在这里是偏乐观的代理,系数 ~0.86**。

**e2e**,锁频 2100 MHz(实测 ~2072),2 个 campaign 各 6 轮、逐轮交替臂序,两臂共享 cache:

| 口径 | 值 |
|---|--:|
| **nsys 核时间(stream 7),自然频率** | **−1.11 ms/predict** ← 结论取这个,最保守 |
| nsys 全部 busy | −1.04 ms |
| e2e 锁频配对 A/B,n=12,**12/12 同号** | **−1.60 ms**(sd 0.62,SE 0.18)@ ~2072 MHz |
| 同上折算到自然频率 2420 MHz | −1.37 ± 0.15 ms |

e2e 比核时间多的 0.26 ms(~1.5 SE)**没有**算作收益:少掉的 12 次 launch 按 1.3 µs/次只有
16 µs,更可能是 e2e 口径里连带少掉的 GPU 间隙。

**位级一致**:`tools/bitexact_prefix_kv.py` 对 18 层 36 个 KV 张量逐字节 sha256 ——
**eager 下 36/36 完全相同**(COMBINED digest 一致,且 `prefix_output_is_none` off=False /
on=True 证明 patch 确实生效)。KV 是 `sample_actions` 从 prefix LM 唯一拿走的东西,
所以"KV 逐位不变"就是这条改动的全部正确性论证。
⚠️ **编译态只到"不比重编一次更糟"**:`max-autotune` 下 prefix 输出本来就跨进程不可复现
(7 次 run 里 `prefix_embs` 出现 2 个值,与臂无关)。用 `--embs-file` 把 SigLIP 输出冻住回放后,
跨臂的 KV worst `max|Δ|` 是 2.000 / 2.250,而**把同一份代码重编一次**的空对照是 1.625 / 1.750
—— 同一量级。固定编译产物后完全确定(同进程 4/4、同 cache 跨进程 digest 相同)。

⚠️ **上线前提(条件性安装)**:RLinf 的 `get_value_from_vlm(prefix_output)` 读的正是这个被丢弃的
hidden state,门是 `use_vlm_value = value_after_vlm and add_value_head`。
`install_skip_last_lm_layer()` 检测到 VLM value head 就**不安装**。实测
`examples/embodiment/config/*_ppo_openpi_pi05*.yaml` 共 19 份:**15 份**两个开关都 True
(→ 不安装);4 份(`behavior_*`、`robotwin_*`)只设了 `add_value_head`、`value_after_vlm`
用默认 `False`(→ 会安装);DSRL / SAC 那几个 `add_value_head: False`(→ 会安装)。
训练路径结构上摸不到这个 patch:joint 分支直接取 `layer.input_layernorm` /
`layer.self_attn.q_proj` / `layer.mlp` 手算,从不调 `GemmaDecoderLayer.forward`。
模块树 / 参数名 / `state_dict` 全不变 ⇒ **RL 权重同步不受影响**。

原始数据:`claude_mem/pi05_rollout_forward/20260728_skip_last_lm_layer/`。

---

## 2e. 2026-07-28 · bit-exact 补验:四项结构性优化在编译路径上的重验

完整文档:`claude_mem/pi05_rollout_forward/RESULTS_bitexact_backfill.md`。
方法:`--freeze-prefix`(先存一次 prefix 的 pad mask + 18 层 KV,之后每个 run 原样回放,
把唯一会跨进程漂的 SigLIP 整个摘掉)+ 每臂各跑两次做空对照。
**26 个同臂空对照,26/26 全部逐位相同**,所以每一条判决都是有分辨力的。

| 项 | 判据 | eager | `max-autotune`(出货) | actions `max\|Δ\|`(编译) |
|---|---|:--:|:--:|--:|
| 预计算 adaRMS 调制量 | 冻结 prefix 四进程门 | **PASS** | **FAIL** | 2.568e-03(300/300,1.08 %) |
| Q/K/V 并成一个 GEMM | 同上 + isolated 组 | PASS(历史) | **FAIL** | 2.431e-03(1.02 %) |
| prefix KV 静态缓冲区 | 同上 + isolated 组 | PASS(历史) | **FAIL** | 2.858e-03(1.19 %) |
| 设备端 `att_masks` | 同进程张量级 | PASS | **PASS** | —(完备证明,bs=1) |
| 仓库剥离等价性 | 同进程双代码树 24 级 digest | — | **PASS** | 0.00e+00,0/300 |
| SigLIP 三路合批(前史) | 同进程三级 A/B | **FAIL** | **FAIL** | 4.582e-03 / 2.528e-03 |

三条 FAIL 的第一处发散全部在 `step0/mean`(第一步去噪的输出),三个 on 臂的
`step0/mean` digest 完全相同(它们本来就是同一个出货配置)—— harness 自洽性的免费交叉检验。

**机制(核级直接量到,不是推的)**:拿真实的 37 个 `dense` 权重和真实的 `cond`,对比 eager
`n.dense(cond)` 与 `torch.compile(F.linear, "max-autotune-no-cudagraphs")` ——
**逐位相同的 norm 数 0 / 37**,112640 / 113664 元素不同,`max|Δ| = 4.394e-3`。
预算表是在编译区**之外**用 eager 建的,被它替换掉的是编译区**内**的投影核,两个核算同一个
GEMM,没有义务给出同样的比特。

⚠️ **明确没做的**:(a) 三条 FAIL **没有在 `--stage1` 下重跑**,严格说只对 base
`max-autotune` 成立;(b) static KV 那条的机制**未证实**(喂给 SDPA 的 K/V 在编译图内取不到);
(c) `att_masks` 只验了 bs=1;(d) SigLIP 合批只验了 3 视角 / bs=1。

⚠️ **方法学**:**"eager 下 bit-exact" 是一个比看上去弱得多的结论。** 三条 eager 结论在编译
路径上全部翻转,原因是同一个:改动只要动了 GEMM 的形状、或把计算挪出编译区、或换了张量的
来源,inductor 就可能换 tile / 换累加分块。**以后凡是要声称 bit-exact,判据必须跑在实际
出货的编译模式下。**

原始数据:`claude_mem/pi05_rollout_forward/20260728_bitexact_backfill/`(34 个 run 的 digest
+ 每个门的 `verdict.txt`)。

---

## 2f. 2026-07-28 · Nsight Compute 占用率剖析 + SwiGLU tile 扫描(证伪)

完整文档:`claude_mem/pi05_rollout_forward/RESULTS_ncu_occupancy.md`,原始数据
`claude_mem/pi05_rollout_forward/ncu_occupancy/`。`ncu` 2025.1.1 在 sm_120 上**完全可用**
(occupancy / stall / 内存管线全部出真数据),用 `--graph-profiling=node` +
`--profile-from-start off` 直接抓手抓 CUDA 图里的单个 kernel,**没有走替代路径**。
⚠️ **ncu 的绝对 duration 未采信**(比 nsys 高 1.12–1.36×);所有 GB/s = ncu 字节 ÷ nsys µs。

**这张卡的可达 DRAM 读带宽 = 1222 GB/s**(独立 STREAM 式基准,2 GB / 4 GB,`torch.randn`
造数,= spec 1344 的 91 %;读写混合 1106–1144)。**之前把 996 当天花板是循环论证** ——
996 就是 `_swiglu_mm_kernel` 自己的数。

| kernel | CTA | 实际 CTA/SM(限制因子) | SM active % | `cyc_active.min` | 全 GPU 有效 occ | DRAM % | GB/s(% of 1222) |
|---|--:|---|--:|--:|--:|--:|---|
| `bmm_7`(P·V) | 64 | 1(**smem** 96 KB) | 46.5 | **0** | 3.9 % | 13.1 | 222(18 %) |
| `bmm_5`(Q·Kᵀ) | 64 | 1(smem) | 41.5 | **0** | 7.0 % | 11.1 | 193(16 %) |
| `clone_mm_8`(o_proj) | 64 | 1(smem 80 KB) | 48.2 | **0** | 4.0 % | 39.6 | 636(52 %) |
| `mm_10`(down_proj) | 64 | 1(smem 80 KB) | 51.3 | **0** | 4.3 % | 50.4 | 752(62 %) |
| `_qkv_rope_kernel` | 88 | 1(smem 72 KB) | 64.2 | **0** | 5.3 % | 44.2 | 740(61 %) |
| `_swiglu_mm_kernel` | 128 | 2(smem 48 KB) | **84.4** | 36731 | 8.2 % | 67.7 | **973(79.6 %)** |

* **六个核全部是 shared memory 卡死的,没有一个是寄存器卡死的**(`bmm_7` 的 254 regs
  允许 2 个 CTA,是 96 KB smem 把它压到 1)⇒ **"降寄存器压力"是无效动作**。
* **`sm__cycles_active.min = 0` 是硬证据**:六个里五个存在"一个 cycle 都没跑过"的 SM。
  64 CTA 的四个核有 **46 / 110 个 SM 全程空转**。
* `eligible warps per scheduler` 全部 ≤ 0.21(硬件上限 12);`not_selected` 有四个核
  **精确为 0.0 %** —— 连"两个 warp 抢发射槽"都从没发生过。**是 warp 不够,不是等 DRAM。**
* MFU 可精确分解成 `每 SM 张量管线 % × SM active %`,六个核全部在 1–3 个百分点内复现
  roofline 的 MFU —— ncu 计数器与 roofline 互证。
* 字节模型对 **DRAM 侧**是准的(4 个权重主导核误差 ≤ 3 %);两个 attention bmm 高估
  是因为模型按 head 数重复计了被 8 个 Q head 共享的 KV(L2/DRAM 8.5–10×)。
  六个核的 **DRAM 写全部为 0**,输出留在 L2 被下一个核吃掉。

**SwiGLU tile 扫描:8 个配置,现行的最优 —— 这条靶子作废。**
(只动 `BLOCK_M`/`BLOCK_N`/`warps`/`stages`,`BLOCK_K` 钉死 64;**8 个配置输出逐位相同**。)

| 变化 | grid(CTA) | 核时间 |
|---|--:|--:|
| **现行 `(64,32,64,4,4)`** | 128 | **18.89 µs** ← 最快 |
| 缩 tile | 256 | 20.63 / 21.75 µs |
| 再缩 | 512 | 22.30 µs |
| `warps` 4 → 8 | — | 慢 17 % |

**CTA 变多单调变慢**,方向与 small-M 那两个核**相反** —— 因为 SwiGLU 已经是六个里唯一
110 个 SM 全部开工的核,再切碎只摊薄每个 CTA 的访存效率。
⚠️ 这张表的 18.89 µs 与 nsys 稳态的 17.35 µs 是**两把尺**,只能组内比较。
剩下那 ~20 %(973 → 1222 GB/s,理论 ~0.63 ms/predict)的来源是 (i) 128 CTA 填 220 个槽位
造成 18/92 的不均衡(`cyc_active` min/max = 0.85)、(ii) A 矩阵被 128 个 N-tile 重读 ~13 MB
吃掉 L2 带宽(L2/DRAM = 2.03,L2 hit 50.1 %)—— 两条都不是 tile 参数能解的。
**结论:要改结构,不是调 tile。**

---

## 3. 2026-07-27 · 与 `dexmal/realtime-vla` 的正面对比

完整文档:`claude_mem/pi05_rollout_forward/HEADTOHEAD_realtime_vla_pro5k.md`。

对方代码:`github.com/dexmal/realtime-vla @ b86a942`,入口 `pi05_infer.py::Pi05Inference`。
配置逐项对齐:π0.5、3 views、224×224 / 256 patch per view、200 language token、
**968 prefix token**、K=10、chunk 50、bs=1、bf16、action dim 32。
未能对齐的一项:他们的权重是随机 N(0, 0.02)(他们的 loader 只吃 JAX checkpoint 转换出的
pickle,我们只有 HF safetensors)。这一项的影响被单独测掉了:N(0,0.02) 与 N(0,1) 分别给出
43.09 / 43.30 ms,即只要数值非退化,数字对权重尺度不敏感。

同一 scope(从 CPU 上的 uint8 相机帧开始,到 action chunk D2H 结束):

| | n | mean | median | sd | min | max |
|---|--:|--:|--:|--:|--:|--:|
| theirs,scope B e2e | 30 | **43.407** | 43.377 | 0.204 | 43.047 | 43.985 |
| ours(当天 build),scope B wall clock | 30 | **44.55** | 44.57 | — | 43.95 | 44.87 |

**当天结论:对方快 1.14 ms(2.6 %)。** 时钟归一化(cycle 而非 ms)后:prefix 打平
(我们 63.6 vs 他们 64.5 Gcycles,我们少 1.4 %);denoise 他们领先 14.3 %
(30.1 vs 35.1 Gcycles)。他们每个 denoise step 跑 165 个 kernel,我们当时 306。

后续的 kernel fusion 把 expert block 拉到 163 kernels/step(对方 165),两个融合块的
执行时间反超:SwiGLU 312 µs vs 他们 380,QKV+RoPE 132 µs vs 他们 144。

⚠️ **43.41 ms 与本仓库当前的 42.90 ms 不是配对测量** —— 相隔一天、不同 session、
不同 build。两者的差(0.51 ms)小于本机记录在案的 ±0.7 ms rebuild variance,所以
**不能**据此宣称超越。唯一做过配对的正面对比就是上表的 43.41 vs 44.55。

⚠️ 另注:早期笔记把 `dexmal/realtime-vla` 和 `limxdynamics/FluxVLA` 混为一谈。
它们是两个不同的仓库;"他们每步重算 adaRMS"、"44.89 ms 打平" 这两条来自 FluxVLA fork,
**不适用于** realtime-vla —— realtime-vla 在 `Pi05Inference.__init__` 里就把
37 × 10 个 adaRMS 投影全部预计算了,和我们一样。

---

## 4. 2026-07-27 · 零权重伪影(对照实验)

`ZEROWEIGHT_control_ours.md`,artifacts 在 `zeroweight_control/`。

在**我们自己**的 stack 上复现:真实权重 48.47 ms @ 2251 MHz,`torch.zeros` 权重
42.78 ms @ 2559 MHz;kernel census 完全相同(3903 launches,零差异);
时间比 1.1330 vs 时钟比 1.1368 —— 吻合到 0.33 %。

在 realtime-vla 的 `benchmark.py` 上同样成立:它把权重留成 `torch.empty(...)`
(driver 返回全零页,实测 2,826,721,040 个元素,0 个非零),同一份代码在我们的配置下
零权重 37.0 ms、随机权重 43.1 ms,时间比 1.186 vs 时钟比 1.182,差 0.4 %。

结论:**永远不要在有功耗墙的 GPU 上用 `torch.empty` / `torch.zeros` 权重测延迟。**
