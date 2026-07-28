# 测量记录 / Measurement log

本文件是 `README.md` 里那些数字的原始记录。每一节都注明测量日期、机器、配置和
产出的 artifact,便于逐条复核。

统一环境:RTX PRO 5000 72 GB Blackwell(GB202,sm_120,110 SM,1344 GB/s,
**300 W 功耗墙**),容器 `pi05bench`,torch 2.7.1+cu128,nsys 2026.1.2。
配置:π0.5,bs=1,`pi05_turtle`(action_horizon 50),K=10 denoise steps,
968 prefix token,3 × 128² 相机(transform 内 resize 到 224),bf16,
checkpoint `RLinf-Pi05-LIBERO-SFT`。

---

## 1. 2026-07-28 · 独立包 vs RLinf 参考路径(Stage-1 关闭)

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
| 端到端 actions,`pi05_infer` vs RLinf path,固定 seed,`[1,50,6]` float64 | **bitwise equal,`max\|Δ\| = 0.00e+00`** |

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

## 2. 2026-07-28 · `--stage1`(手写 denoise CUDA graph)

Stage-1 的机制随抽取一起搬了过来,但没有任何代码调用它,所以上面每一个数字都是
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

**怎么确认 graph 真的生效。** `graphNodeId` **不是**判据:Stage-1 关闭时,2160 个
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

设计上就是无损的:`flow_ode` 的 `x_t_std == 0`,所以 graph 里捕获的 Euler 更新
(`x_t_next = x_t_mean`)在代数上就是 eager 的 `x_t_mean + noise * 0`;而 eager 的
`sample_noise` 抽样留在 graph 外,全局 RNG 消耗不变。

Profiles:`claude_mem/pi05_rollout_forward/20260728_stage1_pi05infer_pro5k/`
(`stage1_on.nsys-rep`、`stage1_off.nsys-rep`,加 sqlite 导出、A/B 日志和两份工具输出)。

⚠️ 与旧 idle 数字对比的一个坑:2026-07-28 Stage-1 之前的那份 profile
(`pi05_infer_runs/nsys_pi05infer.sqlite`)用同一工具测到 1461.4 µs wall / 214.7 µs
idle,而今天的 off 臂是 1390.0 / 142.2 —— 但两者 GPU busy **完全一致**(1246.7 vs
1247.8 µs/step)。72 µs/step 的差是那次采集里的 host-side stall,不是代码变化。
**只在同一 session 拍的 profile 之间比较 idle。**

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

⚠️ **43.41 ms 与本仓库当前的 43.16 ms 不是配对测量** —— 相隔一天、不同 session、
不同 build。两者的差(0.25 ms)小于本机记录在案的 ±0.7 ms rebuild variance,所以
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
