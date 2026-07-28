# pi05-infer

Standalone π0.5 / π0 **batch-size-1 inference** path, extracted from RLinf.

This repo exists for two reasons:

1. **Rescue.** The complete, measured, bit-exact-validated set of π0.5 bs=1
   optimizations lived only inside a running container's `site-packages`, untracked by
   any repository, downstream of an `install.sh` `cp -r` whose upstream source is
   pristine. One `install.sh` re-run would have deleted ~245 lines of validated work.
   Commit 1 of this repo is that rescue, verbatim.
2. **Separation.** The two largest optimizations (the adaRMS modulation table, −4 ms,
   and the Stage-1 denoise CUDA graph, −2.5 ms) live in the *orchestration* layer,
   which upstream openpi does not have. That layer was tangled into RLinf's RL model
   class. Here it is a ~750-line inference engine with no RLinf imports.

Cumulative effect of the optimizations this code carries: **52.60 → 43.7 ms** e2e
`predict_action_batch` (−17 %), all bit-exact.

---

## What is in here

```
pi05_infer/
  engine.py            OpenPi0Inference: the pure inference orchestration.
                       predict_action_batch -> sample_actions -> sample_mean_var_val
                       -> get_velocity -> get_suffix_out, plus _build_prefix_cache,
                       enable_torch_compile, the Stage-1 denoise CUDA graph, the
                       adaRMS table and invalidate_weight_derived_caches.
  builder.py           build_model(): checkpoint + norm-stats + transform wiring.
  dataconfig/          minimal subset of RLinf's openpi dataconfigs (turtle + libero).
  _vendored/           verbatim copies of RLinf helpers with no RLinf dependency:
                       base_policy.py, cuda_graph.py, nvtx.py
  gemma/               the Gemma fork the ACTION EXPERT runs: modeling_gemma.py
                       (+245 L over transformers) + rlinf_fused_denoise.py (two
                       Triton fusions). Imported; nothing else uses it.
  openpi_patched/      the two openpi files we modified: pi0_pytorch.py +
                       gemma_pytorch.py. Imported instead of openpi's copies.
bench/
  standalone_infer_bench.py   latency benchmark (e2e, phases, nsys, action dump)
tools/
  isolation_check.py          proves expert = pi05_infer.gemma, prefix = transformers
  bitgate.py                  bit-exactness gate for the two Triton fusion kernels
  ab_rlinf_reference.py       reference arm: same harness driving the RLinf path
_extract_src/                 the un-refactored RLinf sources this was extracted from
```

### The prefix / expert split

`import pi05_infer` pulls in **its own** model code. Nothing here reads
`transformers.models.gemma` or `openpi.models_pytorch.pi0_pytorch` for the parts we
modified:

```
engine.OpenPi0Inference
  └─ pi05_infer.openpi_patched.pi0_pytorch.PI0Pytorch
       └─ pi05_infer.openpi_patched.gemma_pytorch.PaliGemmaWithExpertModel
            ├─ paligemma    = transformers.PaliGemmaForConditionalGeneration   ← STOCK
            │                   └─ text tower via AutoModel.from_config
            │                        → transformers.models.gemma.modeling_gemma
            └─ gemma_expert = pi05_infer.gemma.modeling_gemma.GemmaForCausalLM ← OURS
```

That one construction site is the whole enforcement mechanism, and
`tools/isolation_check.py` asserts it module by module. The separation is not
cosmetic: the +4 ms regression during the fusion work happened because PaliGemma's
*prefix* language model is also a Gemma, so overwriting `transformers/models/gemma/`
globally made a kernel tuned for the 50-token denoise suffix fire on the 968-token
prefix. With the expert holding its own module, that class of bug cannot occur —
prefix and expert are now different classes from different files.

**What is ours vs. what is imported.** Only the symbols we actually changed are
vendored. `modeling_gemma.py` imports every unmodified symbol it needs
(`GemmaConfig`, `PreTrainedModel`, `GenerationMixin`, `GradientCheckpointingLayer`,
`ALL_ATTENTION_FUNCTIONS`, `create_causal_mask`, the rope utils, the output
dataclasses, `ACT2FN`, `Cache`/`DynamicCache`, `Unpack`, `auto_docstring` …) from the
installed `transformers`; `pi0_pytorch.py` imports `openpi.models.gemma` and
`openpi.models_pytorch.preprocessing_pytorch` from the installed openpi. See
`EXTRACTION_NOTES.md` §8 for the full boundary, including the one thing this does
*not* buy independence from (openpi's own `transformers_replace` patch, which the
**prefix** needs).

### Vendored from where

| what | source | rev |
|---|---|---|
| `engine.py` | `rlinf/models/embodiment/openpi/openpi_action_model.py` | `cbb9d2fc` |
| `builder.py` | `rlinf/models/embodiment/openpi/__init__.py` (`get_model`) | `cbb9d2fc` |
| `_vendored/base_policy.py` | `rlinf/models/embodiment/base_policy.py` | `cbb9d2fc` |
| `_vendored/cuda_graph.py` | `rlinf/utils/cuda_graph.py` | `cbb9d2fc` |
| `_vendored/nvtx.py` | `nvtx_range` from `rlinf/utils/utils.py` | `cbb9d2fc` |
| `dataconfig/` | `rlinf/models/embodiment/openpi/dataconfig/` + `policies/` | `cbb9d2fc` |
| `bench/standalone_infer_bench.py` | `benchmarks/pi05_infer/standalone_infer_bench.py` | `cbb9d2fc` |
| `tools/bitgate.py` | `claude_mem/pi05_rollout_forward/kernel_fusion/scripts/bitgate.py` | — |
| `gemma/`, `openpi_patched/` | `pi05bench` container `site-packages` | 2026-07-28 |

### What was dropped

Everything RL: the SFT / NFT / DSRL-SAC forwards, the value head and every
`add_value_head` branch, `ExploreNoiseNet`, the `flow_sde` / `flow_cps` / `flow_noise`
samplers, the `chains` / `denoise_inds` / log-prob / value bookkeeping in
`sample_actions`, and the `forward_inputs` dict returned by `predict_action_batch`
(which now returns just the actions). Config went from 34 fields to 8.

Three RL statements were deliberately **kept** because they sit inside the measured
per-denoise-step region and deleting them would change the kernel count — see
`EXTRACTION_NOTES.md` §2.1.

---

## Install (into the existing benchmark container)

No image rebuild. The package is installed editable, with `--no-deps` so the
container's pinned torch / transformers / openpi are untouched:

```bash
# from the analysis box
tar czf - --exclude=.git pi05-infer | ssh <gpubox> \
    'tar xzf - -C /home/chenchaox/project/RLinf_pi0.5_inference'

# on the GPU box
docker exec -w /workspace/rlinf_pub/pi05-infer pi05bench \
    /opt/venv/openpi/bin/pip install -e . --no-deps --no-build-isolation
```

`pi05-infer` does **not** touch `site-packages`; it only adds a path entry. The
container stays in its current working state so it can be A/B'd against.

> The container's `transformers/models/gemma/` still carries the old global
> overwrite (that is deliberate — it is the reference arm of the A/B). The package
> no longer *uses* it for the expert. Because `torch.library` namespaces are
> process-global and both copies of `rlinf_fused_denoise.py` get imported in the
> same process, this package registers its custom ops as `pi05_infer::gate_up_swiglu`
> / `pi05_infer::qkv_rope_kv` rather than `rlinf::*`; otherwise the second
> registration would raise and silently disable the fusions.

## Run the benchmark

```bash
docker exec -w /workspace/rlinf_pub/pi05-infer pi05bench \
  /opt/venv/openpi/bin/python bench/standalone_infer_bench.py \
      --model-path /workspace/rlinf_pub/models/RLinf-Pi05-LIBERO-SFT \
      --config-name pi05_turtle --iters 30

# phase breakdown
... --phases
# dump actions for a numerical A/B, and record SM clock / power in the timed window
... --dump-actions /tmp/a.pt --clocks-json /tmp/clocks.json
# Stage 1: hand-captured denoise CUDA graph (opt-in, see below)
... --stage1
```

### `--stage1` — the hand-captured denoise CUDA graph

`--stage1` captures **one complete flow_ode denoise step** (expert forward + value +
Euler + logprob) into a `torch.cuda.CUDAGraph` and replays it for every step, so one
replay replaces the expert-only inductor cudagraph *plus* all the eager glue between
launches. It is **opt-in**; the default path is unchanged so existing measurements
stay reproducible.

Two things the flag handles for you, both of which are load-bearing:

* It rewrites `--compile-mode max-autotune` to `max-autotune-no-cudagraphs`. Inductor's
  own cudagraphs cannot be nested inside a hand-captured one; the failure mode is
  `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten`.
  `capture_cuda_graph` rejects the cudagraph-emitting modes outright.
* It asserts `is_cuda_graph_enabled()` **and** `_denoise_graph_captured` after warmup.
  The `CUDAGraph` is captured lazily, on the first eval-shaped `sample_actions`, so a
  shape-signature mismatch would otherwise fall back to the eager loop with no symptom
  other than the runtime. That silent fallback is exactly how this flag came to be
  missing from the package in the first place.

The adaRMS modulation table is built *before* capture (it runs capture-illegal eager
ops) and the per-step index into it is a device gather, so both survive capture; the
captured signature is printed, e.g.
`((1, 50, 32), torch.float32, (1, 32), (1, 968), 10)`.

**The GPU is shared and power-capped at 300 W.** Check `pgrep -f
"run_stage1|standalone_infer"` before timing, never run two timing jobs at once (a
concurrent job cost 1.5 ms of CPU contention once), and only compare runs at
comparable SM clocks — a cold 30-iteration run and a sustained one differ by ~8 %.
Rebuild variance is ±0.7 ms, so a single run cannot resolve a small regression; use
`tools/ab_rlinf_reference.py` for a paired comparison instead of guessing.

## Verification

```bash
# 0. isolation: expert must be pi05_infer.gemma, PaliGemma prefix must be transformers
/opt/venv/openpi/bin/python tools/isolation_check.py            # prints ISOLATION_OK

# 1. bit-exactness of the two Triton fusion kernels vs inductor's compiled output
/opt/venv/openpi/bin/python tools/bitgate.py                    # the vendored copy
/opt/venv/openpi/bin/python tools/bitgate.py \
    /opt/venv/openpi/lib/python3.11/site-packages/transformers/models/gemma

# 2. numerical A/B of the whole path, fixed seed
/opt/venv/openpi/bin/python tools/ab_rlinf_reference.py --dump-actions /tmp/ref.pt
/opt/venv/openpi/bin/python bench/standalone_infer_bench.py --dump-actions /tmp/new.pt
python -c "import torch;a=torch.load('/tmp/ref.pt');b=torch.load('/tmp/new.pt');print((a-b).abs().max())"

# 3. nsys (2026 only — 2025.x cannot read its own output on this GPU; export the
#    sqlite with the same 2026 binary)
bash tools/prof.sh nsys_pi05infer pi05infer 12    # this package
bash tools/prof.sh nsys_rlinf     rlinf     12    # the reference arm
bash tools/prof.sh nsys_stage1    stage1    12    # this package + --stage1

# 3b. Stage-1 paired A/B (4 alternating rounds x 30 iters) and per-step GPU idle
bash tools/ab_stage1.sh 4 30
/opt/venv/openpi/bin/python tools/ab_stage1_summary.py \
    /workspace/rlinf_pub/pi05_infer_runs/ab_stage1
/opt/venv/openpi/bin/python tools/step_idle.py <off.sqlite> <on.sqlite>

# 4. per-stream rollup (prefix = stream 7, denoise cudagraph = stream 157) and
#    kernels per denoise step
/opt/venv/openpi/bin/python tools/stream_summary.py <sqlite> 12 10
/opt/venv/openpi/bin/python tools/denoise_kernels.py            # both profiles
/opt/venv/openpi/bin/python tools/ksum.py <sqlite> 7 12         # kernel categories
```

### Measured 2026-07-28

RTX PRO 5000 72 GB Blackwell (sm_120), 300 W cap, driver-idle GPU, one job at a time.
bs=1, `pi05_turtle` (action_horizon 50), 10 denoise steps, 3 × 128² cameras,
`torch.compile max-autotune`, checkpoint `RLinf-Pi05-LIBERO-SFT`.

**Latency** — 30 iterations after 8 warmup calls, serialized, plain wall clock:

| | RLinf path (arm A, reference) | `pi05_infer` (arm B) | Δ |
|---|--:|--:|--:|
| e2e `predict_action_batch`, CPU wall clock, mean | 44.53 ms | **44.01 ms** | −0.52 ms |
| … p50 | 44.50 | 43.95 | |
| … min / max | 44.01 / 45.35 | 43.39 / 45.08 | |
| GPU event span, mean | 44.51 ms | 43.99 ms | −0.52 ms |
| SM clock during window | not sampled | 2445 MHz (2430–2452), 227.6 W | |

Both arms are cold 30-iteration runs in the same session, ~0.3–0.8 ms above the
43.74 ms historical figure — inside the documented ±0.7 ms rebuild variance and the
~8 % cold/sustained clock spread. Arm B is *faster* by 0.52 ms, which the nsys data
below attributes exactly to the 6 removed per-predict RL-bookkeeping kernels.

**Bit-exactness**

| check | result |
|---|---|
| `fused_gate_up_swiglu` vs inductor `max-autotune-no-cudagraphs` | bitwise equal, `max|Δ| = 0.00e+00` |
| `fused_qkv_rope_kv` q / k / v vs inductor | bitwise equal, `max|Δ| = 0.00e+00` |
| the same two gates against the **rescued** `pi05_infer/gemma` copy | bitwise equal, `max|Δ| = 0.00e+00` |
| end-to-end actions, `pi05_infer` vs RLinf path, fixed seed, `[1,50,6]` float64 | **bitwise equal, `max|Δ| = 0.00e+00`** |

### Re-measured 2026-07-28, after the package was switched onto its own `gemma/`

Same box, same harness. Paired B A B A in one session, 30 iterations each:

| run | arm | mean | p50 | min / max | clocks |
|---|---|--:|--:|--:|---|
| B1 | `pi05_infer` (vendored expert) | **44.29 ms** | 44.27 | 43.52 / 45.21 | 2445 MHz, 240.1 W |
| A1 | RLinf reference | 44.37 | 44.21 | 43.75 / 45.32 | — |
| B2 | `pi05_infer` (vendored expert) | **44.05 ms** | 44.03 | 43.56 / 44.42 | 2438 MHz, 208.8 W |
| A2 | RLinf reference | 44.97 | 44.91 | 44.34 / 45.70 | — |

B mean 44.17 vs A mean 44.67 → Δ = −0.50 ms, reproducing the pre-change −0.52 ms to
0.02 ms at comparable SM clocks. Absolute B moved +0.16 ms vs the 44.01 ms above,
inside the ±0.7 ms rebuild variance.

| check (after the change) | result |
|---|---|
| isolation, `tools/isolation_check.py` | `ISOLATION_OK` — expert `pi05_infer.gemma.modeling_gemma`, prefix `transformers.models.gemma.modeling_gemma` |
| `tools/bitgate.py` on `pi05_infer/gemma` (both fusions, q/k/v) | bitwise equal, `max|Δ| = 0.00e+00` |
| end-to-end actions vs the RLinf path, fixed seed | **bitwise equal, `max|Δ| = 0.00e+00`** |
| denoise kernels/step | 234.90, unchanged |
| prefix stream-7 kernels/predict | 1018.00, unchanged |
| every per-category kernel count, streams 7 / 157 / 158 | identical pre vs post, both arms |

Full detail, including the pre/post nsys table and what the vendoring does *not*
isolate, in `EXTRACTION_NOTES.md` §8.

**Kernels** — nsys 2026.1.2, `-t cuda,nvtx --cuda-graph-trace=node
--gpu-metrics-devices=cuda-visible`, 12 predicts inside `cudaProfilerApi`:

| | arm A (RLinf) | arm B (`pi05_infer`) |
|---|--:|--:|
| denoise, kernels/step (stream 157 + everything in its window) | 234.90 | **234.90** |
| … of which stream 157 (inductor denoise cudagraph) | 171.00 | 171.00 |
| **prefix, stream 7, kernels/predict** | 1024.00 | **1018.00** |
| prefix, stream 7, GPU busy µs/predict | 23771.8 | 23863.9 (+0.39 %) |
| stream 158, kernels/step | 35.60 | 35.60 |
| total kernels/predict | 3090 | 3084 |

The stream-157 kernel-category table is identical between the two arms (same 22
categories, same counts, times differing by <0.1 %). The prefix delta is 6 kernels
per predict, all in the deleted RL bookkeeping — `index_elementwise_kernel` −1 (the
`log_probs[arange, denoise_inds]` gather), `reduce_kernel` −1 (the values mean), and
−4 across the `torch.stack` copies of `chains` / `log_probs` / `values`. **No GEMM,
attention or Triton kernel changed count** (`cutlass::Kernel2` 75.00, `triton_tem_fused_mm`
54.00, `triton_tem_fused_bmm` 36.00 … identical), so the prefix is structurally
untouched; its +0.39 % busy time is run-to-run noise on a strictly smaller kernel set.

Caveat on the historical **238 kernels/step**: `tools/denoise_kernels.py` measures
234.90 on *both* arms. Its window ends at the last stream-157 kernel and so excludes
the trailing eager glue of the final denoise step; widening it spills into the next
predict's prefix. The 3-kernel gap is a definitional difference, not a regression —
arm A, i.e. the code that produced the 238 figure, measures 234.90 with this tool too.

### Measured 2026-07-28, `--stage1` (hand-captured denoise CUDA graph)

Same box and harness. The Stage-1 machinery came across with the extraction but
nothing called it, so every number above was measured with the *eager* denoise loop.

**Paired A/B**, 4 alternating rounds, separate processes (the arms need different
compile modes), 30 iterations after 8 warmup calls, serialized, plain wall clock
(`tools/ab_stage1.sh 4 30` → `tools/ab_stage1_summary.py`):

| round | off (max-autotune) | on (`--stage1`) | Δ | SM clock off / on | W off / on |
|---|--:|--:|--:|---|---|
| 1 | 44.16 ms | **43.07 ms** | −1.09 | 2428 / 2430 MHz | 216.3 / 215.9 |
| 2 | 44.28 | **43.30** | −0.98 | 2438 / 2432 | 211.9 / 209.8 |
| 3 | 43.50 | **43.09** | −0.41 | 2433 / 2423 | 219.1 / 217.5 |
| 4 | 44.41 | **43.16** | −1.25 | 2448 / 2430 | 208.7 / 210.0 |
| **grand mean** | **44.08 ms** | **43.16 ms** | **−0.93 ms** (sd 0.36, n=4) | | |

Every round is negative and the two arms of a round sit within ~20 MHz of each other,
so the effect is larger than the clock drift the pairing is there to cancel. 43.16 ms
is **below the 43.74 ms RLinf all-fusions + Stage-1 reference**.

**Per-denoise-step GPU idle** (`tools/step_idle.py`, both profiles shot back to back
with identical flags; anchor `_qkv_rope_kernel`, 18×/step, 108 predict-internal steps):

| build | step wall | busy | idle | idle % |
|---|--:|--:|--:|--:|
| `--stage1` **off** | 1390.0 µs | 1247.8 | 142.2 | 10.2 % |
| `--stage1` **on** | **1294.2 µs** | 1233.7 | **60.5 µs** | **4.7 %** |

−95.8 µs/step × 10 steps = **−0.96 ms/predict**, which accounts for the −0.93 ms
paired wall-clock delta on its own. Total GPU busy per predict is flat (40.32 → 40.26
ms), i.e. the win is pure launch-gap removal, and the collateral cost of dropping
`vision_tower`'s inductor cudagraph (unavoidable: `--stage1` switches the whole build
to `-no-cudagraphs`) is ≈ +0.08 ms of prefix busy — inside the noise.

**How to tell the graph is actually live.** `graphNodeId` is *not* a discriminator:
with Stage 1 off, all 2160 denoise kernels already carry one, because inductor emits
its own cudagraph. The reliable signals are:

| signal | off | on |
|---|---|---|
| `denoise/expert_forward` NVTX ranges | 10 / predict | **0** (body is inside the graph) |
| distinct `graphId`s | 2 (expert 20520 kern + vision 4272) | **1** (28560 kern) |
| kernels/step on stream 157 | 171 | **238** |
| `_denoise_graph_captured` asserted by the bench | n/a | True |

238 kernels/step is exactly the historical shipped-build figure — the whole step body,
not just the expert block, is now one graph node set.

**Bit-exactness**, fixed seed, `[1,50,6]` compared in float64:

| check | result |
|---|---|
| `--stage1` on vs off | **bitwise equal, `max\|Δ\| = 0.00e+00`** |
| `--stage1` on vs the RLinf reference path | **bitwise equal, `max\|Δ\| = 0.00e+00`** |
| `--stage1` off vs the RLinf reference path | bitwise equal, `max\|Δ\| = 0.00e+00` |

Lossless as designed: `flow_ode` has `x_t_std == 0`, so the Euler update captured in
the graph (`x_t_next = x_t_mean`) is algebraically the eager `x_t_mean + noise * 0`,
and the eager `sample_noise` draw is kept outside the graph so global RNG consumption
is unchanged.

Profiles: `claude_mem/pi05_rollout_forward/20260728_stage1_pi05infer_pro5k/`
(`stage1_on.nsys-rep`, `stage1_off.nsys-rep`, + sqlite exports, the A/B logs and the
two tool outputs).

⚠️ One caveat on comparing against older idle numbers: the 2026-07-28 pre-Stage-1
profile (`pi05_infer_runs/nsys_pi05infer.sqlite`) measures 1461.4 µs wall / 214.7 µs
idle with this same tool, versus 1390.0 / 142.2 for today's off arm — at **identical**
GPU busy (1246.7 vs 1247.8 µs/step). The 72 µs/step difference is host-side stall in
that older capture, not a code change. Only compare idle figures from profiles shot in
the same session.

## Not done

- the **three-way merge** of the openpi-side files (`patches/` × fork × runtime) and
  vendoring `model.py` + `array_typing.py` — it would change behaviour (the
  `array_typing` typecheck patch is currently *not* applied; enabling it moves e2e by
  ~3.2 ms), so it needs its own A/B;
- **plan stage 3**, rewiring RLinf's `openpi_action_model.py` onto this engine;
- **plan stage 4**, restoring the container's `site-packages` to openpi-pristine and
  clearing the 11 `.bak_*` files. The vendoring makes this possible; it was not done
  because the patched copy is the reference arm of the A/B.

See `EXTRACTION_NOTES.md` §7–§8.
