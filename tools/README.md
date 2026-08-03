# `tools/`

31 scripts in four groups. Only the first group is meant to be run by a reader; the
other three are the machinery behind the numbers, kept so the numbers can be re-derived.

Set the checkpoint once instead of passing it every time:

```bash
export PI05_MODEL_PATH=/path/to/RLinf-Pi05-LIBERO-SFT
```

Both `.sh` files in group 1 -- `bitexact_gate.sh` and `run_bitexact_backfill.sh` --
default their interpreter to `PY=/opt/venv/openpi/bin/python`, which exists only in the
author's container. Outside it, run them as `PY=$(which python) tools/<script>.sh ...`.

## 1. Verification gates -- portable, run these

These take every path they need as an argument (or from `$PI05_MODEL_PATH`) and run
anywhere the package is installed. They are the ones the root README points at.

| script | checks |
|---|---|
| `isolation_check.py` | the vendoring boundary: expert = `pi05_infer.gemma`, prefix = stock `transformers` |
| `bitgate.py` | the two hand-written Triton fusion kernels |
| `bitexact_denoise_gemms.py` | the small-M `mm` retile (`down_proj`, `o_proj`) |
| `bitexact_denoise_bmms.py` | the attention `bmm` retile and the `Q*K^T` tile pin |
| `bitexact_prefix_kv.py` | the prefix last-layer skip |
| `bitexact_prefix_qkv.py` | the fused prefix QKV |
| `bitexact_rope_hoist.py` | the hoisted denoise step invariants (`RLINF_HOIST_STEP_INVARIANTS`) |
| `bitexact_gate.sh` | end-to-end numerical A/B, four processes, always with an empty control |
| `run_bitexact_backfill.sh` | driver for the compiled-path gates in group 2 (`siglip\|extraction\|prefix\|adarms\|qkv\|kvstatic\|attmask`) |

The two digests must match. A gate declares INCONCLUSIVE, never PASS, when its own null
control fails.

Invocations per gate -- running the wrong number is silent:

| gate | invocations |
|---|---|
| `isolation_check.py`, `bitgate.py`, `bitexact_prefix_qkv.py`, `bitexact_rope_hoist.py` | one; both arms run in-process |
| `bitexact_denoise_gemms.py` | two: `RLINF_SMALL_M_MM=0` then `=1`, one shared `TORCHINDUCTOR_CACHE_DIR` |
| `bitexact_denoise_bmms.py` | two: same with `RLINF_SMALL_M_BMM` |
| `bitexact_prefix_kv.py` | three: `RLINF_SKIP_LAST_LM_LAYER=0 --out off.json`, `=1 --out on.json`, `--compare off.json on.json` |

## 2. Compiled-path gate workers and diagnostics

The first three are what `run_bitexact_backfill.sh` drives -- run them through it unless
you are debugging a stage. The last two exist for when a gate comes back disagreeing.

| script | what it settles |
|---|---|
| `bitexact_siglip_batch.py` | does batching the three camera views into one SigLIP call move the numbers |
| `bitexact_extraction.py` | does `pi05_infer` still compute what RLinf computes, both arms in one process. Needs `--rlinf-root` (or `$RLINF_ROOT`) pointing at an RLinf checkout |
| `bitexact_compiled_toggles.py` | the four structural optimizations (adaRMS table, fused QKV, static prefix-KV, device-side `att_masks`) on the compiled path, by monkeypatching the seam |
| `bitexact_adarms_dense.py` | follow-up to the `adarms` stage: do eager `dense(cond)` and inductor's compiled `addmm` agree on that shape |
| `determinism_probe.py` | where a `--dump-actions` run stops being reproducible across processes -- digests every intermediate instead of only the final actions |

## 3. Readers, probes and summarisers -- portable, but situational

| script | input / caveat |
|---|---|
| `denoise_kernels.py` | kernels per denoise step, counted on the GPU timeline |
| `ksum.py` | kernel-count summary for a denoise stream, plus a category rollup |
| `stream_summary.py` | per-stream kernel rollup |
| `step_idle.py` | per-denoise-step GPU idle, on the kernel timeline |
| `prefix_census.py` | NVTX-attributed, per-stream kernel census |
| `ab_stage1_summary.py` | turns an `ab_stage1.sh` output directory into a paired A/B table |
| `peak_bf16_gemm.py` | achievable dense bf16 tensor-core throughput of the card it runs on |
| `prefix_sol_probe.py` | clock-resolved timing of the prefix phase, isolated and in situ |
| `ab_rlinf_reference.py` | the RLinf reference arm of the A/B. Needs `--rlinf-root` (or `$RLINF_ROOT`) and an importable RLinf install; not part of the package |
| `sglang_pi05_bench.py` | in-process bs=1 latency of SGLang's pi0.5 pipeline, for comparison against `bench/standalone_infer_bench.py` |

The first five read an nsys `.sqlite` you pass in: portable, but they assume this model's
stream layout (7 = prefix, 157 = denoise, 158 = vision) and an nsys **2026.1.2** export.

## 4. Measurement drivers -- author's machine, not portable

The six `ab_*.sh` and `prof.sh` were written against one box and one container:

* **All seven** hard-code `export CUDA_VISIBLE_DEVICES=0` and default their output
  directory to a path under `/workspace` (`AB_OUT`, and `PROF_OUT` for `prof.sh`,
  override the output; the device you have to edit).
* **Six of seven** hard-code `cd /workspace/rlinf_pub/pi05-infer`. Only
  `ab_small_m_bmm.sh` resolves the repository from `$0`. That `cd` you have to edit.
* **Three** -- `ab_prefix_qkv.sh`, `ab_skip_last_lm_layer.sh`, `ab_small_m_bmm.sh` --
  call `nvidia-smi -lgc` / `-rgc` on GPU 0 to lock the SM clock, so they need the
  privilege to do that and a card no other job is using.
* The six `ab_*.sh` default `PY=/opt/venv/openpi/bin/python`. `prof.sh` has no such
  override: it hard-codes both that interpreter and `NSYS=/opt/nsys2026/...`.

They are kept in the repository because they are the exact commands behind the numbers
quoted in the README and in the source comments -- the recipe matters even when the paths
do not. **They will not run unmodified elsewhere.**

## Why the measurement discipline looks paranoid

These scripts are shaped the way they are because of specific failures:
separate inductor cache dirs once produced a **sign-flipped** A/B result; an unlocked
clock read -1.05 ms where the real effect was -0.32; and a four-round A/B whose null
control read -4.5 % read +0.1 % at twelve rounds. The controls are not ceremony.
