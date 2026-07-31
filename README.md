**English** | [简体中文](README.zh-CN.md)

# pi05-infer

A standalone **bs=1 inference engine for the π0.5 action expert**, extracted from
[RLinf](https://github.com/RLinf/RLinf) and optimized for the **RTX PRO 5000
(GB202 / sm_120, Blackwell)**. Every item is an algebraically equivalent transform —
**no quantization, no change of sampler, no reduction in denoising steps**.

## Results

End-to-end `predict_action_batch`: **52.60 ms → 42.90 ms (−9.70 ms, −18.4 %)**,
against a `torch.compile max-autotune` baseline.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ledger_dark.png">
  <img alt="Three-panel optimization ledger: the prehistory before this repository (a different ruler), the paired end-to-end waterfall from 52.60 to 42.90 ms, and the same optimizations accounted per denoising step" src="docs/ledger_light.png">
</picture>

Three panels, **three different rulers — they do not chain**: the prehistory before this
repository (a different measurement protocol), this repository's paired end-to-end ledger
(52.60 → 42.90 ms, the headline), and the same optimizations accounted per denoising step
(GPU busy 2025.6 → 1185.0 µs/step, 347 → 217 kernels/step). Derivations:
[opt.md § ledger](opt.md#s-ledger), [§ per-step](opt.md#s-per-step).

Five more items landed after the ledger was closed. ⚠️ **Different ruler — do not add them
onto 42.90** (their absolute baselines come from separate sessions, some clock-locked,
some not, so they are not part of the paired chain and are not in the chart):

| Optimization | Gain | Commit | Details |
|---|--:|---|---|
| Small-`M` mm tile candidates (`down_proj` / `o_proj`) | **−0.88 ms/predict** | `ca4ae39` | [opt.md §3.1](opt.md#s-3-1) |
| Skip the dead compute in the prefix LM's last layer — ⚠️ **conditionally installed, see caveat 3** | **−1.11 ms/predict** | `72af442` | [opt.md](opt.md#s-after-ledger) |
| Retile the P·V attention `bmm` | **−0.18 ms/predict** | `ff237bf` | [opt.md §3.2b](opt.md#s-3-2b) |
| Hoist the step-invariant work out of the denoise loop | **−0.32 ± 0.05 ms/predict** | `0ed3ca2` | [opt.md](opt.md#s-hoist) |
| Merge the prefix LM's Q/K/V projections into one GEMM | **−0.61 ± 0.22 ms/predict** | `d7cf3c2` | [opt.md](opt.md#s-prefix-qkv) |

<a id="r-config"></a>
**Measured under** π0.5, batch 1, **K = 10** Euler steps, **968 prefix tokens**, action
chunk 50, **bf16 throughout**; RTX PRO 5000 72 GB (GB202, sm_120, 110 SMs, **300 W cap**),
checkpoint `RLinf-Pi05-LIBERO-SFT`, torch 2.7.1+cu128, nsys 2026.1.2
([full config](opt.md#s-roadmap)).

## Caveats

1. **Numerics** — every transform is algebraically equivalent, but **bit-identity is tiered by compile path** (some items are bit-identical under eager only, not under the shipping `max-autotune`): [opt.md § correctness](opt.md#s-correctness).
2. **Reference points** — the two dashed lines mark where reference implementations sit; **neither is a paired measurement and no win/loss is claimed** ([opt.md § baselines](opt.md#s-baselines)).
3. **Skipping the prefix LM's last layer is conditionally installed** — it declines to install when a VLM value head is detected, and **15 of the 19 published pi0.5 PPO configs hit that condition** (kill switch `RLINF_SKIP_LAST_LM_LAYER=0`).

## Install and run

Use the existing RLinf benchmark container image — **no Docker rebuild required**. Install
editable and with `--no-deps`, so the torch / transformers / openpi versions pinned inside
the container are left alone:

```bash
docker exec -w /path/to/pi05-infer pi05bench \
    /opt/venv/openpi/bin/pip install -e . --no-deps --no-build-isolation

# benchmark
/opt/venv/openpi/bin/python bench/standalone_infer_bench.py \
    --model-path /path/to/RLinf-Pi05-LIBERO-SFT --config-name pi05_turtle --iters 30
... --stage1        # enable the hand-captured denoise CUDA graph (opt-in; every number
                    # from ledger row 7 onwards was measured with it on)
... --phases        # per-phase timing
... --dump-actions /tmp/a.pt --clocks-json /tmp/clocks.json   # numerical A/B + SM clock/power
```

`pi05-infer` does not touch `site-packages` (it only adds one path entry), so the container
stays pristine and can serve as the reference arm of an A/B.
`--stage1` rewrites `max-autotune` into `max-autotune-no-cudagraphs` and asserts after
warmup that the graph really was captured — otherwise it falls back to the eager loop
**silently** ([opt.md](opt.md#s-stage1)).

<a id="r-verify"></a>

## Verification: numerical agreement

```bash
python tools/isolation_check.py          # expert = pi05_infer.gemma, prefix = transformers

# kernel / GEMM / KV-level bit-exactness gates
python tools/bitgate.py                  # the two Triton fusion kernels
python tools/bitexact_denoise_gemms.py   # small-M mm retile
python tools/bitexact_denoise_bmms.py    # P·V bmm retile
python tools/bitexact_prefix_kv.py       # prefix last-layer skip
python tools/bitexact_prefix_qkv.py      # fused prefix QKV

# structural optimizations on the compiled path (frozen prefix + four-process control gate)
bash tools/run_bitexact_backfill.sh <stage>   # siglip|extraction|prefix|adarms|adarms_eager|qkv|kvstatic|attmask

# end-to-end numerical A/B -- ⚠️ four processes, always with an empty control; declares
# INCONCLUSIVE (never PASS) unless both same-arm controls come back clean
GATE_OFF="RLINF_SMALL_M_MM=0" GATE_ON="RLINF_SMALL_M_MM=1" \
  tools/bitexact_gate.sh /tmp/gate_small_m --stage1 --iters 1 --warmup 4
```

Every optimization has a kill switch, and the OFF arm exercises a verified fallback path
([opt.md](opt.md#s-fallback)).

<a id="r-layout"></a>

## Repository layout

```
pi05_infer/    the engine (engine.py, the vendored action-expert Gemma + Triton fusion
               kernels, prefix_last_layer.py, prefix_qkv_fused.py, inductor_mm_tiles.py)
bench/         standalone_infer_bench.py -- latency bench
tools/         isolation check, bit-exactness gates, paired A/B drivers, profile analysis
docs/          make_charts.py (regenerates the charts) + MEASUREMENTS.md
_extract_src/  the original RLinf files before extraction (not refactored)
```

`import pi05_infer` routes **only the action expert** through the vendored Gemma; the
PaliGemma **prefix** keeps stock transformers ([opt.md § isolation](opt.md#s-isolation)).
Per-file inventory: [opt.md § inventory](opt.md#s-inventory).

## Further reading

* **[`opt.md`](opt.md)** (Chinese) — the complete optimization record: why/how/how-much per
  item, the correctness argument, the measurement methodology, the traps, and the
  approaches explicitly ruled out.
* **[`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md)** — the raw per-A/B measurement archive.
* **[`EXTRACTION_NOTES.md`](EXTRACTION_NOTES.md)** — the extraction boundary against RLinf.

## License and provenance

Apache-2.0 ([`LICENSE`](LICENSE)). Vendors code from HuggingFace Transformers,
[openpi](https://github.com/Physical-Intelligence/openpi) (via the
[RLinf/openpi](https://github.com/RLinf/openpi) fork) and
[RLinf](https://github.com/RLinf/RLinf); per-file modifications are listed in
[`NOTICE`](NOTICE). `dexmal/realtime-vla` is referenced as a peer; none of its code is
reused.
