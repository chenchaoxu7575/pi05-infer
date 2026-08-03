# `tools/`

Two kinds of script live here, and only one kind is meant for you.

## 1. Verification gates — portable, run these

These take every path they need as an argument (or from `$PI05_MODEL_PATH`) and run
anywhere the package is installed. They are the ones the root README points at.

| script | checks |
|---|---|
| `isolation_check.py` | the vendoring boundary: expert = `pi05_infer.gemma`, prefix = stock `transformers` |
| `bitgate.py` | the two hand-written Triton fusion kernels |
| `bitexact_denoise_gemms.py` | the small-M `mm` retile (`down_proj`, `o_proj`) |
| `bitexact_denoise_bmms.py` | the attention `bmm` retile and the `Q·Kᵀ` tile pin |
| `bitexact_prefix_kv.py` | the prefix last-layer skip |
| `bitexact_prefix_qkv.py` | the fused prefix QKV |
| `bitexact_gate.sh` | end-to-end numerical A/B, four processes, always with an empty control |
| `run_bitexact_backfill.sh` | the compiled-path gates (`siglip\|extraction\|prefix\|adarms\|…`) |

Set the checkpoint once instead of passing it every time:

```bash
export PI05_MODEL_PATH=/path/to/RLinf-Pi05-LIBERO-SFT
```

Every gate compares two arms and prints a digest; **the two digests must match**. They
are written to declare INCONCLUSIVE rather than PASS when their own null control fails.

## 2. Measurement drivers — author's machine, not portable

`ab_*.sh` and `prof.sh` hard-code `cd /workspace/rlinf_pub/pi05-infer` and an output
directory under `/workspace`, and several assume a specific container name, a second
RLinf checkout, or an `nvidia-smi` the caller owns exclusively.

They are kept in the repository because they are the exact commands behind the numbers
quoted in the README and in the source comments — the recipe matters even when the paths
do not. **They will not run unmodified elsewhere.** `AB_OUT` / `PROF_OUT` / `PY` override
the output and interpreter; the `cd` you have to edit.

Also in this class: `denoise_kernels.py`, `ksum.py`, `stream_summary.py`, `step_idle.py`
and `prefix_census.py` — these read an nsys `.sqlite` you pass in and are portable, but
they assume this model's stream layout (7 = prefix, 157 = denoise, 158 = vision).

## Why the measurement discipline looks paranoid

Both classes of script exist in the shape they do because of specific failures:
separate inductor cache dirs once produced a **sign-flipped** A/B result; an unlocked
clock read −1.05 ms where the real effect was −0.32; and a four-round A/B whose null
control read −4.5 % read +0.1 % at twelve rounds. The controls are not ceremony.
