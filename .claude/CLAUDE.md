# pi05-infer — working conventions

π0.5 action-expert inference, bs=1, optimized for RTX PRO 5000 (GB202 / sm_120).

## Documentation scope — this one is a hard rule

**The root `README.md` carries only two things: the results, and how to install and run it.**

Keep: the headline number, the ledger chart, install / run / verify commands, links out.
Everything else — per-optimization narratives, measurement blocks, ruler discussions,
correctness arguments, traps, rejected approaches — lives in **`opt.md`**.

- `README.md` (English) and `README.zh-CN.md` must stay content-identical.
- `docs/MEASUREMENTS.md` is the raw per-A/B archive; `EXTRACTION_NOTES.md` is the RLinf boundary.
- **Moving something out of the README means moving it *into* `opt.md`, not deleting it.**
  No measured number may be dropped without a home.

## Claims that must not be overstated

- ⛔ **Never write "every optimization is bit-exact."** Bit-identity is *tiered by compile
  path*: some items are bit-identical only under eager, not under the shipping
  `max-autotune`. The transforms are all algebraically equivalent — say that instead.
- ⛔ **No win/loss claim against reference implementations.** The chart's dashed lines are
  not paired measurements. The only paired head-to-head we have, we lost by 1.14 ms.
- ⛔ **Different rulers do not chain.** The 52.60 → 42.90 ledger is a paired chain;
  post-ledger items have separate baselines and must never be added onto 42.90.
- Optimizations with install conditions (e.g. skipping the prefix LM's last layer declines
  to install when a VLM value head is present) must carry that condition wherever the gain
  is quoted.

## Measurement discipline — violating these makes a result invalid, not just noisy

- **e2e is plain wall clock only.** nsys wall clock carries ~2.7 ms of overhead.
- **Paired A/B: same source toggled, serial, ≥4 alternating rounds.** Both arms must share
  one `TORCHINDUCTOR_CACHE_DIR` — separate caches let autotune re-pick untouched shapes and
  have produced a *sign-flipped* result.
- **Lock the clocks.** An unlocked probe once read −1.05 ms where the real effect was
  −0.32. During a whole predict the card draws 145–205 W against a 300 W cap, so it is
  boost-limited and the arms do not self-align. `nvidia-smi -lgc` is silently dropped when
  persistence mode is off — use `-pm 1` and re-set per arm.
- ⛔ **Never build weights with `torch.empty` / `zeros`.** CUDA returns zeroed pages; almost
  no bits flip, the power cap is never hit, and the clock runs ~29 % high.
- **Never run two timing jobs at once** — and note the lock only binds those who take it:
  an agent running *tests* on the same GPU will still contaminate a timing run.
- **Split by stream**: 7 = prefix, 157 = denoise, 158 = vision. A denoise change that moves
  stream 7 is a bug; a +4 ms regression once hid entirely in prefix.
- nsys must be **2026.1.2** (2025.x reports all-zero GPU counters on GB202).

## Calibration constants (measured on this card — do not substitute spec or derived values)

- Achievable DRAM read bandwidth **1222 GB/s** (spec 1344).
- Achievable dense bf16 **194.9 TFLOP/s**. Archived 240 is 23 % high and probably a
  zero-weight measurement; 268 is a derivation, 37 % high.
- 2377 MHz is the *application* clock, not boost; `clocks.max.sm` is 3090. Under a dense
  GEMM the 300 W cap pulls it to 1.78–1.86 GHz. `peak(f) = 110 × f × 1024 × 0.965`.
- 110 SMs, 100 KB shared memory per SM, 96 MB L2.

## Architecture invariants

- **The vendoring boundary is deliberate**: the action expert uses `pi05_infer.gemma.*`;
  the PaliGemma prefix and SigLIP use **stock transformers**. The seam is
  `PaliGemmaWithExpertModel.__init__`. It exists so that a denoise kernel change cannot
  reach the 968-token prefix — that mistake once cost +6.5 ms and was only visible in a
  per-stream profile.
- **Optimize the prefix with surgical patches, not by extending the vendoring.** Precedent:
  `prefix_last_layer.py` swaps one instance's `forward` via `types.MethodType`, leaving the
  module tree, parameter names and `state_dict` untouched so weight sync is unaffected.
- **CUDA graphs record addresses.** Buffers a captured graph reads must be refilled with
  `copy_` and never reallocated. Weight-derived caches go stale silently on an RL weight
  sync — refresh them in place via `refresh_derived_weights()`.
- Every optimization needs a kill switch (`RLINF_*=0`).

## Things already settled — do not re-litigate

- **Tile tuning on the denoise GEMMs is exhausted.** SwiGLU (8 configs) and `qkv_rope`
  (9 configs) are both at their optimum; on both, more CTAs is monotonically worse and
  `warps` 4→8 is worse. `down_proj`/`o_proj` were retuned once; the remaining corners are
  reachable and simply slower.
- `BLOCK_K` is **per-shape**, not globally 128 (SwiGLU 64, `down_proj` 128, and one bmm
  shape has no stable choice at all). `BLOCK_M`/`BLOCK_N`/`warps`/`stages` are numerically
  inert — 17 configs measured.
- **Metrics improving ≠ kernel faster ≠ e2e faster.** ncu's occupancy signal pointed the
  wrong way twice: adding warps to SwiGLU cost 17 %, adding CTAs to the P·V bmm cost 30 %.
  The dispatch-savings theory is disproved — e2e gain equals the kernel-time reduction, so
  convert kernel counts to time with measured durations, never a per-launch constant.
- Rejected with measurements: fp8/quantization (precision is off the table), whole-model
  single graph, Triton-only GEMM backends, GeGLU epilogue fusion on the prefix (the entry
  fee of leaving cuBLAS exceeds the prize).
