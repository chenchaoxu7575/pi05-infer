#!/usr/bin/env python3
"""Regenerate the three README charts (light + dark PNG for each).

    python docs/make_charts.py [--sqlite <stage1_on.sqlite> --sqlite-off <stage1_off.sqlite>]

Every number plotted here is a measurement, and every measurement has a source:

* the **ledger** (chart 1) is a chain of paired plain-wall-clock A/Bs, one per
  optimization, each recorded in its own results note.  It is hard-coded below
  with the note it came from, because those runs cannot be replayed from a
  profile.
* the **denoise kernel breakdown** (chart 2) and the **phase split** (chart 3)
  are derived from nsys 2026.1.2 sqlite exports.  Pass ``--sqlite`` /
  ``--sqlite-off`` to re-derive them instead of using the recorded values; the
  script prints what it derived so it can be diffed against the constants.

Design follows a validated categorical palette (blue / orange / aqua) that
passes CVD separation, normal-vision separation and lightness-band checks in
both light and dark mode.  The light-mode aqua sits below 3:1 against the light
surface, so every aqua mark carries a visible direct label.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

OUT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------

THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        s1="#2a78d6",  # slot 1 - blue
        s2="#eb6834",  # slot 2 - orange
        s3="#1baf7a",  # slot 3 - aqua
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        s1="#3987e5",
        s2="#d95926",
        s3="#199e70",
    ),
}


def _style(t: dict) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": t["ink"],
            "axes.labelcolor": t["ink2"],
            "xtick.color": t["muted"],
            "ytick.color": t["ink2"],
            "axes.edgecolor": t["axis"],
        }
    )


def _chrome(ax, t: dict, xgrid: bool = True) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(1.0)
    if xgrid:
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=t["grid"], linewidth=1.0)
    ax.tick_params(length=0)


_KAPPA = 0.5522847498307936  # circle -> cubic-Bezier control-point ratio


def _rounded_hbar(ax, y, x0, x1, height, color, radius_pt=4.0, zorder=3):
    """Horizontal bar with ~4pt-rounded ends, drawn in data coordinates.

    matplotlib has no rounded bars.  The obvious construction -- a
    ``FancyBboxPatch`` with ``boxstyle="Round"`` and ``mutation_aspect``
    carrying the x->y scale ratio -- is *not* usable here: ``mutation_aspect``
    divides the box height, runs the corner rounding, then scales back, so at
    the aspect ratios these charts use (chart 2 is ~0.038 data units of y per
    data unit of x) the intermediate box degenerates and the patch renders at
    full row height, one solid band per bar.  Chart 1 (aspect ~1.4) survives
    it, which is why only chart 2 ever looked broken.

    So the path is built explicitly instead: a rectangle whose four corners are
    circular arcs of ``radius_pt`` points, with the point->data conversion done
    separately per axis so the arcs are round *on screen*.  No mutation, no
    aspect handling, identical output on any matplotlib version.

    Call it only after the axes limits and the subplot geometry are final --
    the conversion reads ``ax.get_window_extent()``.
    """
    fig = ax.figure
    bbox = ax.get_window_extent()
    xr = ax.get_xlim()[1] - ax.get_xlim()[0]
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    px_per_x = bbox.width / xr
    px_per_y = bbox.height / yr

    xa, xb = min(x0, x1), max(x0, x1)
    ya, yb = y - height / 2.0, y + height / 2.0

    # one radius in pixels, clamped so it can never exceed half the bar in
    # either direction, then converted back into each axis' data units
    r_px = min(
        radius_pt * fig.dpi / 72.0,
        (xb - xa) / 2.0 * px_per_x,
        height / 2.0 * px_per_y,
    )
    rx = r_px / px_per_x
    ry = r_px / px_per_y
    k = 1.0 - _KAPPA

    verts = [
        (xa + rx, ya),  # start after the bottom-left corner
        (xb - rx, ya),  # bottom edge
        (xb - rx * k, ya), (xb, ya + ry * k), (xb, ya + ry),  # bottom-right arc
        (xb, yb - ry),  # right edge
        (xb, yb - ry * k), (xb - rx * k, yb), (xb - rx, yb),  # top-right arc
        (xa + rx, yb),  # top edge
        (xa + rx * k, yb), (xa, yb - ry * k), (xa, yb - ry),  # top-left arc
        (xa, ya + ry),  # left edge
        (xa, ya + ry * k), (xa + rx * k, ya), (xa + rx, ya),  # bottom-left arc
        (xa + rx, ya),
    ]
    C, L, M = MplPath.CURVE4, MplPath.LINETO, MplPath.MOVETO
    codes = [M, L, C, C, C, L, C, C, C, L, C, C, C, L, C, C, C, MplPath.CLOSEPOLY]

    p = PathPatch(
        MplPath(verts, codes),
        linewidth=0,
        facecolor=color,
        zorder=zorder,
    )
    ax.add_patch(p)
    return p


# --------------------------------------------------------------------------
# Chart 1 - the optimization ledger
# --------------------------------------------------------------------------
# e2e = predict_action_batch, plain CPU wall clock, 30 iters after 8 warmups,
# serialized, single job on the box.  RTX PRO 5000 (GB202/sm_120), 300 W cap.
# pi0.5, bs=1, K=10 Euler steps, 968 prefix tokens, action chunk 50, bf16.
#
# (label, e2e_after_ms, delta_ms, source note)
LEDGER = [
    ("baseline: torch.compile max-autotune", 52.60, None,
     "adarms_cache_impl/RESULTS_adarms_cache.md"),
    ("+ adaRMS modulation table", 49.77, -2.83,
     "adarms_cache_impl/RESULTS_adarms_cache.md"),
    ("+ Stage-1 denoise CUDA graph", 47.73, -2.04,
     "adarms_cache_impl/RESULTS_adarms_cache.md UPDATE 1"),
    ("+ fused QKV weight", 45.61, -2.12,
     "adarms_cache_impl/RESULTS_adarms_cache.md UPDATE 2"),
    ("+ static KV buffer", 45.10, -0.51,
     "adarms_cache_impl/RESULTS_adarms_cache.md UPDATE 3"),
    ("+ att_masks built on device", 44.79, -0.31,
     "attmask_fix/RESULTS_attmask.md"),
    ("+ SwiGLU & QKV+RoPE epilogue fusion", 43.74, -1.05,
     "kernel_fusion/RESULTS_fusion.md (own paired A/B: 44.87 -> 43.74, -1.14)"),
    ("standalone package + --stage1", 43.16, -0.58,
     "20260728_stage1_pi05infer_pro5k/ab_stage1_summary.txt (paired 44.08 -> 43.16)"),
    ("+ drop the dead timestep conditioning", 42.90, -0.30,
     "20260728_adarms_cond/ab_summary.txt (paired 43.20 -> 42.90, sd 0.07, n=4)"),
]

# dexmal/realtime-vla @ b86a942, our config, our GPU, same day, real (non-zero)
# weights, our e2e scope.  HEADTOHEAD_realtime_vla_pro5k.md sec.2.
PEER_MS = 43.41
PEER_LABEL = "dexmal/realtime-vla, matched config: 43.41 ms"


def chart_ledger(mode: str) -> None:
    t = THEMES[mode]
    _style(t)
    fig, ax = plt.subplots(figsize=(10.5, 4.9), dpi=200)

    base = LEDGER[0][1]
    rows = LEDGER[1:]
    n = len(rows)
    ys = list(range(n))[::-1]

    ax.set_xlim(41.9, 53.7)
    ax.set_ylim(-0.75, n - 0.05)
    fig.subplots_adjust(left=0.295, right=0.985, top=0.845, bottom=0.145)
    _chrome(ax, t)

    ax.axvline(base, color=t["s1"], lw=1.8, zorder=2)
    ax.text(base - 0.15, n - 0.10, f"baseline  {base:.2f} ms", ha="right",
            va="top", fontsize=10, fontweight="bold", color=t["s1"])
    ax.axvline(PEER_MS, color=t["s2"], lw=1.6, ls=(0, (4, 3)), zorder=2)
    ax.text(PEER_MS + 0.16, n - 0.10, PEER_LABEL, ha="left", va="top",
            fontsize=8.5, color=t["s2"])
    ax.text(PEER_MS + 0.16, n - 0.42, "separate session - not a paired A/B",
            ha="left", va="top", fontsize=7.5, color=t["muted"])

    prev = base
    for y, (label, after, delta, _src) in zip(ys, rows):
        _rounded_hbar(ax, y, after, prev, 0.54, t["s3"])
        ax.plot([prev, prev], [y - 0.42, y + 0.70], color=t["axis"], lw=1.0,
                ls=(0, (2, 2)), zorder=1)
        ax.text(after - 0.14, y, f"{delta:+.2f}", ha="right", va="center",
                fontsize=10.5, fontweight="bold", color=t["ink"])
        last = y == ys[-1]
        ax.text(prev + 0.14, y, f"{after:.2f} ms", ha="left", va="center",
                fontsize=11 if last else 9,
                fontweight="bold" if last else "normal",
                color=t["ink"] if last else t["ink2"])
        prev = after

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlabel("end-to-end predict_action_batch latency (ms), plain wall clock",
                  fontsize=9.5)

    fig.suptitle("pi0.5 action expert, bs=1: 52.60 -> 42.90 ms  (-9.70 ms, -18.4%)",
                 x=0.012, ha="left", fontsize=14, fontweight="bold", color=t["ink"])
    fig.text(0.012, 0.895,
             "every step bit-exact against the unoptimised path (max|delta| = 0.00e+00); "
             "bar length = the ms that step bought",
             ha="left", fontsize=9, color=t["ink2"])
    fig.text(0.012, 0.015,
             "RTX PRO 5000 (GB202/sm_120), 300 W cap - K=10 Euler steps, 968 prefix tokens, "
             "chunk 50, bf16, no quantization - x-axis starts at 41.9 ms (waterfall: bars encode deltas)",
             ha="left", fontsize=7.5, color=t["muted"])
    fig.savefig(OUT / f"ledger_{mode}.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# Chart 2 - denoise per-kernel breakdown, current build
# --------------------------------------------------------------------------
# Source: 20260728_adarms_cond/prof_skip1.sqlite, stream 157
# (the hand-captured Stage-1 denoise graph), 12 predicts x 10 steps = 120 steps.
# 217 kernels/step, 1185.0 us/step.
#
# (label, us_per_step, kernels_per_step, group)
#   group 0 = inductor-generated Triton
#   group 1 = eager per-step glue (the remaining headroom)
#   group 2 = our hand-written fused kernels
DENOISE = [
    ("SwiGLU: gate/up GEMM + gelu(g)*u  [fused]", 312.41, 18, 2),
    ("down_proj GEMM", 271.19, 18, 0),
    ("o_proj GEMM (+ transpose prologue)", 152.46, 18, 0),
    ("QKV GEMM + RoPE -> static KV  [fused]", 131.79, 18, 2),
    ("attention BMM  P*V", 110.40, 18, 0),
    ("attention BMM  Q*K^T", 69.27, 18, 0),
    ("adaRMS: gated residual + RMSNorm + scale/shift", 57.29, 42, 0),
    ("eager per-step glue (embed_suffix, Euler, log-prob, position ids)", 50.63, 49, 1),
    ("masked softmax", 29.57, 18, 0),
]
DENOISE_TOTAL_US = 1185.01
DENOISE_TOTAL_KERNELS = 217.0

GROUP_NAMES = {
    0: "inductor Triton (torch.compile max-autotune-no-cudagraphs)",
    1: "eager per-step glue - was 99.7 us / 70 k, half of it was dead code",
    2: "hand-written fused Triton kernel (this project)",
}


def derive_denoise(sqlite_path: str, n_steps: float = 120.0) -> None:
    """Re-derive chart 2 from a profile and print it next to the constants."""
    c = sqlite3.connect(sqlite_path)
    rows = c.execute(
        "select s.value, count(*), sum(k.end - k.start) / 1e3 "
        "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.demangledName "
        "where k.streamId = 157 group by 1"
    ).fetchall()

    def bucket(name: str) -> str:
        if name == "_swiglu_mm_kernel":
            return "SwiGLU: gate/up GEMM + gelu(g)*u  [fused]"
        if name == "triton_tem_fused_mm_10":
            return "down_proj GEMM"
        if name == "triton_tem_fused_clone_mm_8":
            return "o_proj GEMM (+ transpose prologue)"
        if name == "_qkv_rope_kernel":
            return "QKV GEMM + RoPE -> static KV  [fused]"
        if name == "triton_tem_fused_bmm_7":
            return "attention BMM  P*V"
        if name == "triton_tem_fused_bmm_5":
            return "attention BMM  Q*K^T"
        if "softmax" in name:
            return "masked softmax"
        if "mean_mul_pow_rsqrt" in name:
            return "adaRMS: gated residual + RMSNorm + scale/shift"
        return "eager per-step glue (embed_suffix, Euler, log-prob, position ids)"

    agg: dict[str, list[float]] = {}
    for name, cnt, us in rows:
        b = bucket(name)
        a = agg.setdefault(b, [0.0, 0.0])
        a[0] += us / n_steps
        a[1] += cnt / n_steps
    print(f"derived from {sqlite_path} (stream 157, {n_steps:.0f} steps):")
    for label, us, k, _g in DENOISE:
        d_us, d_k = agg.get(label, (0.0, 0.0))
        flag = "" if abs(d_us - us) < 0.05 and abs(d_k - k) < 0.05 else "   <-- DIFFERS"
        print(f"  {label:66s} {d_us:8.2f} us/step ({us:8.2f})  "
              f"{d_k:6.2f} k/step ({k:6.2f}){flag}")
    print(f"  {'TOTAL':66s} {sum(v[0] for v in agg.values()):8.2f} us/step "
          f"({DENOISE_TOTAL_US:8.2f})  {sum(v[1] for v in agg.values()):6.2f} k/step "
          f"({DENOISE_TOTAL_KERNELS:6.2f})")


def chart_denoise(mode: str) -> None:
    t = THEMES[mode]
    _style(t)
    colors = {0: t["s1"], 1: t["s2"], 2: t["s3"]}
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=200)

    n = len(DENOISE)
    ys = list(range(n))[::-1]
    # headroom on the right so the longest value label ("312.5 us  18/step",
    # on the top bar) clears the figure edge instead of being clipped
    ax.set_xlim(0, 432)
    ax.set_ylim(-0.7, n - 0.3)
    fig.subplots_adjust(left=0.415, right=0.985, top=0.875, bottom=0.205)
    _chrome(ax, t)

    for y, (label, us, kern, grp) in zip(ys, DENOISE):
        _rounded_hbar(ax, y, 0, us, 0.56, colors[grp])
        ax.text(us + 5, y, f"{us:.1f} us   {kern:g}/step", ha="left", va="center",
                fontsize=9, color=t["ink"] if grp == 2 else t["ink2"],
                fontweight="bold" if grp == 2 else "normal")

    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in DENOISE], fontsize=9)
    ax.set_xlabel("GPU time per denoise step (us)", fontsize=9.5)

    handles = [
        plt.Line2D([], [], marker="s", ls="", markersize=9, color=colors[g],
                   label=GROUP_NAMES[g])
        for g in (2, 0, 1)
    ]
    # figure-level, below the axes: inside the axes the legend sat on top of
    # the two shortest bars' rows
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.010, 0.048),
               ncol=3, frameon=False, fontsize=8.5, labelcolor=t["ink2"],
               handletextpad=0.6, borderpad=0.2, columnspacing=1.6)

    fig.suptitle("Where a denoise step goes: 1185.0 us, 217 kernels", x=0.012,
                 ha="left", fontsize=14, fontweight="bold", color=t["ink"])
    fig.text(0.012, 0.912,
             "current build, nsys 2026.1.2, stream 157 (Stage-1 captured graph), "
             "12 predicts x 10 steps - 10 steps = 11.85 ms of the 42.90 ms predict",
             ha="left", fontsize=9, color=t["ink2"])
    fig.text(0.012, 0.013,
             "the 37 adaRMS dense(cond) projections that used to cost 395 us/step here "
             "(triton_per_fused_addmm_0, 300 instances) are gone - 0 instances",
             ha="left", fontsize=7.5, color=t["muted"])
    fig.savefig(OUT / f"denoise_{mode}.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# Chart 3 - phase split of GPU busy time
# --------------------------------------------------------------------------
# Source: 20260728_stage1_pi05infer_pro5k/stage1_off.sqlite, 12 predicts.
# Streams: 7 = PaliGemma prefix LM, 157 = denoise expert (inductor cudagraph),
# 158 = SigLIP vision tower.  The --stage1 build merges 158 into 7 and pulls the
# eager glue into 157, so the three-way split is read off the off arm; total GPU
# busy per predict is the same to within 0.06 ms (40.32 vs 40.26).
PHASES = [
    ("prefix: PaliGemma LM over 968 tokens", 24.10, 0),
    ("denoise: 10 x action expert", 11.40, 1),
    ("prefix: SigLIP vision, 3 views", 4.82, 2),
]
PHASE_TOTAL = 40.32


def derive_phases(sqlite_path: str, n_predicts: float = 12.0) -> None:
    c = sqlite3.connect(sqlite_path)
    rows = c.execute(
        "select streamId, count(*), sum(end - start) / 1e6 "
        "from CUPTI_ACTIVITY_KIND_KERNEL group by 1 order by 3 desc"
    ).fetchall()
    print(f"derived from {sqlite_path} ({n_predicts:.0f} predicts):")
    tot = 0.0
    for stream, cnt, ms in rows:
        tot += ms / n_predicts
        print(f"  stream {stream:4d}: {ms / n_predicts:7.2f} ms/predict  "
              f"{cnt / n_predicts:8.1f} kernels/predict")
    print(f"  TOTAL      : {tot:7.2f} ms/predict  (recorded {PHASE_TOTAL:.2f})")


def chart_phases(mode: str) -> None:
    t = THEMES[mode]
    _style(t)
    colors = {0: t["s1"], 1: t["s3"], 2: t["s2"]}
    fig, ax = plt.subplots(figsize=(10.5, 2.8), dpi=200)

    ax.set_xlim(0, PHASE_TOTAL)
    ax.set_ylim(-1.02, 0.72)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(length=0)
    # right < 0.99 so the last segment's right-aligned label ("prefix: SigLIP
    # vision, 3 views") keeps a margin instead of running into the figure edge
    fig.subplots_adjust(left=0.012, right=0.972, top=0.66, bottom=0.10)

    gap = PHASE_TOTAL * 0.0045  # 2px surface gap between segments
    x = 0.0
    for i, (label, ms, grp) in enumerate(PHASES):
        last = i == len(PHASES) - 1
        _rounded_hbar(ax, 0.30, x, x + ms - (0 if last else gap), 0.52, colors[grp])
        ax.text(x + ms / 2, 0.30, f"{ms:.2f} ms", ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="#ffffff")
        lx, ha = (x + ms, "right") if last else (x, "left")
        ax.text(lx, -0.12, label, ha=ha, va="top", fontsize=9.5, color=t["ink"])
        ax.text(lx, -0.44, f"{100 * ms / PHASE_TOTAL:.1f}% of GPU busy", ha=ha,
                va="top", fontsize=9, color=t["muted"])
        x += ms

    ax.text(0, -0.88, "GPU busy per predict: 40.32 ms total "
            "(stage1_off profile, 12 predicts; streams 7 / 157 / 158)",
            ha="left", va="top", fontsize=8, color=t["muted"])

    fig.suptitle("Amdahl: the denoise loop is 28.3% of GPU busy", x=0.012,
                 ha="left", fontsize=14, fontweight="bold", color=t["ink"])
    fig.text(0.012, 0.855,
             "the 968-token prefix is 71.7% - even a free denoise loop caps the "
             "whole-predict speedup at 1.39x",
             ha="left", fontsize=9, color=t["ink2"])
    fig.savefig(OUT / f"phases_{mode}.png")
    plt.close(fig)


# --------------------------------------------------------------------------


def _shrink(colors: int = 64) -> None:
    """Palette-quantise the PNGs (flat-colour charts) - ~3.5x smaller, no visible loss."""
    try:
        from PIL import Image
    except ImportError:
        return
    for f in sorted(OUT.glob("*.png")):
        im = Image.open(f).convert("RGB")
        im.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE).save(
            f, optimize=True
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", help="stage1_on.sqlite - re-derive chart 2")
    ap.add_argument("--sqlite-off", help="stage1_off.sqlite - re-derive chart 3")
    args = ap.parse_args()

    if args.sqlite:
        derive_denoise(args.sqlite)
    if args.sqlite_off:
        derive_phases(args.sqlite_off)

    for mode in ("light", "dark"):
        chart_ledger(mode)
        chart_denoise(mode)
        chart_phases(mode)
    _shrink()
    print(f"wrote 6 PNGs to {OUT}")


if __name__ == "__main__":
    main()
