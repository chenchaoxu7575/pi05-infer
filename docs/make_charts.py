#!/usr/bin/env python3
"""Regenerate the four README charts (light + dark PNG for each).

    python docs/make_charts.py [--sqlite <stage1_on.sqlite> --sqlite-off <stage1_off.sqlite>]

Every number plotted here is a measurement, and every measurement has a source:

* the **e2e ledger** (chart 1) is a chain of paired plain-wall-clock A/Bs, one
  per optimization, each recorded in its own results note.  It is hard-coded
  below with the note it came from, because those runs cannot be replayed from
  a profile.  Its upper panel is the *prehistory* of that ledger -- older runs
  taken on a **different measurement harness**, kept visually separate on
  purpose (see PREHISTORY below).
* the **denoise ledger** (chart 2) is the same idea on us/denoise-step.  It too
  is split into two panels because the early numbers use a different ruler
  (NVTX wall clock per step) from the later ones (nsys stream-157 GPU busy).
* the **denoise kernel breakdown** (chart 3) and the **phase split** (chart 4)
  are derived from nsys 2026.1.2 sqlite exports.  Pass ``--sqlite`` /
  ``--sqlite-off`` to re-derive them instead of using the recorded values; the
  script prints what it derived so it can be diffed against the constants.

Source notes live in ``claude_mem/pi05_rollout_forward/`` (not shipped in this
repo); every hard-coded constant below carries its file:line.

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

# --- prehistory: the runs that came *before* the ledger above -----------------
# ALL of these were taken on a DIFFERENT measurement harness: the full RLinf
# Ray + EnvWorker training path with nsys attached, on a different box
# (10.172.160.142, 4x RTX PRO 5000), 2026-07-03, nsys 2025.3.1.  e2e there is
# "the sum of the per-phase CPU wall clocks under nsys"
# (claude_mem/pi05_rollout_forward/nsys_sm120/README.md:104), NOT the plain
# standalone wall clock the ledger above uses.
#
# The step from 58.9 to the 52.60 ledger baseline is therefore NOT an
# optimization.  It is explicitly documented as a change of ruler:
#   opt_validation/OPTIMIZATION_LOG.md:12  "纯推理基线(同上配置) | 53.1 ms |
#       换测法:去掉 Ray/worker 口径约 6ms | 纯推理"
#   opt_validation/OPTIMIZATION_LOG.md:16-17  "两种测法别相减 ... 两者差约 6ms
#       是测法口径,不是优化"  ("do not subtract the two rulers ... the ~6 ms
#       difference is measurement scope, not an optimization")
#   HANDOFF_min_infer_repro.md:13  "142 的 58.9 是 full-worker + nsys 口径"
# Both sides run the *same* compile config (torch.compile max-autotune, RLinf
# PR #968); PROFILES_INDEX.md:343 labels the E-series rep "#968 max-autotune".
#
# E1 and E2 are single-factor arms off E0, NOT a cumulative chain: E2 has
# typecheck still ON (nsys_sm120/README.md:96-102, column "typecheck关").
# Deltas below are therefore all "vs E0".
#
# (label, e2e_ms, delta_vs_E0, source)
PREHISTORY = [
    ("eager - no torch.compile", 270.6, None,
     "PROFILES_INDEX.md:348 (opt_eager/)"),
    ("naive torch.compile: max-autotune (RLinf #968)", 65.8, None,
     "PROFILES_INDEX.md:349 (opt_E0/) - the prehistory baseline"),
    ("E1: openpi typecheck off        (single factor)", 63.0, -2.8,
     "PROFILES_INDEX.md:350 (opt_E1/)"),
    ("E2: SigLIP 3-view batching      (single factor)", 61.7, -4.1,
     "PROFILES_INDEX.md:351 (opt_E2/)"),
    ("E3: both, on top of compile", 58.9, -6.9,
     "PROFILES_INDEX.md:352 (opt_E3/) - -10.5% vs E0"),
]
# dexmal/realtime-vla @ b86a942, our config, our GPU, same day, real (non-zero)
# weights, our e2e scope.  HEADTOHEAD_realtime_vla_pro5k.md:473 (scope B, n=30,
# sd 0.20).  The only *paired* head-to-head we have is that same line: theirs
# 43.41 vs ours-that-day 44.55, i.e. theirs faster by 1.14 ms.
PEER_MS = 43.41
# limxdynamics/FluxVLA @ 7f9f774 - a DIFFERENT repo from dexmal/realtime-vla.
# opt_validation/fluxvla_baseline.md:20 - 44.9 ms/predict at 968 prefix tokens
# (their own default is a lighter 560-token config that runs 31.1 ms; that one
# must never be compared against our 968-token numbers).
# NOT config-matched to us, in both directions:
#   - chunk 10, not 50 (HEADTOHEAD_realtime_vla_pro5k.md:456) -> cheaper for them
#   - their timer skips the CPU preprocessing ours includes, ~2-3 ms
#     (opt_validation/fluxvla_baseline.md:10-11)        -> cheaper for them
#   - their loop recomputes adaRMS + the time MLP every step and does a device
#     sync per call (HEADTOHEAD_realtime_vla_pro5k.md:451-455) -> costlier
# PROFILES_INDEX.md:437-441 retracts the old "their code at our config =
# 44.89 ms, dead heat" reading of this run: what was retracted is the
# *attribution* (wrong repo) and the "dead heat" verdict, not the measurement.
FLUX_MS = 44.9


def _peer_key(ax, t, x, y, color, ls, title, sub, dash=0.30, pad=0.42, dy=0.30):
    """One entry of the in-axes peer key: a short dash + two lines of text.

    ``dash`` and ``pad`` are in x data units, ``dy`` in y data units, so the
    same helper serves both the ms-scaled and the us-scaled chart.
    """
    bb = dict(facecolor=t["surface"], edgecolor="none", pad=1.4)
    ax.plot([x, x + dash], [y, y], color=color, lw=1.6, ls=ls, zorder=5,
            solid_capstyle="butt")
    ax.text(x + pad, y, title, ha="left", va="center", fontsize=8.2, color=color,
            zorder=5, bbox=bb)
    ax.text(x + pad, y - dy, sub, ha="left", va="center", fontsize=7.2,
            color=t["muted"], zorder=5, bbox=bb)


def chart_ledger(mode: str) -> None:
    t = THEMES[mode]
    _style(t)
    fig = plt.figure(figsize=(10.5, 8.6), dpi=200)
    # top = prehistory (5 rows), bottom = the paired ledger (8 rows)
    axp = fig.add_axes((0.360, 0.700, 0.625, 0.163))
    ax = fig.add_axes((0.360, 0.098, 0.625, 0.462))

    # ---------------- top panel: prehistory, absolute bars -----------------
    npre = len(PREHISTORY)
    ypre = list(range(npre))[::-1]
    axp.set_xlim(0, 322)
    axp.set_ylim(-0.60, npre - 0.22)
    _chrome(axp, t)

    for y, (label, ms, delta, _src) in zip(ypre, PREHISTORY):
        _rounded_hbar(axp, y, 0, ms, 0.52, t["s1"])
        txt = f"{ms:.1f} ms" if delta is None else f"{ms:.1f} ms   ({delta:+.1f} vs E0)"
        axp.text(ms + 5, y, txt, ha="left", va="center", fontsize=8.6,
                 color=t["ink"] if delta is None else t["ink2"])
    axp.set_yticks(ypre)
    axp.set_yticklabels([r[0] for r in PREHISTORY], fontsize=8.6)
    axp.tick_params(axis="x", labelsize=8)

    # where the ledger below starts, on this panel's (different) ruler
    axp.axvline(LEDGER[0][1], color=t["s3"], lw=1.4, ls=(0, (3, 2.5)), zorder=2)
    axp.text(LEDGER[0][1] + 6, npre - 0.42,
             "52.60 ms - where the ledger below starts (same config as E3, other ruler)",
             ha="left", va="center", fontsize=7.4, color=t["s3"])

    fig.text(0.012, 0.884,
             "PREHISTORY - full RLinf worker + nsys, 2026-07-03, a different box.  "
             "Absolute bars, not a waterfall.",
             ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=t["s1"])
    fig.text(0.012, 0.866,
             "E1 and E2 are single-factor arms off E0, not a cumulative chain.",
             ha="left", va="bottom", fontsize=7.6, color=t["muted"])

    # ---------------- the ruler change, spelled out between the panels -----
    fig.text(0.012, 0.632,
             "58.9 -> 52.60 ms is a RULER CHANGE, not an optimization: full Ray + EnvWorker "
             "path under nsys  ->  standalone bench, plain wall clock, another box.",
             ha="left", va="bottom", fontsize=8.2, fontweight="bold", color=t["s2"])
    fig.text(0.012, 0.612,
             "Both sides run the same torch.compile max-autotune config.  The two panels "
             "must not be chained into a single speedup.",
             ha="left", va="bottom", fontsize=7.6, color=t["muted"])

    # ---------------- bottom panel: the paired ledger waterfall ------------
    base = LEDGER[0][1]
    rows = LEDGER[1:]
    n = len(rows)
    ys = list(range(n))[::-1]

    ax.set_xlim(41.9, 53.7)
    ax.set_ylim(-2.45, n - 0.05)
    _chrome(ax, t)

    ax.axvline(base, color=t["s1"], lw=1.8, zorder=2)
    ax.text(base - 0.15, n - 0.10, f"baseline  {base:.2f} ms", ha="right",
            va="top", fontsize=10, fontweight="bold", color=t["s1"])

    # the vertical peer lines stop above the key block so they do not run
    # through its text (ymin is an axes fraction, y = -0.75 in data units)
    vmin = (-0.75 - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.axvline(PEER_MS, ymin=vmin, color=t["s2"], lw=1.6, ls=(0, (4, 3)), zorder=2)
    ax.axvline(FLUX_MS, ymin=vmin, color=t["muted"], lw=1.4, ls=(0, (1.6, 2.2)),
               zorder=2)
    # peer key parked below the waterfall
    _peer_key(ax, t, 42.0, -1.20, t["s2"], (0, (4, 3)),
              "dexmal/realtime-vla @b86a942: 43.41 ms",
              "config-matched, same GPU - but a separate session: not a paired A/B",
              dash=0.26, pad=0.38, dy=0.36)
    _peer_key(ax, t, 42.0, -1.95, t["muted"], (0, (1.6, 2.2)),
              "limxdynamics/FluxVLA @7f9f774 - a different repo: 44.9 ms",
              "968 tok but chunk 10, and their timer skips CPU preproc - NOT config-matched",
              dash=0.26, pad=0.38, dy=0.36)

    prev = base
    for y, (label, after, delta, _src) in zip(ys, rows):
        _rounded_hbar(ax, y, after, prev, 0.54, t["s3"])
        ax.plot([prev, prev], [y - 0.42, y + 0.70], color=t["axis"], lw=1.0,
                ls=(0, (2, 2)), zorder=1)
        # bbox = surface: the value labels overrun the peer rules, and the
        # label must read on top of them rather than be crossed out by them
        lbb = dict(facecolor=t["surface"], edgecolor="none", pad=1.0)
        ax.text(after - 0.14, y, f"{delta:+.2f}", ha="right", va="center",
                fontsize=10.5, fontweight="bold", color=t["ink"], bbox=lbb,
                zorder=4)
        last = y == ys[-1]
        ax.text(prev + 0.14, y, f"{after:.2f} ms", ha="left", va="center",
                fontsize=11 if last else 9,
                fontweight="bold" if last else "normal",
                color=t["ink"] if last else t["ink2"], bbox=lbb, zorder=4)
        prev = after

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlabel("end-to-end predict_action_batch latency (ms), plain wall clock",
                  fontsize=9.5)
    fig.text(0.012, 0.578,
             "PAIRED LEDGER - standalone bench, plain wall clock, one A/B per row.  "
             "Waterfall: bar length = the ms that row bought.",
             ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=t["s3"])

    fig.suptitle("pi0.5 action expert, bs=1: 52.60 -> 42.90 ms  (-9.70 ms, -18.4%)",
                 x=0.012, y=0.978, ha="left", fontsize=14, fontweight="bold",
                 color=t["ink"])
    fig.text(0.012, 0.936,
             "every ledger step bit-exact against the unoptimised path "
             "(max|delta| = 0.00e+00) - no quantization, no fewer denoise steps",
             ha="left", fontsize=9, color=t["ink2"])
    fig.text(0.012, 0.012,
             "RTX PRO 5000 (GB202/sm_120), 300 W cap - K=10 Euler steps, 968 prefix tokens, "
             "chunk 50, bf16 - lower panel x-axis starts at 41.9 ms (waterfall: bars encode deltas)",
             ha="left", fontsize=7.5, color=t["muted"])
    fig.savefig(OUT / f"ledger_{mode}.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# Chart 2 - the denoise-step ledger (us per denoise step)
# --------------------------------------------------------------------------
# Same idea as chart 1, on a different y-axis: what one *denoise step* costs.
#
# MAIN PANEL ruler: GPU busy per denoise step = union of kernel intervals over
# the denoise loop, / 10 steps.  At bs=1 nothing overlaps, so this equals the
# naive sum of kernel durations (HEADTOHEAD_realtime_vla_pro5k.md:274), which
# is how the later rows were measured (nsys 2026.1.2, stream 157).  All rows
# and BOTH peers sit on this one ruler.
#
#   2025.6 / 347 k  MFU_denoise_analysis.md:93  "OURS baseline (pre-cache)",
#                   denoise-loop busy 20.256 ms/predict, kernels/step 347
#   1620.9 / 346 k  MFU_denoise_analysis.md:94  "OURS + adaRMS cache",
#                   16.209 ms/predict - same table, same session => PAIRED
#   1368.0 / 305 k  PROFILES_INDEX.md:142 = kernel_fusion/RESULTS_fusion.md:20
#                   (base.nsys-rep, both fusions off), 20 predicts
#   1236.0 / 238 k  PROFILES_INDEX.md:146 = kernel_fusion/RESULTS_fusion.md:23
#                   (f_guard.nsys-rep, shipped).  RESULTS_fusion.md:24 states
#                   the delta as -132.0 (-9.6 %) => PAIRED with 1368.0
#   1185.0 / 217 k  pi05-infer commit 62aa78e; docs/MEASUREMENTS.md:193,205.
#                   Its own paired baseline is 1232.3 (prof_skip0, same
#                   session), i.e. -47.3; on this chain the drop reads -51.0.
#
# Caveats deliberately encoded in the chart text:
#   - row 2 lumps four optimizations and is CROSS-SESSION (07-25 -> 07-27), not
#     a paired A/B.  Note also MFU_denoise_analysis.md:104: the Stage-1 graph is
#     *not* faster in GPU-busy terms - it converts idle into busy (idle/predict
#     3.74 -> 0.93 ms).  Its win shows up in wall clock, not here.
#   - the 07-28 per-kernel census of the 238-kernel build reads 1232.6
#     (HANDOFF_20260728.md:58) and its step_idle union reads 1233.3 - same
#     build as 1236.0, different session.  Not charted as a step.
#
# (label, us_after, delta, kernels, paired?, source)
DEN_BASE = (2025.6, 347, "MFU_denoise_analysis.md:93 (pre-adaRMS-cache baseline)")
DEN_LEDGER = [
    ("+ adaRMS modulation table", 1620.9, -404.7, 346, True,
     "MFU_denoise_analysis.md:94 (same table, same session)"),
    ("+ fused QKV, static KV, on-device masks (+ Stage-1 graph)", 1368.0, -252.9, 305, False,
     "PROFILES_INDEX.md:142 - cross-session lump, four changes"),
    ("+ SwiGLU & QKV+RoPE epilogue fusion", 1236.0, -132.0, 238, True,
     "kernel_fusion/RESULTS_fusion.md:24 (-132.0, -9.6%)"),
    ("+ drop the dead timestep conditioning", 1185.0, -51.0, 217, True,
     "MEASUREMENTS.md:193 (own paired baseline 1232.3 -> 1185.0, -47.3)"),
]

# Earlier rungs on a DIFFERENT ruler: wall clock per step, i.e. GPU busy *plus*
# per-step idle and CPU.  Kept in their own panel; they must not be chained
# onto the panel below.
#   2255  PROFILES_INDEX.md:324 - denoise/loop NVTX span 22.554 ms / 10, nsys
#         2025.3.1, 2026-07-09 pro5k baseline.  (Busy in that same profile is
#         21.887 ms -> 2188.7 us/step; RESULTS_bubble_timeline.md:108.)
#   2286 -> 1881  adarms_cache_impl/RESULTS_adarms_cache.md:14 - denoise/loop
#         "sync-timed" (torch.cuda.synchronize per phase, no profiler),
#         22.86 -> 18.81 ms/predict, the same A/B that moved e2e 52.60 -> 49.77.
# NOT charted: the oldest figure, "2115 us/step @ 417 kernels"
# (HANDOFF_latency_investigation.md:39, opt_validation/denoise_kernel_trace.md:10).
# Its own doc says the source sqlite is not in the tree (PROFILES_INDEX.md:417),
# and 2.115 ms is exactly the `denoise/expert_forward` span in the sibling
# profile (RESULTS_bubble_timeline.md:113) while `denoise/loop` there is 2.255 -
# so the 2115 figure is probably expert-only, a narrower scope than every other
# number here.  Left out rather than charted on a guess.
DEN_WALL_MARK = (2255.0, "2026-07-09 baseline, NVTX denoise/loop span / 10")
DEN_WALL_PAIR = (2286.0, 1881.0, "adaRMS modulation table, paired sync-timed wall clock")

# Peers, on the main panel's ruler (union of kernel intervals per step).
#   1191.0 / 165  HEADTOHEAD_realtime_vla_pro5k.md:271 (sd 13.2), dexmal/
#                 realtime-vla @b86a942, config-matched, same GPU, same day.
#                 Its paired counterpart is our 1368.7 / 306 from that session
#                 (= the 1368.0 / 305 row here, a different capture of the same
#                 build).  1185.0 vs 1191.0 is CROSS-SESSION - not a reversal.
#   1419.0 / 205  MFU_denoise_analysis.md:96,98 - limxdynamics/FluxVLA @7f9f774,
#                 same table and same ruler as our 2025.6 / 1620.9 rows.
#                 (fluxvla_per_module.md:39 gives 1404 us/step from marker-to-
#                 marker spans; MFU:316 reconciles the two at ~1408.)
#                 Chunk 10, not 50 - a 5x smaller action-token suffix than ours.
DEN_PEER_RTVLA = (1191.0, 165)
DEN_PEER_FLUX = (1419.0, 205)


def chart_denoise_ledger(mode: str) -> None:
    t = THEMES[mode]
    _style(t)
    fig = plt.figure(figsize=(10.5, 7.0), dpi=200)
    axw = fig.add_axes((0.375, 0.735, 0.610, 0.102))   # wall-clock panel
    ax = fig.add_axes((0.375, 0.128, 0.610, 0.442))    # GPU-busy ledger

    XL, XR = 1055.0, 2400.0

    # ---------------- top panel: the older wall-clock ruler ----------------
    axw.set_xlim(XL, XR)
    axw.set_ylim(-0.55, 1.55)
    _chrome(axw, t)
    axw.tick_params(axis="x", labelsize=8)

    mark, mark_lbl = DEN_WALL_MARK
    axw.plot([mark], [1.0], marker="D", markersize=6, color=t["s1"], zorder=3)
    axw.text(mark - 24, 1.0, f"{mark:.0f} us", ha="right", va="center",
             fontsize=8.6, color=t["ink2"])

    wb, wa, wlbl = DEN_WALL_PAIR
    _rounded_hbar(axw, 0.0, wa, wb, 0.42, t["s1"])
    axw.text(wa - 24, 0.0, f"{wa - wb:+.0f}", ha="right", va="center",
             fontsize=9, fontweight="bold", color=t["ink"])
    axw.text(wb + 20, 0.0, f"{wa:.0f} us", ha="left", va="center", fontsize=8.6,
             color=t["ink2"])
    axw.set_yticks([1.0, 0.0])
    axw.set_yticklabels([mark_lbl, wlbl], fontsize=8.4)

    fig.text(0.012, 0.858,
             "EARLIER RUNGS, WALL-CLOCK RULER - GPU busy plus per-step idle and CPU.  "
             "Cross-session, two different tools.",
             ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=t["s1"])
    fig.text(0.012, 0.840,
             "The 2255 diamond's own profile reads 2189 us/step of GPU busy.",
             ha="left", va="bottom", fontsize=7.6, color=t["muted"])

    fig.text(0.012, 0.652,
             "These two panels use two different rulers and must not be chained.  "
             "Wall clock includes the launch gaps that the GPU-busy ruler below excludes.",
             ha="left", va="bottom", fontsize=8.2, fontweight="bold", color=t["s2"])
    fig.text(0.012, 0.632,
             "On today's build those two rulers read 1242.7 us (wall) vs 1185.0 us (busy) "
             "- 57.7 us/step of GPU idle left.",
             ha="left", va="bottom", fontsize=7.6, color=t["muted"])

    # ---------------- bottom panel: GPU-busy ledger ------------------------
    base, base_k, _ = DEN_BASE
    n = len(DEN_LEDGER)
    ys = list(range(n))[::-1]
    ax.set_xlim(XL, XR)
    ax.set_ylim(-2.55, n - 0.02)
    _chrome(ax, t)

    ax.axvline(base, color=t["s1"], lw=1.8, zorder=2)
    ax.text(base - 16, n - 0.06, f"baseline  {base:.1f} us   {base_k} kern",
            ha="right", va="top", fontsize=9.5, fontweight="bold", color=t["s1"])

    prv, prk = DEN_PEER_RTVLA
    pfv, pfk = DEN_PEER_FLUX
    vmin = (-0.72 - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.axvline(prv, ymin=vmin, color=t["s2"], lw=1.6, ls=(0, (4, 3)), zorder=2)
    ax.axvline(pfv, ymin=vmin, color=t["muted"], lw=1.4, ls=(0, (1.6, 2.2)),
               zorder=2)

    prev = base
    for y, (label, after, delta, kern, paired, _src) in zip(ys, DEN_LEDGER):
        _rounded_hbar(ax, y, after, prev, 0.50, t["s3"])
        ax.plot([prev, prev], [y - 0.40, y + 0.68], color=t["axis"], lw=1.0,
                ls=(0, (2, 2)), zorder=1)
        lbb = dict(facecolor=t["surface"], edgecolor="none", pad=1.0)
        ax.text(after - 16, y, f"{delta:+.1f}", ha="right", va="center",
                fontsize=10, fontweight="bold", color=t["ink"], bbox=lbb,
                zorder=4)
        last = y == ys[-1]
        ax.text(prev + 16, y, f"{after:.1f} us   {kern} kern", ha="left",
                va="center", fontsize=10 if last else 8.8,
                fontweight="bold" if last else "normal",
                color=t["ink"] if last else t["ink2"], bbox=lbb, zorder=4)
        if not paired:
            ax.text(prev + 16, y - 0.30, "cross-session, not paired",
                    ha="left", va="center", fontsize=7.2, color=t["s2"])
        prev = after

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in DEN_LEDGER], fontsize=9)
    ax.set_xlabel("GPU busy per denoise step (us) - union of kernel intervals / 10 steps",
                  fontsize=9.5)
    fig.text(0.012, 0.588,
             "GPU-BUSY LEDGER - one ruler for every row and both peers.  "
             "Waterfall: bar length = the us/step that row bought.",
             ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=t["s3"])

    _peer_key(ax, t, XL + 10, -1.05, t["s2"], (0, (4, 3)),
              f"dexmal/realtime-vla @b86a942: {prv:.1f} us, {prk} kern/step",
              "config-matched - but paired against our 1368.7 of that day, not against 1185.0",
              dash=34, pad=50, dy=0.34)
    _peer_key(ax, t, XL + 10, -1.85, t["muted"], (0, (1.6, 2.2)),
              f"limxdynamics/FluxVLA @7f9f774 - a different repo: {pfv:.1f} us, {pfk} kern/step",
              "chunk 10, not 50 - a 5x smaller action-token suffix.  NOT config-matched",
              dash=34, pad=50, dy=0.34)

    fig.suptitle("One denoise step: 2025.6 -> 1185.0 us GPU busy  (-41.5%), 347 -> 217 kernels",
                 x=0.012, y=0.972, ha="left", fontsize=14, fontweight="bold",
                 color=t["ink"])
    fig.text(0.012, 0.922,
             "K=10 of these per predict - 20.26 ms -> 11.85 ms of GPU busy.  "
             "Every row bit-exact (max|delta| = 0.00e+00)",
             ha="left", fontsize=9, color=t["ink2"])
    fig.text(0.012, 0.012,
             "RTX PRO 5000 (GB202/sm_120) - nsys, stream 157 - the 238-kernel build re-measured "
             "next session reads 1232.3-1233.3 us (same build, not a step)",
             ha="left", fontsize=7.5, color=t["muted"])
    fig.savefig(OUT / f"denoise_ledger_{mode}.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# Chart 3 - denoise per-kernel breakdown, current build
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
    1: "eager per-step glue - was 99.7 us / 70 k",
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
        chart_denoise_ledger(mode)
        chart_denoise(mode)
        chart_phases(mode)
    _shrink()
    print(f"wrote 8 PNGs to {OUT}")


if __name__ == "__main__":
    main()
