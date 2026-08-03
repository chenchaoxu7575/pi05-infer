#!/usr/bin/env python3
"""Regenerate the three README charts (light + dark PNG for each).

    python docs/make_charts.py [--sqlite <on.sqlite> --sqlite-off <off.sqlite>]

Chart 1's ledger rows are hard-coded because paired A/B runs cannot be replayed
from a profile; each row carries the note it came from. Charts 2 and 3 come from
nsys sqlite exports and can be re-derived with ``--sqlite``, which prints what it
derived so it can be diffed against the constants below.

WARNING: chart 1's three panels use three different rulers and must not be chained.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

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

    matplotlib has no rounded bars, and ``FancyBboxPatch`` + ``mutation_aspect``
    degenerates at the aspect ratios these charts use. So the path is built
    explicitly, with the point->data conversion done per axis so the arcs are
    round on screen.

    WARNING: call only after the axes limits and subplot geometry are final -- the
    conversion reads ``ax.get_window_extent()``.
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
# Chart 1 - the optimization ledger (three stacked panels)
# --------------------------------------------------------------------------
# Rows are named verb + object, never by internal codename.
#
# --- panel 2: the paired ledger. e2e = predict_action_batch, plain CPU wall
# clock, 30 iters after 8 warmups, serialized. RTX PRO 5000, 300 W cap, bs=1,
# K=10 Euler steps, 968 prefix tokens, chunk 50, bf16.
#
# (label, e2e_after_ms, delta_ms, source note)
LEDGER = [
    ("baseline: torch.compile max-autotune", 52.60, None,
     "adarms_cache_impl/RESULTS_adarms_cache.md"),
    ("precompute the adaRMS modulation", 49.77, -2.83,
     "adarms_cache_impl/RESULTS_adarms_cache.md"),
    ("capture the denoise step as one CUDA graph", 47.73, -2.04,
     "adarms_cache_impl/RESULTS_adarms_cache.md UPDATE 1 (code name: Stage-1)"),
    ("merge Q/K/V into a single GEMM", 45.61, -2.12,
     "adarms_cache_impl/RESULTS_adarms_cache.md UPDATE 2"),
    ("keep the prefix KV in a static buffer", 45.10, -0.51,
     "adarms_cache_impl/RESULTS_adarms_cache.md UPDATE 3"),
    ("build the attention mask on the GPU", 44.79, -0.31,
     "attmask_fix/RESULTS_attmask.md"),
    ("fuse GeGLU and RoPE into the GEMM epilogue", 43.74, -1.05,
     "kernel_fusion/RESULTS_fusion.md (own paired A/B: 44.87 -> 43.74, -1.14)"),
    ("extract to a standalone package, graph on", 43.16, -0.58,
     "20260728_stage1/ab_stage1_summary.txt (paired 44.08 -> 43.16)"),
    ("delete the timestep conditioning nothing reads", 42.90, -0.30,
     "20260728_adarms_cond/ab_summary.txt (paired 43.20 -> 42.90, sd 0.07, n=4)"),
]

# --- panel 1: prehistory, on a DIFFERENT harness -- the full RLinf Ray +
# EnvWorker path under nsys, on another box, where e2e means "sum of the
# per-phase CPU wall clocks under nsys". Same compile config on both sides.
#
# WARNING: 58.9 -> 52.60 is a change of ruler, NOT an optimization. The two are ~6 ms
# apart because of measurement scope; never subtract them.
# WARNING: these rows are single-factor arms off the compile-only row, not a chain.
#
# (label, e2e_ms, delta_vs_compile_only, source)
PREHISTORY = [
    ("run eager - no compiler at all", 270.6, None,
     "the profile index (opt_eager/)"),
    ("compile the graph: torch.compile max-autotune", 65.8, None,
     "the profile index (opt_E0/) - the prehistory baseline"),
    ("skip the runtime type checks           (on its own)", 63.0, -2.8,
     "the profile index (opt_E1/)"),
    ("batch the 3 camera views into one ViT  (on its own)", 61.7, -4.1,
     "the profile index (opt_E2/)"),
    ("both of the above, on top of the compiler", 58.9, -6.9,
     "the profile index (opt_E3/) - -10.5% vs compile-only"),
]

# --- panel 3: the same optimizations, per denoise step.
# Ruler: union of kernel intervals over the denoise loop / 10 steps. At bs=1
# nothing overlaps, so this equals the sum of kernel durations. All rows and both
# peers sit on this one ruler; each row's own source is in its tuple below.
#
# WARNING: row 2 lumps four optimizations and is cross-session, not a paired A/B --
# marked in the chart. The captured graph is not faster in GPU-busy terms at all;
# it converts idle into busy, so its win shows up only in wall clock.
#
# (label, us_after, delta, kernels, paired?, source)
DEN_BASE = (2025.6, 347, "the denoise MFU analysis (pre-adaRMS-cache baseline)")
DEN_LEDGER = [
    ("precompute the adaRMS modulation", 1620.9, -404.7, 346, True,
     "the denoise MFU analysis (same table, same session)"),
    ("merge Q/K/V, static KV, mask on GPU (+ the graph)", 1368.0, -252.9, 305, False,
     "the profile index - cross-session lump, four changes"),
    ("fuse GeGLU and RoPE into the GEMM epilogue", 1236.0, -132.0, 238, True,
     "the kernel-fusion results (-132.0, -9.6%)"),
    ("delete the timestep conditioning nothing reads", 1185.0, -51.0, 217, True,
     "the measurement log (own paired baseline 1232.3 -> 1185.0, -47.3)"),
]

# --- the two reference implementations, on both panel 2 and panel 3 --------
# dexmal/realtime-vla @ b86a942: our config, our GPU, same day, real weights, our
# e2e scope. The only paired head-to-head we have is that run, and we lost it:
# theirs 43.41 vs ours-that-day 44.55.
PEER_MS = 43.41
PEER_US, PEER_KERN = 1191.0, 165
# limxdynamics/FluxVLA @ 7f9f774 - a DIFFERENT repo from dexmal/realtime-vla.
# 44.9 ms/predict at 968 prefix tokens. Their own default is a lighter 560-token
# config at 31.1 ms, which must never be compared against our 968-token numbers.
#
# WARNING: not config-matched, in both directions: chunk 10 not 50, and their timer
# skips the ~2-3 ms of CPU preprocessing ours includes (both cheaper for them);
# their loop recomputes adaRMS every step and syncs per call (costlier).
FLUX_MS = 44.9
FLUX_US, FLUX_KERN = 1419.0, 205


# Provenance stamp. Each chart is a snapshot of one commit and main has moved on;
# not redrawn, because grafting today's numbers onto a paired chain is the
# chained-rulers error these charts exist to warn about.
STAMPS = {
    "ledger": "chart state: pi05-infer @ 62aa78e\n"
              "rows measured 2026-07-11..07-28\n"
              "main has moved since -- see README for its e2e",
    "denoise": "chart state: pi05-infer @ 62aa78e\n"
               "profiled 2026-07-28, nsys 2026.1.2, 12 predicts\n"
               "main is now 190 kernels/step -- not re-derived",
    "phases": "chart state: pi05-infer @ 62aa78e\n"
              "profiled 2026-07-28, nsys 2026.1.2, 12 predicts\n"
              "the split moves as either side changes",
}


def _stamp(fig, t: dict, which: str) -> None:
    # Top-right: the bottom edge is taken and one long line collides with the title.
    fig.text(0.988, 0.992, STAMPS[which], ha="right", va="top", linespacing=1.5,
             fontsize=6.6, color=t["muted"], style="italic")


def _peer_key(ax, t, x, y, color, ls, title, sub, dash, pad, dy):
    """One entry of the in-axes peer key: a short dash + two lines of text.

    ``dash`` and ``pad`` are in x data units, ``dy`` in y data units, so the
    same helper serves both the ms-scaled and the us-scaled panel.  The text
    carries a surface-coloured bbox because the key sits under the waterfall,
    where the vertical peer rules and the x grid would otherwise cross it.
    """
    bb = dict(facecolor=t["surface"], edgecolor="none", pad=1.4)
    ax.plot([x, x + dash], [y, y], color=color, lw=1.6, ls=ls, zorder=5,
            solid_capstyle="butt")
    ax.text(x + pad, y, title, ha="left", va="center", fontsize=8.2, color=color,
            zorder=5, bbox=bb)
    ax.text(x + pad, y - dy, sub, ha="left", va="center", fontsize=7.2,
            color=t["muted"], zorder=5, bbox=bb)


def _waterfall(ax, t, base, rows, fmt, dfmt, gap, dgap, label_fs):
    """Draw a descending waterfall: each bar spans [after, previous].

    ``rows`` are ``(label, after, delta, extra)`` where ``extra`` is either
    ``None`` or a short caveat printed under the row's value label.  ``fmt``
    and ``dfmt`` render the value and the delta (the two panels carry a
    different number of decimals), ``gap``/``dgap`` are the x-offsets of the
    value and delta labels in data units.
    """
    n = len(rows)
    ys = list(range(n))[::-1]
    lbb = dict(facecolor=t["surface"], edgecolor="none", pad=1.0)
    prev = base
    for y, (label, after, delta, extra) in zip(ys, rows):
        _rounded_hbar(ax, y, after, prev, 0.52, t["s3"])
        ax.plot([prev, prev], [y - 0.40, y + 0.68], color=t["axis"], lw=1.0,
                ls=(0, (2, 2)), zorder=1)
        ax.text(after - dgap, y, dfmt(delta), ha="right", va="center",
                fontsize=label_fs + 1, fontweight="bold", color=t["ink"],
                bbox=lbb, zorder=4)
        last = y == ys[-1]
        ax.text(prev + gap, y, fmt(after), ha="left", va="center",
                fontsize=label_fs + 1.2 if last else label_fs,
                fontweight="bold" if last else "normal",
                color=t["ink"] if last else t["ink2"], bbox=lbb, zorder=4)
        if extra:
            ax.text(prev + gap, y - 0.30, extra, ha="left", va="center",
                    fontsize=7.2, color=t["s2"], zorder=4)
        prev = after
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=label_fs + 0.5)


def chart_ledger(mode: str) -> None:
    """The whole story in one figure: prehistory, e2e ledger, denoise ledger."""
    t = THEMES[mode]
    _style(t)
    fig = plt.figure(figsize=(10.5, 12.0), dpi=200)
    axp = fig.add_axes((0.360, 0.800, 0.625, 0.108))   # 1: prehistory
    ax = fig.add_axes((0.360, 0.398, 0.625, 0.300))    # 2: e2e ledger
    axd = fig.add_axes((0.360, 0.075, 0.625, 0.235))   # 3: denoise ledger

    fig.suptitle("pi0.5 action expert, bs=1: 52.60 -> 42.90 ms  (-9.70 ms, -18.4%)",
                 x=0.012, y=0.988, ha="left", fontsize=14, fontweight="bold",
                 color=t["ink"])
    # NOT "every step is bit-exact": bit-identity here is tiered by compile path.
    # Algebraic equivalence is what holds for every row unqualified.
    fig.text(0.012, 0.961,
             "every ledger step is an algebraically equivalent transform - no quantization, "
             "no change of sampler, no fewer denoise steps",
             ha="left", fontsize=9, color=t["ink2"])

    # ---------------- panel 1: prehistory, absolute bars -------------------
    npre = len(PREHISTORY)
    ypre = list(range(npre))[::-1]
    axp.set_xlim(0, 322)
    axp.set_ylim(-0.60, npre - 0.22)
    _chrome(axp, t)

    for y, (label, ms, delta, _src) in zip(ypre, PREHISTORY):
        _rounded_hbar(axp, y, 0, ms, 0.52, t["s1"])
        txt = f"{ms:.1f} ms" if delta is None else \
            f"{ms:.1f} ms   ({delta:+.1f} vs compile-only)"
        axp.text(ms + 5, y, txt, ha="left", va="center", fontsize=8.6,
                 color=t["ink"] if delta is None else t["ink2"])
    axp.set_yticks(ypre)
    axp.set_yticklabels([r[0] for r in PREHISTORY], fontsize=8.6)
    axp.tick_params(axis="x", labelsize=8)

    axp.axvline(LEDGER[0][1], color=t["s3"], lw=1.4, ls=(0, (3, 2.5)), zorder=2)
    axp.text(LEDGER[0][1] + 6, npre - 0.42,
             "52.60 ms - where panel 2 starts (same config as the last row, other ruler)",
             ha="left", va="center", fontsize=7.4, color=t["s3"])

    fig.text(0.012, 0.933,
             "1  BEFORE THIS REPO - full RLinf worker + nsys, 2026-07-03, a different box.  "
             "Absolute bars, not a waterfall.",
             ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=t["s1"])
    fig.text(0.012, 0.918,
             "the last three rows are two single-factor arms and their combination, "
             "all measured against the compile-only row - not a cumulative chain",
             ha="left", va="bottom", fontsize=7.6, color=t["muted"])

    # the ruler change, spelled out between panel 1 and panel 2
    fig.text(0.012, 0.756,
             "58.9 -> 52.60 ms is a RULER CHANGE, not an optimization: full Ray + EnvWorker "
             "path under nsys  ->  standalone bench, plain wall clock, another box.",
             ha="left", va="bottom", fontsize=8.2, fontweight="bold", color=t["s2"])
    fig.text(0.012, 0.740,
             "Both sides run the same torch.compile max-autotune config.  Panel 1 and "
             "panel 2 must not be chained into a single speedup.",
             ha="left", va="bottom", fontsize=7.6, color=t["muted"])

    # ---------------- panel 2: the paired e2e ledger -----------------------
    base = LEDGER[0][1]
    rows = [(lb, af, dl, None) for lb, af, dl, _s in LEDGER[1:]]
    ax.set_xlim(41.9, 53.7)
    ax.set_ylim(-2.45, len(rows) - 0.05)
    _chrome(ax, t)

    ax.axvline(base, color=t["s1"], lw=1.8, zorder=2)
    ax.text(base - 0.15, len(rows) - 0.10, f"baseline  {base:.2f} ms", ha="right",
            va="top", fontsize=10, fontweight="bold", color=t["s1"])
    vmin = (-0.75 - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.axvline(PEER_MS, ymin=vmin, color=t["s2"], lw=1.6, ls=(0, (4, 3)), zorder=2)
    ax.axvline(FLUX_MS, ymin=vmin, color=t["muted"], lw=1.4, ls=(0, (1.6, 2.2)),
               zorder=2)

    _waterfall(ax, t, base, rows, lambda v: f"{v:.2f} ms",
               lambda d: f"{d:+.2f}", 0.14, 0.14, 9.3)
    ax.set_xlabel("end-to-end predict_action_batch latency (ms), plain wall clock",
                  fontsize=9.5)
    _peer_key(ax, t, 42.0, -1.20, t["s2"], (0, (4, 3)),
              "dexmal/realtime-vla @b86a942: 43.41 ms",
              "config-matched, same GPU - but a separate session: not a paired A/B",
              0.26, 0.38, 0.36)
    _peer_key(ax, t, 42.0, -1.95, t["muted"], (0, (1.6, 2.2)),
              "limxdynamics/FluxVLA @7f9f774 - a different repo: 44.9 ms",
              "968 tok but chunk 10, and their timer skips CPU preproc - NOT config-matched",
              0.26, 0.38, 0.36)

    fig.text(0.012, 0.716,
             "2  THIS REPO, END TO END - standalone bench, plain wall clock, "
             "one paired A/B per row.  Waterfall: bar length = the ms that row bought.",
             ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=t["s3"])

    # ---------------- panel 3: the same rows, per denoise step -------------
    dbase, dbase_k, _ = DEN_BASE
    drows = [(lb, af, dl, None if paired else "cross-session, not paired")
             for lb, af, dl, _k, paired, _s in DEN_LEDGER]
    axd.set_xlim(1055, 2400)
    axd.set_ylim(-2.55, len(drows) - 0.02)
    _chrome(axd, t)

    axd.axvline(dbase, color=t["s1"], lw=1.8, zorder=2)
    axd.text(dbase - 16, len(drows) - 0.06,
             f"baseline  {dbase:.1f} us   {dbase_k} kern", ha="right", va="top",
             fontsize=9.5, fontweight="bold", color=t["s1"])
    dvmin = (-0.72 - axd.get_ylim()[0]) / (axd.get_ylim()[1] - axd.get_ylim()[0])
    axd.axvline(PEER_US, ymin=dvmin, color=t["s2"], lw=1.6, ls=(0, (4, 3)), zorder=2)
    axd.axvline(FLUX_US, ymin=dvmin, color=t["muted"], lw=1.4, ls=(0, (1.6, 2.2)),
                zorder=2)

    after_k = {af: k for _lb, af, _d, k, _p, _s in DEN_LEDGER}
    _waterfall(axd, t, dbase, drows,
               lambda v: f"{v:.1f} us   {after_k[v]} kern",
               lambda d: f"{d:+.1f}", 16, 16, 8.8)
    axd.set_xlabel("GPU busy per denoise step (us) - union of kernel intervals / 10 steps",
                   fontsize=9.5)
    _peer_key(axd, t, 1065, -1.05, t["s2"], (0, (4, 3)),
              f"dexmal/realtime-vla @b86a942: {PEER_US:.1f} us, {PEER_KERN} kern/step",
              "config-matched - but paired against our 1368.7 of that day, not against 1185.0",
              34, 50, 0.34)
    _peer_key(axd, t, 1065, -1.85, t["muted"], (0, (1.6, 2.2)),
              f"limxdynamics/FluxVLA @7f9f774 - a different repo: {FLUX_US:.1f} us, "
              f"{FLUX_KERN} kern/step",
              "chunk 10, not 50 - a 5x smaller action-token suffix.  NOT config-matched",
              34, 50, 0.34)

    fig.text(0.012, 0.330,
             "3  WHERE THOSE MS CAME FROM - the same work, counted per denoise step "
             "(10 of them per predict)",
             ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=t["s3"])
    fig.text(0.012, 0.315,
             "2025.6 -> 1185.0 us GPU busy (-41.5%), 347 -> 217 kernels.  A third ruler: it "
             "excludes the launch gaps panel 2 includes; row 2 lumps four of panel 2's rows",
             ha="left", va="bottom", fontsize=7.6, color=t["muted"])

    fig.text(0.012, 0.010,
             "RTX PRO 5000 (GB202/sm_120), 300 W cap - K=10 Euler steps, 968 prefix tokens, "
             "chunk 50, bf16 - panels 2 and 3 are zoomed waterfalls: bars encode deltas, not totals",
             ha="left", fontsize=7.5, color=t["muted"])
    _stamp(fig, t, "ledger")
    fig.savefig(OUT / f"ledger_{mode}.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# Chart 2 - denoise per-kernel breakdown
# --------------------------------------------------------------------------
# Source: 20260728_adarms_cond/prof_skip1.sqlite, stream 157, 120 steps.
#
# (label, us_per_step, kernels_per_step, group)
#   group 0 = inductor-generated Triton
#   group 1 = eager per-step glue (the remaining headroom)
#   group 2 = our hand-written fused kernels
DENOISE = [
    ("GeGLU: gate/up GEMM + gelu(g)*u  [fused]", 312.41, 18, 2),
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
        # "_swiglu_mm_kernel" is an older name for the same kernel;
        # archived profiles still carry it.
        if name in ("_geglu_mm_kernel", "_swiglu_mm_kernel"):
            return "GeGLU: gate/up GEMM + gelu(g)*u  [fused]"
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
             "nsys 2026.1.2, stream 157 (Stage-1 captured graph), "
             "12 predicts x 10 steps - 10 steps = 11.85 ms of the 42.90 ms predict",
             ha="left", fontsize=9, color=t["ink2"])
    fig.text(0.012, 0.013,
             "the 37 adaRMS dense(cond) projections that used to cost 395 us/step here "
             "(triton_per_fused_addmm_0, 300 instances) are gone - 0 instances",
             ha="left", fontsize=7.5, color=t["muted"])
    _stamp(fig, t, "denoise")
    fig.savefig(OUT / f"denoise_{mode}.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# Chart 3 - phase split of GPU busy time
# --------------------------------------------------------------------------
# Source: 20260728_stage1/stage1_off.sqlite, 12 predicts. Streams 7 = prefix LM,
# 157 = denoise expert, 158 = SigLIP. Read off the OFF arm because --stage1
# merges the streams; total GPU busy matches to within 0.06 ms.
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
    _stamp(fig, t, "phases")
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
