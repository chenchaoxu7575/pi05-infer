#!/usr/bin/env python3
"""Regenerate the three README charts (light + dark PNG for each).

    python docs/make_charts.py [--sqlite <denoise.sqlite>]

    ledger    the same three blocks of work, on end-to-end wall clock
    denoise   the same three blocks, on GPU time per denoise step
    phases    where the shipping build's model-inference GPU time goes

--sqlite prints the per-step total and top kernels of an nsys export, to check
a D_* constant below against its source.
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
        (xb - rx * k, ya),
        (xb, ya + ry * k),
        (xb, ya + ry),  # bottom-right arc
        (xb, yb - ry),  # right edge
        (xb, yb - ry * k),
        (xb - rx * k, yb),
        (xb - rx, yb),  # top-right arc
        (xa + rx, yb),  # top edge
        (xa + rx * k, yb),
        (xa, yb - ry * k),
        (xa, yb - ry),  # top-left arc
        (xa, ya + ry),  # left edge
        (xa, ya + ry * k),
        (xa + rx * k, ya),
        (xa + rx, ya),  # bottom-left arc
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
# Chart 1 - eager to current, one ruler
# --------------------------------------------------------------------------
# Five arms, one session, six rounds of rotating order, each delta paired within
# a round. Span is sample_actions (--model-only).
#
#   EAGER   --no-compile
#   BASE    torch.compile max-autotune, all twelve optimizations off
#   C1      + CPU overhead
#   C2      + denoise-step work removed
#   C3      + kernel fusion & optimization   (= shipping defaults)
#
# UNLOCKED, stock configuration -- the card as shipped, 300 W cap, no -lgc. That
# is what a reader has. It costs some resolution: base..c3 all draw 293-301 W
# against the cap, so the SM clock falls along the chain and each arm is compared
# against a predecessor running faster. ENV is printed per arm so that is visible
# rather than hidden. The locked chain, where every arm runs at one clock, is in
# docs/locked_clock.md and is never chained onto anything here.
#
# None -> the chart renders that value as a placeholder.
EAGER = 124.71
BASE = 51.57
C1 = 48.16
C2 = 42.90
C3 = 39.64

# SM clock and board power, median over each arm's timed window
ENV = {
    "eager": (2325, 172),
    "base": (2437, 295),
    "c1": (2362, 301),
    "c2": (2317, 301),
    "c3": (2220, 301),
}
POWER_CAP_W = 300

CATS = [
    (
        "CPU overhead",
        [
            "capture one denoise step as a graph, replay it for every step",
            "hoist the step-invariant mask / position ids / rotary table",
            "prefix KV into a static buffer",
            "attention mask built on device",
        ],
    ),
    (
        "denoise-step work removed",
        [
            "precompute the adaRMS modulation table, 37 projections -> 1 gather",
            "drop the timestep conditioning nothing reads",
            "drop the prefix LM's last-layer tail",
        ],
    ),
    (
        "kernel fusion & optimization",
        [
            "GeGLU and RoPE fused into the GEMM epilogue",
            "Q/K/V into a single GEMM, expert and prefix",
            "retuned down_proj / o_proj tiles, pinned the Q*K^T tile",
        ],
    ),
]


_ARMS = ("base", "c1", "c2", "c3")


def _ms(v):
    return "--.--" if v is None else f"{v:.2f}"


def _env_footer():
    """SM clock and power per arm, in chain order -- the cap decay, made visible."""
    decay = "  ->  ".join(f"{a} {ENV[a][0]} MHz {ENV[a][1]} W" for a in _ARMS)
    e_mhz, e_w = ENV["eager"]
    return f"{decay}          eager {e_mhz} MHz {e_w} W, never near the cap"


def chart_ledger(mode: str) -> None:
    """Descending waterfall: each category picks up where the previous ended."""
    t = THEMES[mode]
    _style(t)
    colors = [t["s1"], t["s2"], t["s3"]]

    stages = [BASE, C1, C2, C3]
    have = all(v is not None for v in stages)
    # placeholder geometry so the layout is reviewable before the data lands
    geo = stages if have else [50.0, 47.0, 45.2, 43.0]

    fig = plt.figure(figsize=(10.5, 6.4), dpi=200)
    rect = (0.345, 0.150, 0.620, 0.560)
    ax = fig.add_axes(rect)
    lo = geo[3] - (geo[0] - geo[3]) * 0.10
    hi = geo[0] + (geo[0] - geo[3]) * 0.06
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, 2.6)
    _chrome(ax, t)
    ax.set_yticks([])
    ax.set_xlabel("model inference per predict (ms)", fontsize=9.5)

    for i, ((label, items), c) in enumerate(zip(CATS, colors)):
        y = 2 - i
        left, right = geo[i + 1], geo[i]
        _rounded_hbar(ax, y, left, right, 0.33, c, radius_pt=3.0)
        ax.text(
            left - (hi - lo) * 0.012,
            y,
            f"-{_ms(right - left) if have else '--.--'}",
            ha="right",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=t["ink"],
        )
        ax.text(
            right + (hi - lo) * 0.012,
            y,
            _ms(geo[i]) if have else "--.--",
            ha="left",
            va="center",
            fontsize=9.5,
            color=t["muted"],
        )
        if i == 2:
            ax.text(
                left,
                y - 0.40,
                f"{_ms(C3)} ms",
                ha="left",
                va="top",
                fontsize=11,
                fontweight="bold",
                color=t["ink"],
            )

    ax.axvline(geo[0], color=t["axis"], lw=1.0, ls=(0, (4, 3)), zorder=1)

    fig.text(
        0.035,
        0.038,
        _env_footer(),
        ha="left",
        va="bottom",
        fontsize=8,
        color=t["muted"],
    )

    fig.suptitle(
        f"pi0.5 action expert, bs=1:   {_ms(BASE)} -> {_ms(C3)} ms   "
        f"({'-' + _ms(BASE - C3) if have else '--.--'} ms,"
        f" {f'-{100 * (BASE - C3) / BASE:.1f}%' if have else '--.-%'})",
        x=0.035,
        y=0.960,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
        color=t["ink"],
    )
    fig.text(
        0.035,
        0.905,
        f"stock configuration -- unlocked clock, {POWER_CAP_W} W cap",
        ha="left",
        fontsize=9,
        color=t["muted"],
    )
    fig.text(
        0.035,
        0.855,
        "The baseline is already compiled. torch.compile gets there first:",
        ha="left",
        fontsize=9.5,
        color=t["ink2"],
    )
    gain = EAGER - BASE if (EAGER is not None and BASE is not None) else None
    fig.text(
        0.035,
        0.800,
        f"eager {_ms(EAGER)} ms  ->  torch.compile max-autotune {_ms(BASE)} ms"
        f"   ({'-' + _ms(gain) if gain is not None else '--.--'} ms,"
        f" {f'-{100 * gain / EAGER:.1f}%' if gain is not None else '--.-%'})",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=t["ink"],
    )

    # each text block is centred on its own bar, in figure coords
    def _bar_fig_y(d):
        return rect[1] + rect[3] * (d + 0.6) / 3.2

    for i, ((label, items), c) in enumerate(zip(CATS, colors)):
        fy = _bar_fig_y(2 - i) + 0.016 + (len(items) - 1) * 0.0155
        fig.patches.append(
            plt.Rectangle(
                (0.035, fy - 0.004),
                0.011,
                0.024,
                transform=fig.transFigure,
                facecolor=c,
                edgecolor="none",
                zorder=5,
            )
        )
        fig.text(
            0.055,
            fy,
            label,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=t["ink"],
        )
        y = fy - 0.040
        for it in items:
            fig.text(0.055, y, it, ha="left", va="bottom", fontsize=8.5, color=t["ink2"])
            y -= 0.031

    if not have:
        fig.text(
            0.965,
            0.025,
            "PLACEHOLDER -- measurement in flight",
            ha="right",
            va="bottom",
            fontsize=9,
            color=t["s2"],
            fontweight="bold",
        )
    fig.savefig(OUT / f"ledger_{mode}.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# Chart 2 - the same three categories, measured on one denoise step
# --------------------------------------------------------------------------
# Same arms as chart 1, profiled under nsys with the clock locked, so a
# per-kernel measurement is not reading clock drift. GPU time on the denoise
# stream, per step; a predict runs ten steps.
#
# None -> rendered as a placeholder.
D_BASE = 2477.56
D_C1 = 2252.04
D_C2 = 1792.18
D_C3 = 1237.83


def chart_denoise(mode: str) -> None:
    """Descending waterfall of GPU time per denoise step."""
    t = THEMES[mode]
    _style(t)
    colors = [t["s1"], t["s2"], t["s3"]]

    stages = [D_BASE, D_C1, D_C2, D_C3]
    have = all(v is not None for v in stages)
    geo = stages if have else [2000.0, 1750.0, 1500.0, 1240.0]

    fig = plt.figure(figsize=(10.5, 4.6), dpi=200)
    rect = (0.345, 0.200, 0.620, 0.560)
    ax = fig.add_axes(rect)
    lo = geo[3] - (geo[0] - geo[3]) * 0.10
    hi = geo[0] + (geo[0] - geo[3]) * 0.06
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, 2.6)
    _chrome(ax, t)
    ax.set_yticks([])
    ax.set_xlabel("GPU time per denoise step (us)", fontsize=9.5)

    for i, ((label, _items), c) in enumerate(zip(CATS, colors)):
        y = 2 - i
        left, right = geo[i + 1], geo[i]
        _rounded_hbar(ax, y, left, right, 0.46, c, radius_pt=3.0)
        ax.text(
            left - (hi - lo) * 0.012,
            y,
            f"-{right - left:.0f}" if have else "--",
            ha="right",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=t["ink"],
        )
        fy = rect[1] + rect[3] * (y + 0.6) / 3.2
        fig.text(0.330, fy, label, ha="right", va="center", fontsize=10.5, color=t["ink"])

    ax.axvline(geo[0], color=t["axis"], lw=1.0, ls=(0, (4, 3)), zorder=1)

    fig.suptitle(
        f"One denoise step:   {_ms(D_BASE)} -> {_ms(D_C3)} us"
        f"   ({'-' + f'{D_BASE - D_C3:.0f}' if have else '--'} us,"
        f" {f'-{100 * (D_BASE - D_C3) / D_BASE:.1f}%' if have else '--.-%'})",
        x=0.035,
        y=0.945,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=t["ink"],
    )
    fig.text(
        0.035,
        0.845,
        "nsys, CUDA stream 157, clock locked 1897 MHz",
        ha="left",
        fontsize=9,
        color=t["ink2"],
    )

    if not have:
        fig.text(
            0.965,
            0.035,
            "PLACEHOLDER -- measurement in flight",
            ha="right",
            va="bottom",
            fontsize=9,
            color=t["s2"],
            fontweight="bold",
        )
    fig.savefig(OUT / f"denoise_{mode}.png")
    plt.close(fig)


def derive_denoise(sqlite_path: str, n_steps: float = 120.0) -> None:
    """Print per-step GPU time on the denoise stream, to fill the constants above."""
    con = sqlite3.connect(sqlite_path)
    names = dict(con.execute("select id, value from StringIds"))
    rows = con.execute(
        "select shortName, sum(end - start) from CUPTI_ACTIVITY_KIND_KERNEL "
        "where streamId = 157 group by shortName"
    ).fetchall()
    total = sum(v for _, v in rows) / 1e3 / n_steps
    print(f"{sqlite_path}: {total:.2f} us/step over {len(rows)} distinct kernels")
    for sid, v in sorted(rows, key=lambda r: -r[1])[:8]:
        print(f"  {names.get(sid, sid)[:52]:<52} {v / 1e3 / n_steps:8.2f}")


# --------------------------------------------------------------------------
# Chart 3 - where the shipping build spends GPU time
# --------------------------------------------------------------------------
# Model inference only: GPU kernels attributed to their innermost enclosing NVTX
# range, so preprocessing and the output transform are outside it. Commit
# eccaeb6, shipping defaults, clock locked 1897 MHz, 12 predicts.
# Re-derive with: tools/prefix_census.py <sqlite> 12
PHASES = [
    ("prefix: SigLIP vision, 3 views", 5933.7),
    ("prefix: PaliGemma LM over 968 tokens", 24443.2),
    ("denoise: 10 x action expert", 12391.5),
    ("everything else", 116.5),
]

# Each phase is normalized against the roofline that actually binds it, so the
# three percentages are comparable. Peak is 206.2 TFLOP/s bf16 and 1222 GB/s at
# the 1897 MHz these phases were timed at; the knee is 169 FLOP/byte. The two
# prefix intensities are upper bounds (weight bytes, no ncu DRAM counter), hence
# ">"; the achieved FLOP/s behind their verdicts is shape-exact and needs no
# byte count. The per-phase roofline derivation lives in the internal record.
ROOFLINE = {
    "prefix: SigLIP vision, 3 views": (
        "compute-bound",
        ">800 FLOP/byte",
        "54-57% of peak FLOP/s",
    ),
    "prefix: PaliGemma LM over 968 tokens": (
        "compute-bound",
        ">1000 FLOP/byte",
        "75-79% of peak FLOP/s",
    ),
    "denoise: 10 x action expert": (
        "memory-bound",
        "56 FLOP/byte",
        "46% of peak DRAM BW",
    ),
}


def chart_phases(mode: str) -> None:
    """One bar: GPU busy per predict, split by phase."""
    t = THEMES[mode]
    _style(t)
    colors = [t["s2"], t["s1"], t["s3"], t["muted"]]
    total = sum(v for _, v in PHASES)

    fig = plt.figure(figsize=(10.5, 3.55), dpi=200)
    ax = fig.add_axes((0.035, 0.575, 0.930, 0.140))
    ax.set_xlim(0, total)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")

    x = 0.0
    for (label, us), c in zip(PHASES, colors):
        _rounded_hbar(ax, 0, x, x + us, 0.95, c, radius_pt=3.0)
        if us / total > 0.05:
            ax.text(
                x + us / 2,
                0,
                f"{us / 1e3:.2f} ms",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="#ffffff",
            )
        x += us

    fig.suptitle(
        f"Model inference, {total / 1e3:.2f} ms of GPU time per predict",
        x=0.035,
        y=0.945,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=t["ink"],
    )
    fig.text(
        0.035,
        0.840,
        "preprocessing and the output transform are outside "
        "this; the prefix runs once, the denoise loop ten times.",
        ha="left",
        fontsize=9,
        color=t["ink2"],
    )

    x = 0.0
    for (label, us), c in zip(PHASES, colors):
        if us / total > 0.02:
            fx = 0.035 + 0.930 * (x + us / 2) / total
            ax.figure.text(
                fx, 0.500, label, ha="center", va="top", fontsize=9, color=t["ink"]
            )
            ax.figure.text(
                fx,
                0.410,
                f"{100 * us / total:.1f}%",
                ha="center",
                va="top",
                fontsize=9,
                color=t["muted"],
            )
            bound = ROOFLINE.get(label)
            if bound is not None:
                ax.figure.text(
                    fx,
                    0.285,
                    bound[0],
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    fontweight="bold",
                    color=c,
                )
                for k, line in enumerate(bound[1:]):
                    ax.figure.text(
                        fx,
                        0.195 - k * 0.080,
                        line,
                        ha="center",
                        va="top",
                        fontsize=8.5,
                        color=t["ink2"],
                    )
        x += us
    fig.savefig(OUT / f"phases_{mode}.png")
    plt.close(fig)


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sqlite", default=None)
    args = ap.parse_args()
    if args.sqlite:
        derive_denoise(args.sqlite)
    for mode in ("light", "dark"):
        chart_ledger(mode)
        chart_denoise(mode)
        chart_phases(mode)
    _shrink()
    print(f"wrote 6 PNGs to {OUT}")


if __name__ == "__main__":
    main()
