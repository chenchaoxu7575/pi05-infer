#!/usr/bin/env python3
"""Regenerate the two README charts (light + dark PNG for each).

    python docs/make_charts.py [--sqlite <on.sqlite> --sqlite-off <off.sqlite>]

Chart 1 groups the paired A/B ledger into three blocks of work. Chart 2 is the
per-kernel breakdown of one denoise step, from an nsys sqlite export; --sqlite
re-derives it and prints what it derived so it can be diffed against the
constants below.
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
# Chart 1 - improvement over the original baseline
# --------------------------------------------------------------------------
# Baseline is the extracted model as it arrived: torch.compile max-autotune, none
# of this work applied. e2e = predict_action_batch, plain CPU wall clock, 30 iters
# after 8 warmups, serialized, RTX PRO 5000 at its 300 W default.
#
# Each ledger row is its own paired A/B against the row above it, so the deltas
# are additive; the three category totals are sums of those deltas and add to 9.70.
BASE, LEDGER_END, NOW = 52.60, 42.90, 40.50

CATS = [
    (
        "CPU overhead",
        3.44,
        [
            "capture one flow_ode step as a graph, replay it for every step",
            "prefix KV into a static buffer",
            "attention mask built on device",
            "extract to a standalone package",
        ],
    ),
    (
        "denoise-step work removed",
        3.09,
        [
            "precompute the adaRMS modulation table, 37 projections -> 1 gather",
            "drop the timestep conditioning nothing reads",
        ],
    ),
    (
        "kernel fusion & optimization",
        3.17,
        [
            "GeGLU and RoPE fused into the GEMM epilogue",
            "Q/K/V into a single GEMM",
        ],
    ),
]

SINCE = (
    "since then, each on its own baseline: fused prefix QKV, retiled "
    "down_proj / o_proj, pinned the Q*K^T tile"
)


def chart_ledger(mode: str) -> None:
    t = THEMES[mode]
    _style(t)
    colors = [t["s1"], t["s2"], t["s3"]]
    span = BASE - LEDGER_END

    fig = plt.figure(figsize=(10.5, 5.6), dpi=200)
    ax = fig.add_axes((0.035, 0.660, 0.930, 0.090))
    ax.set_xlim(0, span)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")

    x = 0.0
    for (label, ms, _items), c in zip(CATS, colors):
        _rounded_hbar(ax, 0, x, x + ms, 0.95, c, radius_pt=3.0)
        ax.text(
            x + ms / 2,
            0.06,
            f"-{ms:.2f}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=t["surface"],
        )
        ax.text(
            x + ms / 2,
            -0.26,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color=t["surface"],
        )
        x += ms

    fig.suptitle(
        f"pi0.5 action expert, bs=1:   {BASE:.2f} -> {LEDGER_END:.2f} ms   "
        f"(-{span:.2f} ms, -{100 * span / BASE:.1f}%)",
        x=0.035,
        y=0.950,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
        color=t["ink"],
    )
    fig.text(
        0.035,
        0.888,
        "end-to-end predict_action_batch, plain wall clock.  Baseline is the model "
        "as extracted: torch.compile max-autotune, none of this applied.",
        ha="left",
        fontsize=9.5,
        color=t["ink2"],
    )
    fig.text(
        0.035,
        0.775,
        f"{BASE:.2f} ms",
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=t["ink2"],
    )
    fig.text(
        0.965,
        0.775,
        f"{LEDGER_END:.2f} ms",
        ha="right",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=t["ink"],
    )

    y = 0.565
    for (label, ms, items), c in zip(CATS, colors):
        fig.patches.append(
            plt.Rectangle(
                (0.035, y - 0.004),
                0.012,
                0.028,
                transform=fig.transFigure,
                facecolor=c,
                edgecolor="none",
                zorder=5,
            )
        )
        fig.text(
            0.057,
            y,
            label,
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=t["ink"],
        )
        fig.text(
            0.965,
            y,
            f"-{ms:.2f} ms",
            ha="right",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=t["ink"],
        )
        y -= 0.052
        for it in items:
            fig.text(0.057, y, it, ha="left", va="bottom", fontsize=9.5, color=t["ink2"])
            y -= 0.040
        y -= 0.018

    fig.text(
        0.035,
        0.030,
        SINCE,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=t["muted"],
        style="italic",
    )
    fig.savefig(OUT / f"ledger_{mode}.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# Chart 2 - denoise per-kernel breakdown
# --------------------------------------------------------------------------
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
        if name == "triton_tem_fused_mm_6":
            return "down_proj GEMM"
        if name == "triton_tem_fused_clone_mm_4":
            return "o_proj GEMM (+ transpose prologue)"
        if name == "_qkv_rope_kernel":
            return "QKV GEMM + RoPE -> static KV  [fused]"
        if name == "triton_tem_fused_bmm_3":
            return "attention BMM  P*V"
        if name == "triton_tem_fused_bmm_1":
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
        print(
            f"  {label:66s} {d_us:8.2f} us/step ({us:8.2f})  "
            f"{d_k:6.2f} k/step ({k:6.2f}){flag}"
        )
    print(
        f"  {'TOTAL':66s} {sum(v[0] for v in agg.values()):8.2f} us/step "
        f"({DENOISE_TOTAL_US:8.2f})  {sum(v[1] for v in agg.values()):6.2f} k/step "
        f"({DENOISE_TOTAL_KERNELS:6.2f})"
    )


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
        ax.text(
            us + 5,
            y,
            f"{us:.1f} us   {kern:g}/step",
            ha="left",
            va="center",
            fontsize=9,
            color=t["ink"] if grp == 2 else t["ink2"],
            fontweight="bold" if grp == 2 else "normal",
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in DENOISE], fontsize=9)
    ax.set_xlabel("GPU time per denoise step (us)", fontsize=9.5)

    handles = [
        plt.Line2D(
            [], [], marker="s", ls="", markersize=9, color=colors[g], label=GROUP_NAMES[g]
        )
        for g in (2, 0, 1)
    ]
    # figure-level, below the axes: inside the axes the legend sat on top of
    # the two shortest bars' rows
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.010, 0.048),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=t["ink2"],
        handletextpad=0.6,
        borderpad=0.2,
        columnspacing=1.6,
    )

    fig.suptitle(
        "Where a denoise step goes: 1185.0 us, 217 kernels",
        x=0.012,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=t["ink"],
    )
    fig.text(
        0.012,
        0.912,
        "nsys 2026.1.2, stream 157 (Stage-1 captured graph), "
        "",
        ha="left",
        fontsize=9,
        color=t["ink2"],
    )
    fig.text(
        0.012,
        0.013,
        "the 37 adaRMS dense(cond) projections that used to cost 395 us/step here "
        "(triton_per_fused_addmm_0, 300 instances) are gone - 0 instances",
        ha="left",
        fontsize=7.5,
        color=t["muted"],
    )
    fig.savefig(OUT / f"denoise_{mode}.png")
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
    ap.add_argument("--sqlite-off", default=None)
    args = ap.parse_args()
    if args.sqlite:
        derive_denoise(args.sqlite)
    for mode in ("light", "dark"):
        chart_ledger(mode)
        chart_denoise(mode)
    _shrink()
    print(f"wrote 4 PNGs to {OUT}")


if __name__ == "__main__":
    main()
