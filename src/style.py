"""Anthropic warm theme — shared colour tokens and matplotlib theming.

Palette draws from a warm, terracotta-toned design language with
cool green and blue accents. Applied once at notebook startup so
every plot in the project looks consistent.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
from matplotlib import font_manager

# ── Anthropic warm theme tokens ──────────────────────────────────────────

BG         = "#FDF8F4"
SURFACE    = "#F7F0EA"
GRID       = "#E8DDD4"
TEXT       = "#3D3229"
TEXT_MUTED = "#8A7E74"

PALETTE = [
    "#E07A5F", "#81B29A", "#F2CC8F",
    "#457B9D", "#9C6644", "#B5838D",
    "#6D6875", "#E9C46A", "#264653", "#F4A261",
]

HIGHLIGHT = "#E07A5F"

PRIMARY   = PALETTE[0]
SECONDARY = PALETTE[3]
ACCENT    = PALETTE[4]

DIVERGING = [PALETTE[3], "#A8C4D0", BG, PALETTE[5], PALETTE[0]]


def _available_sans_fonts() -> list[str]:
    candidates = [
        "DejaVu Sans", "Nimbus Sans", "Liberation Sans",
        "Arial", "Helvetica Neue", "Segoe UI",
    ]
    available = []
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            available.append(name)
        except ValueError:
            continue
    return available or ["DejaVu Sans"]


def apply_warm_theme(dpi: int = 140) -> None:
    """Register and activate the warm theme globally."""
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "figure.dpi": dpi,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": _available_sans_fonts(),
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
        "axes.labelcolor": TEXT_MUTED,
        "axes.titlesize": 14,
        "axes.titleweight": "600",
        "axes.titlecolor": TEXT,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "text.color": TEXT,
        "legend.frameon": False,
    })


def theme_axes(ax, title="", xlabel="", ylabel=""):
    """Apply the warm theme to a single axes (post-hoc touch-up)."""
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=14, fontweight="600", color=TEXT, pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=TEXT_MUTED, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=TEXT_MUTED, labelpad=8)
    ax.tick_params(colors=TEXT_MUTED, labelsize=10, length=0)
