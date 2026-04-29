"""Reporting plots used in the modeling and profiling sections."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from .pure_premium import lorenz_curve
from . import style


def plot_roc(y_true: pd.Series, y_pred: np.ndarray, *,
             title: str = "Frequency model ROC") -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.fill_between(fpr, tpr, alpha=0.15, color=style.PRIMARY)
    ax.plot(fpr, tpr, lw=2.5, color=style.PRIMARY)
    ax.plot([0, 1], [0, 1], "--", color=style.SECONDARY, lw=1, alpha=0.6)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_calibration(y_true: pd.Series, y_pred: np.ndarray, *,
                     n_bins: int = 10,
                     title: str = "Frequency calibration") -> plt.Figure:
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins,
                                             strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, marker="o", lw=2.5, color=style.PRIMARY,
            markeredgecolor=style.BG, markeredgewidth=0.5, markersize=7)
    ax.plot([0, 1], [0, 1], "--", color=style.SECONDARY, lw=1, alpha=0.6)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed claim rate")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_severity_diagnostics(y_true: pd.Series, y_pred: np.ndarray
                              ) -> plt.Figure:
    """Two-panel severity check: predicted vs actual, and residual ratios."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # predicted vs actual
    axes[0].scatter(y_pred, y_true, s=4, alpha=0.25, color=style.PRIMARY,
                    edgecolors="none")
    lim = (max(1, np.min(y_pred)), max(np.max(y_true), np.max(y_pred)))
    axes[0].plot(lim, lim, "--", color=style.ACCENT, lw=1.5)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("predicted (EUR, log)")
    axes[0].set_ylabel("observed (EUR, log)")
    axes[0].set_title("Severity: predicted vs observed")
    # residual ratio
    ratio = y_true / np.clip(y_pred, 1e-6, None)
    axes[1].scatter(y_pred, ratio, s=4, alpha=0.25, color=style.SECONDARY,
                    edgecolors="none")
    axes[1].axhline(1.0, color=style.ACCENT, linestyle="--", lw=1.5)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("predicted (EUR, log)")
    axes[1].set_ylabel("observed / predicted")
    axes[1].set_title("Severity residual ratio")
    fig.tight_layout()
    return fig


def plot_lorenz(pp_df: pd.DataFrame, *,
                title: str = "Loss concentration (Lorenz curve)"
                ) -> plt.Figure:
    pop, cum = lorenz_curve(pp_df)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.fill_between(pop, cum, pop, alpha=0.15, color=style.PRIMARY)
    ax.plot(pop, cum, lw=2.5, color=style.PRIMARY, label="portfolio")
    ax.plot([0, 1], [0, 1], "--", color=style.SECONDARY, lw=1, alpha=0.6,
            label="equal risk")
    ax.set_xlabel("cumulative share of policies (low to high risk)")
    ax.set_ylabel("cumulative share of expected loss")
    ax.set_title(title)
    ax.legend(frameon=True, facecolor=style.BG, edgecolor=style.SURFACE)
    fig.tight_layout()
    return fig


def plot_tier_bars(tier_summary_df: pd.DataFrame) -> plt.Figure:
    """Side-by-side bars: portfolio share vs loss share per tier."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(tier_summary_df))
    w = 0.35
    ax.bar(x - w/2, tier_summary_df["pct_of_portfolio"], w,
           label="% of portfolio", color=style.SECONDARY, edgecolor="none")
    ax.bar(x + w/2, tier_summary_df["pct_of_total_loss"], w,
           label="% of expected loss", color=style.PRIMARY,
           edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(tier_summary_df.index.astype(str))
    ax.set_ylabel("percent")
    ax.set_title("Risk tier: share of portfolio vs share of expected loss")
    ax.legend(frameon=True, facecolor=style.BG, edgecolor=style.SURFACE)
    fig.tight_layout()
    return fig


def plot_pure_premium_distribution(pp_df: pd.DataFrame) -> plt.Figure:
    """Histogram of per-policy pure premium across the full portfolio."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pp_df["pure_premium"], bins=60, color=style.PRIMARY,
            edgecolor="none")
    mean_pp = pp_df["pure_premium"].mean()
    ax.axvline(mean_pp, color=style.ACCENT, linestyle="--", lw=1.5,
               label=f"mean = {mean_pp:.0f} EUR")
    ax.set_xlabel("pure premium (EUR)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of per-policy pure premium")
    ax.legend(frameon=True, facecolor=style.BG, edgecolor=style.SURFACE)
    fig.tight_layout()
    return fig
