"""Exploratory plots and tabular summaries used in the EDA section."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from .config import CATEGORICAL_VARS, NUMERIC_VARS
from . import style


def factor_target_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """For each factor variable, return a long-format table with: level,
    n (count of policies in that level), mean of target. For frequency target
    this becomes the empirical claim probability per level; for severity, the
    average claim cost per level."""
    rows = []
    for col in CATEGORICAL_VARS:
        grp = df.groupby(col, observed=True)[target].agg(["count", "mean"])
        for level, (cnt, mu) in grp.iterrows():
            rows.append({"variable": col, "level": str(level),
                         "n": int(cnt), "mean_target": float(mu)})
    return pd.DataFrame(rows)


def plot_target_distribution(df: pd.DataFrame, target: str, *,
                             log_scale: bool = False, bins: int = 50) -> plt.Figure:
    """Histogram of the target variable, optionally on log10 scale."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if log_scale:
        data = df[target].clip(lower=1)
        ax.hist(np.log10(data), bins=bins, color=style.PRIMARY,
                edgecolor="none")
        ax.set_xlabel(f"log10({target})")
    else:
        ax.hist(df[target], bins=bins, color=style.PRIMARY,
                edgecolor="none")
        ax.set_xlabel(target)
    ax.set_ylabel("count")
    ax.set_title(f"Distribution of {target}" + (" (log10)" if log_scale else ""))
    fig.tight_layout()
    return fig


def plot_categorical_target(df: pd.DataFrame, target: str,
                            ncols: int = 3) -> plt.Figure:
    """Bar plot of mean(target) per level for every categorical variable."""
    n = len(CATEGORICAL_VARS)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, CATEGORICAL_VARS):
        grp = df.groupby(col, observed=True)[target].mean().sort_index()
        colors = [style.PALETTE[i % len(style.PALETTE)]
                  for i in range(len(grp))]
        ax.bar(grp.index.astype(str), grp.values, color=colors,
               edgecolor="none")
        ax.set_title(f"mean({target}) by {col}")
        ax.tick_params(axis="x", rotation=30)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_numeric_target(df: pd.DataFrame, target: str,
                        ncols: int = 2) -> plt.Figure:
    """Smoothed mean(target) versus each numeric covariate using 20
    quantile bins."""
    n = len(NUMERIC_VARS)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, (ax, col) in enumerate(zip(axes, NUMERIC_VARS)):
        binned = pd.qcut(df[col], q=20, duplicates="drop")
        grp = df.groupby(binned, observed=True).agg(
            x=(col, "mean"), y=(target, "mean")
        )
        c = style.PALETTE[i % len(style.PALETTE)]
        ax.plot(grp["x"], grp["y"], marker="o", lw=2, color=c,
                markeredgecolor=style.BG, markeredgewidth=0.5, markersize=6)
        ax.fill_between(grp["x"], grp["y"], alpha=0.12, color=c)
        ax.set_xlabel(col); ax.set_ylabel(f"mean({target})")
        ax.set_title(f"{target} vs {col} (binned)")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Pearson correlation among numeric covariates."""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "anthropic_div", style.DIVERGING, N=256)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(df[NUMERIC_VARS].corr(), annot=True, fmt=".2f",
                cmap=cmap, center=0, ax=ax,
                linewidths=0.5, linecolor=style.BG,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson correlation: numeric covariates")
    fig.tight_layout()
    return fig
