"""Customer profiling: convert per-policy pure premiums into actionable
risk tiers and describe what kind of customer each tier contains."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import CATEGORICAL_VARS, NUMERIC_VARS


def assign_tiers(pp_df: pd.DataFrame, *,
                 quantiles: tuple[float, ...] = (0.5, 0.8, 0.95)
                 ) -> pd.Series:
    """Bucket policies into risk tiers based on quantile thresholds of the
    pure premium. The default cuts give: bottom 50% (Low), 50-80% (Medium),
    80-95% (High), top 5% (Very High) — a typical actuarial tiering for
    portfolio steering."""
    cuts = pp_df["pure_premium"].quantile(list(quantiles)).values
    edges = [-np.inf, *cuts, np.inf]
    labels = ["Low", "Medium", "High", "Very High"]
    if len(edges) - 1 != len(labels):
        # Fallback for non-default quantiles — generate generic labels
        labels = [f"Tier {i+1}" for i in range(len(edges) - 1)]
    return pd.cut(pp_df["pure_premium"], bins=edges, labels=labels)


def tier_summary(df: pd.DataFrame, pp_df: pd.DataFrame, tiers: pd.Series
                 ) -> pd.DataFrame:
    """One row per tier with size, average pure premium, and a few headline
    covariate means. Lets management see at a glance what kind of customers
    sit in each tier."""
    joined = df.copy()
    joined["pure_premium"] = pp_df["pure_premium"].values
    joined["tier"] = tiers.values
    out = joined.groupby("tier", observed=True).agg(
        n=("pure_premium", "size"),
        mean_pp=("pure_premium", "mean"),
        median_pp=("pure_premium", "median"),
        mean_age=("age", "mean"),
        mean_carVal=("carVal", "mean"),
        mean_density=("density", "mean"),
    )
    out["pct_of_portfolio"] = out["n"] / out["n"].sum() * 100
    out["pct_of_total_loss"] = (
        joined.groupby("tier", observed=True)["pure_premium"].sum()
        / joined["pure_premium"].sum() * 100
    )
    return out


def tier_factor_profile(df: pd.DataFrame, tiers: pd.Series, factor: str
                        ) -> pd.DataFrame:
    """Cross-tab of tier x factor level showing what % of each tier sits in
    each level of the factor. Helpful for narrative: "65% of the Very High
    tier are male" type sentences."""
    tab = pd.crosstab(tiers, df[factor], normalize="index") * 100
    tab.columns = [str(c) for c in tab.columns]
    return tab.round(1)


def describe_top_segment(df: pd.DataFrame, pp_df: pd.DataFrame,
                         tier: pd.Series, segment_label: str = "Very High"
                         ) -> dict[str, object]:
    """Plain-English description of the worst tier — used by the closing
    'recommendation' cell of the notebook."""
    mask = tier == segment_label
    sub = df.loc[mask]
    pp = pp_df.loc[mask, "pure_premium"]
    overall_mean = pp_df["pure_premium"].mean()
    profile = {}
    for c in CATEGORICAL_VARS:
        # Most common level + its share — gives a quick narrative anchor
        vc = sub[c].value_counts(normalize=True)
        if len(vc):
            profile[c] = f"{vc.index[0]} ({vc.iloc[0]*100:.0f}%)"
    for c in NUMERIC_VARS:
        profile[c] = round(float(sub[c].mean()), 1)
    return {
        "segment": segment_label,
        "n_policies": int(mask.sum()),
        "share_of_portfolio_pct": round(float(mask.mean()) * 100, 1),
        "mean_pure_premium": round(float(pp.mean()), 2),
        "ratio_to_overall_mean": round(float(pp.mean() / overall_mean), 2),
        "typical_profile": profile,
    }
