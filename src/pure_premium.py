"""Combine frequency and severity into a per-policy pure premium.

Because the two datasets cannot be linked at the individual level we score
the *frequency* dataset as the canonical portfolio (it contains every
policy, claim or no claim) and use the trained severity model to predict
what each of those policies would cost *if* it had a claim. Multiplying
gives the expected loss per policy — the pure premium.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .frequency_model import predict_proba
from .severity_model import predict_mean


def compute_pure_premium(
    portfolio_X: pd.DataFrame,
    freq_model: sm.GLM,
    sev_model: sm.GLM,
    sev_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Score every row in ``portfolio_X`` and return a dataframe with:
        - p_claim: predicted probability of a claim
        - exp_severity: predicted EUR cost given a claim occurred
        - pure_premium: p_claim * exp_severity (the headline number)

    We pass column names explicitly because frequency and severity were each
    fit on potentially-different one-hot column sets; reindexing aligns them.
    """
    p = predict_proba(freq_model, portfolio_X)

    # Reindex so the severity model sees the columns it was fit on
    if sev_columns is not None:
        X_sev = portfolio_X.reindex(columns=sev_columns, fill_value=0.0)
    else:
        X_sev = portfolio_X
    s = predict_mean(sev_model, X_sev)

    out = pd.DataFrame({
        "p_claim": p,
        "exp_severity": s,
        "pure_premium": p * s,
    }, index=portfolio_X.index)
    return out


def calibration_check(pp_df: pd.DataFrame, observed_total_loss: float
                      ) -> dict[str, float]:
    """Balance-property check: the sum of predicted pure premiums should
    approximate the total observed claims cost in the portfolio. Big gaps
    flag a miscalibrated frequency or severity model."""
    pred_total = float(pp_df["pure_premium"].sum())
    return {
        "predicted_total_loss": pred_total,
        "observed_total_loss": float(observed_total_loss),
        "ratio_pred_over_obs": pred_total / observed_total_loss,
    }


def lorenz_curve(pp_df: pd.DataFrame, ascending: bool = True
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative-share curves for plotting. Sorting customers from low to
    high pure premium and tracking cumulative loss-share gives the actuarial
    Lorenz curve; the area between it and the 45-degree line is the Gini —
    a one-number summary of how concentrated risk is in the tail."""
    s = pp_df["pure_premium"].sort_values(ascending=ascending).to_numpy()
    cum = np.cumsum(s) / s.sum()
    pop = np.linspace(1.0 / len(s), 1.0, len(s))
    return pop, cum
