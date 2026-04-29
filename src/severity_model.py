"""Severity model: expected claim cost given a claim occurred.

Standard actuarial choice is a Gamma GLM with log link — the log link gives a
multiplicative structure (consistent with the frequency model) and Gamma
handles strictly-positive, right-skewed cost data with variance proportional
to mean^2.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error


def fit_gamma_glm(X: pd.DataFrame, y: pd.Series) -> sm.GLM:
    """Gamma GLM with log link. Strictly-positive responses required — the
    data loader already filters out non-positive claim sizes."""
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.GLM(
        y.astype(float), X_const,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    )
    # method="lbfgs" is more robust than IRLS when columns are nearly collinear
    return model.fit(maxiter=200)


def predict_mean(model: sm.GLM, X: pd.DataFrame) -> np.ndarray:
    """Predicted E[claim size | claim] for each row of X."""
    X_const = sm.add_constant(X, has_constant="add")
    return np.asarray(model.predict(X_const))


def evaluate_severity(model: sm.GLM, X: pd.DataFrame,
                      y: pd.Series) -> dict[str, float]:
    """Out-of-sample diagnostics on the original (EUR) scale:
    - MAE: median-style typical error
    - RMSE: penalises large mistakes (relevant for high-severity tail)
    - mean_pred / observed_mean: balance check; should be close
    - gini-ish: rank-quality, useful for risk ordering
    """
    yhat = predict_mean(model, X)
    return {
        "mae": float(mean_absolute_error(y, yhat)),
        "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
        "mean_pred": float(yhat.mean()),
        "observed_mean": float(y.mean()),
        "rank_corr": float(pd.Series(y).corr(pd.Series(yhat), method="spearman")),
    }


def coefficient_table(model: sm.GLM) -> pd.DataFrame:
    """Same structure as the frequency table but with multiplicative effects
    (exp(coef)) instead of odds ratios, reflecting the log-link semantics."""
    params = model.params
    conf = model.conf_int()
    return pd.DataFrame({
        "coef": params,
        "std_err": model.bse,
        "z": model.tvalues,
        "p_value": model.pvalues,
        "ci_low": conf[0],
        "ci_high": conf[1],
        "mult_effect": np.exp(params),
    })
