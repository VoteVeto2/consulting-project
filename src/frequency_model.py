"""Frequency model: probability that a policy has at least one MD claim.

The target ``claimNumbMD`` is binary (0/1) so we use a logistic regression
(Bernoulli GLM with logit link) — not Poisson, since we don't observe a
count. This is the cleanest match to the documented data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


def fit_logit(X: pd.DataFrame, y: pd.Series) -> sm.GLM:
    """Fit a Bernoulli GLM (logit link). statsmodels gives us coefficient
    p-values and confidence intervals out of the box, which the GLM section
    of the report leans on."""
    # add_constant prepends an intercept column. has_constant='add' is a
    # safety net in case the matrix already has one; statsmodels will then
    # not double-add.
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.GLM(y.astype(float), X_const, family=sm.families.Binomial())
    return model.fit(maxiter=200)


def predict_proba(model: sm.GLM, X: pd.DataFrame) -> np.ndarray:
    """Return P(claim=1 | x) for each row of X, with the same column-order
    convention used at fit time."""
    X_const = sm.add_constant(X, has_constant="add")
    return np.asarray(model.predict(X_const))


def evaluate_frequency(model: sm.GLM, X: pd.DataFrame,
                       y: pd.Series) -> dict[str, float]:
    """Out-of-sample diagnostics:
    - AUC: rank-quality (does the model order high-risk above low-risk?)
    - log-loss: probabilistic calibration penalty
    - brier: mean squared error of the probability — combines calibration + sharpness
    - mean_pred / observed_rate: balance/calibration sanity check
    """
    p = predict_proba(model, X)
    return {
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "mean_pred": float(p.mean()),
        "observed_rate": float(y.mean()),
    }


def coefficient_table(model: sm.GLM) -> pd.DataFrame:
    """Tidy coefficient table with odds ratios — easier to read than raw
    log-odds when communicating with management."""
    params = model.params
    conf = model.conf_int()
    out = pd.DataFrame({
        "coef": params,
        "std_err": model.bse,
        "z": model.tvalues,
        "p_value": model.pvalues,
        "ci_low": conf[0],
        "ci_high": conf[1],
        "odds_ratio": np.exp(params),
    })
    return out
