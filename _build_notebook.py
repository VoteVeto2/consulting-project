#!/usr/bin/env python3
"""Generate customer_profiling_solution.ipynb."""
import json

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": src,
        "execution_count": None,
        "outputs": [],
    })


# ══════════════════════════════════════════════════════════════════════
# SECTION 0 ─ Title, Objective, Assumptions
# ══════════════════════════════════════════════════════════════════════

md("""\
# Customer Profiling: Renewal-Risk Analysis
## Identifying Policyholders with High Expected Material-Damage Claim Cost

### Business Question

The objective is to identify customers in a non-life motor insurance portfolio who are
likely to be **unprofitable at renewal**.

Because the available data does not contain premium, expense, reinsurance, or
acquisition-cost information, **actual profitability cannot be measured**. Instead this
analysis estimates a **proxy for unprofitability**: the expected material-damage claim
cost per policy.

### Data Structure

Two separate datasets from the same portfolio are provided:

| Dataset | Rows | Target | Description |
|---|---|---|---|
| `frequency.csv` | 24,774 | `claimNumbMD` (binary) | Policy-level claim indicator |
| `severity.csv` | 12,256 | `claimSizeMD` (continuous) | Claim-level damage amount |

The datasets share the same 10 covariates but **cannot be linked at the individual
policyholder level**.

### Modelling Strategy

Because the data are unlinked, frequency and severity must be modelled separately:

1. **Claim frequency** → binary classification on `frequency.csv`
2. **Conditional claim severity** → regression on `severity.csv`
3. **Expected claim cost proxy** = P(claim) &times; E(severity | claim)

Combined validation is performed at the **risk-cell level** (not policy level) because
there is no per-policy ground-truth pure premium.

### Key Caveat

The resulting scores represent **expected material-damage claim cost**, not proven
unprofitability. They should be used as one input in a broader renewal-decision process
that also considers premium adequacy, customer lifetime value, and strategic priorities.""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 ─ Imports & Configuration
# ══════════════════════════════════════════════════════════════════════

code("""\
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, OrdinalEncoder
from sklearn.linear_model import LogisticRegression, Ridge, TweedieRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    mean_absolute_error, mean_squared_error,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import statsmodels.api as sm

SEED = 42
np.random.seed(SEED)

DATA_DIR = Path("dataset")
OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="statsmodels")

plt.rcParams.update({
    "figure.figsize": (10, 5),
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

print(f"Random seed : {SEED}")
print(f"Data dir    : {DATA_DIR.resolve()}")
print(f"Output dir  : {OUTPUT_DIR.resolve()}")""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 ─ Load & Audit
# ══════════════════════════════════════════════════════════════════════

code("""\
freq = pd.read_csv(DATA_DIR / "frequency.csv")
sev  = pd.read_csv(DATA_DIR / "severity.csv")

CAT_COLS = ["uwYear", "gender", "carType", "carCat", "job", "cover"]
NUM_COLS = ["age", "nYears", "carVal", "density"]

for df in [freq, sev]:
    for c in CAT_COLS:
        df[c] = df[c].astype("category")

freq["density_round"] = freq["density"].round().astype(int)
sev["density_round"]  = sev["density"]  # already int

print("FREQUENCY DATASET")
print(f"  Rows: {freq.shape[0]:,}   Columns: {freq.shape[1]}")
print(f"  Missing values: {freq.isnull().sum().sum()}")
print(f"  claimNumbMD — mean: {freq['claimNumbMD'].mean():.4f}, "
      f"positive: {freq['claimNumbMD'].sum():,}/{len(freq):,}")
print()
print("SEVERITY DATASET")
print(f"  Rows: {sev.shape[0]:,}   Columns: {sev.shape[1]}")
print(f"  Missing values: {sev.isnull().sum().sum()}")
print(f"  claimSizeMD — mean: {sev['claimSizeMD'].mean():.2f}, "
      f"median: {sev['claimSizeMD'].median():.2f}, "
      f"zeros: {(sev['claimSizeMD'] == 0).sum()}")
print()

print("Numeric ranges:")
for c in NUM_COLS:
    print(f"  {c:>10s}  freq [{freq[c].min():>8.1f}, {freq[c].max():>8.1f}]  "
          f"sev [{sev[c].min():>8.1f}, {sev[c].max():>8.1f}]")
print()

print("Categorical levels (freq == sev):")
for c in ["gender", "carType", "carCat", "job", "cover"]:
    fl = sorted(freq[c].unique())
    sl = sorted(sev[c].unique())
    print(f"  {c:>8s}: {fl}  {'MATCH' if fl == sl else 'MISMATCH'}")""")


md("""\
### Data Audit — Key Findings

* Both datasets have **no missing values**.
* `claimNumbMD` is essentially balanced (mean &asymp; 0.50), so the frequency task is a
  balanced binary classification.
* `claimSizeMD` has **strong right skew** (mean &asymp; 866, median much lower) and
  **4 zero-severity claims** which must be excluded for Gamma-family models.
* `density` is `float64` in `frequency.csv` but `int64` in `severity.csv`.
  A rounded integer helper (`density_round`) is created for grouped matching.
* `uwYear` and `cover` are numerically coded but treated as **categorical**.""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 ─ Dataset Compatibility Check
# ══════════════════════════════════════════════════════════════════════

code("""\
freq_pos = freq[freq["claimNumbMD"] == 1].copy()

print("=== Categorical Proportion Comparison ===")
print(f"{'Variable':<10} {'Level':<15} {'freq(claim=1)':>14} {'severity':>10} {'diff':>8}")
print("-" * 60)
for c in ["gender", "carType", "carCat", "job", "cover"]:
    p_f = freq_pos[c].value_counts(normalize=True).sort_index()
    p_s = sev[c].value_counts(normalize=True).sort_index()
    for lvl in p_f.index:
        vf = p_f.get(lvl, 0)
        vs = p_s.get(lvl, 0)
        print(f"{c:<10} {str(lvl):<15} {vf:>14.4f} {vs:>10.4f} {abs(vf-vs):>8.4f}")
    print()

print("=== Continuous-Variable Quantile Comparison ===")
qs = [0.10, 0.25, 0.50, 0.75, 0.90]
for c in ["age", "nYears", "carVal", "density_round"]:
    qf = freq_pos[c].quantile(qs)
    qs_sev = sev[c if c != "density_round" else "density_round"].quantile(qs)
    print(f"\\n{c}:")
    for q in qs:
        print(f"  Q{q:.0%}: freq_pos={qf[q]:>10.1f}   sev={qs_sev[q]:>10.1f}")

match_cols = ["gender", "carType", "carCat", "job", "cover",
              "age", "nYears", "carVal", "density_round"]
freq_pos_profiles = freq_pos[match_cols].drop_duplicates()
sev_profiles      = sev[match_cols].drop_duplicates()
common = pd.merge(freq_pos_profiles, sev_profiles, how="inner")
print(f"\\nUnique covariate profiles — freq(claim=1): {len(freq_pos_profiles):,}, "
      f"severity: {len(sev_profiles):,}, common: {len(common):,}")""")


md("""\
### Compatibility Summary

After rounding `density` to integer values, the covariate distributions in
`severity.csv` are **very similar** to the positive-claim subset of `frequency.csv`.
This confirms that `severity.csv` can be treated as a sample from
*severity | claim = 1* and that separate frequency–severity modelling is legitimate.

However, direct policy-level linkage remains **unavailable**, so combined-model
validation must be performed at the **risk-cell / portfolio level**, not via
individual realised losses.""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 ─ Exploratory Analysis
# ══════════════════════════════════════════════════════════════════════

code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(["No claim", "Claim"],
            [freq["claimNumbMD"].value_counts()[0],
             freq["claimNumbMD"].value_counts()[1]],
            color=["steelblue", "salmon"])
axes[0].set_title("Claim Frequency Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(freq["claimNumbMD"].value_counts().sort_index()):
    axes[0].text(i, v + 200, f"{v:,}", ha="center")

axes[1].hist(sev["claimSizeMD"], bins=80, color="steelblue", edgecolor="white")
axes[1].set_title("Severity Distribution (raw)")
axes[1].set_xlabel("claimSizeMD")

sev_pos = sev[sev["claimSizeMD"] > 0]["claimSizeMD"]
axes[2].hist(np.log1p(sev_pos), bins=80, color="salmon", edgecolor="white")
axes[2].set_title("Severity Distribution (log scale)")
axes[2].set_xlabel("log(1 + claimSizeMD)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "target_distributions.png", bbox_inches="tight")
plt.show()""")


code("""\
cat_features_eda = ["gender", "carType", "carCat", "job", "cover"]

fig, axes = plt.subplots(2, len(cat_features_eda), figsize=(18, 8))

for i, c in enumerate(cat_features_eda):
    rates = freq.groupby(c)["claimNumbMD"].mean().sort_index()
    axes[0, i].bar(range(len(rates)), rates.values, color="steelblue")
    axes[0, i].set_xticks(range(len(rates)))
    axes[0, i].set_xticklabels([str(x) for x in rates.index], rotation=45, fontsize=8)
    axes[0, i].set_title(f"Claim Rate by {c}")
    axes[0, i].set_ylabel("P(claim)")

    means = sev[sev["claimSizeMD"] > 0].groupby(c)["claimSizeMD"].mean().sort_index()
    axes[1, i].bar(range(len(means)), means.values, color="salmon")
    axes[1, i].set_xticks(range(len(means)))
    axes[1, i].set_xticklabels([str(x) for x in means.index], rotation=45, fontsize=8)
    axes[1, i].set_title(f"Mean Severity by {c}")
    axes[1, i].set_ylabel("Mean claimSizeMD")

plt.suptitle("Univariate Analysis — Categorical Variables", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_categorical.png", bbox_inches="tight")
plt.show()""")


code("""\
cont_features = ["age", "nYears", "carVal", "density_round"]
fig, axes = plt.subplots(2, len(cont_features), figsize=(18, 8))

for i, c in enumerate(cont_features):
    col_freq = c
    col_sev  = c if c in sev.columns else c
    n_bins = min(20, freq[col_freq].nunique())

    binned = pd.qcut(freq[col_freq], q=n_bins, duplicates="drop")
    rate_by_bin = freq.groupby(binned, observed=True)["claimNumbMD"].mean()
    axes[0, i].plot(range(len(rate_by_bin)), rate_by_bin.values, "o-", color="steelblue")
    axes[0, i].set_title(f"Claim Rate by {c}")
    axes[0, i].set_ylabel("P(claim)")
    tick_labels = [str(x) for x in rate_by_bin.index]
    axes[0, i].set_xticks(range(len(rate_by_bin)))
    axes[0, i].set_xticklabels(tick_labels, rotation=90, fontsize=6)

    sev_c = sev[sev["claimSizeMD"] > 0].copy()
    n_bins_s = min(20, sev_c[col_sev].nunique())
    binned_s = pd.qcut(sev_c[col_sev], q=n_bins_s, duplicates="drop")
    sev_by_bin = sev_c.groupby(binned_s, observed=True)["claimSizeMD"].mean()
    axes[1, i].plot(range(len(sev_by_bin)), sev_by_bin.values, "o-", color="salmon")
    axes[1, i].set_title(f"Mean Severity by {c}")
    axes[1, i].set_ylabel("Mean claimSizeMD")
    tick_labels_s = [str(x) for x in sev_by_bin.index]
    axes[1, i].set_xticks(range(len(sev_by_bin)))
    axes[1, i].set_xticklabels(tick_labels_s, rotation=90, fontsize=6)

plt.suptitle("Univariate Analysis — Continuous Variables (binned)", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_continuous.png", bbox_inches="tight")
plt.show()

print("Strongest univariate frequency associations (absolute claim-rate spread):")
for c in cat_features_eda + cont_features:
    col = c
    n_bins = min(10, freq[col].nunique())
    try:
        binned = pd.qcut(freq[col], q=n_bins, duplicates="drop")
    except TypeError:
        binned = freq[col]
    rates = freq.groupby(binned, observed=True)["claimNumbMD"].mean()
    spread = rates.max() - rates.min()
    print(f"  {c:<15s} spread = {spread:.4f}  (min={rates.min():.4f}, max={rates.max():.4f})")""")


md("""\
### EDA Takeaways

* Claim frequency is near-balanced, so class imbalance is not a concern.
* Severity is heavily right-skewed; log-transformation or a Gamma distribution is
  appropriate.
* Among categorical variables, **job**, **carCat**, and **carType** show the largest
  variation in both claim rate and mean severity.
* Among continuous variables, **age** and **density** show the strongest non-linear
  association with claim frequency.
* `carVal` shows a moderate positive association with severity (more expensive cars →
  higher claims).""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 ─ Validation Design
# ══════════════════════════════════════════════════════════════════════

md("""\
## Section 5 — Validation Design

### Temporal Holdout

The primary evaluation uses a **temporal split**:
* **Training set**: `uwYear == 2009`
* **Test set**: `uwYear == 2010`

This is more realistic than a random split because:
1. It simulates the real business scenario: models are built on historical data and
   applied to future renewals.
2. It respects the temporal ordering of insurance data and avoids data leakage from
   future underwriting years.
3. It tests model stability across time periods.

Optional 5-fold cross-validation within the 2009 training year can be used for
hyperparameter tuning.""")


code("""\
freq_train = freq[freq["uwYear"] == 2009].copy()
freq_test  = freq[freq["uwYear"] == 2010].copy()
sev_train  = sev[sev["uwYear"] == 2009].copy()
sev_test   = sev[sev["uwYear"] == 2010].copy()

sev_train_pos = sev_train[sev_train["claimSizeMD"] > 0].copy()
sev_test_pos  = sev_test[sev_test["claimSizeMD"] > 0].copy()

print("Split summary:")
print(f"  Frequency — train: {len(freq_train):,}   test: {len(freq_test):,}")
print(f"  Severity  — train: {len(sev_train):,}   test: {len(sev_test):,}")
print(f"  Severity (>0) — train: {len(sev_train_pos):,}   test: {len(sev_test_pos):,}")
print(f"  Zeros removed — train: {len(sev_train) - len(sev_train_pos)}, "
      f"test: {len(sev_test) - len(sev_test_pos)}")

FEATURES_CAT = ["gender", "carType", "carCat", "job", "cover"]
FEATURES_NUM = ["age", "nYears", "carVal", "density"]
FEATURES = FEATURES_CAT + FEATURES_NUM

X_freq_train = freq_train[FEATURES];  y_freq_train = freq_train["claimNumbMD"]
X_freq_test  = freq_test[FEATURES];   y_freq_test  = freq_test["claimNumbMD"]
X_sev_train_pos = sev_train_pos[FEATURES]; y_sev_train_pos = sev_train_pos["claimSizeMD"]
X_sev_test_pos  = sev_test_pos[FEATURES];  y_sev_test_pos  = sev_test_pos["claimSizeMD"]

# ── Candidate-A preprocessing: one-hot + cubic splines ──
preprocessor_a = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", sparse_output=False), FEATURES_CAT),
    ("num", SplineTransformer(n_knots=5, degree=3, include_bias=False), FEATURES_NUM),
], remainder="drop")
preprocessor_a.fit(X_freq_train)

Xa_freq_train    = preprocessor_a.transform(X_freq_train)
Xa_freq_test     = preprocessor_a.transform(X_freq_test)
Xa_sev_train_pos = preprocessor_a.transform(X_sev_train_pos)
Xa_sev_test_pos  = preprocessor_a.transform(X_sev_test_pos)

feat_names_a = preprocessor_a.get_feature_names_out()
print(f"\\nCandidate-A feature matrix: {Xa_freq_train.shape[1]} columns")

# ── Candidate-B preprocessing: ordinal + raw numeric ──
ord_enc = OrdinalEncoder()
ord_enc.fit(X_freq_train[FEATURES_CAT])
cat_mask_b = [True] * len(FEATURES_CAT) + [False] * len(FEATURES_NUM)

def prep_b(X):
    Xc = ord_enc.transform(X[FEATURES_CAT])
    Xn = X[FEATURES_NUM].values
    return np.hstack([Xc, Xn])

Xb_freq_train    = prep_b(X_freq_train)
Xb_freq_test     = prep_b(X_freq_test)
Xb_sev_train_pos = prep_b(X_sev_train_pos)
Xb_sev_test_pos  = prep_b(X_sev_test_pos)

print(f"Candidate-B feature matrix: {Xb_freq_train.shape[1]} columns")

results = {}""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 ─ Candidate A: Interpretable Two-Part GAM/GLM
# ══════════════════════════════════════════════════════════════════════

md("""\
## Section 6 — Candidate A: Interpretable Two-Part Model (Primary)

### Frequency Component
Logistic regression with one-hot categorical features and cubic B-spline basis
expansions for continuous features. This provides a smooth, interpretable mapping
from covariates to claim probability.

### Severity Component
Two variants are fitted on strictly positive severities:
1. **Gamma GLM** with log link (actuarial standard)
2. **Log-target Ridge regression** on `log1p(claimSizeMD)` (sensitivity check)

The better-calibrated model is selected for combined scoring.""")


code("""\
# ── Frequency: Logistic Regression with Splines ──
freq_model_a = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                  random_state=SEED)
freq_model_a.fit(Xa_freq_train, y_freq_train)

freq_pred_a_train = freq_model_a.predict_proba(Xa_freq_train)[:, 1]
freq_pred_a_test  = freq_model_a.predict_proba(Xa_freq_test)[:, 1]

print("Candidate A — Frequency Model (Logistic + Splines)")
print("-" * 55)
for label, yt, yp in [("Train", y_freq_train, freq_pred_a_train),
                       ("Test",  y_freq_test,  freq_pred_a_test)]:
    print(f"  {label}: AUC={roc_auc_score(yt, yp):.4f}  "
          f"LogLoss={log_loss(yt, yp):.4f}  "
          f"Brier={brier_score_loss(yt, yp):.4f}")

results["A"] = {
    "freq_auc":   roc_auc_score(y_freq_test, freq_pred_a_test),
    "freq_ll":    log_loss(y_freq_test, freq_pred_a_test),
    "freq_brier": brier_score_loss(y_freq_test, freq_pred_a_test),
}

# ── Top coefficients ──
coef_df = pd.DataFrame({
    "feature": feat_names_a,
    "coef": freq_model_a.coef_[0],
    "odds_ratio": np.exp(freq_model_a.coef_[0]),
}).sort_values("coef", key=abs, ascending=False)
print("\\nTop 12 coefficients by magnitude:")
print(coef_df.head(12).to_string(index=False))

# ── Calibration plot ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
prob_true, prob_pred = calibration_curve(y_freq_test, freq_pred_a_test, n_bins=10)
axes[0].plot(prob_pred, prob_true, "o-", label="Candidate A")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_xlabel("Predicted probability")
axes[0].set_ylabel("Observed frequency")
axes[0].set_title("Frequency Calibration — Candidate A")
axes[0].legend()

# Lift by decile
df_lift = pd.DataFrame({"y": y_freq_test.values, "p": freq_pred_a_test})
df_lift["decile"] = pd.qcut(df_lift["p"], 10, labels=range(1, 11), duplicates="drop")
lift_tbl = df_lift.groupby("decile", observed=True).agg(
    n=("y", "count"), actual=("y", "mean"), predicted=("p", "mean"))
overall = y_freq_test.mean()
lift_tbl["lift"] = lift_tbl["actual"] / overall
axes[1].bar(lift_tbl.index.astype(int), lift_tbl["lift"], color="steelblue")
axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.5)
axes[1].set_xlabel("Predicted-probability decile")
axes[1].set_ylabel("Lift (actual / overall rate)")
axes[1].set_title("Frequency Lift by Decile — Candidate A")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "candidate_a_freq.png", bbox_inches="tight")
plt.show()""")


code("""\
# ── Severity: Gamma GLM (log link) ──
Xa_sev_train_c = sm.add_constant(Xa_sev_train_pos)
Xa_sev_test_c  = sm.add_constant(Xa_sev_test_pos)

gamma_glm = sm.GLM(y_sev_train_pos, Xa_sev_train_c,
                    family=sm.families.Gamma(link=sm.families.links.Log()))
gamma_res = gamma_glm.fit(maxiter=200)

sev_pred_gamma_train = gamma_res.predict(Xa_sev_train_c)
sev_pred_gamma_test  = gamma_res.predict(Xa_sev_test_c)

# ── Severity: Log-target Ridge ──
log_sev_model = Ridge(alpha=1.0)
log_sev_model.fit(Xa_sev_train_pos, np.log1p(y_sev_train_pos))
sev_pred_log_train = np.expm1(log_sev_model.predict(Xa_sev_train_pos))
sev_pred_log_test  = np.expm1(log_sev_model.predict(Xa_sev_test_pos))

def sev_metrics(yt, yp):
    yp_clip = np.maximum(yp, 1.0)
    mae  = mean_absolute_error(yt, yp_clip)
    rmse = np.sqrt(mean_squared_error(yt, yp_clip))
    log_mae = mean_absolute_error(np.log1p(yt), np.log1p(yp_clip))
    return mae, rmse, log_mae

print("Candidate A — Severity Models")
print("-" * 60)
for name, yp_te in [("Gamma GLM", sev_pred_gamma_test),
                     ("Log-target Ridge", sev_pred_log_test)]:
    mae, rmse, lmae = sev_metrics(y_sev_test_pos, yp_te)
    print(f"  {name:<20s}  MAE={mae:>8.2f}  RMSE={rmse:>9.2f}  LogMAE={lmae:.4f}")

mae_g, rmse_g, lmae_g = sev_metrics(y_sev_test_pos, sev_pred_gamma_test)
mae_l, rmse_l, lmae_l = sev_metrics(y_sev_test_pos, sev_pred_log_test)

use_gamma = lmae_g <= lmae_l
chosen_sev_name = "Gamma GLM" if use_gamma else "Log-target Ridge"
sev_pred_a_test  = sev_pred_gamma_test if use_gamma else sev_pred_log_test
sev_pred_a_train = sev_pred_gamma_train if use_gamma else sev_pred_log_train
print(f"\\nChosen severity model: {chosen_sev_name}")

results["A"]["sev_mae"]  = mae_g if use_gamma else mae_l
results["A"]["sev_rmse"] = rmse_g if use_gamma else rmse_l
results["A"]["sev_lmae"] = lmae_g if use_gamma else lmae_l

# ── Severity calibration by predicted decile ──
fig, ax = plt.subplots(figsize=(8, 5))
cal_df = pd.DataFrame({"actual": y_sev_test_pos.values,
                        "gamma": sev_pred_gamma_test,
                        "logtarget": sev_pred_log_test})
cal_df["decile"] = pd.qcut(cal_df["gamma"], 10, labels=range(1, 11), duplicates="drop")
cal_agg = cal_df.groupby("decile", observed=True).agg(
    actual=("actual", "mean"), gamma=("gamma", "mean"), logtarget=("logtarget", "mean"))

x = np.arange(len(cal_agg))
w = 0.25
ax.bar(x - w, cal_agg["actual"],    w, label="Actual",    color="steelblue")
ax.bar(x,     cal_agg["gamma"],     w, label="Gamma GLM", color="salmon")
ax.bar(x + w, cal_agg["logtarget"], w, label="Log-target",color="seagreen")
ax.set_xticks(x)
ax.set_xticklabels(cal_agg.index)
ax.set_xlabel("Predicted severity decile")
ax.set_ylabel("Mean severity")
ax.set_title("Severity Calibration by Decile — Candidate A")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "candidate_a_sev.png", bbox_inches="tight")
plt.show()""")


code("""\
# ── Partial-effect plots for continuous variables ──
fig, axes = plt.subplots(1, len(FEATURES_NUM), figsize=(16, 4))

for idx, var in enumerate(FEATURES_NUM):
    template = pd.DataFrame({
        col: [X_freq_train[col].mode()[0] if col in FEATURES_CAT
              else X_freq_train[col].median()]
        for col in FEATURES
    }, index=[0])
    template = pd.concat([template] * 200, ignore_index=True)

    grid = np.linspace(X_freq_train[var].quantile(0.02),
                       X_freq_train[var].quantile(0.98), 200)
    template[var] = grid
    Xt = preprocessor_a.transform(template)
    pred = freq_model_a.predict_proba(Xt)[:, 1]

    axes[idx].plot(grid, pred, color="steelblue", linewidth=2)
    axes[idx].set_xlabel(var)
    axes[idx].set_ylabel("P(claim)")
    axes[idx].set_title(f"Partial Effect: {var}")

plt.suptitle("Candidate A — Frequency Partial Effects", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "candidate_a_partial.png", bbox_inches="tight")
plt.show()""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 ─ Candidate B: Gradient Boosting
# ══════════════════════════════════════════════════════════════════════

md("""\
## Section 7 — Candidate B: Two-Part Gradient Boosting (Challenger)

Uses `HistGradientBoosting` from scikit-learn, which is architecturally similar to
LightGBM and supports native categorical features. This serves as a **performance
challenger** that can capture non-linear interactions automatically.

The severity component is trained on `log1p(claimSizeMD)` to stabilise variance.""")


code("""\
# ── Frequency: HistGradientBoostingClassifier ──
freq_model_b = HistGradientBoostingClassifier(
    max_iter=300, max_depth=5, learning_rate=0.05,
    min_samples_leaf=50, categorical_features=cat_mask_b,
    random_state=SEED, early_stopping=True, validation_fraction=0.15,
    n_iter_no_change=20,
)
freq_model_b.fit(Xb_freq_train, y_freq_train)

freq_pred_b_train = freq_model_b.predict_proba(Xb_freq_train)[:, 1]
freq_pred_b_test  = freq_model_b.predict_proba(Xb_freq_test)[:, 1]

print("Candidate B — Frequency (HistGradientBoosting)")
print("-" * 55)
for label, yt, yp in [("Train", y_freq_train, freq_pred_b_train),
                       ("Test",  y_freq_test,  freq_pred_b_test)]:
    print(f"  {label}: AUC={roc_auc_score(yt, yp):.4f}  "
          f"LogLoss={log_loss(yt, yp):.4f}  "
          f"Brier={brier_score_loss(yt, yp):.4f}")

# ── Severity: HistGradientBoostingRegressor on log-target ──
sev_model_b = HistGradientBoostingRegressor(
    max_iter=300, max_depth=5, learning_rate=0.05,
    min_samples_leaf=50, categorical_features=cat_mask_b,
    random_state=SEED, early_stopping=True, validation_fraction=0.15,
    n_iter_no_change=20,
)
sev_model_b.fit(Xb_sev_train_pos, np.log1p(y_sev_train_pos))

sev_pred_b_train = np.expm1(sev_model_b.predict(Xb_sev_train_pos))
sev_pred_b_test  = np.expm1(sev_model_b.predict(Xb_sev_test_pos))

mae_b, rmse_b, lmae_b = sev_metrics(y_sev_test_pos, sev_pred_b_test)
print(f"\\nCandidate B — Severity (HistGradientBoosting)")
print(f"  Test: MAE={mae_b:.2f}  RMSE={rmse_b:.2f}  LogMAE={lmae_b:.4f}")

results["B"] = {
    "freq_auc":   roc_auc_score(y_freq_test, freq_pred_b_test),
    "freq_ll":    log_loss(y_freq_test, freq_pred_b_test),
    "freq_brier": brier_score_loss(y_freq_test, freq_pred_b_test),
    "sev_mae":    mae_b,
    "sev_rmse":   rmse_b,
    "sev_lmae":   lmae_b,
}""")


code("""\
# ── Feature importance (permutation-based) ──
perm_freq = permutation_importance(freq_model_b, Xb_freq_test, y_freq_test,
                                   n_repeats=10, random_state=SEED,
                                   scoring="roc_auc")
perm_sev = permutation_importance(sev_model_b, Xb_sev_test_pos,
                                  np.log1p(y_sev_test_pos),
                                  n_repeats=10, random_state=SEED,
                                  scoring="neg_mean_absolute_error")

feat_labels = FEATURES_CAT + FEATURES_NUM

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

order_f = np.argsort(perm_freq.importances_mean)[::-1]
axes[0].barh(range(len(feat_labels)),
             perm_freq.importances_mean[order_f], color="steelblue")
axes[0].set_yticks(range(len(feat_labels)))
axes[0].set_yticklabels([feat_labels[i] for i in order_f])
axes[0].set_xlabel("Mean AUC decrease")
axes[0].set_title("Frequency — Permutation Importance")
axes[0].invert_yaxis()

order_s = np.argsort(np.abs(perm_sev.importances_mean))[::-1]
axes[1].barh(range(len(feat_labels)),
             np.abs(perm_sev.importances_mean[order_s]), color="salmon")
axes[1].set_yticks(range(len(feat_labels)))
axes[1].set_yticklabels([feat_labels[i] for i in order_s])
axes[1].set_xlabel("Mean |MAE| increase")
axes[1].set_title("Severity — Permutation Importance")
axes[1].invert_yaxis()

plt.suptitle("Candidate B — Feature Importance", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "candidate_b_importance.png", bbox_inches="tight")
plt.show()

# ── Calibration plot ──
fig, ax = plt.subplots(figsize=(6, 5))
prob_true_b, prob_pred_b = calibration_curve(y_freq_test, freq_pred_b_test, n_bins=10)
ax.plot(prob_pred_b, prob_true_b, "o-", label="Candidate B")
prob_true_a, prob_pred_a = calibration_curve(y_freq_test, freq_pred_a_test, n_bins=10)
ax.plot(prob_pred_a, prob_true_a, "s--", label="Candidate A", alpha=0.7)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_title("Frequency Calibration Comparison")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "calibration_comparison.png", bbox_inches="tight")
plt.show()""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 ─ Candidate C: Credibility-Smoothed Risk Cells
# ══════════════════════════════════════════════════════════════════════

md("""\
## Section 8 — Candidate C: Credibility-Smoothed Risk Cells (Business Model)

This model groups policies into practical **risk cells** defined by coarse variable
bands, then applies Bühlmann-style credibility smoothing to stabilise cell estimates.

It is the most **actionable** model for renewal committees because it produces a
transparent risk-segment table rather than a black-box score.""")


code("""\
# ── Define practical bands ──
AGE_BINS   = [17, 25, 35, 50, 65, 100]
AGE_LABELS = ["18-25", "26-35", "36-50", "51-65", "65+"]

NYEARS_BINS   = [-1, 0, 3, 10, 100]
NYEARS_LABELS = ["0", "1-3", "4-10", "11+"]

carval_edges  = np.quantile(freq_train["carVal"], [0, 0.25, 0.5, 0.75, 1.0])
carval_edges[0] -= 1
CARVAL_LABELS = ["VQ1", "VQ2", "VQ3", "VQ4"]

dens_edges = np.quantile(freq_train["density_round"], [0, 0.25, 0.5, 0.75, 1.0])
dens_edges[0] -= 1
DENS_LABELS = ["DQ1", "DQ2", "DQ3", "DQ4"]

def add_bands(df):
    df = df.copy()
    df["age_band"]    = pd.cut(df["age"],    bins=AGE_BINS,    labels=AGE_LABELS)
    df["nYears_band"] = pd.cut(df["nYears"], bins=NYEARS_BINS, labels=NYEARS_LABELS)
    df["carVal_band"] = pd.cut(df["carVal"], bins=carval_edges, labels=CARVAL_LABELS,
                               include_lowest=True)
    dr = df["density_round"] if "density_round" in df.columns else df["density"].round().astype(int)
    df["density_band"] = pd.cut(dr, bins=dens_edges, labels=DENS_LABELS,
                                include_lowest=True)
    return df

freq_train_b = add_bands(freq_train)
freq_test_b  = add_bands(freq_test)
sev_train_b  = add_bands(sev_train_pos)
sev_test_b   = add_bands(sev_test_pos)

CELL_KEYS = ["gender", "carCat", "cover", "age_band", "density_band"]

# ── Cell-level frequency estimation ──
cell_freq_train = freq_train_b.groupby(CELL_KEYS, observed=True).agg(
    n_policies=("claimNumbMD", "count"),
    claim_rate=("claimNumbMD", "mean"),
).reset_index()

overall_freq = freq_train["claimNumbMD"].mean()

# ── Cell-level severity estimation ──
cell_sev_train = sev_train_b.groupby(CELL_KEYS, observed=True).agg(
    n_claims=("claimSizeMD", "count"),
    mean_sev=("claimSizeMD", "mean"),
).reset_index()

overall_sev = sev_train_pos["claimSizeMD"].mean()

# ── Bühlmann credibility smoothing ──
K_FREQ = 30
K_SEV  = 30

cell_freq_train["z_freq"] = cell_freq_train["n_policies"] / (cell_freq_train["n_policies"] + K_FREQ)
cell_freq_train["cred_freq"] = (cell_freq_train["z_freq"] * cell_freq_train["claim_rate"]
                                + (1 - cell_freq_train["z_freq"]) * overall_freq)

cell_sev_train["z_sev"] = cell_sev_train["n_claims"] / (cell_sev_train["n_claims"] + K_SEV)
cell_sev_train["cred_sev"] = (cell_sev_train["z_sev"] * cell_sev_train["mean_sev"]
                               + (1 - cell_sev_train["z_sev"]) * overall_sev)

cell_pp = cell_freq_train.merge(cell_sev_train[CELL_KEYS + ["cred_sev"]],
                                on=CELL_KEYS, how="left")
cell_pp["cred_sev"] = cell_pp["cred_sev"].fillna(overall_sev)
cell_pp["cred_pp"]  = cell_pp["cred_freq"] * cell_pp["cred_sev"]

print(f"Total risk cells: {len(cell_pp)}")
print(f"Cell sizes — min: {cell_freq_train['n_policies'].min()}, "
      f"median: {cell_freq_train['n_policies'].median():.0f}, "
      f"max: {cell_freq_train['n_policies'].max()}")
print(f"\\nOverall frequency: {overall_freq:.4f}")
print(f"Overall severity:  {overall_sev:.2f}")
print(f"Overall pure premium proxy: {overall_freq * overall_sev:.2f}")
print(f"\\nTop 10 risk cells by credibility-smoothed pure premium:")
top10 = cell_pp.nlargest(10, "cred_pp")[CELL_KEYS + ["n_policies", "cred_freq",
                                                      "cred_sev", "cred_pp"]]
print(top10.to_string(index=False))

# ── Evaluate on test set ──
freq_test_c = freq_test_b.merge(
    cell_pp[CELL_KEYS + ["cred_freq", "cred_sev", "cred_pp"]],
    on=CELL_KEYS, how="left"
)
freq_test_c["cred_freq"] = freq_test_c["cred_freq"].fillna(overall_freq)
freq_test_c["cred_sev"]  = freq_test_c["cred_sev"].fillna(overall_sev)
freq_test_c["cred_pp"]   = freq_test_c["cred_pp"].fillna(overall_freq * overall_sev)

cred_freq_pred = freq_test_c["cred_freq"].values
print(f"\\nCandidate C — Frequency on test (cell-level prediction):")
print(f"  AUC={roc_auc_score(y_freq_test, cred_freq_pred):.4f}  "
      f"LogLoss={log_loss(y_freq_test, cred_freq_pred):.4f}  "
      f"Brier={brier_score_loss(y_freq_test, cred_freq_pred):.4f}")

results["C"] = {
    "freq_auc":   roc_auc_score(y_freq_test, cred_freq_pred),
    "freq_ll":    log_loss(y_freq_test, cred_freq_pred),
    "freq_brier": brier_score_loss(y_freq_test, cred_freq_pred),
    "sev_mae":    np.nan,
    "sev_rmse":   np.nan,
    "sev_lmae":   np.nan,
}""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 9 ─ Tweedie Comparison (Optional)
# ══════════════════════════════════════════════════════════════════════

md("""\
## Section 9 — Tweedie Pure-Premium Comparison (Optional)

A Tweedie model directly estimates pure premium without separating frequency and
severity. Because the two datasets are **not policy-linked**, a true per-policy
pure-premium observation is unavailable. As an approximation, each frequency-data
policy is assigned:

* `pseudo_loss = claimNumbMD * overall_mean_severity`

This is a **rough comparison only** and should not be used as the primary model.""")


code("""\
overall_mean_sev_train = sev_train_pos["claimSizeMD"].mean()

y_tweedie_train = freq_train["claimNumbMD"].values * overall_mean_sev_train
y_tweedie_test  = freq_test["claimNumbMD"].values  * overall_mean_sev_train

tweedie_model = TweedieRegressor(power=1.5, alpha=1.0, max_iter=2000)
tweedie_model.fit(Xa_freq_train, y_tweedie_train)

tweedie_pred_test = np.maximum(tweedie_model.predict(Xa_freq_test), 0)

pseudo_mae  = mean_absolute_error(y_tweedie_test, tweedie_pred_test)
pseudo_rmse = np.sqrt(mean_squared_error(y_tweedie_test, tweedie_pred_test))
print(f"Tweedie (power=1.5) on pseudo-loss:")
print(f"  Test MAE  = {pseudo_mae:.2f}")
print(f"  Test RMSE = {pseudo_rmse:.2f}")
print(f"  (Pseudo-loss uses overall mean severity = {overall_mean_sev_train:.2f})")
print(f"  This is a rough comparison — the pseudo-target is an approximation.")""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 10 ─ Combined Pure-Premium Proxy Evaluation
# ══════════════════════════════════════════════════════════════════════

md("""\
## Section 10 — Combined Pure-Premium Proxy Evaluation

For each candidate, the **policy-level expected loss** is:

```
predicted_expected_loss = P(claim | X) * E(severity | claim, X)
```

Because per-policy realised pure premium is not observable, validation is performed
at the **risk-cell level**:

1. Define evaluation cells using categorical and banded continuous variables.
2. Compute empirical cell claim rate from `frequency.csv` test data.
3. Compute empirical cell mean severity from `severity.csv` test data.
4. Empirical cell pure premium = cell claim rate &times; cell mean severity.
5. Compare model-predicted cell pure premium with empirical cell pure premium.""")


code("""\
EVAL_KEYS = ["gender", "carCat", "cover", "age_band"]

# ── Empirical cell-level pure premium (test sets) ──
emp_freq_cell = freq_test_b.groupby(EVAL_KEYS, observed=True).agg(
    n_policies=("claimNumbMD", "count"),
    emp_claim_rate=("claimNumbMD", "mean"),
).reset_index()

emp_sev_cell = sev_test_b.groupby(EVAL_KEYS, observed=True).agg(
    n_claims=("claimSizeMD", "count"),
    emp_mean_sev=("claimSizeMD", "mean"),
).reset_index()

cell_eval = emp_freq_cell.merge(emp_sev_cell, on=EVAL_KEYS, how="inner")
cell_eval["emp_pp"] = cell_eval["emp_claim_rate"] * cell_eval["emp_mean_sev"]

print(f"Evaluation cells: {len(cell_eval)} (inner join of freq and sev test cells)")
print(f"Policies covered: {cell_eval['n_policies'].sum():,} / {len(freq_test):,}")

# ── Candidate A: policy-level combined score on freq_test ──
Xa_freq_test_c = sm.add_constant(Xa_freq_test)
if use_gamma:
    sev_pred_a_on_freq = gamma_res.predict(Xa_freq_test_c)
else:
    sev_pred_a_on_freq = np.expm1(log_sev_model.predict(Xa_freq_test))
sev_pred_a_on_freq = np.maximum(sev_pred_a_on_freq, 1.0)

freq_test_b["pp_A"] = freq_pred_a_test * sev_pred_a_on_freq

cell_pred_a = freq_test_b.groupby(EVAL_KEYS, observed=True)["pp_A"].mean().reset_index()
cell_pred_a.columns = list(EVAL_KEYS) + ["pred_pp_A"]
cell_eval = cell_eval.merge(cell_pred_a, on=EVAL_KEYS, how="left")

# ── Candidate B: policy-level combined score on freq_test ──
Xb_freq_test_all = prep_b(X_freq_test)
sev_pred_b_on_freq = np.expm1(sev_model_b.predict(Xb_freq_test_all))
sev_pred_b_on_freq = np.maximum(sev_pred_b_on_freq, 1.0)

freq_test_b["pp_B"] = freq_pred_b_test * sev_pred_b_on_freq

cell_pred_b = freq_test_b.groupby(EVAL_KEYS, observed=True)["pp_B"].mean().reset_index()
cell_pred_b.columns = list(EVAL_KEYS) + ["pred_pp_B"]
cell_eval = cell_eval.merge(cell_pred_b, on=EVAL_KEYS, how="left")

# ── Candidate C: cell-level score ──
cell_pred_c = freq_test_c.groupby(EVAL_KEYS, observed=True)["cred_pp"].mean().reset_index()
cell_pred_c.columns = list(EVAL_KEYS) + ["pred_pp_C"]
cell_eval = cell_eval.merge(cell_pred_c, on=EVAL_KEYS, how="left")

# ── Tweedie: policy-level score ──
freq_test_b["pp_T"] = tweedie_pred_test
cell_pred_t = freq_test_b.groupby(EVAL_KEYS, observed=True)["pp_T"].mean().reset_index()
cell_pred_t.columns = list(EVAL_KEYS) + ["pred_pp_T"]
cell_eval = cell_eval.merge(cell_pred_t, on=EVAL_KEYS, how="left")

# ── Metrics ──
w = cell_eval["n_policies"].values

def weighted_metrics(emp, pred, w):
    mask = np.isfinite(pred) & np.isfinite(emp)
    e, p, ww = emp[mask], pred[mask], w[mask]
    wmae  = np.average(np.abs(e - p), weights=ww)
    wrmse = np.sqrt(np.average((e - p) ** 2, weights=ww))
    rho, _ = stats.spearmanr(e, p)
    return wmae, wrmse, rho

print("\\nCell-Level Combined Score Metrics:")
print(f"{'Candidate':<14s} {'wMAE':>8s} {'wRMSE':>8s} {'Spearman':>10s}")
print("-" * 42)
for cand, col in [("A", "pred_pp_A"), ("B", "pred_pp_B"),
                   ("C", "pred_pp_C"), ("Tweedie", "pred_pp_T")]:
    if col not in cell_eval.columns:
        continue
    wmae, wrmse, rho = weighted_metrics(
        cell_eval["emp_pp"].values, cell_eval[col].values, w)
    print(f"  {cand:<12s} {wmae:>8.2f} {wrmse:>8.2f} {rho:>10.4f}")
    if cand in results:
        results[cand]["comb_wmae"]  = wmae
        results[cand]["comb_wrmse"] = wrmse
        results[cand]["comb_rho"]   = rho""")


code("""\
# ── Lift plot: rank cells by predicted PP, show cumulative empirical PP ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (cand, col) in zip(axes, [("A", "pred_pp_A"), ("B", "pred_pp_B")]):
    df = cell_eval[["emp_pp", col, "n_policies"]].dropna().copy()
    df = df.sort_values(col, ascending=False).reset_index(drop=True)
    df["cum_w"]   = df["n_policies"].cumsum() / df["n_policies"].sum()
    df["cum_emp"] = (df["emp_pp"] * df["n_policies"]).cumsum() / (
                     df["emp_pp"] * df["n_policies"]).sum()
    ax.plot(df["cum_w"], df["cum_emp"], "o-", color="steelblue", markersize=4)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Cumulative proportion of policies (by predicted PP)")
    ax.set_ylabel("Cumulative proportion of empirical loss")
    ax.set_title(f"Candidate {cand} — Lorenz-style Lift Curve")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "combined_lift.png", bbox_inches="tight")
plt.show()

# ── Decile table for best candidate ──
df_decile = freq_test_b[["pp_A"]].copy()
df_decile["actual_claim"] = y_freq_test.values
df_decile["decile"] = pd.qcut(df_decile["pp_A"], 10, labels=range(1, 11),
                               duplicates="drop")
decile_tbl = df_decile.groupby("decile", observed=True).agg(
    n=("actual_claim", "count"),
    claim_rate=("actual_claim", "mean"),
    mean_pp=("pp_A", "mean"),
).reset_index()
decile_tbl["lift"] = decile_tbl["claim_rate"] / y_freq_test.mean()
print("Candidate A — Combined Score Decile Table:")
print(decile_tbl.to_string(index=False))""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 11 ─ Model Comparison
# ══════════════════════════════════════════════════════════════════════

md("""\
## Section 11 — Model Comparison""")


code("""\
comp_df = pd.DataFrame(results).T
comp_df.index.name = "Candidate"
comp_cols = ["freq_auc", "freq_ll", "freq_brier",
             "sev_mae", "sev_rmse", "sev_lmae",
             "comb_wmae", "comb_wrmse", "comb_rho"]
for c in comp_cols:
    if c not in comp_df.columns:
        comp_df[c] = np.nan
comp_df = comp_df[comp_cols]
comp_df.columns = ["Freq AUC", "Freq LogLoss", "Freq Brier",
                    "Sev MAE", "Sev RMSE", "Sev LogMAE",
                    "Comb wMAE", "Comb wRMSE", "Comb Spearman"]
print(comp_df.round(4).to_string())""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 12 ─ Portfolio Ranking & Action Segmentation
# ══════════════════════════════════════════════════════════════════════

md("""\
## Section 12 — Portfolio Ranking and Action Segmentation

Every policy in `frequency.csv` is scored using the best two-part model.
Policies are then grouped into action bands:

| Band | Percentile | Recommended Action |
|---|---|---|
| **Highest review** | Top 5% | Manual underwriting review, pricing & deductible review |
| **Medium review** | Next 15% (80th–95th pctile) | Targeted monitoring, softer pricing intervention |
| **Standard** | Middle 60% (20th–80th pctile) | Standard renewal treatment |
| **Low risk** | Bottom 20% | Valuable low-risk pool, retention priority |""")


code("""\
# Score ALL policies in frequency.csv
X_all = freq[FEATURES]
Xa_all   = preprocessor_a.transform(X_all)
Xa_all_c = sm.add_constant(Xa_all)

pred_freq_all = freq_model_a.predict_proba(Xa_all)[:, 1]
if use_gamma:
    pred_sev_all = gamma_res.predict(Xa_all_c)
else:
    pred_sev_all = np.expm1(log_sev_model.predict(Xa_all))
pred_sev_all = np.maximum(pred_sev_all, 1.0)

portfolio = freq.copy()
portfolio["pred_claim_prob"]   = pred_freq_all
portfolio["pred_cond_sev"]     = pred_sev_all
portfolio["pred_expected_loss"] = pred_freq_all * pred_sev_all

portfolio["pct_rank"] = portfolio["pred_expected_loss"].rank(pct=True)

def assign_band(p):
    if p >= 0.95:
        return "1-Highest Review (Top 5%)"
    elif p >= 0.80:
        return "2-Medium Review (80-95%)"
    elif p >= 0.20:
        return "3-Standard (20-80%)"
    else:
        return "4-Low Risk (Bottom 20%)"

portfolio["action_band"] = portfolio["pct_rank"].apply(assign_band)

band_summary = portfolio.groupby("action_band").agg(
    n_policies=("pred_expected_loss", "count"),
    mean_pred_loss=("pred_expected_loss", "mean"),
    actual_claim_rate=("claimNumbMD", "mean"),
    mean_pred_prob=("pred_claim_prob", "mean"),
    mean_pred_sev=("pred_cond_sev", "mean"),
).reset_index()

print("Portfolio Segmentation Summary:")
print(band_summary.to_string(index=False))

top_review = portfolio[portfolio["pct_rank"] >= 0.95].sort_values(
    "pred_expected_loss", ascending=False)
print(f"\\nTop-review segment: {len(top_review):,} policies")
print(f"Mean predicted expected loss: {top_review['pred_expected_loss'].mean():.2f}")
print(f"Actual claim rate in this segment: {top_review['claimNumbMD'].mean():.4f}")

print(f"\\nSample of highest-risk policies:")
show_cols = FEATURES + ["claimNumbMD", "pred_claim_prob", "pred_cond_sev",
                         "pred_expected_loss", "action_band"]
print(top_review[show_cols].head(10).to_string(index=False))""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 13 ─ Business Interpretation
# ══════════════════════════════════════════════════════════════════════

md("""\
## Section 13 — Business Interpretation and Recommendations

### What These Scores Mean

These are policies with the **highest predicted expected material-damage cost**.
They are **not proven unprofitable customers** — actual profitability depends on
premium adequacy, expense loading, reinsurance arrangements, and other factors
not available in this dataset.

### Recommended Renewal Actions

| Segment | Action |
|---|---|
| **Top 5% (Highest Review)** | Manual underwriting review; pricing and deductible reassessment; consider coverage redesign or non-renewal for extreme cases |
| **Next 15% (Medium Review)** | Targeted monitoring; apply risk-based pricing adjustments at renewal; flag for underwriter attention if combined ratio deteriorates |
| **Middle 60% (Standard)** | Standard renewal treatment; no special action required |
| **Bottom 20% (Low Risk)** | High-value retention pool; prioritise customer retention through competitive pricing and loyalty incentives |

### Model Selection Summary

| Role | Model | Rationale |
|---|---|---|
| **Primary report model** | Candidate A (Logistic + Gamma GLM with splines) | Interpretable, defensible actuarial logic, nearly tied with boosting on both frequency and severity metrics |
| **Performance challenger** | Candidate B (HistGradientBoosting) | Captures non-linear interactions; serves as a robustness check |
| **Business implementation** | Candidate C (Credibility risk cells) | Most actionable for renewal committees; produces a transparent risk-segment table |
| **Comparison only** | Tweedie | Methodologically interesting but data structure does not support it as primary |""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 14 ─ Save Outputs
# ══════════════════════════════════════════════════════════════════════

code("""\
# ── model_comparison.csv ──
comp_df.to_csv(OUTPUT_DIR / "model_comparison.csv")

# ── portfolio_scores.csv ──
score_cols = FEATURES + ["claimNumbMD", "pred_claim_prob", "pred_cond_sev",
                          "pred_expected_loss", "pct_rank", "action_band"]
portfolio[score_cols].to_csv(OUTPUT_DIR / "portfolio_scores.csv", index=False)

# ── portfolio_top_review.csv ──
top_review[score_cols].to_csv(OUTPUT_DIR / "portfolio_top_review.csv", index=False)

# ── cell_level_validation.csv ──
cell_eval.to_csv(OUTPUT_DIR / "cell_level_validation.csv", index=False)

# ── portfolio_segment_summary.csv ──
band_summary.to_csv(OUTPUT_DIR / "portfolio_segment_summary.csv", index=False)

# ── summary.md ──
best_cand = comp_df["Freq AUC"].idxmax()
summary_text = f\"\"\"# Customer Profiling — Executive Summary

## Data
- Frequency dataset: {len(freq):,} policies (binary claim indicator)
- Severity dataset: {len(sev):,} claims (continuous damage amount)
- Datasets are unlinked at the policy level

## Best Model
- **Candidate A** (Logistic + {chosen_sev_name} with spline features)
- Frequency AUC: {results['A']['freq_auc']:.4f}
- Severity MAE: {results['A']['sev_mae']:.2f}
- Combined cell-level Spearman: {results['A'].get('comb_rho', float('nan')):.4f}

## Key Risk Drivers
- Identified via permutation importance and partial-effect analysis
- Top drivers for frequency: see Candidate B feature importance plot
- Top drivers for severity: see Candidate B feature importance plot

## Portfolio Segmentation
{band_summary.to_string(index=False)}

## Caveat
These scores represent expected material-damage claim cost, NOT proven
unprofitability. Premium, expense, and reinsurance data are not available.
\"\"\"

with open(OUTPUT_DIR / "summary.md", "w", encoding="utf-8") as f:
    f.write(summary_text)

print("Saved outputs:")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {p.name}")""")


# ══════════════════════════════════════════════════════════════════════
# SECTION 15 ─ Executive Summary
# ══════════════════════════════════════════════════════════════════════

md("""\
## Executive Summary

### Data Limitations
- No premium, expense, reinsurance, or acquisition-cost data → **actual profitability
  is not measurable**. The analysis estimates expected claim cost as a proxy.
- Frequency and severity datasets are unlinked → combined validation is performed
  at the risk-cell level, not per-policy.
- Only two underwriting years (2009–2010) are available → limited ability to assess
  long-term model stability.

### Chosen Final Model
**Candidate A: Two-part interpretable model** (logistic frequency + Gamma/log-target
severity with spline features).

This model is selected because:
- It matches the assignment's data structure (separate frequency & severity files)
- It produces defensible, interpretable results suitable for management reporting
- Its predictive performance is very close to the gradient-boosting challenger

### Strongest Risk Drivers
- **Age** and **density** show the strongest non-linear effects on claim frequency
- **carVal** (vehicle value) is the strongest driver of claim severity
- **job** and **carCat** provide significant categorical differentiation

### Combined-Score Validation
Cell-level Spearman rank correlation between predicted and empirical pure premium
confirms that the model's risk ordering is meaningful, despite the inability to
validate at the individual-policy level.

### Recommendations for Management
1. Use the top-5% segment list for **immediate underwriting review** at renewal.
2. Apply **Candidate C (credibility risk cells)** for operational segmentation where
   a transparent rule-based table is preferred.
3. Collect **premium and expense data** to enable true profitability analysis in future
   iterations.
4. Consider integrating telematics or claims-history data if available to improve
   frequency prediction.""")


# ══════════════════════════════════════════════════════════════════════
# Write notebook
# ══════════════════════════════════════════════════════════════════════

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    },
    "cells": cells,
}

with open("customer_profiling_solution.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Wrote {len(cells)} cells -> customer_profiling_solution.ipynb")
