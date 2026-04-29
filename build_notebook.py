"""Generate the customer-profiling notebook programmatically.

Keeping the notebook in code-form makes it easy to regenerate when the
analysis evolves. Run with: uv run python build_notebook.py
"""
from __future__ import annotations
import nbformat as nbf
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "notebook" / "customer_profiling.ipynb"

nb = nbf.v4.new_notebook()
cells = []

def md(txt: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(txt.strip()))

def code(txt: str) -> None:
    cells.append(nbf.v4.new_code_cell(txt.strip()))

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
# Customer Profiling — Identifying Unprofitable Policyholders

**Course:** Statistical Consulting (KU Leuven)
**Approach:** Frequency-severity GLM decomposition with risk-tier profiling

We model claim probability and claim cost separately, then multiply to get
the **pure premium** — the expected loss per policy. Customers whose pure
premium far exceeds the portfolio average are the "unprofitable" profiles
management wants to identify. All reusable logic lives in `src/`; this
notebook focuses on narrative and results.

$$\widehat{\text{Pure Premium}}(\mathbf{x})
   = \underbrace{\hat{p}(\mathbf{x})}_{\text{frequency}}
   \;\times\;
   \underbrace{\hat{\mu}(\mathbf{x})}_{\text{severity}}$$
""")

code(r"""
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import (
    data_loader, eda, preprocessing,
    frequency_model, severity_model,
    pure_premium, profiling, plots,
    style,
)
from src.config import FREQ_TARGET, SEV_TARGET

# Activate the Anthropic warm palette globally
style.apply_warm_theme()

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
## 1. Data Loading & Exploration

We have two datasets from the same car-insurance portfolio (2009–2010) that
**cannot be linked at the individual level**:

| Dataset | Records | Target | Description |
|---|---|---|---|
| `frequency.csv` | 24,774 | `claimNumbMD` (0/1) | Did the policy have a claim? |
| `severity.csv`  | 12,256 | `claimSizeMD` (EUR) | Total claim cost (claims only) |
""")

# ── 1.1 Load ─────────────────────────────────────────────────────────────────
md("### 1.1 Loading and schema checks")
code(r"""
freq_df = data_loader.load_frequency()
sev_df  = data_loader.load_severity()

print(data_loader.quick_summary(freq_df, "frequency"))
print(data_loader.quick_summary(sev_df,  "severity"))
freq_df.head()
""")

# ── 1.2 EDA frequency ───────────────────────────────────────────────────────
md(r"""
### 1.2 Frequency EDA

Key checks: class balance, mean claim-rate per factor level, and numeric
covariate trends. The goal is to anticipate which covariates will be
significant before we fit the model.
""")

code(r"""
print(f"Overall claim rate: {freq_df[FREQ_TARGET].mean():.3f}")
print(freq_df[FREQ_TARGET].value_counts().rename("count").to_string())
""")

code(r"""
freq_factor_table = eda.factor_target_table(freq_df, FREQ_TARGET)
freq_factor_table
""")

code(r"""
fig = eda.plot_categorical_target(freq_df, FREQ_TARGET); plt.show()
""")

code(r"""
fig = eda.plot_numeric_target(freq_df, FREQ_TARGET); plt.show()
""")

code(r"""
fig = eda.correlation_heatmap(freq_df); plt.show()
""")

# ── 1.3 EDA severity ────────────────────────────────────────────────────────
md(r"""
### 1.3 Severity EDA

Severity data is **conditional on a claim** — only policies that had at
least one accident appear here. The cost distribution is heavily
right-skewed, so we use a log axis.
""")

code(r"""
print(sev_df[SEV_TARGET].describe().round(1).to_string())
""")

code(r"""
fig = eda.plot_target_distribution(sev_df, SEV_TARGET, log_scale=True); plt.show()
""")

code(r"""
sev_factor_table = eda.factor_target_table(sev_df, SEV_TARGET)
sev_factor_table
""")

code(r"""
fig = eda.plot_categorical_target(sev_df, SEV_TARGET); plt.show()
""")

code(r"""
fig = eda.plot_numeric_target(sev_df, SEV_TARGET); plt.show()
""")

md(r"""
**EDA takeaways:**

- **Frequency** is roughly balanced (~50% claim rate). Age, density, and
  job are strong movers. Retired drivers claim much less often; unemployed
  and young drivers claim more.
- **Severity** has a median around 563 EUR but a long right tail (max ~13k).
  Retired and unemployed drivers have the highest average cost when they
  *do* claim — the opposite pattern to frequency for retirees.
- Numeric trends are smooth enough that the GLM linear-in-link assumption
  should hold as a first cut.
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODELING
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
## 2. Frequency-Severity Models

We fit each component separately using `statsmodels` GLMs (coefficient
p-values and CIs out of the box). 80/20 train-test splits for evaluation.
""")

# ── 2.1 Frequency ────────────────────────────────────────────────────────────
md(r"""
### 2.1 Frequency — Logistic GLM (Bernoulli, logit link)

The target is binary (claim yes/no), so a logistic regression is the
natural fit. We stratify the split on the target to preserve class balance.
""")

code(r"""
X_tr, X_te, y_tr, y_te = preprocessing.split_xy(
    freq_df, FREQ_TARGET, stratify_target=True,
)
print(f"train: {X_tr.shape}  |  test: {X_te.shape}")

freq_fit = frequency_model.fit_logit(X_tr, y_tr)
print(freq_fit.summary().tables[0])
""")

code(r"""
freq_metrics = frequency_model.evaluate_frequency(freq_fit, X_te, y_te)
pd.Series(freq_metrics, name="frequency").round(4)
""")

code(r"""
freq_coef = frequency_model.coefficient_table(freq_fit)
freq_coef.sort_values("p_value").round(4)
""")

code(r"""
y_pred_freq = frequency_model.predict_proba(freq_fit, X_te)
fig1 = plots.plot_roc(y_te, y_pred_freq); plt.show()
fig2 = plots.plot_calibration(y_te, y_pred_freq); plt.show()
""")

md(r"""
**Frequency takeaways:** AUC ~0.69, well-calibrated (predicted mean
tracks observed rate). Significant drivers: `age` (older → fewer claims),
`density` (urban → more claims), `job_Retired` (OR ~0.51), `carType_E`
(OR ~1.44), `gender_Male` (OR ~1.28).
""")

# ── 2.2 Severity ─────────────────────────────────────────────────────────────
md(r"""
### 2.2 Severity — Gamma GLM (log link)

Claim costs are positive and right-skewed; the Gamma family with log link
is the standard actuarial choice. The log link gives multiplicative effects
that combine naturally with the frequency model.
""")

code(r"""
Xs_tr, Xs_te, ys_tr, ys_te = preprocessing.split_xy(sev_df, SEV_TARGET)
print(f"train: {Xs_tr.shape}  |  test: {Xs_te.shape}")

sev_fit = severity_model.fit_gamma_glm(Xs_tr, ys_tr)
print(sev_fit.summary().tables[0])
""")

code(r"""
sev_metrics = severity_model.evaluate_severity(sev_fit, Xs_te, ys_te)
pd.Series(sev_metrics, name="severity").round(4)
""")

code(r"""
sev_coef = severity_model.coefficient_table(sev_fit)
sev_coef.sort_values("p_value").round(4)
""")

code(r"""
y_pred_sev = severity_model.predict_mean(sev_fit, Xs_te)
fig = plots.plot_severity_diagnostics(ys_te, y_pred_sev); plt.show()
""")

md(r"""
**Severity takeaways:** Mean predicted cost tracks the observed mean
(balanced). Individual predictions are noisy (as expected — claim size
variability is inherent). Notable cross-effect: `carType_E` *increases*
frequency but *decreases* severity; retirees claim *rarely* but *expensively*.
This is exactly why the two-model decomposition matters.
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PURE PREMIUM
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
## 3. Pure Premium & Portfolio Calibration

We score every policy in the frequency dataset (the full portfolio) to get
$\widehat{\text{PP}} = \hat{p} \times \hat{\mu}$, then run a
**balance-property check**: does the predicted total loss approximate the
observed total in the severity dataset?
""")

# ── 3.1 Compute ──────────────────────────────────────────────────────────────
md("### 3.1 Computing per-policy pure premium")
code(r"""
X_freq_full = preprocessing.build_design_matrix(freq_df)
X_sev_full  = preprocessing.build_design_matrix(sev_df)
sev_columns = list(X_sev_full.columns)

pp_df = pure_premium.compute_pure_premium(
    X_freq_full, freq_fit, sev_fit, sev_columns=sev_columns,
)
pp_df.describe().round(2)
""")

# ── 3.2 Calibration ─────────────────────────────────────────────────────────
md("### 3.2 Balance-property check")
code(r"""
calib = pure_premium.calibration_check(
    pp_df, observed_total_loss=sev_df[SEV_TARGET].sum(),
)
pd.Series(calib).round(2)
""")

# ── 3.3 Distribution & concentration ─────────────────────────────────────────
md("### 3.3 Loss distribution and concentration")
code(r"""
fig = plots.plot_pure_premium_distribution(pp_df); plt.show()
""")

code(r"""
fig = plots.plot_lorenz(pp_df); plt.show()
""")

md(r"""
**Calibration & concentration:** The predicted-to-observed ratio is ~1.0 —
the combined model is well-balanced at portfolio level. The Lorenz curve
shows meaningful risk concentration: a small share of policies drives a
disproportionate share of expected loss.
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PROFILING & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
## 4. Customer Profiling & Recommendations

We define four risk tiers by pure-premium quantile:

| Tier | Quantile range | Interpretation |
|---|---|---|
| Low | bottom 50% | Profitable, retain |
| Medium | 50–80% | Average, monitor |
| High | 80–95% | Above average, consider repricing |
| **Very High** | **top 5%** | **Unprofitable, action required** |
""")

# ── 4.1 Tier summary ────────────────────────────────────────────────────────
md("### 4.1 Risk-tier summary")
code(r"""
tiers = profiling.assign_tiers(pp_df)
tier_tbl = profiling.tier_summary(freq_df, pp_df, tiers)
tier_tbl.round(2)
""")

code(r"""
fig = plots.plot_tier_bars(tier_tbl); plt.show()
""")

# ── 4.2 Unprofitable profile ────────────────────────────────────────────────
md("### 4.2 Profile of the unprofitable segment")
code(r"""
top_profile = profiling.describe_top_segment(
    freq_df, pp_df, tiers, segment_label="Very High",
)
top_profile
""")

# ── 4.3 Factor breakdown ────────────────────────────────────────────────────
md("### 4.3 Factor-level breakdown across tiers")
code(r"""
for f in ["gender", "carType", "carCat", "job", "cover", "uwYear"]:
    print(f"\n--- {f} (% of each tier) ---")
    print(profiling.tier_factor_profile(freq_df, tiers, f))
""")

# ── 4.4 Recommendations ─────────────────────────────────────────────────────
md(r"""
### 4.4 Recommendations to management

> **Definition used:** A customer profile is *unprofitable* if its predicted
> pure premium significantly exceeds the portfolio average. Without premium
> data, this is the cleanest pure-risk proxy available.

**Actions at renewal:**

1. **Reprice the Very High tier** — their expected loss is ~2.4x the book
   average. Current pricing almost certainly doesn't reflect that.
2. **Adjust coverage on selected profiles** — raise deductibles or restrict
   MD cover for the worst micro-segments (young, urban, unemployed males).
3. **Monitor the High tier** — not the worst offenders individually, but at
   15% of the portfolio a small repricing compounds to significant savings.

**Caveats:**

- No premium data → "unprofitable" is relative, not absolute.
- Frequency and severity are modelled on disjoint datasets; we assume the
  same covariate relationships hold in unseen no-claim records.
- A gradient-boosting robustness check would strengthen conclusions.
""")

# ══════════════════════════════════════════════════════════════════════════════
nb["cells"] = cells
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NB_PATH)
print(f"Notebook written: {NB_PATH}")
