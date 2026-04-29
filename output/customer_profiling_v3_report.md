# Customer Profiling for Renewal Action — Analysis Report (v3)

**Course:** Statistical Consulting (KU Leuven)
**Date:** April 2026
**Analysis:** `notebook/customer_profiling_v3.ipynb`

---

## 1. Executive Summary

We estimated the **expected technical loss** for each of the 24,774 policies
in a Belgian motor-insurance portfolio using a two-part actuarial pricing
model. The primary model — a logistic GLM for claim frequency and a Gamma
GLM for claim severity — is well-calibrated (predicted/observed total loss
ratio = 1.00), stable across underwriting years, and confirmed by a
gradient-boosting machine-learning challenger.

**Key result:** the top 5 % of policies ("Very High" tier) account for
12 % of total expected loss at 2.4x the portfolio average. These are
predominantly young (~23 yr), male, unemployed policyholders in urban areas
without additional MD cover.

Because premium data are unavailable, we cannot measure true profitability.
Instead, we identify the customers with the **highest expected loss** —
those most likely to be underpriced at current rates. This is the standard
actuarial proxy for commercial risk.

---

## 2. Methodology

### 2.1 Problem framing

The assignment asks to identify "unprofitable" customers for renewal action.
Two constraints shape the approach:

1. **No premium data** — profitability requires revenue; we only have cost.
   The deliverable is therefore a *technical risk ranking*, not a literal
   profit/loss statement.
2. **No individual linkage** — frequency and severity datasets share
   covariates but cannot be joined at the policy level. We model each
   component separately and multiply.

### 2.2 Model structure

$$\widehat{\text{Expected Loss}}(x)
   = \underbrace{\hat{p}(x)}_{\text{P(claim | x)}}
   \times
   \underbrace{\hat{s}(x)}_{\text{E[cost | claim, x]}}$$

| Component | Model | Family / Link | Target |
|---|---|---|---|
| Frequency | Logistic GLM | Binomial / logit | `claimNumbMD` (0/1) |
| Severity (primary) | Gamma GLM | Gamma / log | `claimSizeMD` (EUR) |
| Severity (challenger) | Lognormal OLS | Normal / identity on log scale | log(`claimSizeMD`) |
| ML challenger (both) | Gradient Boosting | 300 trees, depth 4 | Same targets |

### 2.3 Validation strategy

| Check | Method | Result |
|---|---|---|
| Out-of-sample accuracy | 80/20 train-test split | Freq AUC = 0.69, Sev MAE = 634 EUR |
| Portfolio calibration | Predicted vs observed total loss | Ratio = 1.00 |
| Cell-level calibration | 90 common risk cells (job x carType x age band) | Correlation = 0.91 |
| Year stability | Train 2009 -> test 2010, and vice versa | Freq AUC stable at 0.685-0.687 |
| Gender sensitivity | Refit without gender | Freq AUC drops 0.003; Sev MAE +1.2 EUR |
| ML challenger | Gradient-boosted trees | Freq AUC = 0.70; ranking Spearman = 0.88 |

---

## 3. Data Audit

### 3.1 Key observations

| Question | Finding |
|---|---|
| Is this the full portfolio? | Unlikely — 49.6 % claim rate suggests balanced/sampled data. Rankings valid; absolute probabilities are relative. |
| Claimant count mismatch? | 12,277 claimants in frequency vs 12,252 in severity (gap = 0.2 %). Immaterial. |
| Covariate alignment? | Max categorical difference = 0.1 ppt; numeric means match to 0.1. No reweighting needed. |
| Cover = 0 structural zeros? | Cover = 0 policies still claim at 53.2 %. Not structural zeros; all included. |

### 3.2 Data summary

| Dataset | Records | Target | Description |
|---|---|---|---|
| `frequency.csv` | 24,774 | `claimNumbMD` (0/1) | Full portfolio — did the policy have a claim? |
| `severity.csv` | 12,252 | `claimSizeMD` (EUR) | Claims only — total claim cost |

**Covariates (10):** `uwYear` (2009/2010), `gender`, `carType` (A–E),
`carCat` (Small/Medium/Large), `job` (5 levels), `cover` (0/1), `age`,
`nYears`, `carVal`, `density`.

---

## 4. Model Results

### 4.1 Frequency model — significant drivers

| Variable | Odds Ratio | Direction | Interpretation |
|---|---|---|---|
| `age` | 0.97 per year | Older = fewer claims | Strongest continuous driver |
| `density` | 1.004 per unit | Urban = more claims | Second-strongest |
| `job_Retired` | 0.51 | Much fewer claims | Retirees are low-frequency risks |
| `carType_E` | 1.44 | More claims | High-performance / larger cars |
| `gender_Male` | 1.28 | More claims | Males claim more often |
| `job_Unemployed` | 1.24 | More claims | Higher social risk |
| `nYears` | 0.98 per year | Longer tenure = fewer claims | Loyalty effect |
| `cover_1` | 0.91 | Slightly fewer claims | MD cover policyholders claim less |

### 4.2 Severity model — significant drivers

| Variable | Multiplicative Effect | Direction | Interpretation |
|---|---|---|---|
| `job_Retired` | 1.92x | Much costlier | Rare but expensive when they do claim |
| `job_Unemployed` | 1.25x | Costlier | Higher average claim |
| `carType_E` | 0.79x | Cheaper claims | *Opposite* to frequency — key cross-effect |
| `carType_D` | 0.81x | Cheaper claims | Similar pattern |
| `gender_Male` | 1.15x | Costlier | Males have higher average claim cost |
| `age` | 0.99 per year | Older = cheaper | Opposite to retired-job effect |
| `density` | 1.002 per unit | Urban = costlier | Consistent with frequency |
| `cover_1` | 0.88x | Cheaper | MD-covered policies cost less per claim |

**Critical cross-effect:** `carType_E` increases frequency but *decreases*
severity. Retirees claim rarely but expensively. This is why the two-part
decomposition matters — a single combined model would obscure these
opposing patterns.

### 4.3 Model comparison

**Frequency:**

| Model | AUC | Brier Score | Log Loss |
|---|---|---|---|
| Logistic GLM | 0.689 | 0.223 | 0.636 |
| Gradient Boosting | **0.704** | **0.218** | **0.626** |

**Severity:**

| Model | MAE (EUR) | RMSE (EUR) | Rank Correlation |
|---|---|---|---|
| Gamma GLM | **634** | 907 | -0.01 |
| Lognormal OLS | 634 | 908 | -0.04 |
| Gradient Boosting | 638 | **907** | **0.22** |

The GLM is chosen as primary for transparency and tariff compatibility.
The GBM confirms that the same variables dominate (age, density, job) and
that the GLM ranking is robust (Spearman correlation between GLM and GBM
pure premiums = 0.88).

---

## 5. Portfolio Scoring

### 5.1 Expected loss distribution

| Statistic | Expected Loss (EUR) | Risk Index |
|---|---|---|
| Mean | 428 | 1.00 |
| Median | 384 | 0.90 |
| Std dev | 232 | 0.54 |
| Min | 42 | 0.10 |
| Max | 1,675 | 3.91 |

- **5.4 % of policies** have a Risk Index above 2.0 (double the average).
- **Gini coefficient:** 0.296 — moderate risk concentration.
- **Lorenz curve:** the bottom 50 % of policies contribute ~29 % of expected
  loss; the top 5 % contribute ~12 %.

### 5.2 Risk-tier summary

| Tier | Quantile | N | Mean EL (EUR) | Risk Index | % of Portfolio | % of Loss |
|---|---|---|---|---|---|---|
| Low | Bottom 50 % | 12,387 | 249 | 0.58x | 50 % | 29 % |
| Medium | 50–80th | 7,432 | 483 | 1.13x | 30 % | 34 % |
| High | 80–95th | 3,716 | 715 | 1.67x | 15 % | 25 % |
| **Very High** | **Top 5 %** | **1,239** | **1,029** | **2.4x** | **5 %** | **12 %** |

---

## 6. Customer Profiles

### 6.1 Frequency-severity quadrant decomposition

Each policy is classified by whether predicted frequency and severity are
above or below the portfolio median:

| Quadrant | N | % Portfolio | % Loss | Mean EL | Mean Risk Index |
|---|---|---|---|---|---|
| Low freq / Low sev | 8,459 | 34 % | 19 % | 234 | 0.55 |
| High freq / Low sev | 3,928 | 16 % | 15 % | 403 | 0.94 |
| Low freq / High sev | 3,928 | 16 % | 12 % | 323 | 0.76 |
| **High freq / High sev** | **8,459** | **34 %** | **54 %** | **682** | **1.59** |

The High freq / High sev quadrant contains 34 % of policies but drives
**54 % of expected loss**. Nearly all Very High tier policies (99.8 %) fall
in this quadrant.

### 6.2 Very High tier profile (top 5 %)

| Attribute | Value |
|---|---|
| Policies | 1,239 |
| Mean expected loss | 1,029 EUR (2.4x average) |
| Typical age | 23 years |
| Gender | Male (85 %) |
| Job | Unemployed (63 %) |
| Car type | A (36 %) |
| Car category | Small (43 %) |
| Cover | 0 — no MD cover (88 %) |
| Density | 237 (high urban density) |

### 6.3 Surrogate decision tree — simplified risk rules

A depth-3 decision tree trained to identify the top-10 % risk policies
produces the following primary rule:

> **Highest-risk leaf:** density > 207 AND age <= 30 AND cover = 0
>
> These are young drivers in high-density urban areas without MD cover.

Secondary splits further distinguish by unemployment status and specific
age thresholds. The tree provides a simple, actionable screening rule for
underwriting review.

---

## 7. Renewal Action Matrix

### 7.1 Quadrant-based actions

| Quadrant | Risk Pattern | Recommended Action | Priority |
|---|---|---|---|
| **High freq / High sev** | Many costly claims | Strong repricing (+30–50 %), stricter underwriting, possible non-renewal | **Immediate** |
| **High freq / Low sev** | Many small claims | Moderate repricing, raise deductible, repair-network steering | High |
| **Low freq / High sev** | Rare but costly | Underwriting review, coverage/limit audit, car valuation check | Medium |
| **Low freq / Low sev** | Attractive risk | Retain — standard renewal, possible loyalty discount | Standard |

### 7.2 Tier-based actions

| Risk Group | Share | Risk Index | Primary Action |
|---|---|---|---|
| Very High (top 5 %) | 5 % | 2.4x | Strong repricing, strict UW, non-renewal review |
| High (80–95th pctl) | 15 % | 1.7x | Moderate repricing, deductible adjustment |
| Medium (50–80th) | 30 % | 1.1x | Monitor, selective repricing |
| Low (bottom 50 %) | 50 % | 0.6x | Retain, standard renewal |

### 7.3 Specific recommendations

1. **Immediate repricing of the Very High tier** — 1,239 policies at 2.4x
   average expected loss. Current pricing almost certainly does not reflect
   this risk level.

2. **Deductible and coverage adjustment for High freq / Low sev policies** —
   these generate many small claims. Raising deductibles and steering to
   partner repair networks can reduce claim costs without losing the customer.

3. **Underwriting review for Low freq / High sev policies** — rare but
   costly claims suggest coverage limits or car valuations may be inadequate.
   Retired policyholders are over-represented here.

4. **Retain and protect the Low tier** — half the portfolio at below-average
   risk. Consider loyalty pricing to prevent adverse selection.

---

## 8. Robustness & Caveats

### 8.1 What the analysis can and cannot say

| Can say | Cannot say |
|---|---|
| Which policies have the highest expected loss | Which policies are truly unprofitable (need premium) |
| How risk decomposes into frequency and severity | Whether frequency and severity are residually dependent |
| That the ranking is stable across years and models | That absolute probability levels are correct (likely sampled) |

### 8.2 Year stability

Models trained on 2009 predict 2010 with essentially the same accuracy as
the reverse direction (AUC 0.685–0.687). No evidence of temporal drift in
the risk structure.

### 8.3 Gender and EU law

The Test-Achats ruling (2012) requires unisex pricing for new EU contracts.
Removing gender reduces frequency AUC by only 0.003 and severity MAE by
1.2 EUR. **The gender-free model is fully viable for compliant deployment.**
Gender was used here for diagnostic purposes only.

### 8.4 Severity distribution

Gamma GLM and Lognormal OLS produce nearly identical predictions (MAE
difference < 1 EUR). Results are not sensitive to the distributional
assumption. Both underpredict in the extreme tail, but the *ranking* is
what matters for profiling.

### 8.5 Limitations

- **No premium data** — expected loss != profitability. Adding premium data
  would enable loss-ratio analysis and sharpen the actionable output.
- **No individual linkage** — residual frequency-severity dependence beyond
  shared covariates cannot be captured.
- **~50 % claim rate** — if the dataset is balanced/sampled, absolute
  probabilities need recalibration before production deployment.
- **Severity tail** — GLMs systematically underpredict extreme claims. The
  GBM challenger partially addresses this (rank corr = 0.22 vs ~0 for GLMs).

---

## 9. Technical Appendix

### 9.1 Software

- Python 3.13, statsmodels 0.14+, scikit-learn 1.5+, pandas 2.2+, numpy 2.0+
- All reusable logic in `src/`; analysis in `notebook/customer_profiling_v3.ipynb`

### 9.2 Reproducibility

- Random state fixed at 42 for all splits and models
- 80/20 stratified train-test split for frequency; simple split for severity
- Full-data refit used for final portfolio scoring

### 9.3 Files

| File | Description |
|---|---|
| `notebook/customer_profiling_v3.ipynb` | Full analysis notebook with all code and plots |
| `build_notebook_v3.py` | Programmatic notebook generator |
| `src/` | Reusable source modules (data loading, modeling, profiling, plotting) |
| `dataset/frequency.csv` | Frequency data (24,774 policies) |
| `dataset/severity.csv` | Severity data (12,252 claims) |
| `output/customer_profiling_v3_report.md` | This report |
