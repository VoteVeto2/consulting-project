# Customer Profiling v4 -- Results Summary

## 1. Objective

Identify motor-insurance policyholders with high expected material-damage claim cost as a proxy for unprofitability at renewal. Actual profitability is unmeasurable because the data contains no premium, expense, reinsurance, or acquisition-cost information. The deliverable is a per-policy "expected claim cost" score: P(claim) x E(severity | claim).

## 2. Data

| Dataset | Rows | Target | Description |
|---|---|---|---|
| `frequency.csv` | 24,774 | `claimNumbMD` (binary) | Policy-level claim indicator |
| `severity.csv` | 12,256 | `claimSizeMD` (continuous) | Claim-level damage amount |

Key data characteristics:

- **No missing values** in either dataset.
- `claimNumbMD` has mean ~0.50 -- the frequency task is near-balanced binary classification, so class imbalance is not a concern.
- `claimSizeMD` is heavily right-skewed (mean ~866, median much lower) with 4 zero-severity claims excluded for Gamma-family models. Log-transformation or a Gamma distribution is appropriate.
- `density` is `float64` in frequency but `int64` in severity. A rounded integer helper (`density_round`) was created for grouped matching.
- Covariates: `gender`, `carType`, `carCat`, `job`, `cover` (categorical); `age`, `nYears`, `carVal`, `density` (continuous); `uwYear` (used as the temporal split variable).
- The datasets **cannot be linked** at the individual policy level. However, covariate distributions between the positive-claim subset of `frequency.csv` and `severity.csv` are near-identical (proportions differ by <0.001 for all categorical levels; continuous quantiles match exactly). This confirms the severity data can be treated as a sample from *severity | claim = 1*, and the two-part modelling approach is valid.

## 3. EDA Highlights

- **Frequency**: `job`, `carCat`, and `carType` show the largest variation in claim rate among categorical features. Among continuous features, `age` and `density` show the strongest non-linear associations with claim frequency (highest binned claim-rate spread).
- **Severity**: `carVal` (vehicle value) has a moderate positive association with severity -- more expensive cars produce higher claims. `carCat` and `carType` also contribute significant variation in mean severity.

## 4. Validation Design

- **Temporal holdout**: train on `uwYear = 2009`, test on `uwYear = 2010`. This simulates real business conditions (models built on historical data, applied to future renewals), respects temporal ordering, and avoids data leakage.
- **Combined validation**: performed at **risk-cell level** (gender x carCat x cover x age_band), yielding 60 evaluation cells covering all 12,587 test policies. Cell-level evaluation is necessary because the unlinked datasets prevent per-policy ground-truth pure premium calculation.

## 5. Candidate Models

### 5.1 Candidate A -- Interpretable Two-Part Model (Primary)

**Frequency**: Logistic regression with one-hot encoded categorical features and cubic B-spline basis expansions for continuous variables. Provides smooth, interpretable partial effects.

| Metric | Train | Test |
|---|---|---|
| AUC | 0.6962 | **0.6938** |
| LogLoss | 0.6305 | 0.6339 |
| Brier | 0.2203 | 0.2218 |

Minimal train-test gap indicates no significant overfitting.

Top coefficient drivers (by magnitude):
- `age_sp_1`: coef = +1.61 (odds ratio 5.0) -- young age is the dominant frequency driver
- `job_Retired`: coef = -1.00 (OR 0.37) -- retired policyholders have substantially lower claim probability
- `density_sp_1`: coef = -0.88 (OR 0.42) -- non-linear density effect
- `nYears_sp_0`: coef = +0.78 (OR 2.17) -- newer policies have higher claim probability

**Severity**: Two variants fitted on strictly positive severities:
1. Gamma GLM with log link (actuarial standard)
2. Log-target Ridge regression on `log1p(claimSizeMD)` (sensitivity check)

The better-calibrated variant was selected for combined scoring.

| Metric | Test |
|---|---|
| MAE | **631.53** |
| RMSE | **1059.69** |
| LogMAE | 0.9791 |

### 5.2 Candidate B -- Two-Part Gradient Boosting (Challenger)

**Frequency**: `HistGradientBoostingClassifier` (scikit-learn) with `max_iter=300`, `max_depth=5`, `learning_rate=0.05`, `min_samples_leaf=50`, early stopping. Supports native categorical features.

| Metric | Train | Test |
|---|---|---|
| AUC | 0.7301 | 0.6914 |
| LogLoss | 0.6074 | 0.6355 |
| Brier | 0.2101 | 0.2225 |

Larger train-test gap (AUC 0.73 vs 0.69) compared to Candidate A, suggesting mild overfitting despite regularisation and early stopping.

**Severity**: `HistGradientBoostingRegressor` trained on `log1p(claimSizeMD)` to stabilise variance.

| Metric | Test |
|---|---|
| MAE | 632.93 |
| RMSE | 1060.70 |
| LogMAE | 0.9816 |

**Permutation importance** identified:
- Frequency top drivers: `age`, `density`, `carType`, `job`
- Severity top drivers: `carVal`, `carCat`, `carType`

### 5.3 Candidate C -- Credibility-Smoothed Risk Cells (Business Model)

Policies grouped into practical risk cells defined by binned covariates (gender x carCat x cover x age_band x density_band), then Buhlmann credibility smoothing applied (K=30 for both frequency and severity) to stabilise small-cell estimates.

- Total risk cells: 239; cell sizes range from 1 to 164 (median 43)
- Overall portfolio frequency: 0.4842, severity: 838.45, pure premium: 405.98
- Highest-risk cells: young males with small/medium/large cars, high-density areas, no comprehensive cover -- predicted pure premiums up to ~964

| Metric | Test |
|---|---|
| Freq AUC | 0.6533 |
| Freq LogLoss | 0.6579 |

Frequency discrimination is lower than A/B because cell-level predictions are coarser. However, this is the most **transparent and actionable** model for renewal committees.

### 5.4 Tweedie (Optional Comparison)

A Tweedie GLM (power=1.5) was fitted directly on a pseudo-loss target (`claimNumbMD * overall_mean_severity`). This is methodologically interesting but the data structure (unlinked files, pseudo-target approximation) does not support it as a primary model.

## 6. Combined Pure-Premium Evaluation (Cell-Level)

All candidates produce a per-policy expected loss score. These are aggregated to risk cells and compared against empirical cell-level pure premium (claim_rate x mean_severity on the test sets).

| Candidate | wMAE | wRMSE | Spearman |
|---|---|---|---|
| **A** | 223.14 | 258.46 | **0.8700** |
| B | 221.67 | 259.45 | **0.8729** |
| C | 79.41 | 119.93 | 0.7351 |
| Tweedie | 104.17 | 168.11 | 0.8521 |

Interpretation:
- **Spearman rank correlation** is the most important metric here: it measures whether the model correctly orders risk cells from cheapest to most expensive. Both A and B achieve ~0.87 -- strong monotonic agreement with empirical losses.
- Candidate C has lower wMAE/wRMSE because credibility-smoothed cell estimates are naturally closer to cell-level empirical averages (they are fitted at that granularity), but its Spearman correlation is weaker (0.74), indicating coarser discrimination.
- Candidate A essentially matches B on all combined metrics, while being fully interpretable.

## 7. Full Model Comparison

| Candidate | Freq AUC | Freq LogLoss | Freq Brier | Sev MAE | Sev RMSE | Sev LogMAE | Comb wMAE | Comb wRMSE | Cell Spearman |
|---|---|---|---|---|---|---|---|---|---|
| **A** | **0.6938** | **0.6339** | **0.2218** | **631.53** | **1059.69** | **0.9791** | 223.14 | 258.46 | 0.8700 |
| B | 0.6914 | 0.6355 | 0.2225 | 632.93 | 1060.70 | 0.9816 | 221.67 | 259.45 | **0.8729** |
| C | 0.6533 | 0.6579 | 0.2328 | -- | -- | -- | 79.41 | 119.93 | 0.7351 |

Candidate A wins or ties on every frequency and severity metric. On the combined score, A and B are statistically indistinguishable. A is the recommended model because interpretability provides defensible actuarial logic for management.

## 8. Portfolio Segmentation (Candidate A, Full Portfolio)

All 24,774 policies in `frequency.csv` are scored with Candidate A's combined model. Policies are ranked by predicted expected loss and segmented into action bands:

| Segment | Policies | Mean Predicted Loss | Actual Claim Rate | Mean P(claim) | Mean E(sev) |
|---|---|---|---|---|---|
| Top 5% (Highest Review) | 1,239 | 596.04 | 0.808 | 0.791 | 753.29 |
| Next 15% (Medium Review) | 3,716 | 405.04 | 0.703 | 0.696 | 582.17 |
| Middle 60% (Standard) | 14,865 | 198.68 | 0.483 | 0.475 | 423.79 |
| Bottom 20% (Low Risk) | 4,954 | 91.91 | 0.298 | 0.280 | 356.31 |

Key observations:
- The top-5% segment has **2.7x the claim rate** (0.81 vs 0.30) and **6.5x the predicted loss** (596 vs 92) of the bottom 20%.
- Predicted probabilities closely track actual claim rates across all bands, confirming good calibration.
- The severity gradient is less steep than the frequency gradient (753 vs 356, or ~2.1x), meaning frequency is the dominant driver of loss variation across segments.

## 9. Strongest Risk Drivers

**Frequency** (claim probability):
- **Age**: youngest drivers (18-25) have the highest claim probability; strong non-linear effect via spline coefficients (odds ratio up to 5.0 for the first spline basis)
- **Density**: non-linear; intermediate densities can lower claim probability
- **Job**: retired policyholders significantly less likely to claim (OR 0.37); unemployed slightly more likely (OR 1.26)
- **nYears**: newer policyholders (low tenure) have higher claim probability

**Severity** (claim cost given claim):
- **carVal**: dominant driver; higher vehicle values produce higher claim costs
- **carCat** and **carType**: meaningful categorical differentiation in mean severity

## 10. Recommended Renewal Actions

| Segment | Action |
|---|---|
| **Top 5% (Highest Review)** | Manual underwriting review; pricing and deductible reassessment; consider coverage redesign or non-renewal for extreme cases |
| **Next 15% (Medium Review)** | Targeted monitoring; apply risk-based pricing adjustments at renewal; flag for underwriter attention if combined ratio deteriorates |
| **Middle 60% (Standard)** | Standard renewal treatment; no special action required |
| **Bottom 20% (Low Risk)** | High-value retention pool; prioritise customer retention through competitive pricing and loyalty incentives |

## 11. Model Roles

| Role | Model | Rationale |
|---|---|---|
| **Primary report model** | Candidate A (Logistic + Gamma/Log-target Ridge with splines) | Interpretable, defensible actuarial logic, nearly tied with boosting on all metrics |
| **Performance challenger** | Candidate B (HistGradientBoosting) | Captures non-linear interactions automatically; serves as a robustness check confirming A's adequacy |
| **Business implementation** | Candidate C (Credibility risk cells) | Most actionable for renewal committees; produces a transparent risk-segment table rather than a per-policy score |
| **Comparison only** | Tweedie | Methodologically interesting but data structure (unlinked files, pseudo-target) does not support it as primary |

## 12. Caveats

- Scores represent **expected material-damage claim cost**, not proven unprofitability. Actual profitability depends on premium adequacy, expense loading, reinsurance arrangements, and other factors unavailable in this dataset.
- Only **two underwriting years** (2009--2010) are available. Long-term model stability cannot be assessed.
- Combined validation is **cell-level only** -- no per-policy ground truth for pure premium exists due to unlinked datasets.
- Candidate B shows mild overfitting (train AUC 0.73 vs test 0.69) despite regularisation; this reinforces the preference for the more parsimonious Candidate A.
- Collecting **premium and expense data** would enable true profitability analysis in future iterations. Integrating telematics or claims-history data could improve frequency prediction.
