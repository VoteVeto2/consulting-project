# Practical Plan: Insurance Customer Profiling

## 1. Literature Foundations

### Core Concept: Frequency-Severity Decomposition

The standard actuarial framework decomposes the expected loss (pure premium) per policy as:

$$\text{Pure Premium} = \mathbb{E}[N] \times \mathbb{E}[X \mid X > 0]$$

where $N$ is the claim count and $X$ is the claim size given a claim occurred. Frequency and severity are modeled separately on their respective datasets, then multiplied to obtain a per-profile expected cost.

### Key References

| Area | Reference | Why it matters |
|---|---|---|
| Frequency-severity framework | Klugman, Panjer & Willmot (2019). *Loss Models: From Data to Decisions*, 5th ed. Wiley | Standard textbook establishing $\mathbb{E}[L] = \mathbb{E}[N] \times \mathbb{E}[X]$ |
| GLMs for insurance | de Jong & Heller (2008). *Generalized Linear Models for Insurance Data*. Cambridge UP | Go-to practical GLM reference for insurance |
| Insurance pricing with GLMs | Ohlsson & Johansson (2010). *Non-Life Insurance Pricing with GLMs*. Springer | Highly practical, insurance-specific |
| Claim count modeling | Denuit, Maréchal, Pitrebois & Walhin (2007). *Actuarial Modelling of Claim Counts*. Wiley | Definitive treatment of frequency models, hurdle/ZIP models |
| Combining component models | Frees, Derrig & Meyers (2014). *Predictive Modeling Applications in Actuarial Science*, Vol. 1. Cambridge UP | Chapters on combining separate frequency and severity models |
| Ratemaking standard | Werner & Modlin (2016). *Basic Ratemaking*, 5th ed. CAS | Industry-standard reference on multiplicative models |
| GLM theory | McCullagh & Nelder (1989). *Generalized Linear Models*, 2nd ed. Chapman & Hall | Canonical GLM reference |
| Boosting vs. GLM benchmark | Noll, Salzmann & Wüthrich (2020). "Case Study: French Motor Third-Party Liability Claims." SSRN 3164764 | Benchmarks GLM vs. gradient boosting on similar data |
| SHAP interpretability | Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS | Standard for model-agnostic explanations |
| Micro-level reserving (KU Leuven) | Antonio & Plat (2014). "Micro-level stochastic loss reserving." *Scand. Actuar. J.* | Seminal individual-level claims modeling paper |
| Hierarchical claims modeling | Frees & Valdez (2008). "Hierarchical Insurance Claims Modeling." *JASA*, 103(484) | Multi-level modeling for policyholder profiling |

---

## 2. The "Unprofitable" Problem — Scoping the Question

We have no premium data. We cannot directly compute a loss ratio. So before picking a method, we must define what "unprofitable" means in this context.

### Our working definition

> A customer profile is **unprofitable** if its predicted pure premium significantly exceeds the portfolio average.

**Justification:** In a competitive market, premiums converge toward average risk. Customers whose expected loss is well above average are likely underpriced relative to their risk, hence unprofitable. This is a standard actuarial proxy when actual premium data is unavailable.

**Important assumption to state explicitly:** We are assuming the current premium structure is approximately uniform or at least does not perfectly price for risk. If the insurer already risk-prices accurately, high-risk customers may not be unprofitable. We flag this limitation.

---

## 3. Overall Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐
│   EDA &     │     │   Model     │     │  Profile &          │
│   Cleaning  │ ──> │   Building  │ ──> │  Recommend          │
└─────────────┘     └─────────────┘     └─────────────────────┘
                          │
               ┌──────────┴──────────┐
               │                     │
        ┌──────┴──────┐       ┌──────┴──────┐
        │  Frequency  │       │  Severity   │
        │  Model      │       │  Model      │
        │ (24,774)    │       │ (12,256)    │
        └──────┬──────┘       └──────┬──────┘
               │                     │
               └──────────┬──────────┘
                          │
                   Pure Premium
                 = freq × sev
```

### Phase 1: EDA & Data Preparation

- **Frequency data:** Check the claim rate distribution (~49.6% claim rate — unusually high, worth noting). Tabulate all factor variables against the target. Look for class imbalances within covariate levels.
- **Severity data:** Plot claim size distribution (expect heavy right skew). Check for outliers. Consider log-transform for visualization. Examine mean/variance by covariate level.
- **Both datasets:** Check for missing values, verify covariate coding, examine correlations, check for multicollinearity (e.g., `carType` vs. `carCat`, `age` vs. `nYears`).
- **The `cover` variable:** Understand what it means — material damage cover included or not. This likely affects *what* can be claimed and *how much*, so it is crucial in both models.

### Phase 2: Model Building (details per solution below)

### Phase 3: Combine & Profile

- Compute predicted pure premium per covariate profile: $\hat{\pi} = \hat{p}(\text{claim}) \times \hat{\mu}(\text{severity})$
- Rank profiles by pure premium relative to portfolio average
- Segment into risk tiers (e.g., Low / Medium / High / Very High)
- Identify the covariate patterns that drive each tier

### Phase 4: Deliverable to Management

- Risk profile table: which customer characteristics drive high expected loss
- Ranked visualization (e.g., Lorenz curve / cumulative loss by customer decile)
- Concrete recommended actions per segment (reprice, restrict coverage, monitor)
- Honest caveats about what we cannot conclude without premium data

---

## 4. Solution A (Recommended): GLM Frequency-Severity

This is the industry-standard approach, well-understood by actuaries and regulators, interpretable, and directly maps to the business question.

### Frequency Model

| Aspect | Choice | Rationale |
|---|---|---|
| Distribution | Logistic regression (Bernoulli) since target is binary 0/1 | `claimNumbMD` is binary, not a count |
| Link function | Logit | Standard for binary outcomes |
| Covariates | All 10, with possible interactions | Start full, reduce by significance/AIC |
| Validation | 80/20 train-test split; metrics: AUC, log-loss, calibration plot | |

> **Note:** Although classical frequency models use Poisson for claim counts, our target is binary (at least one claim: yes/no), so logistic regression is appropriate here. If we had actual counts (0, 1, 2, 3, ...) we would use Poisson/NB.

### Severity Model

| Aspect | Choice | Rationale |
|---|---|---|
| Distribution | Gamma GLM | Standard for positive, right-skewed, continuous cost data; variance proportional to mean² |
| Link function | Log | Ensures positive predictions; multiplicative structure on covariates |
| Covariates | All 10 | Same covariate set; start full, reduce by AIC/BIC |
| Alternatives to test | Lognormal (OLS on log-cost), Inverse Gaussian | Compare via AIC, QQ-plots, residual diagnostics |
| Validation | 80/20 train-test split; metrics: MAE, RMSE on original scale, deviance | |

### Combining

For any covariate profile $\mathbf{x}$:

$$\widehat{\text{Pure Premium}}(\mathbf{x}) = \hat{p}(\mathbf{x}) \times \hat{\mu}(\mathbf{x})$$

where $\hat{p}$ comes from the frequency model and $\hat{\mu}$ from the severity model.

**Calibration check (balance property):** The sum of predicted pure premiums across the frequency dataset should approximate the total observed claims in the severity dataset. This is a sanity check, not a formal test, but deviations signal model issues.

### Profiling

- Compute $\widehat{\text{Pure Premium}}(\mathbf{x})$ for every unique covariate profile (or for every row in the frequency dataset)
- Rank by pure premium; define thresholds for unprofitability (e.g., top 20%, or > 150% of portfolio mean)
- Report the coefficient table: which variables increase/decrease expected loss, with confidence intervals
- Create customer profiles: "A young male driver with car type E, living in a high-density area, with a new policy, is predicted to cost X EUR/year — Y% above average"

### Strengths of Solution A

- Industry standard, trusted by actuaries and regulators
- Fully interpretable coefficients (multiplicative effects on log scale)
- Small number of parameters, low overfitting risk
- Direct mapping to actuarial concepts (relativities, rating factors)

### Weaknesses of Solution A

- Assumes linear effects (on link scale) and no interactions unless explicitly added
- May miss complex nonlinear patterns
- Requires manual interaction/spline specification

---

## 5. Solution B (Alternative): Gradient Boosting (XGBoost)

A modern ML approach that captures nonlinearities and interactions automatically, at the cost of interpretability.

### Frequency Model

| Aspect | Choice |
|---|---|
| Algorithm | XGBoost with `binary:logistic` objective |
| Features | All 10 covariates; categorical variables one-hot encoded or native categorical handling |
| Hyperparameter tuning | 5-fold CV on learning rate, max depth, min child weight, subsample |
| Metrics | AUC, log-loss |

### Severity Model

| Aspect | Choice |
|---|---|
| Algorithm | XGBoost with `reg:gamma` objective (Gamma deviance loss) |
| Features | Same 10 covariates |
| Tuning | Same CV strategy |
| Metrics | MAE, RMSE, Gamma deviance |

### Combining & Profiling

Same multiplication approach as Solution A: $\hat{\pi} = \hat{p} \times \hat{\mu}$.

**Interpretability recovery** using SHAP values:
- SHAP summary plot: global ranking of which features drive pure premium
- SHAP dependence plots: show nonlinear effects (e.g., how does risk change with age?)
- SHAP interaction values: detect important two-way effects the GLM would miss
- Use SHAP-based clustering to define customer segments

### Strengths of Solution B

- Captures nonlinearities and interactions without manual specification
- Often better predictive accuracy on tabular data
- SHAP provides post-hoc interpretability

### Weaknesses of Solution B

- Less transparent than GLM to actuarial/regulatory audiences ("black box" concern)
- Risk of overfitting if not carefully tuned
- Not standard in insurance regulatory filings
- Harder to extract simple "rating factor" tables for management

---

## 6. Solution C (Alternative): Two-Stage Clustering Approach

A segmentation-first approach that directly answers "which customer groups are unprofitable?" rather than scoring individual policies.

### Step 1: Build frequency and severity models (GLM or XGBoost — either works)

Use Solution A or B to get predicted pure premium per covariate profile.

### Step 2: Cluster on risk-driving features

- Use **k-prototypes** (Huang, 1998) for mixed numerical/categorical covariates, or encode categoricals and use k-means
- Input features: the 10 covariates + predicted pure premium as an additional feature
- Choose $k$ via silhouette score or elbow method (try $k = 3$ to $7$)

### Step 3: Profile each cluster

- For each cluster, report: average predicted pure premium, size (% of portfolio), dominant covariate values
- Label clusters as "Low Risk", "Medium Risk", "High Risk", etc.
- Compare cluster average pure premium to portfolio average to identify unprofitable segments

### Step 4: Management deliverable

Instead of individual scores, deliver **named customer segments** with clear descriptions:
- *"Segment 3: Young urban drivers (age < 25, density > 200, car type D/E) — 12% of portfolio, 180% of average expected cost → recommend premium increase or coverage restriction at renewal"*

### Strengths of Solution C

- Produces immediately actionable segments rather than continuous scores
- Easier for management to understand and act on
- Natural grouping aligns with how insurers typically structure their portfolios

### Weaknesses of Solution C

- Clustering is sensitive to $k$ and feature scaling choices
- Loses granularity compared to individual-level scoring
- Segments may not align with natural risk boundaries

---

## 7. Recommended Approach for the Exam

**Use Solution A (GLM) as the primary analysis, supplemented with elements of B and C.**

| Phase | What to do | Why |
|---|---|---|
| EDA | Thorough exploration of both datasets | Shows you understand the data before modeling |
| Primary model | GLM frequency + GLM severity | Defensible, interpretable, industry standard |
| Robustness check | Fit XGBoost, compare predictions to GLM | Shows awareness of alternatives; if results agree, strengthens confidence |
| Profiling | Compute pure premium profiles + segment into risk tiers | Directly answers the business question |
| Communication | Coefficient interpretation + customer segment descriptions + visualization | Shows you can translate statistics → business |
| Caveats | State the no-premium limitation, the independence assumption, the `cover` variable interpretation | Shows maturity and honesty |

### What the examiner likely wants to see

1. **Scoping:** You recognized that "unprofitable" needs defining, and you justified your proxy
2. **Method choice:** You chose GLM with clear reasoning, not because it's the only thing you know
3. **Two-model structure:** You correctly modeled frequency and severity separately and combined them
4. **Diagnostics:** You checked model assumptions (distribution fits, residual plots, calibration)
5. **Business translation:** You turned statistical output into actionable customer profiles
6. **Honesty:** You stated what you cannot conclude and where assumptions are fragile

---

## 8. Pitfalls to Avoid

- **Don't model only frequency or only severity.** A customer who rarely claims but claims huge amounts is just as unprofitable as one who claims often for small amounts.
- **Don't treat severity data as representative of all customers.** It is conditional on a claim having occurred — this is by design, not a data error.
- **Don't overfit severity on small subgroups.** Some covariate combinations will have very few claims; regularize or pool categories.
- **Don't ignore the `cover` variable.** It likely affects both whether a claim is made and how large it is. Understand what it represents before modeling.
- **Don't deliver a model without a business narrative.** The examiner (echoing the lecture) values communication as much as methodology.

---

## 9. Tools & Implementation

| Task | Recommended tool |
|---|---|
| EDA & visualization | Python (`pandas`, `matplotlib`, `seaborn`) or R (`ggplot2`, `dplyr`) |
| GLM fitting | R (`glm()`) or Python (`statsmodels`) |
| XGBoost | Python (`xgboost`) or R (`xgboost`) |
| SHAP values | Python (`shap` library) |
| Clustering | Python (`scikit-learn`, `kmodes` for k-prototypes) |
| Report | R Markdown, Jupyter Notebook, or Quarto |
