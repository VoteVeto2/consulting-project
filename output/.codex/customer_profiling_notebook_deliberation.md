# Customer Profiling Notebook Deliberation

Date reviewed: 2026-05-03

Files reviewed:

- `task/Customer_profiling_exam_assignment.pdf`
- `dataset/frequency.csv`
- `dataset/severity.csv`
- `notebook/customer_profiling_v2.ipynb`
- `notebook/customer_profiling_v3.ipynb`
- `notebook/customer_profiling_v4.ipynb`
- `notebook/V2_RESULTS_SUMMARY.md`
- `notebook/V3_RESULTS_SUMMARY.md`
- `notebook/V4_RESULTS_SUMMARY.md`

## Verdict

**Best current answer: `notebook/customer_profiling_v3.ipynb`.**

v3 is the strongest submission as-is because it best matches the exam's actual evaluation criteria: it asks the right business questions, uses a defensible frequency-severity model for unlinked data, validates the combined score at portfolio and risk-cell level, and translates the statistical output into renewal actions that management can understand.

My ranking:

1. **v3 - best current submission**
2. **v2 - solid but less complete**
3. **v4 - promising structure, but not safe as-is because the expected-loss scale is under-calibrated**

v4 could become the best version after fixes, but I would not submit it ahead of v3 in its current state.

## What The Assignment Actually Requires

The PDF is intentionally open-ended. It says there is no prescribed method and no single correct answer. The work is judged on whether the analyst:

- asks the right questions,
- makes defensible choices,
- explains the reasoning clearly.

The key technical constraint is that `frequency.csv` and `severity.csv` share covariates but cannot be linked at individual policyholder level. That makes a direct per-policy observed loss ratio impossible. A good answer therefore needs to:

- define "unprofitable" carefully because no premium data is available,
- model claim probability and conditional claim severity separately,
- combine them as `P(claim | x) x E(claim cost | claim, x)`,
- validate the result at aggregate or cell level,
- communicate renewal actions without pretending that expected loss equals true profit.

## Dataset Checks

Raw dataset facts:

| Dataset | Rows | Target | Key check |
|---|---:|---|---|
| `frequency.csv` | 24,774 | `claimNumbMD` | No missing values; claim rate = 49.56% |
| `severity.csv` | 12,256 | `claimSizeMD` | No missing values; 4 zero-cost rows; positive rows = 12,252 |

Additional observations:

- Raw severity total claim cost is **10,615,726 EUR**.
- Observed average claim cost proxy over the frequency portfolio is **428.50 EUR** per policy: `10,615,726 / 24,774`.
- The near-50% claim rate is unusually high for real motor insurance, so the notebooks are right to warn that absolute probabilities may be sampled or balanced and should be treated carefully.
- The positive-claim subset of `frequency.csv` aligns closely with `severity.csv` on the shared covariates, which supports the two-part modelling approach.

## Version Comparison

| Criterion | v2 | v3 | v4 |
|---|---|---|---|
| Frames "unprofitable" despite no premium data | Good | **Best** | Good |
| Handles unlinked frequency/severity structure | Good | **Best** | Good |
| Data audit | Basic | **Strong** | Strong |
| Primary modelling choice | Logistic GLM + Gamma GLM | **Logistic GLM + Gamma GLM, with challengers** | Logistic + splines, selected log-target severity |
| Challenger models | GBM | **Lognormal severity + GBM** | GBM, credibility cells, Tweedie comparison |
| Combined calibration | Balance ratio about 1.00 | **Balance ratio about 1.00, common-cell correlation 0.907** | Cell Spearman about 0.87, but expected-loss level is under-calibrated |
| Robustness checks | Bootstrap CIs | **Year stability, gender sensitivity, tail diagnostics, GBM challenger** | Temporal holdout, cell-level validation |
| Renewal actions | Tier action matrix | **Tier + frequency/severity quadrant action matrix** | Segment actions, less diagnostic decomposition |
| Submission readiness | Good | **Best** | Needs correction before submission |

## v2 Assessment

v2 is a credible frequency-severity playbook. It uses the right architecture, scores the portfolio, builds risk tiers, includes a GBM benchmark, and adds bootstrap uncertainty intervals. It also reports a calibrated pure premium mean around **428 EUR**, matching the empirical portfolio-level loss proxy.

Main limitations:

- The business framing is less explicit than v3.
- It does not include v3's common-cell calibration, year-stability analysis, gender sensitivity check, or severity distribution challenger.
- Renewal actions are useful, but less differentiated by whether the risk comes from frequency, severity, or both.

v2 is defensible, but v3 is a clear improvement.

## v3 Assessment

v3 is the best current notebook.

Strengths:

- It explicitly states that true profitability cannot be measured without premium data.
- It uses the right two-part structure for unlinked data: logistic frequency model plus Gamma severity model.
- It checks whether the high claim rate suggests sampling or balancing.
- It checks the small claimant count mismatch: 12,277 frequency claimants vs 12,252 positive severity rows.
- It verifies covariate alignment between frequency claimants and severity records.
- It tests whether `cover = 0` is a structural zero issue and correctly keeps those rows.
- It validates the combined expected-loss score:
  - mean pure premium around **428 EUR**,
  - predicted/observed total loss ratio about **1.00**,
  - common-cell calibration correlation about **0.907**.
- It includes robustness checks:
  - year stability,
  - gender sensitivity,
  - Gamma vs lognormal severity comparison,
  - GBM challenger,
  - severity tail diagnostics.
- It gives management-ready output:
  - top 5% risk tier,
  - risk index,
  - profile card,
  - surrogate decision tree,
  - frequency/severity quadrant action matrix.

Weaknesses to polish before final submission:

- The main train/test split is random, although year-stability is checked later. A final report could present the year-stability check more prominently.
- The primary model uses gender, then notes that a gender-free model is viable. For an EU deployment narrative, the gender-free model should be the recommended operational version and gender should remain diagnostic.
- The wording around "unprofitable" and non-renewal should stay cautious because no premium data is available.
- The notebook should explicitly state that `severity.csv` has 12,256 raw rows but 12,252 positive rows after dropping four zero-cost rows for Gamma modelling.

These are polish issues, not structural failures.

## v4 Assessment

v4 has good ideas and a cleaner exam-report structure, but I would not submit it as-is.

Strengths:

- It starts with the business caveat: expected claim cost is only a proxy for unprofitability.
- It uses a temporal holdout: train on 2009, test on 2010. That is a strong validation design for renewal work.
- It compares multiple candidate approaches:
  - interpretable two-part model,
  - gradient boosting,
  - credibility-smoothed risk cells,
  - Tweedie pseudo-loss comparison.
- It validates the combined score at risk-cell level because per-policy pure premium is not observable.
- Its top-risk segment is directionally plausible: young, high predicted claim probability, high review priority.

Main issue:

**The selected expected-loss scale is materially under-calibrated.**

The output `data/output/portfolio_scores.csv` shows:

- mean predicted expected loss = **228.16 EUR**,
- total predicted expected loss = **5,652,410 EUR**,
- observed severity total = **10,615,726 EUR**,
- empirical portfolio loss proxy = **428.50 EUR** per policy.

So v4's selected portfolio score is only about **53%** of the empirical total loss level. That is a serious problem for a notebook whose deliverable is expected material-damage claim cost.

The likely cause is the selected log-target severity model:

- v4 trains on `log1p(claimSizeMD)`,
- back-transforms with `expm1`,
- but does not apply a smearing or bias correction.

That creates a predictable downward bias on the original EUR scale. v3's lognormal challenger explicitly uses Duan's smearing factor; v4 does not.

Other v4 concerns:

- The markdown says the primary report model is "Logistic + Gamma GLM with splines", but the executed output chooses **Log-target Ridge** severity.
- Candidate A is said to win or tie on metrics, but the monetary calibration story is weaker than that statement suggests.
- Candidate C has much lower combined wMAE/wRMSE at cell level, but is rejected mainly because its Spearman rank is lower. That may be reasonable for ranking, but it needs a clearer business justification.
- The final actions are useful, but less insightful than v3's frequency/severity quadrant actions.

v4 is the best foundation for a future polished version if the calibration issue is fixed. As-is, the under-calibrated expected-loss scale makes it weaker than v3.

## Recommended Submission Choice

Submit **v3** as the main answer.

If there is time to improve it, borrow these elements from v4:

- the temporal holdout framing,
- the explicit raw-row note for four zero severity records,
- the credibility-cell idea as an operational appendix,
- the concise executive-summary structure.

But keep v3's calibrated two-part GLM, common-cell validation, portfolio balance check, and quadrant-based renewal action matrix. Those are the strongest parts of the current project.

## Final Recommendation

Use `notebook/customer_profiling_v3.ipynb` as the answer to the assignment.

It is not merely the most detailed notebook; it is the one that most defensibly answers management's question under the actual data constraints. It recognizes that "unprofitable" cannot be measured directly, builds an actuarially standard expected-loss proxy, validates that proxy at the right aggregation level, and turns the result into clear renewal actions.
