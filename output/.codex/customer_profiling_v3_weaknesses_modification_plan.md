# v3 Weaknesses And Modification Plan

Target notebook: `notebook/customer_profiling_v3.ipynb`

## Short Assessment

`customer_profiling_v3.ipynb` is the best current version to build on. Its core answer is defensible: it frames "unprofitable" as high expected technical loss, uses a two-part frequency-severity model for unlinked datasets, validates at portfolio and cell level, and turns results into renewal actions.

The weaknesses are mostly presentation, compliance framing, and validation design. The notebook does not need a full rebuild. It needs targeted edits to make the answer more examiner-proof.

## Main Weaknesses

### 1. The word "unprofitable" is still a bit too strong

v3 correctly says true profitability cannot be measured without premium data, but later sections still use terms like "unprofitable profile", "non-renewal", and "current pricing almost certainly does not reflect this risk level".

Why this matters:

- The assignment explicitly asks for unprofitable customers, but the data only supports expected claim cost.
- A strict examiner may penalize language that implies actual profit/loss was measured.

Recommended fix:

- Use "high expected loss", "high technical risk", or "likely underpriced if premiums do not already reflect risk".
- Keep "unprofitable" only as a management term, then immediately define it as a proxy.

Suggested wording:

> Because premium and expense data are unavailable, this analysis cannot identify truly unprofitable customers. It identifies customers with high expected material-damage claim cost, who are the most likely to be underpriced under an average or imperfectly risk-adjusted premium structure.

### 2. Gender is used in the primary model, then handled as a caveat

v3 includes gender in the primary GLM and later reports that excluding gender barely affects performance. This is statistically useful, but the operational recommendation should be clearer.

Why this matters:

- For EU insurance pricing, gender is legally sensitive.
- The notebook mentions the Test-Achats ruling, but the final recommendation should clearly separate diagnostic modelling from deployable pricing.

Recommended fix:

- Keep the with-gender model as a diagnostic/explanatory benchmark.
- Add a final "deployment model" note recommending the gender-free model for pricing or renewal action.
- Reword actions so gender is not used as a decision rule.

Suggested wording:

> Gender was retained in the diagnostic model to understand portfolio risk structure. For operational renewal decisions in an EU context, the gender-free specification should be used; its performance loss is negligible.

### 3. Random 80/20 split appears before temporal validation

v3's main model validation uses a random 80/20 split. It later includes a year-stability check, but the notebook's main validation story would be stronger if the temporal check were elevated.

Why this matters:

- Renewal decisions are forward-looking.
- Training on 2009 and validating on 2010 better mimics the business use case.

Recommended fix:

- Keep the random split for model diagnostics if desired.
- Move or summarize the year-stability analysis earlier in the validation section.
- In the final conclusion, say the model is supported by both random holdout and 2009-to-2010 validation.

Optional stronger fix:

- Recompute the primary reported metrics using 2009 as train and 2010 as test.
- Use the full-data refit only for final portfolio scoring after validation.

### 4. The 12,256 vs 12,252 severity-row distinction should be explicit

The assignment states `severity.csv` has 12,256 rows. v3 loads 12,252 rows because `src/data_loader.py` drops zero or negative claim sizes for Gamma modelling.

Why this matters:

- The notebook could look inconsistent with the assignment PDF unless the four zero-cost rows are explained.
- Dropping zeros is reasonable for Gamma GLM, but it must be stated.

Recommended fix:

- Add a short audit cell before modelling:
  - raw severity rows = 12,256,
  - zero-cost severity rows = 4,
  - positive severity rows used for Gamma = 12,252.

Suggested wording:

> Four severity records have zero claim cost. They are retained in the raw data audit but excluded from Gamma severity modelling because the Gamma family requires strictly positive responses. The effect on total claim cost is zero and the row impact is negligible.

### 5. Severity ranking is weak under the GLM

v3 reports Gamma GLM severity rank correlation around -0.01 and lognormal around -0.04, while GBM severity rank correlation is around 0.22.

Why this matters:

- The Gamma GLM is well-calibrated on mean severity, but weak at ranking individual claim severities.
- The final portfolio ranking may be driven mostly by frequency.

Recommended fix:

- State this limitation directly.
- Emphasize that the GLM is selected for calibration, transparency, and actuarial defensibility, not because it dominates all predictive metrics.
- Add a decomposition note: frequency is the stronger driver of segmentation; severity still matters for profile interpretation and expected-loss scale.

Suggested wording:

> The GLM severity model is calibrated in aggregate but has limited individual-level severity discrimination. This is common with noisy, heavy-tailed claim sizes and only policy-level covariates. The GBM improves severity ranking modestly, so it is retained as a challenger and sensitivity check.

### 6. The action recommendations are occasionally too aggressive

v3 recommends strong repricing, strict underwriting, and possible non-renewal for the highest-risk groups.

Why this matters:

- Without premium, expense, retention, regulation, or fairness constraints, direct non-renewal recommendations can sound overconfident.
- The exam likely rewards business realism.

Recommended fix:

- Change "non-renewal" from a default recommendation to "manual review for potential non-renewal where legally and commercially justified".
- Add that renewal action should combine model score with current premium, claims history, underwriting rules, and customer value.

Suggested action wording:

| Risk group | Better wording |
|---|---|
| Very High | Manual underwriting review; check premium adequacy; consider deductible, coverage, or pricing changes; escalate only extreme cases for non-renewal review |
| High | Targeted repricing or deductible adjustment subject to current premium adequacy |
| Medium | Monitor and review if premium is below technical price |
| Low | Retain, avoid excessive price increases |

### 7. The notebook is strong analytically but could be more exam-reader friendly

v3 has many checks and outputs. The final narrative is good, but an examiner should be able to see the answer quickly.

Why this matters:

- The assignment rewards clear reasoning, not model volume.
- Too many technical sections can hide the main consulting answer.

Recommended fix:

- Add an executive summary near the top, before technical modelling.
- State the final answer in five bullets:
  - definition of unprofitable proxy,
  - method,
  - validation,
  - high-risk profile,
  - recommended renewal actions.

## Proposed Modification Plan

### Phase 1 - Clarify business framing

Edit the introduction and conclusion:

- Define the target as "expected material-damage claim cost", not literal profitability.
- Add the premium-data limitation in the first page.
- Make the working assumption explicit: high expected loss indicates potential unprofitability only if current premiums are not fully risk-adjusted.

Expected impact:

- Stronger alignment with the assignment's "ask the right questions" criterion.

### Phase 2 - Add raw severity audit

Add one short code or markdown section before the existing schema checks:

- Load raw `severity.csv`.
- Count raw rows, zero-cost rows, and positive rows.
- Explain why Gamma modelling uses positive rows only.

Expected impact:

- Removes apparent mismatch between PDF row count and notebook row count.

### Phase 3 - Reframe validation around renewal use

Keep current random holdout metrics, but restructure the validation story:

- Random split: model diagnostic.
- 2009-to-2010 split: business realism.
- Common-cell calibration: best available combined validation because files are unlinked.
- Portfolio balance: sanity check on expected-loss scale.

Expected impact:

- Makes the validation design more persuasive and easier to defend.

### Phase 4 - Make gender handling operationally clean

Modify the gender sensitivity section:

- Present gender-included results as diagnostic.
- Present gender-free model as the recommended operational version for EU pricing or renewal decisions.
- Remove gender from any simplified decision rule or action trigger if possible.

Expected impact:

- Reduces legal/compliance criticism without losing analytical insight.

### Phase 5 - Temper renewal actions

Revise action wording:

- Replace direct "non-renewal" phrasing with "manual review for potential non-renewal in extreme cases".
- Add a note that actions require premium adequacy and underwriting review.
- Keep the quadrant logic because it is one of v3's best contributions.

Expected impact:

- More realistic management recommendation.

### Phase 6 - Add a severity-model limitation note

In the severity and final caveat sections:

- State that GLM severity has weak individual-level rank discrimination.
- Explain why this does not invalidate the analysis:
  - aggregate severity calibration is good,
  - expected-loss scale is balanced,
  - GBM challenger gives similar pure-premium ranking,
  - customer profiling is mostly about relative risk tiers and renewal triage.

Expected impact:

- Shows statistical honesty and prevents overclaiming.

### Phase 7 - Add a concise final executive table

Add a final table like:

| Question | Answer |
|---|---|
| Can we measure true unprofitability? | No, no premium/expense data |
| What do we estimate instead? | Expected material-damage claim cost |
| Recommended model | Two-part GLM, with GBM challenger |
| Validation | Portfolio balance about 1.00; common-cell correlation about 0.91; stable across years |
| Highest-risk segment | Top 5%, about 2.4x average expected loss |
| Management action | Manual renewal review, pricing/deductible/coverage assessment |

Expected impact:

- Makes the notebook read like a consulting deliverable, not only a technical analysis.

## Priority Order

If time is limited, do these first:

1. Add raw severity row/zero explanation.
2. Reword "unprofitable" and non-renewal claims.
3. Make gender-free model the operational recommendation.
4. Elevate temporal validation in the final narrative.
5. Add severity rank-discrimination caveat.

If time allows, then:

6. Recompute primary reported metrics on the 2009-to-2010 split.
7. Add a top-level executive summary.
8. Consider adding credibility risk cells from v4 as an appendix, not as the main model.

## Recommended End State

The final v3-based submission should say:

> We cannot observe true profitability because premium and expense data are missing. We therefore identify high expected material-damage claim cost as a proxy for potential unprofitability. A two-part frequency-severity GLM is the primary model because it matches the unlinked data structure and is interpretable for management. The result is calibrated at portfolio level, validated at common risk-cell level, stable across underwriting years, and supported by a GBM challenger. The top-risk policies should be reviewed at renewal for premium adequacy, deductible/coverage changes, and only in extreme cases potential non-renewal.

That keeps v3's strengths while removing the main ways an examiner could challenge it.
