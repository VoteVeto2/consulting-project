# V5 Results Summary

## What changed from v3

V5 implements all seven modification phases identified in the codex review
of v3 (`output/.codex/customer_profiling_v3_weaknesses_modification_plan.md`).
No code logic was changed — same models, same data, same src modules.
The changes are in framing, narrative structure, and examiner-readiness.

### Phase 1 — Clarified business framing

- Added Section 1.0 defining the target as **expected material-damage
  claim cost**, not literal profitability.
- The premium-data limitation appears on the first page, before any
  modelling.
- Working assumption made explicit: high expected loss indicates potential
  unprofitability only if premiums are not fully risk-adjusted.
- The word "unprofitable" is no longer used as a standalone label.
  Replaced throughout with "high expected loss" or "high expected
  material-damage claim cost."

### Phase 2 — Raw severity audit

- Added Section 1.2 showing the raw severity CSV has **12,256 rows**,
  of which **4 have zero claim cost**.
- The 12,252 positive rows used for Gamma modelling are now explicitly
  reconciled with the assignment PDF's stated row count.
- Zero-cost rows are previewed in-notebook.

### Phase 3 — Validation restructured around renewal use

- Temporal holdout (train 2009, test 2010) **elevated to Section 4.1**
  as the primary business validation — it mimics forward-looking renewal
  decisions.
- Random 80/20 holdout **repositioned as Section 4.2** (model diagnostic),
  with a static summary table referencing Section 2 metrics.
- Validation summary added after common-cell calibration (Section 5.3),
  listing all three complementary checks and explaining why cell-level
  validation is the best available combined test for unlinked datasets.

### Phase 4 — Gender handling operationally clean

- Section 4.3 retitled "diagnostic vs operational model."
- With-gender model explicitly labelled as **diagnostic only**.
- Gender-free model designated as the **recommended operational
  specification** for EU pricing and renewal decisions.
- Cites the Test-Achats ruling (C-236/09, 2011) with the 21 December 2012
  cutoff date.
- Frequency findings in Section 2 now note that gender is "retained for
  diagnostic purposes only — see Section 4.3."

### Phase 5 — Tempered renewal actions

- "Non-renewal" replaced with "manual underwriting review; escalate only
  extreme cases for non-renewal review."
- Added caveat that recommendations should be combined with current
  premium adequacy, claims history, underwriting rules, and
  customer-lifetime-value considerations.
- "No policy should be non-renewed based solely on model score" stated
  explicitly.
- Renewal action tables in both Sections 6.3 and 7.4 use the tempered
  wording.

### Phase 6 — Severity-model limitation note

- After the severity comparison (Section 2.4), a dedicated note explains
  that the GLM severity model has **weak individual-level rank
  discrimination** (Spearman ≈ −0.01).
- States this is common with heavy-tailed data and policy-level covariates.
- Explains why it does not invalidate the analysis: aggregate calibration
  is good, frequency dominates segmentation, GBM challenger confirms
  similar pure-premium rankings.
- The limitation is also referenced in Section 7.3 (key findings) and
  Section 7.5 (caveats).

### Phase 7 — Executive summary table

- Section 0 added at the top of the notebook with a blockquote framing
  statement and a six-row executive table answering the key consulting
  questions.
- Section 7.6 repeats the table (now seven rows, adding the operational
  EU model row) as a closing reference.
- The notebook now reads as a consulting deliverable with a clear top-line
  answer, not only as a technical analysis.

---

## Key results (unchanged from v3)

All numerical results are identical to v3. The modifications are purely
in framing and presentation.

| Metric | Value |
|---|---|
| Frequency GLM AUC (random holdout) | 0.689 |
| Frequency GLM Brier | 0.223 |
| Severity Gamma GLM MAE | 634 EUR |
| Severity Gamma GLM rank corr. | −0.01 |
| Severity GBM rank corr. | 0.22 |
| Frequency GBM AUC | 0.704 |
| GLM vs GBM pure-premium Spearman | 0.882 |
| Common-cell correlation (r) | 0.907 |
| Portfolio balance (pred/obs) | 1.00 |
| Top 5 % mean expected loss | 1,029 EUR |
| Top 5 % risk index | 2.4x |
| Gini coefficient | ~0.23 |

### Temporal stability

| Split | Freq AUC | Freq Brier | Sev MAE | Sev mean pred | Sev mean obs |
|---|---|---|---|---|---|
| 2009→2010 | 0.687 | 0.224 | 645.9 | 836.2 | 892.3 |
| 2010→2009 | 0.685 | 0.225 | 648.1 | 891.1 | 838.4 |

### Gender sensitivity

| Metric | With gender | Without gender | Delta |
|---|---|---|---|
| Freq AUC | 0.6894 | 0.6860 | −0.003 |
| Freq Brier | 0.2229 | 0.2237 | +0.001 |
| Sev MAE | 634.2 | 635.4 | +1.2 |

### Raw severity audit

| Count | Value |
|---|---|
| Raw severity rows | 12,256 |
| Zero-cost rows | 4 |
| Positive rows (Gamma) | 12,252 |

### Typical Very High tier profile (top 5 %)

- 1,239 policies (5.0 % of portfolio)
- Mean expected loss: 1,029 EUR (2.4x portfolio average)
- Age: ~23 yr
- Gender: Male (85 %)
- Job: Unemployed (63 %)
- Car type: A (36 %)
- Car category: Small (43 %)
- Cover: 0 (88 %)
- Density: ~237 (urban)

---

## Notebook structure

| Section | Title | Cells |
|---|---|---|
| 0 | Executive Summary | 1 |
| 1 | Data Audit & Business Questions | 14 |
| 2 | Baseline GLM Models | 14 |
| 3 | Machine-Learning Challenger | 9 |
| 4 | Robustness & Validation | 10 |
| 5 | Expected Loss Scoring | 12 |
| 6 | Customer Profiling & Renewal Actions | 16 |
| 7 | Conclusions & Recommendations | 2 |
| — | Footer | 1 |
| **Total** | | **84** |

---

## Files

- `notebook/customer_profiling_v5.ipynb` — the complete v5 notebook
  (executed, all outputs embedded)
- `_build-v5-notebook.py` — generator script to recreate the notebook
  from source
