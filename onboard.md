# Onboarding Guide

## What is this project?

An insurance customer profiling analysis for the KU Leuven Statistical Consulting exam. The goal: identify motor-insurance policyholders with the **highest expected material-damage claim cost** as a proxy for unprofitability at renewal.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

## Getting started

```bash
git clone https://github.com/VoteVeto2/consulting-project.git
cd consulting-project
uv sync
```

## Running the analysis

Open and run the main notebook:

```bash
uv run jupyter notebook customer_profiling_solution.ipynb
```

Run all cells top-to-bottom. The notebook is self-contained and produces all outputs automatically.

## What the notebook does

| Section | What it covers |
|---|---|
| Sections 0-2 | Data loading, audit, and compatibility check between datasets |
| Sections 3-4 | Exploratory analysis and target distributions |
| Section 5 | Temporal train/test split (2009 vs 2010) |
| Section 6 | **Candidate A** — Interpretable two-part model (logistic + Gamma GLM with splines) |
| Section 7 | **Candidate B** — Gradient boosting challenger (HistGradientBoosting) |
| Section 8 | **Candidate C** — Credibility-smoothed risk cells (business model) |
| Section 9 | Tweedie comparison (optional, for reference only) |
| Sections 10-11 | Combined pure-premium evaluation and model comparison |
| Section 12 | Portfolio ranking and action segmentation |
| Section 13 | Business interpretation and recommendations |

## Key data files

| File | Description |
|---|---|
| `dataset/frequency.csv` | 24,774 policies with binary claim indicator (`claimNumbMD`) |
| `dataset/severity.csv` | 12,256 claims with continuous damage amount (`claimSizeMD`) |

These datasets share 10 covariates but **cannot be linked at the policy level**.

## Output files

All outputs are saved to `data/output/`:

- `portfolio_scores.csv` — every policy scored with expected loss and action band
- `portfolio_top_review.csv` — top 5% policies flagged for underwriting review
- `model_comparison.csv` — side-by-side metrics for all candidates
- `cell_level_validation.csv` — risk-cell-level validation results
- `summary.md` — executive summary
- PNG plots — EDA, calibration, feature importance, lift curves

## Regenerating the notebook

If you need to rebuild the notebook from scratch:

```bash
uv run python _build_notebook.py
uv run jupyter nbconvert --to notebook --execute customer_profiling_solution.ipynb
```

## Important caveats

- These scores are **expected claim cost**, not proven unprofitability (no premium data available).
- Combined validation uses risk-cell aggregation because the two datasets are unlinked.
- The temporal split (2009→2010) provides only a single holdout period.
