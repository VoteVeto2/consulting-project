# CLAUDE.md

## Project overview

KU Leuven Statistical Consulting exam project. Two-part frequency-severity model to identify motor-insurance policyholders with high expected material-damage claim cost (proxy for unprofitability).

## Key constraints

- `frequency.csv` and `severity.csv` share covariates but **cannot be linked at the policy level**.
- No premium, expense, or reinsurance data — actual profitability is unmeasurable.
- Combined validation must be done at risk-cell level, not per-policy.

## Repository layout

```
dataset/                  Raw data (frequency.csv + severity.csv)
customer_profiling_solution.ipynb   Main analysis notebook (run top-to-bottom)
_build_notebook.py        Generator script to recreate the notebook
data/output/              Model outputs, scores, plots, summary
task/                     Exam assignment, plans, literature notes
notebook/                 Legacy notebook versions (v1-v3)
src/                      Legacy modelling modules
```

## Development

```bash
uv sync                    # install dependencies
uv run jupyter notebook customer_profiling_solution.ipynb   # open notebook
uv run python _build_notebook.py   # regenerate notebook from source
```

## Modelling pipeline

1. Temporal holdout: train on uwYear=2009, test on uwYear=2010
2. Candidate A (primary): Logistic + Gamma/log-target GLM with spline features
3. Candidate B (challenger): HistGradientBoosting (frequency + severity)
4. Candidate C (business): Credibility-smoothed risk cells
5. Combined score: P(claim) * E(severity|claim), validated at cell level

## Results (latest run)

- Candidate A: Freq AUC=0.694, Sev MAE=632, Cell Spearman=0.870
- Candidate B: Freq AUC=0.691, Sev MAE=633, Cell Spearman=0.873
- Interpretable model (A) nearly matches boosting (B) — A is the primary recommendation.
