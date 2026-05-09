# Insurance Customer Profiling

Two-part frequency-severity model to identify motor-insurance policyholders with high expected material-damage claim cost. KU Leuven Statistical Consulting exam project.

## Quick start

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/VoteVeto2/consulting-project.git
cd consulting-project
uv sync
uv run jupyter notebook notebook/customer_profiling_v5-final.ipynb
```

Run all cells top-to-bottom. The notebook is structured as a consulting deliverable: executive summary, data audit, baseline GLM models, machine-learning challenger (HistGradientBoosting), robustness and validation checks, expected-loss scoring, customer profiling with renewal-action recommendations, and conclusions.

## Project structure

```
dataset/                          frequency.csv (policies) + severity.csv (claims)
notebook/
  customer_profiling_v5-final.ipynb   Main analysis notebook (84 cells, 8 sections)
  EDA.ipynb                           Exploratory data analysis
  V5_RESULTS_SUMMARY.md              Results documentation and changelog
_build-v5-notebook.py             Generator script to recreate the notebook
src/                              Modelling modules (data loading, GLMs, preprocessing, plots)
task/                             Exam assignment and planning docs
```

## Regenerating the notebook

```bash
uv run python _build-v5-notebook.py
uv run jupyter nbconvert --to notebook --execute notebook/customer_profiling_v5-final.ipynb
```
