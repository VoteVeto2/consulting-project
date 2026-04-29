# Insurance Customer Profiling

Two-part frequency-severity model to identify motor-insurance policyholders with high expected material-damage claim cost. KU Leuven Statistical Consulting exam project.

## Quick start

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/VoteVeto2/consulting-project.git
cd consulting-project
uv sync
uv run jupyter notebook customer_profiling_solution.ipynb
```

Run all cells top-to-bottom. The notebook covers data audit, dataset compatibility checks, EDA, three candidate models (interpretable GLM, gradient boosting, credibility risk cells), combined pure-premium evaluation, and portfolio segmentation with renewal-action recommendations.

See [onboard.md](onboard.md) for a detailed walkthrough.

## Project structure

```
dataset/                  frequency.csv (policies) + severity.csv (claims)
customer_profiling_solution.ipynb   Main analysis notebook
_build_notebook.py        Generator script to recreate the notebook
data/output/              Model outputs, scores, plots, summary
task/                     Exam assignment and planning docs
notebook/                 Legacy notebook versions (v1-v3)
src/                      Legacy modelling modules
```

## Regenerating the notebook

```bash
uv run python _build_notebook.py
uv run jupyter nbconvert --to notebook --execute customer_profiling_solution.ipynb
```
