# Insurance Customer Profiling

Two-part GLM (frequency × severity) to identify unprofitable motor-insurance customers. KU Leuven Statistical Consulting exam project.

## Quick start

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/VoteVeto2/consulting-project.git
cd consulting-project
uv sync
uv run jupyter notebook notebook/customer_profiling_v3.ipynb
```

Run all cells top-to-bottom. The notebook covers data audit, GLM + ML modeling, robustness checks, expected-loss scoring, and customer profiling with renewal actions.

## Project structure

```
dataset/          frequency.csv (policies) + severity.csv (claims)
notebook/         Jupyter notebooks (v1–v3) and result summaries
src/              Modeling modules: data loading, preprocessing, GLMs, profiling, plots
output/           Generated reports
task/             Exam assignment and planning docs
build_notebook_v3.py   Regenerates the v3 notebook from source
```

## Regenerating the notebook

```bash
uv run python build_notebook_v3.py
uv run jupyter nbconvert --to notebook --execute notebook/customer_profiling_v3.ipynb
```
