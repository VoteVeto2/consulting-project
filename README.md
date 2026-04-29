# Insurance Customer Profiling — KU Leuven Statistical Consulting

Two-part GLM (frequency × severity) to identify unprofitable motor-insurance customers, with a gradient-boosting challenger, quadrant-based renewal actions, and a surrogate decision tree for management communication.

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.13 | pinned in `.python-version` |
| [uv](https://docs.astral.sh/uv/) | latest | package + venv manager |

Install `uv` if you don't have it:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## Setup

```bash
git clone https://github.com/VoteVeto2/consulting-project.git
cd consulting-project

# Create venv and install all dependencies from uv.lock
uv sync
```

That's it — `uv sync` reads `pyproject.toml` + `uv.lock` and produces a fully reproducible environment in `.venv/`.

---

## Datasets

Both files live in `dataset/` and are committed to the repo.

| File | Rows | Description |
|------|------|-------------|
| `dataset/frequency.csv` | ~20 k | One row per policy. Target: `claimNumbMD` (binary claim indicator). |
| `dataset/severity.csv` | ~10 k | One row per claim. Target: `claimSizeMD` (claim cost in EUR). |

The two datasets are **not individually linked** — they share the same covariates but cannot be joined at the policy level. The analysis accounts for this via common-cell calibration.

Key variables: `uwYear`, `gender`, `carType`, `carCat`, `job`, `cover` (categorical) + `age`, `nYears`, `carVal`, `density` (numeric).

---

## Running the analysis

### Option A — Jupyter notebook (recommended)

Launch Jupyter and open the latest notebook:

```bash
uv run jupyter notebook notebook/customer_profiling_v3.ipynb
```

Run all cells top-to-bottom. The notebook is self-contained and covers:

1. Data audit & business questions
2. Frequency GLM (logistic) + Severity GLM (Gamma) + Lognormal challenger
3. Gradient-boosting ML challenger
4. Robustness: year-stability, gender-sensitivity, tail diagnostics
5. Expected-loss scoring & Risk Index
6. Customer profiling, quadrant decomposition, renewal action matrix
7. Surrogate decision tree + conclusions

### Option B — Regenerate the notebook from source

The notebooks are built programmatically from `build_notebook_v3.py`:

```bash
uv run python build_notebook_v3.py
```

This overwrites `notebook/customer_profiling_v3.ipynb` with a fresh copy and then execute it with:

```bash
uv run jupyter nbconvert --to notebook --execute notebook/customer_profiling_v3.ipynb \
    --output notebook/customer_profiling_v3.ipynb
```

---

## Project structure

```
consulting-project/
├── dataset/
│   ├── frequency.csv          # Policy-level data (claim indicator)
│   └── severity.csv           # Claim-level data (claim cost)
├── notebook/
│   ├── customer_profiling_v3.ipynb   # Main analysis (latest)
│   ├── customer_profiling_v2.ipynb   # Intermediate version
│   ├── customer_profiling.ipynb      # Baseline version
│   └── V3_RESULTS_SUMMARY.md         # Written summary of v3 findings
├── output/
│   └── customer_profiling_v3_report.md   # Full narrative report
├── src/
│   ├── config.py          # Paths, variable lists, constants
│   ├── data_loader.py     # CSV loading + quick_summary()
│   ├── preprocessing.py   # Design matrix + train/test split
│   ├── frequency_model.py # Logistic GLM fit / evaluate / predict
│   ├── severity_model.py  # Gamma GLM fit / evaluate / predict
│   ├── pure_premium.py    # EL = p_claim × exp_severity, Lorenz curve
│   ├── profiling.py       # Tier assignment + segment descriptions
│   ├── plots.py           # ROC, calibration, tier bar charts
│   ├── eda.py             # Exploratory summaries
│   └── style.py           # Warm matplotlib theme
├── task/
│   ├── Customer_profiling_exam_assignment.pdf
│   └── practical_plan.md
├── build_notebook_v3.py   # Generates customer_profiling_v3.ipynb
├── pyproject.toml
└── uv.lock
```

---

## Key results (v3)

- **Frequency AUC ~0.69** — logistic GLM on balanced sample; gradient boosting adds ~0.01–0.02.
- **Top 5 % of policies** account for ~12 % of expected loss at ~2.4× the portfolio average.
- **Typical high-risk profile:** young (~23 yr), male, unemployed, urban density, no MD cover.
- **Quadrant decomposition** separates high-frequency/low-severity from low-frequency/high-severity risks, enabling differentiated renewal actions.
- Models are **stable across underwriting years** (2009 ↔ 2010) and robust to gender exclusion.

See `output/customer_profiling_v3_report.md` or `notebook/V3_RESULTS_SUMMARY.md` for the full write-up.
