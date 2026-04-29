"""Project-wide constants and paths."""
from __future__ import annotations
from pathlib import Path

# Project root resolves relative to this file (src/config.py -> ../)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "dataset"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

FREQUENCY_CSV: Path = DATA_DIR / "frequency.csv"
SEVERITY_CSV: Path = DATA_DIR / "severity.csv"

# Variable groupings — kept here so every module agrees on what is what
CATEGORICAL_VARS: list[str] = [
    "uwYear", "gender", "carType", "carCat", "job", "cover",
]
NUMERIC_VARS: list[str] = [
    "age", "nYears", "carVal", "density",
]
ALL_FEATURES: list[str] = CATEGORICAL_VARS + NUMERIC_VARS

FREQ_TARGET: str = "claimNumbMD"
SEV_TARGET: str = "claimSizeMD"

RANDOM_STATE: int = 42
