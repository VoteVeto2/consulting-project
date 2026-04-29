"""Load raw CSVs and apply the documented schema (factor levels, dtypes)."""
from __future__ import annotations
import pandas as pd

from .config import (
    FREQUENCY_CSV,
    SEVERITY_CSV,
    CATEGORICAL_VARS,
    FREQ_TARGET,
    SEV_TARGET,
)


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Cast factor variables to pandas Categorical with the levels defined in
    the assignment PDF. Doing this once at load time means every downstream
    function (EDA, modeling, profiling) sees consistent dtypes — and any value
    outside the documented level set surfaces as NaN immediately."""
    df = df.copy()
    # uwYear is documented as a factor with levels {2009, 2010}, not an int
    df["uwYear"] = df["uwYear"].astype(str).astype("category")
    df["gender"] = pd.Categorical(df["gender"], categories=["Female", "Male"])
    df["carType"] = pd.Categorical(df["carType"], categories=list("ABCDE"))
    df["carCat"] = pd.Categorical(df["carCat"], categories=["Small", "Medium", "Large"])
    df["job"] = pd.Categorical(
        df["job"],
        categories=["Employed", "Housewife", "Retired", "Self-employed", "Unemployed"],
    )
    # cover is documented as a 0/1 dummy — keep it as Categorical so it lines up
    # with the other factors when we one-hot encode later
    df["cover"] = pd.Categorical(df["cover"].astype(int), categories=[0, 1])
    return df


def load_frequency() -> pd.DataFrame:
    """Load frequency.csv and apply schema. Target: claimNumbMD (binary 0/1)."""
    df = pd.read_csv(FREQUENCY_CSV)
    df = _coerce_schema(df)
    # Frequency target is binary in the assignment ("at least one accident")
    df[FREQ_TARGET] = df[FREQ_TARGET].astype(int)
    return df


def load_severity() -> pd.DataFrame:
    """Load severity.csv and apply schema. Target: claimSizeMD (continuous EUR)."""
    df = pd.read_csv(SEVERITY_CSV)
    df = _coerce_schema(df)
    df[SEV_TARGET] = df[SEV_TARGET].astype(float)
    # Severity is conditional on a claim having occurred — drop any zero/negative
    # rows defensively (the Gamma GLM requires strictly positive responses)
    df = df.loc[df[SEV_TARGET] > 0].reset_index(drop=True)
    return df


def quick_summary(df: pd.DataFrame, name: str) -> str:
    """One-paragraph human-readable summary used by the notebook header cell."""
    n_rows, n_cols = df.shape
    miss = int(df.isna().sum().sum())
    return (
        f"{name}: {n_rows:,} rows x {n_cols} cols | "
        f"missing values: {miss} | "
        f"dtypes: {dict(df.dtypes.value_counts())}"
    )
