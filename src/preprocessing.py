"""Feature engineering: design matrix construction and train/test split.

Both the GLM and tree models in this project consume the same numeric design
matrix produced here, so encoding choices live in one place.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CATEGORICAL_VARS, NUMERIC_VARS, RANDOM_STATE


def build_design_matrix(df: pd.DataFrame, *,
                        drop_first: bool = True) -> pd.DataFrame:
    """Build a numeric design matrix from the raw dataframe.

    - One-hot encodes the documented factors (drop_first=True for GLMs to
      avoid the dummy-variable trap; the dropped level becomes the reference
      category against which other levels are compared).
    - Leaves numeric covariates as-is.

    Returns a dataframe of dtype float — ready to be passed straight into
    statsmodels or sklearn estimators.
    """
    cat = pd.get_dummies(
        df[CATEGORICAL_VARS], drop_first=drop_first, dtype=float,
        # Stable, readable column names: e.g. "carType_C" instead of "carType_2"
        prefix=CATEGORICAL_VARS, prefix_sep="_",
    )
    num = df[NUMERIC_VARS].astype(float)
    X = pd.concat([num, cat], axis=1)
    return X


def split_xy(df: pd.DataFrame, target: str, *,
             test_size: float = 0.2, stratify_target: bool = False
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split. For the binary frequency target we stratify so the
    claim-rate is preserved in both folds; for continuous severity we don't."""
    X = build_design_matrix(df)
    y = df[target]
    strat = y if stratify_target else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=strat,
    )
    return X_tr, X_te, y_tr, y_te
