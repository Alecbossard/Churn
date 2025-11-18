from typing import List, Tuple

import numpy as np
import pandas as pd


def split_features_and_target(
    df: pd.DataFrame,
    target_col: str = "ChurnFlag",
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Split a dataframe into features (X) and target (y).

    Also returns the list of numeric and categorical feature columns.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # Target
    y = df[target_col]

    # Drop target and some non-feature columns
    X = df.drop(columns=[target_col])

    # Drop ID or redundant text target if present
    for col in ["customerID", "Churn"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    return X, y, numeric_cols, categorical_cols
