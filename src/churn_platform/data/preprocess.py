import os
from typing import Tuple

import numpy as np
import pandas as pd

from churn_platform.data.loader import get_project_root


def clean_telco_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for the Telco churn dataset.

    - Drop duplicate customerID rows (if any)
    - Convert TotalCharges to numeric and handle errors
    - Drop rows with missing target (Churn)
    - Create a numeric target column ChurnFlag (0/1)
    """
    df = raw_df.copy()

    # Drop duplicate customers just in case
    if "customerID" in df.columns:
        df = df.drop_duplicates(subset="customerID")

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Fill NaNs with median value
        median_total = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_total)

    # Drop rows with missing Churn (should not happen, but safe)
    df = df.dropna(subset=["Churn"])

    # Create numeric target
    df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Optional: sanity check
    if df["ChurnFlag"].isna().any():
        raise ValueError("Some rows have invalid Churn values that could not be mapped.")

    return df


def save_interim_data(df: pd.DataFrame, filename: str = "telco_clean.csv") -> str:
    """
    Save a cleaned/intermediate version of the dataset to data/interim/.
    Return the full path to the saved file.
    """
    root_dir = get_project_root()
    interim_dir = os.path.join(root_dir, "data", "interim")
    os.makedirs(interim_dir, exist_ok=True)

    path = os.path.join(interim_dir, filename)
    df.to_csv(path, index=False)

    return path
