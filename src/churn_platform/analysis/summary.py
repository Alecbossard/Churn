from typing import List

import pandas as pd

from churn_platform.data.loader import load_processed_split


def load_full_data() -> pd.DataFrame:
    """
    Load train + val + test and concatenate them into a single dataframe.
    """
    train_df = load_processed_split("train")
    val_df = load_processed_split("val")
    test_df = load_processed_split("test")

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    return full_df


def compute_global_churn_rate(df: pd.DataFrame) -> float:
    """
    Compute global churn rate.
    """
    return df["ChurnFlag"].mean()


def churn_rate_by_category(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Compute churn rate and counts for a categorical column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")

    grouped = (
        df.groupby(column)["ChurnFlag"]
        .agg(
            count="size",
            churn_rate="mean",
        )
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )

    return grouped


def churn_rate_by_numeric_bin(
    df: pd.DataFrame,
    column: str,
    bins: int = 5,
) -> pd.DataFrame:
    """
    Bin a numeric column and compute churn rate inside each bin.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")

    binned_col = f"{column}_bin"

    df = df.copy()
    df[binned_col] = pd.qcut(df[column], q=bins, duplicates="drop")

    grouped = (
        df.groupby(binned_col)["ChurnFlag"]
        .agg(
            count="size",
            churn_rate="mean",
        )
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )

    return grouped


def compute_churn_summaries(
    df: pd.DataFrame,
    cat_cols: List[str],
    num_cols_for_bins: List[str],
) -> dict:
    """
    Compute several churn summary tables.
    Returns a dict of dataframes.
    """
    summaries = {}

    # Global churn
    global_rate = compute_global_churn_rate(df)
    summaries["global_churn_rate"] = pd.DataFrame(
        {"metric": ["global_churn_rate"], "value": [global_rate]}
    )

    # Categorical columns
    for col in cat_cols:
        summaries[f"churn_by_{col}"] = churn_rate_by_category(df, col)

    # Numeric columns (binned)
    for col in num_cols_for_bins:
        summaries[f"churn_by_{col}_bin"] = churn_rate_by_numeric_bin(df, col)

    return summaries
