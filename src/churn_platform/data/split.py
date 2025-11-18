import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_platform.data.loader import get_project_root


def train_val_test_split(
    df: pd.DataFrame,
    target_col: str = "ChurnFlag",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the cleaned dataframe into train / validation / test sets.

    - First split: train+val vs test
    - Second split: train vs val

    The splits are stratified on the target column.
    """
    # First split: train+val and test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )

    # Compute relative validation size inside the train+val part
    val_relative = val_size / (1.0 - test_size)

    # Second split: train and val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        stratify=train_val_df[target_col],
        random_state=random_state,
    )

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prefix: str = "telco",
) -> None:
    """
    Save train/val/test splits as CSV files under data/processed/.
    """
    root_dir = get_project_root()
    processed_dir = os.path.join(root_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    train_path = os.path.join(processed_dir, f"{prefix}_train.csv")
    val_path = os.path.join(processed_dir, f"{prefix}_val.csv")
    test_path = os.path.join(processed_dir, f"{prefix}_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train split to: {train_path}")
    print(f"Saved val split   to: {val_path}")
    print(f"Saved test split  to: {test_path}")
