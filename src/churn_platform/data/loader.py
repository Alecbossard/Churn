import os
import pandas as pd


def get_project_root() -> str:
    """
    Return the absolute path to the project root folder.

    Example:
    C:/Users/Alec/PycharmProjects/user-analytics-churn-platform
    """
    # loader.py is in: project_root/src/churn_platform/data/loader.py
    # We need to go up four levels to reach project_root.
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )
    )


def get_raw_csv_path() -> str:
    """
    Return the path to the raw Telco CSV file.
    """
    root_dir = get_project_root()

    csv_path = os.path.join(
        root_dir,
        "data",
        "raw",
        "telco_customer_churn.csv"
    )

    return csv_path


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw Telco CSV and return a pandas DataFrame.
    """
    csv_path = get_raw_csv_path()

    print(f"Using raw CSV path: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = pd.read_csv(csv_path)

    return data


def get_processed_csv_path(split: str, prefix: str = "telco") -> str:
    """
    Return the path to a processed CSV file (train / val / test).
    """
    root_dir = get_project_root()

    filename = f"{prefix}_{split}.csv"

    csv_path = os.path.join(
        root_dir,
        "data",
        "processed",
        filename
    )

    return csv_path


def load_processed_split(split: str, prefix: str = "telco") -> pd.DataFrame:
    """
    Load a processed split (train / val / test) and return a DataFrame.
    """
    csv_path = get_processed_csv_path(split, prefix)

    print(f"Using processed CSV path for '{split}': {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed CSV file not found: {csv_path}")

    data = pd.read_csv(csv_path)

    return data
