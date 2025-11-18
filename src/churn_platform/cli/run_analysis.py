import os

from churn_platform.analysis.summary import (
    load_full_data,
    compute_churn_summaries,
)
from churn_platform.data.loader import get_project_root


def main() -> None:
    print("Loading full dataset (train + val + test)...")
    df = load_full_data()
    print(f"Full shape: {df.shape}")

    # Choose some key categorical columns
    categorical_cols = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "Partner",
        "Dependents",
        "PhoneService",
    ]

    # Choose some key numeric columns for binning
    numeric_cols_for_bins = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    print("Computing churn summaries...")
    summaries = compute_churn_summaries(df, categorical_cols, numeric_cols_for_bins)

    # Save all summaries under reports/
    root_dir = get_project_root()
    reports_dir = os.path.join(root_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    for name, table in summaries.items():
        filename = f"{name}.csv"
        path = os.path.join(reports_dir, filename)
        table.to_csv(path, index=False)
        print(f"Saved {name} to: {path}")

    # Print a few highlights in console (example)
    global_rate = summaries["global_churn_rate"]["value"].iloc[0]
    print(f"\nGlobal churn rate: {global_rate:.3f}")

    print("\nTop rows for churn_by_Contract:")
    print(summaries["churn_by_Contract"].head())

    print("\nTop rows for churn_by_tenure_bin:")
    print(summaries["churn_by_tenure_bin"].head())

    print("\nAnalysis finished.")


if __name__ == "__main__":
    main()
