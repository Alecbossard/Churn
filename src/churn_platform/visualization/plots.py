import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_target_distribution(data: pd.DataFrame) -> None:
    """
    Plot the distribution of the target variable 'Churn'.
    """
    plt.figure(figsize=(4, 4))

    sns.countplot(
        x="Churn",
        data=data
    )

    plt.title("Target distribution (Churn)")
    plt.xlabel("Churn")
    plt.ylabel("Number of customers")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(data: pd.DataFrame) -> None:
    """
    Plot the distributions of numeric variables.
    """
    numeric_cols = data.select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        print("No numeric columns found.")
        return

    data[numeric_cols].hist(
        bins=30,
        figsize=(12, 8)
    )

    plt.suptitle("Distribution of numeric variables")
    plt.tight_layout()
    plt.show()


def plot_categorical_vs_churn(data: pd.DataFrame) -> None:
    """
    Plot some categorical variables against Churn.
    """
    cat_cols = data.select_dtypes(include="object").columns

    # Remove ID-like columns if present
    cols_to_exclude = ["customerID"]
    cat_cols = [c for c in cat_cols if c not in cols_to_exclude]

    cols_to_plot = []

    # Only keep categorical columns with a reasonable number of categories
    for col in cat_cols:
        if data[col].nunique() <= 10:
            cols_to_plot.append(col)

    # Keep only a few for a first look
    cols_to_plot = cols_to_plot[:4]

    for col in cols_to_plot:
        plt.figure(figsize=(6, 4))

        sns.countplot(
            x=col,
            hue="Churn",
            data=data
        )

        plt.xticks(rotation=45)
        plt.title(f"{col} vs Churn")
        plt.xlabel(col)
        plt.ylabel("Number of customers")
        plt.grid(True, axis="y")

        plt.tight_layout()
        plt.show()

def plot_feature_importances(
    importances_df: pd.DataFrame,
    title: str = "Feature importances",
    max_features: int = 20,
    output_path: str | None = None,
) -> None:
    """
    Plot top feature importances from a dataframe with columns:
    - 'feature'
    - 'importance'
    """

    df = importances_df.sort_values("importance", ascending=False).head(max_features)

    plt.figure(figsize=(8, 6))

    # Reverse order so the most important is on top
    features = df["feature"].tolist()[::-1]
    values = df["importance"].tolist()[::-1]

    plt.barh(features, values)
    plt.xlabel("Importance")
    plt.title(title)
    plt.grid(True, axis="x")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved feature importance plot to: {output_path}")

    plt.show()
