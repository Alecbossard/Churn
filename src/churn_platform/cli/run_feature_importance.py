import os
import pandas as pd

from churn_platform.data.loader import get_project_root
from churn_platform.models.persistence import load_model
from churn_platform.visualization.plots import plot_feature_importances


def extract_importances_from_pipeline(model) -> pd.DataFrame:
    """
    Extract feature importances from a fitted pipeline:
    - model["preprocessor"] : ColumnTransformer
    - model["classifier"]   : tree-based model with feature_importances_
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        raise ValueError("Classifier has no 'feature_importances_' attribute.")

    # Get feature names after preprocessing (numeric + one-hot categorical)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback: generic names
        n_features = len(classifier.feature_importances_)
        feature_names = [f"feature_{i}" for i in range(n_features)]

    importances = classifier.feature_importances_

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    return df


def main() -> None:
    root_dir = get_project_root()

    reports_dir = os.path.join(root_dir, "reports", "feature_importances")
    figures_dir = os.path.join(root_dir, "reports", "figures")

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    models_info = [
        ("Random Forest tuned", "rf_tuned.joblib"),
        ("XGBoost baseline", "xgb_baseline.joblib"),
    ]

    for display_name, filename in models_info:
        print(f"\n=== {display_name} ({filename}) ===")

        try:
            model = load_model(filename)
        except FileNotFoundError:
            print(f"Model file not found, skipping: {filename}")
            continue

        importances_df = extract_importances_from_pipeline(model)

        # Save CSV
        csv_name = f"feature_importances_{filename.replace('.joblib', '')}.csv"
        csv_path = os.path.join(reports_dir, csv_name)
        importances_df.to_csv(csv_path, index=False)
        print(f"Saved importances table to: {csv_path}")

        # Print top 10 in console
        print("\nTop 10 features:")
        print(importances_df.head(10))

        # Plot
        png_name = f"feature_importances_{filename.replace('.joblib', '')}.png"
        png_path = os.path.join(figures_dir, png_name)

        plot_feature_importances(
            importances_df,
            title=f"Feature importances - {display_name}",
            max_features=20,
            output_path=png_path,
        )

    print("\nFeature importance analysis finished.")


if __name__ == "__main__":
    main()
