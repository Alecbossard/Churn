from churn_platform.data.loader import load_processed_split
from churn_platform.features.build_features import split_features_and_target
from churn_platform.models.persistence import load_model
from churn_platform.models.evaluate import evaluate_classifier


def main() -> None:
    print("Loading test split...")
    test_df = load_processed_split("test")
    print(f"Test shape: {test_df.shape}")

    print("Building features for test set...")
    X_test, y_test, _, _ = split_features_and_target(test_df)

    print("Loading trained model...")
    model = load_model(filename="xgb_baseline.joblib")

    print("Evaluating model on test set...")
    metrics = evaluate_classifier(model, X_test, y_test)

    print("Test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print("Test evaluation finished.")


if __name__ == "__main__":
    main()
