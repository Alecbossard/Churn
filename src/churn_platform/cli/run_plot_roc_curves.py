import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score

from churn_platform.data.loader import load_processed_split, get_project_root
from churn_platform.features.build_features import split_features_and_target
from churn_platform.models.dl_models import SimpleMLP
from churn_platform.models.persistence import load_model
import joblib


def get_mlp_proba(X_test_df):
    """
    Load the saved MLP preprocessor + model and return churn probabilities
    for the test dataframe.
    """
    root_dir = get_project_root()
    models_dir = os.path.join(root_dir, "models")

    preprocessor_path = os.path.join(models_dir, "mlp_preprocessor.joblib")
    model_path = os.path.join(models_dir, "mlp_model.pt")

    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        print("MLP preprocessor or model not found, skipping MLP.")
        return None

    # Load preprocessor and transform X_test
    preprocessor = joblib.load(preprocessor_path)
    X_test_np = preprocessor.transform(X_test_df)

    # Build model with correct input_dim
    input_dim = X_test_np.shape[1]
    model = SimpleMLP(input_dim=input_dim, hidden_dim=64)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).numpy()

    return probs


def main() -> None:
    print("Loading test split...")
    test_df = load_processed_split("test")
    print(f"Test shape: {test_df.shape}")

    print("Building features (X, y) for test set...")
    X_test_df, y_test, _, _ = split_features_and_target(test_df)

    # List of classical models (sklearn/xgboost pipelines)
    model_infos = [
        ("Logistic Regression", "logreg_baseline.joblib"),
        ("Random Forest baseline", "rf_baseline.joblib"),
        ("Random Forest tuned", "rf_tuned.joblib"),
        ("XGBoost baseline", "xgb_baseline.joblib"),
    ]

    plt.figure(figsize=(8, 6))

    # Plot ROC for each sklearn/xgboost model
    for display_name, filename in model_infos:
        try:
            model = load_model(filename)
        except FileNotFoundError:
            print(f"Model file not found for {display_name} ({filename}), skipping.")
            continue

        if not hasattr(model, "predict_proba"):
            print(f"Model {display_name} has no predict_proba, skipping.")
            continue

        y_proba = model.predict_proba(X_test_df)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        plt.plot(
            fpr,
            tpr,
            label=f"{display_name} (AUC = {auc:.3f})",
        )

    # Plot ROC for MLP (PyTorch)
    print("Computing ROC for MLP model...")
    mlp_probs = get_mlp_proba(X_test_df)
    if mlp_probs is not None:
        fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp_probs)
        auc_mlp = roc_auc_score(y_test, mlp_probs)

        plt.plot(
            fpr_mlp,
            tpr_mlp,
            label=f"MLP (PyTorch) (AUC = {auc_mlp:.3f})",
        )

    # Diagonal line (random classifier)
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves - churn models")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
