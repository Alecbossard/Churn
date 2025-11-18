import os
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_platform.data.loader import load_processed_split, get_project_root
from churn_platform.features.build_features import split_features_and_target
from churn_platform.models.dl_models import SimpleMLP


def main() -> None:
    # --- 1. Load data ---
    print("Loading train and validation splits...")
    train_df = load_processed_split("train")
    val_df = load_processed_split("val")

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")

    # --- 2. Build features ---
    print("Building features for train set...")
    X_train_df, y_train, numeric_cols, categorical_cols = split_features_and_target(train_df)

    print("Building features for validation set...")
    X_val_df, y_val, _, _ = split_features_and_target(val_df)

    # Preprocessor: scaler for numeric, one-hot for categorical
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,  # dense output for PyTorch
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    print("Fitting preprocessor on train set...")
    X_train_np = preprocessor.fit_transform(X_train_df)
    X_val_np = preprocessor.transform(X_val_df)

    print(f"Train features shape after encoding: {X_train_np.shape}")
    print(f"Val features shape after encoding:   {X_val_np.shape}")

    # --- 3. Convert to PyTorch tensors ---
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

    # Compute class imbalance for pos_weight
    n_pos = float(y_train_tensor.sum().item())
    n_total = float(y_train_tensor.shape[0])
    n_neg = n_total - n_pos

    pos_weight_value = n_neg / n_pos
    print(f"Positive class weight (pos_weight): {pos_weight_value:.3f}")

    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32)

    # --- 4. Create model, loss, optimizer ---
    input_dim = X_train_tensor.shape[1]
    hidden_dim = 64
    model = SimpleMLP(input_dim=input_dim, hidden_dim=hidden_dim)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 30

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()

        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation metrics at each epoch (threshold = 0.5)
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_probs = torch.sigmoid(val_logits).numpy()
            val_pred = (val_probs >= 0.5).astype(int)

            acc = accuracy_score(y_val, val_pred)
            prec = precision_score(y_val, val_pred, zero_division=0)
            rec = recall_score(y_val, val_pred, zero_division=0)
            f1 = f1_score(y_val, val_pred, zero_division=0)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:02d}/{num_epochs} - "
                f"Train loss: {loss.item():.4f} - "
                f"Val acc: {acc:.4f} - Val f1: {f1:.4f}"
            )

    # --- 5. Final evaluation on validation set (threshold 0.5) ---
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_tensor)
        val_probs = torch.sigmoid(val_logits).numpy()
        val_pred = (val_probs >= 0.5).astype(int)

    print("\nFinal evaluation on validation set (MLP, threshold = 0.5):")
    print("Classification report:")
    print(classification_report(y_val, val_pred, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(y_val, val_pred))

    metrics = {
        "accuracy": accuracy_score(y_val, val_pred),
        "precision": precision_score(y_val, val_pred, zero_division=0),
        "recall": recall_score(y_val, val_pred, zero_division=0),
        "f1": f1_score(y_val, val_pred, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_val, val_probs)
    except ValueError:
        metrics["roc_auc"] = float("nan")

    print("Validation metrics (MLP, threshold = 0.5):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # --- 6. Search for best threshold on validation set (by F1) ---
    best_thresh = 0.5
    best_f1 = metrics["f1"]

    for thresh in np.linspace(0.2, 0.8, 13):  # 0.20, 0.25, ..., 0.80
        preds = (val_probs >= thresh).astype(int)
        f1_t = f1_score(y_val, preds, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = float(thresh)

    print(f"\nBest threshold on validation set (by F1): {best_thresh:.2f} (F1 = {best_f1:.4f})")

    # --- 7. Save preprocessor + model for later use (ROC plot, etc.) ---
    root_dir = get_project_root()
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    preprocessor_path = os.path.join(models_dir, "mlp_preprocessor.joblib")
    model_path = os.path.join(models_dir, "mlp_model.pt")

    joblib.dump(preprocessor, preprocessor_path)
    torch.save(model.state_dict(), model_path)

    print(f"\nSaved MLP preprocessor to: {preprocessor_path}")
    print(f"Saved MLP model to: {model_path}")

    print("Training (MLP) finished.")


if __name__ == "__main__":
    main()
