from churn_platform.data.loader import load_processed_split
from churn_platform.features.build_features import split_features_and_target
from churn_platform.models.baselines import (
    create_logreg_baseline,
    create_random_forest_baseline,
    create_xgboost_baseline,
)
from churn_platform.models.evaluate import evaluate_classifier
from churn_platform.models.persistence import save_model


def train_logreg_baseline() -> None:
    """
    Train a baseline logistic regression model and evaluate it on the validation set.
    """
    print("Loading train and validation splits...")
    train_df = load_processed_split("train")
    val_df = load_processed_split("val")

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")

    print("Building features for train set...")
    X_train, y_train, numeric_cols, categorical_cols = split_features_and_target(train_df)

    print("Building features for validation set...")
    X_val, y_val, _, _ = split_features_and_target(val_df)

    print("Creating logistic regression baseline model...")
    model = create_logreg_baseline(numeric_cols, categorical_cols)

    print("Fitting model on train set...")
    model.fit(X_train, y_train)

    print("Evaluating model on validation set...")
    metrics = evaluate_classifier(model, X_val, y_val)

    print("Validation metrics (logistic regression):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print("Saving model...")
    save_model(model, filename="logreg_baseline.joblib")

    print("Training finished.")


def train_random_forest_baseline() -> None:
    """
    Train a baseline random forest model and evaluate it on the validation set.
    """
    print("Loading train and validation splits...")
    train_df = load_processed_split("train")
    val_df = load_processed_split("val")

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")

    print("Building features for train set...")
    X_train, y_train, numeric_cols, categorical_cols = split_features_and_target(train_df)

    print("Building features for validation set...")
    X_val, y_val, _, _ = split_features_and_target(val_df)

    print("Creating random forest baseline model...")
    model = create_random_forest_baseline(numeric_cols, categorical_cols)

    print("Fitting model on train set...")
    model.fit(X_train, y_train)

    print("Evaluating model on validation set...")
    metrics = evaluate_classifier(model, X_val, y_val)

    print("Validation metrics (random forest):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print("Saving model...")
    save_model(model, filename="rf_baseline.joblib")

    print("Training finished.")


def train_xgboost_baseline() -> None:
    """
    Train a baseline XGBoost model and evaluate it on the validation set.
    """
    print("Loading train and validation splits...")
    train_df = load_processed_split("train")
    val_df = load_processed_split("val")

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")

    print("Building features for train set...")
    X_train, y_train, numeric_cols, categorical_cols = split_features_and_target(train_df)

    print("Building features for validation set...")
    X_val, y_val, _, _ = split_features_and_target(val_df)

    print("Creating XGBoost baseline model...")
    model = create_xgboost_baseline(numeric_cols, categorical_cols)

    print("Fitting model on train set...")
    model.fit(X_train, y_train)

    print("Evaluating model on validation set...")
    metrics = evaluate_classifier(model, X_val, y_val)

    print("Validation metrics (XGBoost):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print("Saving model...")
    save_model(model, filename="xgb_baseline.joblib")

    print("Training finished.")
