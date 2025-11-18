from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_platform.data.loader import load_processed_split
from churn_platform.features.build_features import split_features_and_target
from churn_platform.models.evaluate import evaluate_classifier
from churn_platform.models.persistence import save_model


def _build_rf_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    """
    Build a random forest pipeline with preprocessing
    (same idea as your baseline).
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf),
        ]
    )

    return model


def tune_random_forest(
    n_iter: int = 25,
    random_state: int = 42,
) -> Tuple[Pipeline, dict]:
    """
    Tune a random forest pipeline using RandomizedSearchCV on the training set.
    Evaluate the best model on the validation set and save it.

    Returns:
        best_model, best_params
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

    print("Creating random forest pipeline...")
    base_model = _build_rf_pipeline(numeric_cols, categorical_cols)

    # Hyperparameter search space
    param_distributions = {
        "classifier__n_estimators": [100, 200, 300, 500, 800],
        "classifier__max_depth": [None, 5, 8, 12, 16, 20],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
        "classifier__max_features": ["sqrt", "log2", 0.5, 0.8],
    }

    print("Starting RandomizedSearchCV...")
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",  # we target a good global ranking of churners
        n_jobs=-1,
        cv=3,
        verbose=1,
        random_state=random_state,
    )

    search.fit(X_train, y_train)

    print(f"\nBest ROC AUC (CV): {search.best_score_:.4f}")
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    best_model: Pipeline = search.best_estimator_

    print("\nEvaluating best model on validation set...")
    metrics_val = evaluate_classifier(best_model, X_val, y_val)

    print("Validation metrics (tuned random forest):")
    for name, value in metrics_val.items():
        print(f"  {name}: {value:.4f}")

    print("\nSaving tuned random forest model as rf_tuned.joblib ...")
    save_model(best_model, filename="rf_tuned.joblib")

    print("Tuning finished.")

    return best_model, search.best_params_
