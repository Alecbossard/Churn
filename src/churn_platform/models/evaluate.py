from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def evaluate_classifier(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float]:
    """
    Compute basic classification metrics and print a report.
    """
    y_pred = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }

    # If the model supports predict_proba, compute ROC AUC as well
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y, y_proba)

    print("Classification report:")
    print(classification_report(y, y_pred, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))

    return metrics
