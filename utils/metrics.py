"""
metrics.py
----------
Evaluation metrics for BANNet and ARPOS.

Supports:
- Accuracy
- Precision / Recall / F1
- Confusion Matrix
- Regression metrics (MAE, MSE)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)


# -----------------------------------------------------------
# Classification Metrics
# -----------------------------------------------------------
def classification_scores(y_true, y_pred):
    """
    Compute all classification metrics.

    Returns dict:
        {
            "accuracy": ...,
            "precision": ...,
            "recall": ...,
            "f1": ...,
            "confusion_matrix": ...
        }
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }


# -----------------------------------------------------------
# Regression Metrics
# -----------------------------------------------------------
def regression_scores(y_true, y_pred):
    """
    Compute regression evaluation metrics.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }
