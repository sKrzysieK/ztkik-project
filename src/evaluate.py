"""Metryki klasyfikacji i ewaluacja wyników zapisanych w `results/cache/`."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    """Zwraca accuracy oraz precision, recall i F1 klasy spam (1), a ROC-AUC z `scores`.

    `zero_division=0`: klasyfikator, który nigdy nie mówi „spam” (Dummy), ma precision 0,
    a nie ostrzeżenie. Stały wynik (`scores`) daje ROC-AUC 0,5.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }
