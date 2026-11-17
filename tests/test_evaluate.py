import numpy as np
import pytest

from src.evaluate import METRICS, classification_metrics


def test_classification_metrics_for_spam_class():
    y_true = np.array([1, 1, 0, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    scores = np.array([2.0, -0.5, -3.0, -1.0, 0.5, 1.0])

    metrics = classification_metrics(y_true, y_pred, scores)

    assert list(metrics) == list(METRICS)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["roc_auc"] == pytest.approx(8 / 9)  # jedna zła para: (−0.5 spam, 0.5 ham)


def test_classification_metrics_for_constant_classifier():
    y_true = np.array([0, 0, 0, 1])

    metrics = classification_metrics(y_true, np.zeros(4, dtype=int), np.zeros(4))

    assert metrics == {"accuracy": 0.75, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.5}
