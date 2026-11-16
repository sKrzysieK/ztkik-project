"""Model jawny: TF-IDF -> selekcja chi² -> Multinomial NB, eksport wag log-odds."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from sklearn.pipeline import Pipeline


def fit_model(texts_train: list[str], y_train: np.ndarray) -> Pipeline:
    """Stroi hiperparametry 5-fold CV na zbiorze treningowym i zwraca najlepszy `Pipeline`."""
    raise NotImplementedError  # P05


def features(model: Pipeline, texts: list[str]) -> np.ndarray:
    """Zwraca macierz cech `(n, k)`, dense, `float64`."""
    raise NotImplementedError  # P06


def log_odds_weights(model: Pipeline) -> tuple[float, np.ndarray]:
    """Zwraca `b`, `w` takie, że spam <=> `b + x·w > 0`."""
    raise NotImplementedError  # P06
