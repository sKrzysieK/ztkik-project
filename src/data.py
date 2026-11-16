"""Wczytanie SMS Spam Collection, deduplikacja i stratyfikowany podział 80/20."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def load_dataset() -> pd.DataFrame:
    """Zwraca zbiór po deduplikacji: kolumny `label` (1 = spam, 0 = ham) i `text`."""
    raise NotImplementedError  # P02


def load_split() -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Zwraca `texts_train, texts_test, y_train, y_test` (stratyfikowany 80/20, stały seed)."""
    raise NotImplementedError  # P03
