"""Kodowanie stałoprzecinkowe cech i wag do Pythonowych `int`.

Na liczbach całkowitych `phe` ustawia wykładnik szyfrogramu na 0, więc nie zdradza on
wartości, a obliczenie homomorficzne jest dokładne (porównanie `==` z wersją jawną).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def encode_features(X: np.ndarray) -> list[list[int]]:
    """Zwraca `round(x * SCALE)` jako Pythonowe `int`."""
    raise NotImplementedError  # K05


def encode_weights(b: float, w: np.ndarray) -> tuple[int, list[int]]:
    """Zwraca `round(b * SCALE**2)` i `round(w * SCALE)` jako Pythonowe `int`."""
    raise NotImplementedError  # K05


def plain_int_scores(X_int: list[list[int]], b_int: int, w_int: list[int]) -> list[int]:
    """Liczy jawnie to samo co serwer: `b_int + Σ x_int_i · w_int_i` dla każdego wiersza."""
    raise NotImplementedError  # K05


def decode_score(delta_int: int) -> float:
    """Zamienia wynik całkowity z powrotem na log-odds: `delta_int / SCALE**2`."""
    raise NotImplementedError  # K05
