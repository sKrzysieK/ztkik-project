"""Kodowanie stałoprzecinkowe cech i wag do Pythonowych `int`.

Na liczbach całkowitych `phe` ustawia wykładnik szyfrogramu na 0, więc nie zdradza on
wartości, a obliczenie homomorficzne jest dokładne (porównanie `==` z wersją jawną).

Cechy i wagi skalujemy przez `SCALE`, a wyraz wolny przez `SCALE**2`, bo iloczyn
`x_i · w_i` ma już skalę `SCALE**2`. Wynik dekodujemy dzieleniem przez `SCALE**2`.
"""

from __future__ import annotations

import numpy as np

from src.config import SCALE


def _scale_to_ints(values: np.ndarray, scale: int) -> np.ndarray:
    """Zaokrągla `values * scale` do najbliższej liczby całkowitej (połówki do parzystej)."""
    return np.rint(np.asarray(values, dtype=np.float64) * scale).astype(np.int64)


def encode_features(X: np.ndarray) -> list[list[int]]:
    """Zwraca `round(x * SCALE)` jako Pythonowe `int`."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X musi być macierzą 2D, a ma kształt {X.shape}")
    return _scale_to_ints(X, SCALE).tolist()


def encode_weights(b: float, w: np.ndarray) -> tuple[int, list[int]]:
    """Zwraca `round(b * SCALE**2)` i `round(w * SCALE)` jako Pythonowe `int`."""
    w = np.asarray(w)
    if w.ndim != 1:
        raise ValueError(f"w musi być wektorem 1D, a ma kształt {w.shape}")
    b_int = round(float(b) * SCALE**2)
    return b_int, _scale_to_ints(w, SCALE).tolist()


def plain_int_scores(X_int: list[list[int]], b_int: int, w_int: list[int]) -> list[int]:
    """Liczy jawnie to samo co serwer: `b_int + Σ x_int_i · w_int_i` dla każdego wiersza."""
    return [
        b_int + sum(x * w for x, w in zip(row, w_int, strict=True)) for row in X_int
    ]


def decode_score(delta_int: int) -> float:
    """Zamienia wynik całkowity z powrotem na log-odds: `delta_int / SCALE**2`."""
    return delta_int / SCALE**2
