"""Strona serwera: homomorficzny iloczyn skalarny na zaszyfrowanych cechach.

Serwer zna tylko klucz publiczny (zawarty w szyfrogramach) i jawny model. Moduł celowo
nie importuje niczego, co dotyczy klucza prywatnego.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.fixed_point import encode_weights

if TYPE_CHECKING:
    import numpy as np
    from phe import EncryptedNumber


def server_score(
    enc_X: list[list[EncryptedNumber]], b: float, w: np.ndarray
) -> list[EncryptedNumber]:
    """Zwraca `Enc(Δ) = Σ Enc(x_i) · w_i + b` dla każdej wiadomości.

    Wagi są kodowane tak samo jak cechy (`SCALE`), a wyraz wolny przez `SCALE**2`,
    więc wynik ma skalę `SCALE**2` i wykładnik 0, a klient dekoduje go `decode_score`.
    """
    b_int, w_int = encode_weights(b, w)
    return [_score_row(row, b_int, w_int) for row in enc_X]


def _score_row(row: list[EncryptedNumber], b_int: int, w_int: list[int]) -> EncryptedNumber:
    """Liczy `Enc(b_int + Σ x_i · w_int_i)` dla jednego zaszyfrowanego wiersza."""
    if not row:
        raise ValueError("pusty wiersz: brak cech do zsumowania")
    products = (c * w for c, w in zip(row, w_int, strict=True))
    score = next(products)
    for product in products:
        score += product
    return score + b_int
