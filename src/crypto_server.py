"""Strona serwera: homomorficzny iloczyn skalarny na zaszyfrowanych cechach.

Serwer zna tylko klucz publiczny (zawarty w szyfrogramach) i jawny model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from phe import EncryptedNumber


def server_score(
    enc_X: list[list[EncryptedNumber]], b: float, w: np.ndarray
) -> list[EncryptedNumber]:
    """Zwraca `Enc(Δ) = Σ Enc(x_i) · w_i + b` dla każdej wiadomości."""
    raise NotImplementedError  # K09
