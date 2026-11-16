"""Strona klienta: klucze Paillier, szyfrowanie cech, odszyfrowanie wyniku."""

from __future__ import annotations

from typing import TYPE_CHECKING

from phe import PaillierPrivateKey, PaillierPublicKey, generate_paillier_keypair

from src.config import KEY_BITS
from src.fixed_point import encode_features

if TYPE_CHECKING:
    import numpy as np
    from phe import EncryptedNumber


def generate_keys(bits: int = KEY_BITS) -> tuple[PaillierPublicKey, PaillierPrivateKey]:
    """Generuje parę kluczy Paillier o długości modułu `n` równej `bits`."""
    return generate_paillier_keypair(n_length=bits)


def encrypt_vectors(
    pub: PaillierPublicKey, X: np.ndarray, workers: int | None = None
) -> list[list[EncryptedNumber]]:
    """Szyfruje każdą pozycję każdego wiersza (także zera) po zakodowaniu do `int`.

    Szyfrujemy pełny wektor: wysłanie tylko pozycji niezerowych zdradziłoby serwerowi,
    które słowa występują w wiadomości. Wartości całkowite dają wykładnik 0 w każdym
    szyfrogramie, więc wykładnik też nic nie zdradza.
    """
    if workers not in (None, 1):
        raise NotImplementedError("szyfrowanie równoległe: K08")
    return [[pub.encrypt(x) for x in row] for row in encode_features(X)]


def decrypt_predict(
    priv: PaillierPrivateKey, enc_scores: list[EncryptedNumber]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Zwraca `y_pred` (0/1), `scores` (log-odds, float) i surowe wyniki `int`."""
    raise NotImplementedError  # K10
