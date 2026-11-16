"""Strona klienta: klucze Paillier, szyfrowanie cech, odszyfrowanie wyniku."""

from __future__ import annotations

from typing import TYPE_CHECKING

from phe import PaillierPrivateKey, PaillierPublicKey, generate_paillier_keypair

from src.config import KEY_BITS

if TYPE_CHECKING:
    import numpy as np
    from phe import EncryptedNumber


def generate_keys(bits: int = KEY_BITS) -> tuple[PaillierPublicKey, PaillierPrivateKey]:
    """Generuje parę kluczy Paillier o długości modułu `n` równej `bits`."""
    return generate_paillier_keypair(n_length=bits)


def encrypt_vectors(
    pub: PaillierPublicKey, X: np.ndarray, workers: int | None = None
) -> list[list[EncryptedNumber]]:
    """Szyfruje każdą pozycję każdego wiersza (także zera) po zakodowaniu do `int`."""
    raise NotImplementedError  # K07, K08


def decrypt_predict(
    priv: PaillierPrivateKey, enc_scores: list[EncryptedNumber]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Zwraca `y_pred` (0/1), `scores` (log-odds, float) i surowe wyniki `int`."""
    raise NotImplementedError  # K10
