"""Strona klienta: klucze Paillier, szyfrowanie cech, odszyfrowanie wyniku."""

from __future__ import annotations

import multiprocessing
import os
from functools import partial
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

    `workers`: liczba procesów; `None` = wszystkie rdzenie, `1` = bez `multiprocessing`.
    Procesy są uruchamiane metodą `spawn`, więc skrypt wywołujący tę funkcję z
    `workers > 1` musi mieć kod startowy pod `if __name__ == "__main__":`.
    """
    rows = encode_features(X)
    if workers is None:
        workers = os.cpu_count() or 1
    if workers < 1:
        raise ValueError(f"workers musi być >= 1, a jest {workers}")
    workers = min(workers, len(rows))
    if workers <= 1:
        return [_encrypt_row(pub, row) for row in rows]

    # Kilka paczek na proces wyrównuje obciążenie, a nie mnoży narzutu na przesyłanie.
    chunksize = max(1, len(rows) // (workers * 4))
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        return pool.map(partial(_encrypt_row, pub), rows, chunksize=chunksize)


def _encrypt_row(pub: PaillierPublicKey, row: list[int]) -> list[EncryptedNumber]:
    """Szyfruje jeden wiersz; na poziomie modułu, żeby dało się go wysłać do procesu."""
    return [pub.encrypt(x) for x in row]


def decrypt_predict(
    priv: PaillierPrivateKey, enc_scores: list[EncryptedNumber]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Zwraca `y_pred` (0/1), `scores` (log-odds, float) i surowe wyniki `int`."""
    raise NotImplementedError  # K10
