"""Pomiar kosztu protokołu: czas szyfrowania, obliczenia serwera i odszyfrowania na wiadomość.

Dane są syntetyczne (jak TF-IDF z SMS-ów), bo koszt nie zależy od treści wiadomości,
tylko od długości klucza i liczby cech k.

Uruchomienie: `python -m src.benchmark --key-bits 1024 --k 100 --n 5`
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from src.config import SEED
from src.crypto_client import decrypt_predict, encrypt_vectors, generate_keys
from src.crypto_server import server_score

# Średnio kilkanaście różnych słów w SMS-ie; wagi log-odds NB są rzędu kilku jednostek.
WORDS_PER_MESSAGE = 15
WEIGHT_RANGE = 8.0


def synthetic_data(
    n_messages: int, k: int, seed: int = SEED
) -> tuple[np.ndarray, float, np.ndarray]:
    """Zwraca `X` (rzadkie wiersze z normalizacją L2), `b` i `w` o realistycznych zakresach."""
    rng = np.random.default_rng(seed)
    nnz = min(WORDS_PER_MESSAGE, k)
    X = np.zeros((n_messages, k))
    for row in X:
        row[rng.choice(k, size=nnz, replace=False)] = rng.random(nnz)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    w = rng.uniform(-WEIGHT_RANGE, WEIGHT_RANGE, size=k)
    b = float(rng.uniform(-3, 0))
    return X, b, w


def measure(key_bits: int, k: int, n_messages: int, workers: int = 1) -> dict[str, float | int]:
    """Mierzy średni czas na wiadomość (ms) dla każdego etapu i rozmiar szyfrogramu (B).

    Domyślnie `workers=1`, żeby wynik był kosztem jednego rdzenia, niezależnym od maszyny.
    Generowanie kluczy nie wchodzi do pomiaru (robi się je raz).
    """
    X, b, w = synthetic_data(n_messages, k)
    pub, priv = generate_keys(key_bits)

    t0 = time.perf_counter()
    enc_X = encrypt_vectors(pub, X, workers=workers)
    t1 = time.perf_counter()
    enc_scores = server_score(enc_X, b, w)
    t2 = time.perf_counter()
    decrypt_predict(priv, enc_scores)
    t3 = time.perf_counter()

    per_message_ms = 1000 / n_messages
    return {
        "key_bits": key_bits,
        "k": k,
        "n_messages": n_messages,
        "workers": workers,
        "t_enc_ms": (t1 - t0) * per_message_ms,
        "t_server_ms": (t2 - t1) * per_message_ms,
        "t_dec_ms": (t3 - t2) * per_message_ms,
        # Szyfrogram to element Z_{n^2}: tyle bajtów zajmuje po serializacji.
        "ct_bytes": (pub.nsquare.bit_length() + 7) // 8,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key-bits", type=int, default=1024)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--n", type=int, default=5, help="liczba wiadomości")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    result = measure(args.key_bits, args.k, args.n, workers=args.workers)
    for name, value in result.items():
        print(f"{name:>12}: {value:.2f}" if isinstance(value, float) else f"{name:>12}: {value}")


if __name__ == "__main__":
    main()
