"""Atak 3: dlaczego szyfrowanie musi być probabilistyczne (losowe `r` w każdym szyfrogramie).

Szyfrogram Paillier (g = n + 1) to `c = (1 + n·m) · r^n mod n^2`. Porównujemy dwa warianty:

- losowy (`phe`, nasz protokół): nowe `r` przy każdym szyfrowaniu;
- deterministyczny: jedno tajne `r` użyte do wszystkich szyfrogramów (np. błąd implementacji).

W wariancie deterministycznym równe wartości dają równe szyfrogramy, więc serwer widzi
rozkład częstości. Wektory TF-IDF są rzadkie, więc najczęstszy szyfrogram to `Enc(0) = r^n`.
Wtedy dla każdego `c`: `c · Enc(0)^{-1} = 1 + n·m (mod n^2)`, czyli serwer odczytuje `m`
bez klucza prywatnego. W wariancie losowym ta sama procedura nie odzyskuje niczego.

Uruchomienie: `python -m src.attacks.nondeterminism`
"""

from __future__ import annotations

import argparse
import secrets
from collections import Counter

from phe import PaillierPublicKey

from src.benchmark import synthetic_data
from src.config import KEY_BITS
from src.crypto_client import encrypt_vectors, generate_keys
from src.fixed_point import encode_features


def fixed_r_encrypt(pub: PaillierPublicKey, X_int: list[list[int]], r: int) -> list[list[int]]:
    """Wariant deterministyczny: każda wartość szyfrowana z tym samym `r`."""
    r_n = pow(r, pub.n, pub.nsquare)
    return [[(1 + pub.n * m) * r_n % pub.nsquare for m in row] for row in X_int]


def recover_assuming_zero_most_frequent(
    pub: PaillierPublicKey, ciphertexts: list[list[int]]
) -> list[list[int | None]]:
    """Atak serwera: uznaje najczęstszy szyfrogram za `Enc(0)` i odzyskuje z niego resztę.

    Zwraca odzyskaną wartość albo `None`, gdy `c · Enc(0)^{-1} - 1` nie dzieli się przez `n`
    (czyli założenie wspólnego `r` nie zachodzi dla tego szyfrogramu).
    """
    zero_guess, _ = Counter(c for row in ciphertexts for c in row).most_common(1)[0]
    zero_inv = pow(zero_guess, -1, pub.nsquare)

    def recover(c: int) -> int | None:
        u = c * zero_inv % pub.nsquare
        return (u - 1) // pub.n if (u - 1) % pub.n == 0 else None

    return [[recover(c) for c in row] for row in ciphertexts]


def recovered_fraction(recovered: list[list[int | None]], X_int: list[list[int]]) -> float:
    """Odsetek pozycji, dla których atak odzyskał dokładnie prawdziwą wartość."""
    pairs = [(r, x) for r_row, x_row in zip(recovered, X_int) for r, x in zip(r_row, x_row)]
    return sum(r == x for r, x in pairs) / len(pairs)


def run(
    key_bits: int = KEY_BITS, n_messages: int = 30, k: int = 200, workers: int | None = None
) -> dict[str, float | int]:
    """Szyfruje te same syntetyczne wiadomości obu wariantami i porównuje, co widzi serwer."""
    X, _, _ = synthetic_data(n_messages, k)
    X_int = encode_features(X)
    pub, _ = generate_keys(key_bits)

    random_ct = [
        [c.ciphertext(be_secure=False) for c in row]
        for row in encrypt_vectors(pub, X, workers=workers)
    ]
    fixed_ct = fixed_r_encrypt(pub, X_int, r=secrets.randbelow(pub.n - 2) + 2)

    def distinct(rows: list[list[int]]) -> int:
        return len({v for row in rows for v in row})

    return {
        "n_ciphertexts": n_messages * k,
        "distinct_plaintexts": distinct(X_int),
        "distinct_ct_random": distinct(random_ct),
        "distinct_ct_fixed_r": distinct(fixed_ct),
        "recovered_random": recovered_fraction(
            recover_assuming_zero_most_frequent(pub, random_ct), X_int
        ),
        "recovered_fixed_r": recovered_fraction(
            recover_assuming_zero_most_frequent(pub, fixed_ct), X_int
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key-bits", type=int, default=KEY_BITS)
    parser.add_argument("--n", type=int, default=30, help="liczba wiadomości")
    parser.add_argument("--k", type=int, default=200, help="liczba cech")
    args = parser.parse_args()

    result = run(args.key_bits, args.n, args.k)
    print(f"szyfrogramów: {result['n_ciphertexts']}, różnych wartości jawnych: {result['distinct_plaintexts']}")
    for variant, label in (("random", "losowe r"), ("fixed_r", "stałe r")):
        print(
            f"{label:>8}: różnych szyfrogramów {result[f'distinct_ct_{variant}']:>5}, "
            f"odzyskane bez klucza {result[f'recovered_{variant}']:.2%}"
        )


if __name__ == "__main__":
    main()
