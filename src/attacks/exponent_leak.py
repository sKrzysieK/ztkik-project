"""Atak 4: wyciek przez jawny wykładnik (`exponent`), który `phe` dołącza do szyfrogramu.

Przy domyślnym kodowaniu floatów `phe` dobiera wykładnik (podstawa 16) do rzędu wielkości
wartości: `floor((frexp(x)[1] - 53) / 4)`. Daje to -14 dla 0.0 i dla wartości z [1/16, 1),
-15 i mniej dla wartości mniejszych niż 1/16, a -13 dla 1.0. Serwer zna kodowanie
(Kerckhoffs), więc liczy wykładnik zera sam i każdą pozycję z innym wykładnikiem uznaje
za niezerową, czyli wie, że dane słowo występuje w wiadomości.

Przy kodowaniu stałoprzecinkowym (nasz protokół) każda wartość to `int`, a jej wykładnik
to zawsze 0, więc wykładnik nic nie zdradza.

Wykładnik zależy tylko od kodowania, nie od losowości szyfrowania, dlatego liczymy go przez
`EncodedNumber.encode` bez kosztownego szyfrowania (test potwierdza, że jest taki sam).

Uruchomienie: `python -m src.attacks.exponent_leak`
"""

from __future__ import annotations

import argparse
from typing import Literal

import numpy as np
from phe import EncodedNumber, PaillierPublicKey

from src.benchmark import synthetic_data
from src.crypto_client import generate_keys
from src.fixed_point import encode_features

Encoding = Literal["float", "fixed"]


def exponents(pub: PaillierPublicKey, X: np.ndarray, encoding: Encoding) -> np.ndarray:
    """Zwraca macierz wykładników, które serwer zobaczyłby przy szyfrogramach `X`."""
    if encoding == "float":
        values = [[float(x) for x in row] for row in np.asarray(X)]
    elif encoding == "fixed":
        values = encode_features(X)
    else:
        raise ValueError(f"nieznane kodowanie: {encoding!r}")
    return np.array([[EncodedNumber.encode(pub, v).exponent for v in row] for row in values])


def exponent_leak(pub: PaillierPublicKey, X: np.ndarray, encoding: Encoding) -> dict[str, float]:
    """Mierzy, ile serwer odczytuje z samych wykładników.

    - `messages_leaking`: odsetek wiadomości z co najmniej jedną odczytaną pozycją niezerową;
    - `nonzero_revealed`: odsetek wszystkich pozycji niezerowych, które serwer odczytał.

    Serwer uznaje pozycję za niezerową, gdy jej wykładnik różni się od wykładnika zera.
    To nigdy nie daje fałszywego alarmu, bo kodowanie jest deterministyczne.
    """
    zero = 0.0 if encoding == "float" else 0
    zero_exponent = EncodedNumber.encode(pub, zero).exponent
    revealed = exponents(pub, X, encoding) != zero_exponent

    nonzero = np.asarray(X) != 0
    assert not (revealed & ~nonzero).any(), "wykładnik zera musi być stały"
    return {
        "messages_leaking": float(revealed.any(axis=1).mean()),
        "nonzero_revealed": float(revealed.sum() / max(nonzero.sum(), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1000, help="liczba wiadomości")
    parser.add_argument("--k", type=int, default=500, help="liczba cech")
    args = parser.parse_args()

    # Wykładnik nie zależy od długości klucza, więc mały klucz wystarczy.
    pub, _ = generate_keys(512)
    X, _, _ = synthetic_data(args.n, args.k)
    print(f"dane syntetyczne: {args.n} wiadomości × {args.k} cech")
    for encoding in ("float", "fixed"):
        leak = exponent_leak(pub, X, encoding)
        print(
            f"{encoding:>5}: wiadomości z wyciekiem {leak['messages_leaking']:7.2%}, "
            f"odczytane pozycje niezerowe {leak['nonzero_revealed']:7.2%}"
        )


if __name__ == "__main__":
    main()
