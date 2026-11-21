"""Atak 4: wyciek przez jawny wykładnik (`exponent`), który `phe` dołącza do szyfrogramu.

Przy domyślnym kodowaniu floatów `phe` dobiera wykładnik (podstawa 16) do rzędu wielkości
wartości: `floor((frexp(x)[1] - 53) / 4)`. Daje to -14 dla 0.0 i dla wartości z [1/16, 1),
-15 i mniej dla wartości mniejszych niż 1/16, a -13 dla 1.0. Serwer zna kodowanie
(Kerckhoffs), więc liczy wykładnik zera sam i każdą pozycję z innym wykładnikiem uznaje
za niezerową, czyli wie, że dane słowo występuje w wiadomości.

Przy kodowaniu stałoprzecinkowym (nasz protokół) każda wartość to `int`, a jej wykładnik
to zawsze 0, więc wykładnik nic nie zdradza.

Wykładnik zależy tylko od kodowania, nie od losowości szyfrowania ani od długości klucza,
dlatego liczymy go przez `EncodedNumber.encode` bez kosztownego szyfrowania (test potwierdza,
że jest taki sam). Wynik jest więc deterministyczny, a cały zbiór testowy liczy się w sekundy.

Uruchomienie:
- na prawdziwych cechach zbioru testowego, k = 500 i k najlepsze z CV, zapis do
  `results/exponent_leak.csv`: `python -m src.attacks.exponent_leak`
- na danych syntetycznych (zawyża wyciek, wektory gęstsze niż prawdziwe):
  `python -m src.attacks.exponent_leak --synthetic`
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Literal

import numpy as np
from phe import EncodedNumber, PaillierPublicKey

from src.benchmark import synthetic_data
from src.config import RESULTS_DIR, ROOT
from src.crypto_client import generate_keys
from src.data import load_split
from src.fixed_point import encode_features
from src.model import best_params, build_pipeline, features

Encoding = Literal["float", "fixed"]
ENCODINGS: tuple[Encoding, ...] = ("float", "fixed")
RESULT_FILE = RESULTS_DIR / "exponent_leak.csv"
# Wykładnik nie zależy od długości klucza, więc mały klucz wystarczy.
KEY_BITS = 512


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


def run(ks: tuple[int, ...] = (500,)) -> list[dict]:
    """Atak 4 na cechach zbioru testowego dla każdego `k` z `ks` i k najlepszego z CV.

    Model dla każdego `k` ma parametry z CV (`best_params.json`) poza `select__k`, czyli
    jest to ten sam model, którego wektory klient by szyfrował. Zwraca wiersz na parę
    (k, kodowanie) z liczbą wiadomości i pozycji niezerowych, żeby odsetki dało się
    przeliczyć na liczby.
    """
    texts_train, texts_test, y_train, _ = load_split()
    params = best_params()
    pub, _ = generate_keys(KEY_BITS)

    rows = []
    for k in sorted({*ks, params["select__k"]}):
        model = build_pipeline().set_params(**{**params, "select__k": k}).fit(texts_train, y_train)
        X = features(model, texts_test)
        for encoding in ENCODINGS:
            leak = exponent_leak(pub, X, encoding)
            rows.append(
                {
                    "k": k,
                    "encoding": encoding,
                    "n_messages": X.shape[0],
                    "n_nonzero": int(np.count_nonzero(X)),
                    "messages_leaking": round(leak["messages_leaking"], 6),
                    "nonzero_revealed": round(leak["nonzero_revealed"], 6),
                }
            )
    return rows


def save_result(rows: list[dict], path: Path = RESULT_FILE) -> None:
    """Zapisuje wiersze z `run` jako CSV (zawsze LF, jak `benchmark.csv`)."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _print_leak(label: str, leak: dict) -> None:
    print(
        f"{label}: wiadomości z wyciekiem {leak['messages_leaking']:7.2%}, "
        f"odczytane pozycje niezerowe {leak['nonzero_revealed']:7.2%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", action="store_true", help="dane syntetyczne zamiast prawdziwych")
    parser.add_argument("--n", type=int, default=1000, help="liczba wiadomości (tylko --synthetic)")
    parser.add_argument("--k", type=int, default=500, help="liczba cech (tylko --synthetic)")
    args = parser.parse_args()

    if args.synthetic:
        pub, _ = generate_keys(KEY_BITS)
        X, _, _ = synthetic_data(args.n, args.k)
        print(f"dane syntetyczne: {args.n} wiadomości × {args.k} cech")
        for encoding in ENCODINGS:
            _print_leak(f"{encoding:>5}", exponent_leak(pub, X, encoding))
        return

    rows = run()
    save_result(rows)
    print(f"zbiór testowy: {rows[0]['n_messages']} wiadomości")
    for row in rows:
        _print_leak(f"k = {row['k']:>4}, {row['encoding']:>5}", row)
    print(f"zapisano {RESULT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
