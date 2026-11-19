"""Atak 2: co by wyciekło, gdyby klient szyfrował tylko niezerowe pozycje wektora cech.

Wysłanie samych pozycji niezerowych (z ich indeksami) byłoby ok. 300× tańsze (mediana 3
niezerowe cechy na 1000), ale serwer widziałby wtedy, **które** słowa ze słownika
występują w wiadomości — same wartości byłyby nadal zaszyfrowane. Sprawdzamy, ile to
wystarcza:

- `presence`: binarny wektor obecności słów → Bernoulli NB;
- `count`: sama liczba niezerowych pozycji (np. gdy indeksy są ukryte, ale długość
  przesyłki nie) → regresja logistyczna na jednej liczbie.

Serwer zna ekstrakcję cech (Kerckhoffs) i może przygotować oznaczone przykłady sam (CPA),
więc trenujemy na zbiorze treningowym, a oceniamy na testowym — tak jak jawny NB w
`results/metrics.csv`. Szyfrowanie nie jest potrzebne: serwer widziałby indeksy jawnie.

Uruchomienie: `python -m src.attacks.sparse_leak`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

from src.config import RESULTS_DIR, ROOT
from src.data import load_split
from src.evaluate import METRICS, classification_metrics
from src.model import features, fit_best_model

RESULT_FILE = RESULTS_DIR / "attack_sparse.json"


def presence(X: np.ndarray) -> np.ndarray:
    """Zwraca to, co serwer widzi przy wektorze rzadkim: które pozycje są niezerowe (0/1)."""
    return (np.asarray(X) != 0).astype(np.int64)


def nonzero_count(X: np.ndarray) -> np.ndarray:
    """Zwraca kolumnę `(n, 1)` z liczbą niezerowych pozycji w każdej wiadomości."""
    return presence(X).sum(axis=1, keepdims=True)


def attack_metrics(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, classifier
) -> dict[str, float]:
    """Trenuje klasyfikator serwera na `X_train` i zwraca metryki na zbiorze testowym."""
    classifier.fit(X_train, y_train)
    scores = classifier.predict_proba(X_test)[:, 1]
    return classification_metrics(y_test, classifier.predict(X_test), scores)


def run() -> dict:
    """Porównuje atak na obecności słów i na samej ich liczbie z jawnym NB.

    Klasyfikatory serwera są z domyślnymi parametrami (bez strojenia): wynik to dolne
    oszacowanie tego, co przeciwnik mógłby osiągnąć.
    """
    texts_train, texts_test, y_train, y_test = load_split()
    model = fit_best_model(texts_train, y_train)
    X_train, X_test = features(model, texts_train), features(model, texts_test)

    nnz = presence(X_test).sum(axis=1)
    return {
        "n_test": int(len(y_test)),
        "k": int(X_test.shape[1]),
        "median_nonzero": float(np.median(nnz)),
        "presence_bnb": attack_metrics(presence(X_train), y_train, presence(X_test), y_test, BernoulliNB()),
        "count_logreg": attack_metrics(
            nonzero_count(X_train), y_train, nonzero_count(X_test), y_test, LogisticRegression()
        ),
        "plain_nb": classification_metrics(
            y_test, model.predict(texts_test), model.predict_proba(texts_test)[:, 1]
        ),
    }


def save_result(result: dict, path: Path = RESULT_FILE) -> None:
    """Zapisuje wynik ataku jako JSON (metryki zaokrąglone do 4 miejsc)."""
    rounded = {
        key: {m: round(v, 4) for m, v in value.items()} if isinstance(value, dict) else value
        for key, value in result.items()
    }
    path.write_text(json.dumps(rounded, indent=2) + "\n")


def main() -> None:
    result = run()
    save_result(result)
    print(
        f"{result['n_test']} wiadomości testowych, k = {result['k']}, "
        f"mediana niezerowych pozycji: {result['median_nonzero']:.0f}"
    )
    print(f"{'':>28} " + " ".join(f"{m:>9}" for m in METRICS))
    for name, label in (
        ("presence_bnb", "obecność słów (Bernoulli NB)"),
        ("count_logreg", "liczba słów (log. regresja)"),
        ("plain_nb", "jawny NB (odniesienie)"),
    ):
        print(f"{label:>28} " + " ".join(f"{result[name][m]:9.4f}" for m in METRICS))
    print(f"zapisano {RESULT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
