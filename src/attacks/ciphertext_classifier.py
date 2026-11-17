"""Atak 1: czy serwer nauczy się odróżniać spam od hamu na samych szyfrogramach?

Serwer zna klucz publiczny, więc może sam zaszyfrować dowolne oznaczone wiadomości (CPA)
i wytrenować na nich klasyfikator. Z każdego szyfrogramu `c` liczymy cechy, które mogłyby
zdradzić coś o wartości jawnej: najmłodszy bajt, reszty modulo 3, 5, 7, długość w bitach
i symbol Jacobiego `(c / n)`. Random Forest dostaje te cechy dla wszystkich k pozycji.

Paillier jest IND-CPA (przy założeniu DCRA), więc oczekujemy wyniku na poziomie
klasyfikatora większościowego. Dla porównania ten sam Random Forest na jawnych cechach
pokazuje, że przy tej liczbie wiadomości klasyfikator ma się z czego uczyć — jeśli
szyfrogramy czegoś by nie ukrywały, byłoby to widać.

Symbol Jacobiego to przykład cechy, która łamie inne szyfry (Goldwasser–Micali): tutaj
`(c / n) = (r / n)` (bo `(1 + n)^m ≡ 1 mod n`, a `n` jest nieparzyste), więc zależy tylko
od losowego `r`, nie od `m`.

Uruchomienie: `python -m src.attacks.ciphertext_classifier [--n 400] [--key-bits 2048]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gmpy2
import numpy as np
from phe import PaillierPublicKey
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.config import KEY_BITS, RESULTS_DIR, ROOT, SEED
from src.crypto_client import encrypt_vectors, generate_keys
from src.data import load_split
from src.model import features, fit_best_model

N_MESSAGES = 400  # tyle szyfruje „serwer”; k = 1000 i 2048 bit to ok. 2 min na 15 rdzeniach
CV_FOLDS = 5
RESULT_FILE = RESULTS_DIR / "attack_ciphertext.json"

FEATURE_NAMES = ("low_byte", "mod3", "mod5", "mod7", "bit_length", "jacobi")


def ciphertext_features(pub: PaillierPublicKey, ciphertexts: list[list[int]]) -> np.ndarray:
    """Zwraca macierz `(n, k · 6)`: dla każdej pozycji cechy z `FEATURE_NAMES`.

    Długość w bitach liczymy względem `n²` (ile zer wiodących), żeby nie zależała od klucza.
    """
    nsquare_bits = pub.nsquare.bit_length()
    n = gmpy2.mpz(pub.n)

    def position(c: int) -> tuple[int, ...]:
        return (c & 0xFF, c % 3, c % 5, c % 7, nsquare_bits - c.bit_length(), gmpy2.jacobi(c, n))

    return np.array(
        [[value for c in row for value in position(c)] for row in ciphertexts], dtype=np.int64
    )


def cv_scores(X: np.ndarray, y: np.ndarray, classifier) -> dict[str, float]:
    """Accuracy i F1 (spam) z 5-fold stratyfikowanej CV: średnia i odchylenie."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = cross_validate(classifier, X, y, cv=cv, scoring=("accuracy", "f1"))
    return {
        f"{stat}_{name}": round(float(getattr(np, stat)(scores[f"test_{name}"])), 4)
        for name in ("accuracy", "f1")
        for stat in ("mean", "std")
    }


def _forest() -> RandomForestClassifier:
    """Random Forest używany w obu wariantach (szyfrogramy i kontrola na jawnych cechach).

    Ustawienia dobrane tak, żeby RF dobrze radził sobie na jawnych cechach (z domyślnymi
    przy 50 spamach F1 kontroli to tylko ok. 0,34): wagi klas kompensują 13% spamu, a
    większe `max_features` pomaga przy rzadkich cechach, z których mało która jest niezerowa.
    """
    return RandomForestClassifier(
        n_estimators=300, max_features=0.1, class_weight="balanced", random_state=SEED, n_jobs=-1
    )


def run(
    n_messages: int = N_MESSAGES, key_bits: int = KEY_BITS, workers: int | None = None
) -> dict:
    """Szyfruje `n_messages` oznaczonych wiadomości z train i porównuje trzy klasyfikatory.

    Wiadomości bierzemy ze zbioru treningowego (stratyfikowana próbka): serwer szyfruje
    teksty, które sam zna, zbioru testowego nie dotykamy.
    """
    texts_train, _, y_train, _ = load_split()
    texts, _, y, _ = train_test_split(
        texts_train, y_train, train_size=n_messages, stratify=y_train, random_state=SEED
    )
    model = fit_best_model(texts_train, y_train)
    X = features(model, texts)
    pub, _ = generate_keys(key_bits)

    ciphertexts = [
        [c.ciphertext(be_secure=False) for c in row]
        for row in encrypt_vectors(pub, X, workers=workers)
    ]
    X_ct = ciphertext_features(pub, ciphertexts)

    return {
        "n_messages": n_messages,
        "n_spam": int(y.sum()),
        "k": X.shape[1],
        "key_bits": key_bits,
        "ciphertext_rf": cv_scores(X_ct, y, _forest()),
        "plaintext_rf": cv_scores(X, y, _forest()),
        "dummy": cv_scores(X, y, DummyClassifier(strategy="most_frequent")),
    }


def save_result(result: dict, path: Path = RESULT_FILE) -> None:
    """Zapisuje wynik ataku jako JSON (mały plik, commitujemy go)."""
    path.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=N_MESSAGES, help="liczba szyfrowanych wiadomości")
    parser.add_argument("--key-bits", type=int, default=KEY_BITS)
    parser.add_argument("--workers", type=int, help="liczba procesów (domyślnie wszystkie rdzenie)")
    args = parser.parse_args()

    result = run(args.n, args.key_bits, args.workers)
    save_result(result)
    print(
        f"{result['n_messages']} wiadomości ({result['n_spam']} spam), k = {result['k']}, "
        f"klucz {result['key_bits']} bit, {CV_FOLDS}-fold CV"
    )
    for name, label in (
        ("ciphertext_rf", "RF na szyfrogramach"),
        ("plaintext_rf", "RF na jawnych cechach"),
        ("dummy", "klasa większościowa"),
    ):
        s = result[name]
        print(
            f"{label:>22}: accuracy {s['mean_accuracy']:.4f} ± {s['std_accuracy']:.4f}, "
            f"F1 {s['mean_f1']:.4f} ± {s['std_f1']:.4f}"
        )
    print(f"zapisano {RESULT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
