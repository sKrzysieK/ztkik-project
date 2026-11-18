"""Przebieg end-to-end protokołu na zbiorze testowym.

podział → model (parametry z CV) → cechy → klucze → szyfrowanie (klient) → iloczyn
skalarny (serwer) → odszyfrowanie i znak (klient) → `results/cache/scores.npz`.

Zapisujemy tylko wyniki `Δ`, nie szyfrogramy (przy k = 1000 to ok. 530 MB). Szyfrogramy
celowo nie są odtwarzalne (`phe` losuje `r`), ale wyniki są: obliczenie na liczbach
całkowitych jest dokładne, więc dwa przebiegi dają te same `delta_enc`.

Uruchomienie:
- szybka próba: `python -m src.pipeline --limit 50 --key-bits 1024`
- pełny przebieg: `python -m src.pipeline`
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.config import CACHE_DIR, KEY_BITS, ROOT
from src.crypto_client import decrypt_predict, encrypt_vectors, generate_keys
from src.crypto_server import server_score
from src.data import load_split
from src.fixed_point import encode_features, encode_weights, plain_int_scores
from src.model import features, fit_best_model, log_odds_weights

SCORES_FILE = CACHE_DIR / "scores.npz"


def run(
    limit: int | None = None,
    key_bits: int = KEY_BITS,
    workers: int | None = None,
    out: Path = SCORES_FILE,
) -> dict[str, float | int]:
    """Uruchamia protokół na pierwszych `limit` wiadomościach testowych (`None` = wszystkich).

    Zapisuje `y_test`, `delta_enc` (odszyfrowany wynik), `delta_float` (jawny NB bez
    kwantyzacji) i `exact_int_match` (czy odszyfrowany `int` == ten sam wynik liczony jawnie
    na liczbach całkowitych). Zwraca podsumowanie przebiegu z czasami etapów w sekundach.
    """
    texts_train, texts_test, y_train, y_test = load_split()
    if limit is not None:
        texts_test, y_test = texts_test[:limit], y_test[:limit]

    model = fit_best_model(texts_train, y_train)
    X = features(model, texts_test)
    b, w = log_odds_weights(model)
    pub, priv = generate_keys(key_bits)

    t0 = time.perf_counter()
    enc_X = encrypt_vectors(pub, X, workers=workers)
    t1 = time.perf_counter()
    enc_scores = server_score(enc_X, b, w)
    t2 = time.perf_counter()
    y_pred, delta_enc, raw = decrypt_predict(priv, enc_scores)
    t3 = time.perf_counter()

    plain = plain_int_scores(encode_features(X), *encode_weights(b, w))
    exact_int_match = np.array([d == p for d, p in zip(raw, plain, strict=True)])
    delta_float = b + X @ w

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        y_test=y_test,
        delta_enc=delta_enc,
        delta_float=delta_float,
        exact_int_match=exact_int_match,
    )
    return {
        "n_messages": len(y_test),
        "k": X.shape[1],
        "key_bits": key_bits,
        "t_enc_s": t1 - t0,
        "t_server_s": t2 - t1,
        "t_dec_s": t3 - t2,
        "exact_int_match": float(exact_int_match.mean()),
        "accuracy": float((y_pred == y_test).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="tylko pierwsze N wiadomości testowych")
    parser.add_argument("--key-bits", type=int, default=KEY_BITS)
    parser.add_argument("--workers", type=int, help="liczba procesów (domyślnie wszystkie rdzenie)")
    args = parser.parse_args()

    summary = run(args.limit, args.key_bits, args.workers)
    print(
        f"{summary['n_messages']} wiadomości, k = {summary['k']}, klucz {summary['key_bits']} bit\n"
        f"szyfrowanie {summary['t_enc_s']:.1f} s, serwer {summary['t_server_s']:.1f} s, "
        f"odszyfrowanie {summary['t_dec_s']:.2f} s\n"
        f"exact_int_match: {summary['exact_int_match']:.2%}, accuracy: {summary['accuracy']:.4f}\n"
        f"zapisano {SCORES_FILE.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
