"""Odtworzenie wszystkich wyników jednym poleceniem.

Kolejność: (strojenie modelu) → przebieg zaszyfrowany → grupy kontrolne → metryki,
zgodność i błąd kwantyzacji → ataki 1–4 → wykresy. Każdy krok zapisuje swój wynik do
`results/`, więc kolejne kroki czytają z plików i można je też uruchamiać osobno.

Szyfrogramy celowo nie są odtwarzalne (`phe` losuje `r` z `SystemRandom`), ale wyniki są:
dwa przebiegi dają identyczne `metrics.csv`, `agreement.json`, `quantization.json`,
`attack_sparse.json` i `exponent_leak.csv`. Różnią się tylko zmierzone czasy (`pipeline_run.json`).

Uruchomienie (pełny przebieg ok. 15 min na 15 rdzeniach, klucz 2048 bit):
- `python -m src.run_all`
- bez ponownego szyfrowania zbioru testowego (z istniejącego `results/cache/scores.npz`):
  `python -m src.run_all --skip-pipeline`
- ze strojeniem hiperparametrów od nowa (nadpisuje `best_params.json`): `--tune`
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

from src import evaluate, plots
from src.attacks import ciphertext_classifier, exponent_leak, nondeterminism, sparse_leak
from src.baselines import baseline_predictions, fit_logreg, save_baselines
from src.config import KEY_BITS, RESULTS_DIR
from src.data import load_split
from src.model import best_params, fit_best_model, fit_model
from src.pipeline import run as run_pipeline
from src.pipeline import save_summary

NONDETERMINISM_FILE = RESULTS_DIR / "attack_nondeterminism.json"


def _step(name: str, action: Callable[[], object]) -> None:
    """Wykonuje krok i wypisuje jego czas, żeby było widać, gdzie idzie pełny przebieg."""
    print(f"→ {name} ...", flush=True)
    start = time.perf_counter()
    action()
    print(f"  {name}: {time.perf_counter() - start:.1f} s", flush=True)


def tune() -> None:
    """Stroi NB od nowa (5-fold CV na train) i zapisuje `best_params.json`, `cv_results.csv`."""
    texts_train, _, y_train, _ = load_split()
    fit_model(texts_train, y_train)


def pipeline(key_bits: int, workers: int | None) -> None:
    """Pełny przebieg zaszyfrowany na zbiorze testowym → `scores.npz`, `pipeline_run.json`."""
    save_summary(run_pipeline(key_bits=key_bits, workers=workers))


def baselines() -> None:
    """Jawny NB, Dummy i regresja logistyczna na zbiorze testowym → `baselines.npz`."""
    texts_train, texts_test, y_train, y_test = load_split()
    model = fit_best_model(texts_train, y_train)
    logreg = fit_logreg(texts_train, y_train, best_params())
    save_baselines(y_test, baseline_predictions(model, texts_train, y_train, texts_test, logreg))


def evaluation() -> None:
    """Tabela metryk, zgodność i błąd kwantyzacji (tylko z cache)."""
    evaluate.save_metrics_table(evaluate.metrics_table())
    evaluate.save_json(evaluate.agreement(), RESULTS_DIR / "agreement.json")
    evaluate.save_json(evaluate.quantization_error(), RESULTS_DIR / "quantization.json")


def attacks(key_bits: int, workers: int | None) -> None:
    """Ataki 1–4; atak 4 nie szyfruje, więc nie zależy od `key_bits` i `workers`."""
    ciphertext_classifier.save_result(ciphertext_classifier.run(key_bits=key_bits, workers=workers))
    sparse_leak.save_result(sparse_leak.run())
    result = nondeterminism.run(key_bits=key_bits, workers=workers)
    NONDETERMINISM_FILE.write_text(json.dumps(result, indent=2) + "\n")
    exponent_leak.save_result(exponent_leak.run())


def figures() -> None:
    """Wykresy do raportu w `results/figures/`."""
    plots.plot_roc()
    plots.plot_accuracy_vs_k()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key-bits", type=int, default=KEY_BITS)
    parser.add_argument("--workers", type=int, help="liczba procesów (domyślnie wszystkie rdzenie)")
    parser.add_argument("--skip-pipeline", action="store_true", help="użyj istniejącego scores.npz")
    parser.add_argument("--tune", action="store_true", help="stroi hiperparametry od nowa")
    args = parser.parse_args()

    start = time.perf_counter()
    if args.tune:
        _step("strojenie modelu", tune)
    if not args.skip_pipeline:
        _step("przebieg zaszyfrowany", lambda: pipeline(args.key_bits, args.workers))
    _step("grupy kontrolne", baselines)
    _step("metryki", evaluation)
    _step("ataki", lambda: attacks(args.key_bits, args.workers))
    _step("wykresy", figures)
    print(f"gotowe w {(time.perf_counter() - start) / 60:.1f} min, wyniki w results/")


if __name__ == "__main__":
    main()
