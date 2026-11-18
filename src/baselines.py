"""Grupy kontrolne na zbiorze testowym: jawny Naive Bayes i klasyfikator większościowy.

- jawny NB: ten sam model co w wersji zaszyfrowanej, liczony bez szyfrowania i bez
  kwantyzacji — bezpośredni punkt odniesienia;
- `DummyClassifier(strategy="most_frequent")`: zawsze ham; pokazuje, ile daje samo
  niezbalansowanie klas (ok. 87%).

Wyniki trafiają do `results/cache/baselines.npz`, z którego czyta `src.evaluate`.

Uruchomienie: `python -m src.baselines`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.config import CACHE_DIR, ROOT
from src.data import load_split
from src.evaluate import METRICS, classification_metrics
from src.model import features, fit_best_model, log_odds_weights

BASELINES_FILE = CACHE_DIR / "baselines.npz"


def baseline_predictions(
    model: Pipeline, texts_train: list[str], y_train: np.ndarray, texts_test: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Zwraca `{nazwa: (y_pred, scores)}` dla grup kontrolnych na `texts_test`.

    Wynik jawnego NB to log-odds `b + x·w` (ta sama wielkość co `delta_enc`), więc ROC-AUC
    obu wersji jest porównywalne.
    """
    b, w = log_odds_weights(model)
    nb_scores = b + features(model, texts_test) @ w
    nb_pred = model.predict(texts_test)

    # Dummy nie patrzy na cechy, wystarczy mu kolumna zer.
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    X_dummy = np.zeros((len(texts_test), 1))
    return {
        "plain_nb": (nb_pred, nb_scores),
        "dummy": (dummy.predict(X_dummy), dummy.predict_proba(X_dummy)[:, 1]),
    }


def save_baselines(
    y_test: np.ndarray, predictions: dict[str, tuple[np.ndarray, np.ndarray]], path: Path = BASELINES_FILE
) -> None:
    """Zapisuje `y_test` i dla każdej grupy `<nazwa>_pred`, `<nazwa>_score`."""
    arrays = {"y_test": y_test}
    for name, (y_pred, scores) in predictions.items():
        arrays[f"{name}_pred"] = y_pred
        arrays[f"{name}_score"] = scores
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def main() -> None:
    texts_train, texts_test, y_train, y_test = load_split()
    model = fit_best_model(texts_train, y_train)
    predictions = baseline_predictions(model, texts_train, y_train, texts_test)
    save_baselines(y_test, predictions)

    print(f"{'':>9} " + " ".join(f"{m:>9}" for m in METRICS))
    for name, (y_pred, scores) in predictions.items():
        metrics = classification_metrics(y_test, y_pred, scores)
        print(f"{name:>9} " + " ".join(f"{metrics[m]:9.4f}" for m in METRICS))
    print(f"zapisano {BASELINES_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
