"""Grupy kontrolne na zbiorze testowym: jawny Naive Bayes, klasyfikator większościowy
i regresja logistyczna.

- jawny NB: ten sam model co w wersji zaszyfrowanej, liczony bez szyfrowania i bez
  kwantyzacji — bezpośredni punkt odniesienia;
- `DummyClassifier(strategy="most_frequent")`: zawsze ham; pokazuje, ile daje samo
  niezbalansowanie klas (ok. 87%);
- regresja logistyczna na tych samych cechach (TF-IDF i chi² z parametrami NB): też
  liniowa (`b + x·w`), więc działałaby w tym samym protokole Paillier — porównanie jakości.

Wyniki trafiają do `results/cache/baselines.npz`, z którego czyta `src.evaluate`.

Uruchomienie: `python -m src.baselines`
"""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import CACHE_DIR, RESULTS_DIR, ROOT, SEED
from src.data import load_split
from src.evaluate import METRICS, classification_metrics
from src.model import (
    CV_FOLDS,
    REFIT,
    SCORING,
    best_params,
    build_pipeline,
    features,
    fit_best_model,
    log_odds_weights,
)

BASELINES_FILE = CACHE_DIR / "baselines.npz"

# `class_weight="balanced"` kompensuje 13% spamu; C stroimy jak NB: 5-fold CV na train, po F1.
LOGREG_GRID = {"clf__C": [0.1, 1, 10, 100, 1000], "clf__class_weight": [None, "balanced"]}


def fit_logreg(
    texts_train: list[str],
    y_train: np.ndarray,
    nb_params: dict,
    param_grid: dict = LOGREG_GRID,
    out_dir: Path | None = RESULTS_DIR,
) -> Pipeline:
    """Stroi regresję logistyczną na cechach NB (te same TF-IDF i k) i zwraca najlepszy model.

    Zapisuje wybrane parametry i wynik CV do `logreg_params.json` (`out_dir=None`: bez zapisu).
    """
    pipeline = build_pipeline().set_params(
        **{name: value for name, value in nb_params.items() if not name.startswith("nb__")}
    )
    pipeline.steps[-1] = ("clf", LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipeline, param_grid, scoring=SCORING, refit=REFIT, cv=cv, n_jobs=-1)
    search.fit(texts_train, y_train)

    if out_dir is not None:
        best = {
            "params": search.best_params_,
            **{
                f"cv_{name}": round(float(search.cv_results_[f"mean_test_{name}"][search.best_index_]), 6)
                for name in SCORING
            },
        }
        (out_dir / "logreg_params.json").write_text(json.dumps(best, indent=2) + "\n")
    return search.best_estimator_


def baseline_predictions(
    model: Pipeline,
    texts_train: list[str],
    y_train: np.ndarray,
    texts_test: list[str],
    logreg: Pipeline | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Zwraca `{nazwa: (y_pred, scores)}` dla grup kontrolnych na `texts_test`.

    Wynik jawnego NB to log-odds `b + x·w` (ta sama wielkość co `delta_enc`), więc ROC-AUC
    obu wersji jest porównywalne. Regresja logistyczna (jeśli podana) ma wynik
    `decision_function`, czyli też `b + x·w`.
    """
    b, w = log_odds_weights(model)
    nb_scores = b + features(model, texts_test) @ w
    nb_pred = model.predict(texts_test)

    # Dummy nie patrzy na cechy, wystarczy mu kolumna zer.
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    X_dummy = np.zeros((len(texts_test), 1))
    predictions = {
        "plain_nb": (nb_pred, nb_scores),
        "dummy": (dummy.predict(X_dummy), dummy.predict_proba(X_dummy)[:, 1]),
    }
    if logreg is not None:
        predictions["logreg"] = (logreg.predict(texts_test), logreg.decision_function(texts_test))
    return predictions


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
    logreg = fit_logreg(texts_train, y_train, best_params())
    predictions = baseline_predictions(model, texts_train, y_train, texts_test, logreg)
    save_baselines(y_test, predictions)

    print(f"{'':>9} " + " ".join(f"{m:>9}" for m in METRICS))
    for name, (y_pred, scores) in predictions.items():
        metrics = classification_metrics(y_test, y_pred, scores)
        print(f"{name:>9} " + " ".join(f"{metrics[m]:9.4f}" for m in METRICS))
    print(f"zapisano {BASELINES_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
