"""Model jawny: TF-IDF -> selekcja chi² -> Multinomial NB, eksport wag log-odds.

Hiperparametry stroimy 5-fold CV wyłącznie na zbiorze treningowym. TF-IDF i chi² są
częścią `Pipeline`, więc dopasowują się osobno w każdym foldzie i CV nie przecieka.

Uruchomienie (strojenie i zapis `results/best_params.json`, `results/cv_results.csv`):
`python -m src.model`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import RESULTS_DIR, SEED
from src.data import load_split

CV_FOLDS = 5

# k ≤ 1000, bo koszt szyfrowania rośnie liniowo z k (benchmark K13); k < 500 zostaje
# w siatce do wykresu accuracy vs. k. Bigramy i sublinear TF sprawdzamy jako parametry TF-IDF.
PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__sublinear_tf": [False, True],
    "select__k": [100, 200, 300, 500, 750, 1000],
    "nb__alpha": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
}

# Wybieramy po F1 klasy spam: przy 87% ham accuracy słabo odróżnia modele.
SCORING = {"accuracy": "accuracy", "f1": "f1"}
REFIT = "f1"


def build_pipeline() -> Pipeline:
    """Zwraca niewytrenowany `Pipeline` TF-IDF -> `SelectKBest(chi2)` -> `MultinomialNB`."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("select", SelectKBest(chi2)),
            ("nb", MultinomialNB()),
        ]
    )


def tune_model(
    texts_train: list[str], y_train: np.ndarray, param_grid: dict = PARAM_GRID
) -> GridSearchCV:
    """Przeszukuje siatkę 5-fold CV (stratyfikowane, stały seed) i trenuje najlepszy model na całym train."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        build_pipeline(), param_grid, scoring=SCORING, refit=REFIT, cv=cv, n_jobs=-1
    )
    return search.fit(texts_train, y_train)


def save_search(search: GridSearchCV, out_dir: Path = RESULTS_DIR) -> None:
    """Zapisuje `best_params.json` i `cv_results.csv` (do wykresu accuracy vs. k).

    Bez czasów dopasowania, żeby dwa przebiegi dawały identyczne pliki.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    row = _best_row(search)
    best = {
        "params": search.best_params_,
        "cv_folds": CV_FOLDS,
        "refit": REFIT,
        **{f"cv_{name}": round(float(row[f"mean_test_{name}"]), 6) for name in SCORING},
    }
    (out_dir / "best_params.json").write_text(json.dumps(best, indent=2) + "\n")

    results = pd.DataFrame(search.cv_results_)
    columns = [c for c in results if c.startswith("param_")] + [
        f"{stat}_test_{name}" for name in SCORING for stat in ("mean", "std", "rank")
    ]
    results = results[columns].rename(columns=lambda c: c.removeprefix("param_"))
    results.to_csv(out_dir / "cv_results.csv", index=False, float_format="%.6f", lineterminator="\n")


def _best_row(search: GridSearchCV) -> pd.Series:
    """Zwraca wiersz `cv_results_` dla wybranego modelu."""
    return pd.DataFrame(search.cv_results_).iloc[search.best_index_]


def fit_model(
    texts_train: list[str],
    y_train: np.ndarray,
    out_dir: Path | None = RESULTS_DIR,
    param_grid: dict = PARAM_GRID,
) -> Pipeline:
    """Stroi hiperparametry 5-fold CV na zbiorze treningowym i zwraca najlepszy `Pipeline`.

    `out_dir=None` nie zapisuje wyników CV (np. w testach).
    """
    search = tune_model(texts_train, y_train, param_grid)
    if out_dir is not None:
        save_search(search, out_dir)
    return search.best_estimator_


def features(model: Pipeline, texts: list[str]) -> np.ndarray:
    """Zwraca macierz cech `(n, k)`, dense, `float64`.

    To wejście NB: TF-IDF (normalizacja L2, więc wartości z [0, 1]) po selekcji chi².
    Klient liczy je jawnie i dopiero ten wektor szyfruje.
    """
    return model[:-1].transform(texts).toarray().astype(np.float64)


def log_odds_weights(model: Pipeline) -> tuple[float, np.ndarray]:
    """Zwraca `b`, `w` takie, że spam <=> `b + x·w > 0`.

    `b = log P(spam) − log P(ham)`, `w_i = log θ_spam,i − log θ_ham,i` (spam − ham, czyli
    indeks 1 − indeks 0). Przy remisie `MultinomialNB.predict` wybiera ham, tak jak `Δ > 0`.
    """
    nb = model.named_steps["nb"]
    if nb.classes_.tolist() != [0, 1]:
        raise ValueError(f"oczekiwano klas [0, 1] (ham, spam), a są {nb.classes_.tolist()}")
    b = float(nb.class_log_prior_[1] - nb.class_log_prior_[0])
    w = (nb.feature_log_prob_[1] - nb.feature_log_prob_[0]).astype(np.float64)
    return b, w


def main() -> None:
    texts_train, _, y_train, _ = load_split()
    search = tune_model(texts_train, y_train)
    save_search(search)
    row = _best_row(search)
    print(f"najlepsze parametry: {search.best_params_}")
    print(
        f"CV ({CV_FOLDS}-fold, train): accuracy {row['mean_test_accuracy']:.4f} "
        f"± {row['std_test_accuracy']:.4f}, F1 {row['mean_test_f1']:.4f} ± {row['std_test_f1']:.4f}"
    )
    print(f"zapisano {RESULTS_DIR / 'best_params.json'} i {RESULTS_DIR / 'cv_results.csv'}")


if __name__ == "__main__":
    main()
