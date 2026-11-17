"""Metryki klasyfikacji i ewaluacja wyników zapisanych w `results/cache/`.

Moduł nie trenuje ani nie szyfruje niczego: czyta `scores.npz` (`src.pipeline`) i
`baselines.npz` (`src.baselines`), więc tabelę i zgodność można odtworzyć w sekundę.

Uruchomienie: `python -m src.evaluate`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.config import CACHE_DIR, RESULTS_DIR, ROOT

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")

SCORES_FILE = CACHE_DIR / "scores.npz"  # zapisuje src.pipeline
BASELINES_FILE = CACHE_DIR / "baselines.npz"  # zapisuje src.baselines

# Kolejność i nazwy wierszy w tabeli do raportu (raport jest po angielsku).
MODEL_NAMES = {
    "encrypted": "Encrypted NB (Paillier)",
    "plain_nb": "Plaintext NB",
    "logreg": "Logistic regression",
    "dummy": "Majority class",
}
TEX_HEADER = ("Model", "Acc.", "Prec.", "Rec.", "F1", "ROC-AUC")


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    """Zwraca accuracy oraz precision, recall i F1 klasy spam (1), a ROC-AUC z `scores`.

    `zero_division=0`: klasyfikator, który nigdy nie mówi „spam” (Dummy), ma precision 0,
    a nie ostrzeżenie. Stały wynik (`scores`) daje ROC-AUC 0,5.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def metrics_table(
    scores_path: Path = SCORES_FILE, baselines_path: Path = BASELINES_FILE
) -> pd.DataFrame:
    """Zwraca tabelę metryk: wersja zaszyfrowana i wszystkie zapisane grupy kontrolne.

    Wersja zaszyfrowana: predykcja to znak `delta_enc`, a ROC-AUC liczymy z `delta_enc`.
    Oba pliki muszą dotyczyć tych samych wiadomości (np. nie przebiegu z `--limit`).
    """
    scores, baselines = load_cached(scores_path, baselines_path)
    y_test, delta_enc = scores["y_test"], scores["delta_enc"]
    rows = {"encrypted": classification_metrics(y_test, (delta_enc > 0).astype(int), delta_enc)}
    for name in MODEL_NAMES:
        if f"{name}_pred" in baselines:
            rows[name] = classification_metrics(
                y_test, baselines[f"{name}_pred"], baselines[f"{name}_score"]
            )
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(METRICS)).rename_axis("model")


def save_metrics_table(
    table: pd.DataFrame,
    csv_path: Path = RESULTS_DIR / "metrics.csv",
    tex_path: Path = RESULTS_DIR / "metrics.tex",
) -> None:
    """Zapisuje tabelę jako CSV (4 miejsca po przecinku) i jako `tabular` z booktabs do raportu.

    Plik `.tex` zawiera samo `tabular`, żeby podpis i etykietę dodać w raporcie
    (`\\input{../results/metrics.tex}` wewnątrz `table`).
    """
    table.to_csv(csv_path, float_format="%.4f", lineterminator="\n")

    lines = [
        "% Wygenerowane przez `python -m src.evaluate` — nie edytować ręcznie.",
        r"\begin{tabular}{@{}l" + "c" * len(METRICS) + "@{}}",
        r"\toprule",
        " & ".join(TEX_HEADER) + r" \\",
        r"\midrule",
    ]
    for name, row in table.iterrows():
        values = " & ".join(f"{row[m]:.3f}" for m in METRICS)
        lines.append(f"{MODEL_NAMES[name]} & {values} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n")


def load_cached(scores_path: Path, baselines_path: Path) -> tuple[dict, dict]:
    """Wczytuje oba pliki cache i sprawdza, że dotyczą tych samych wiadomości."""
    with np.load(scores_path) as scores, np.load(baselines_path) as baselines:
        scores, baselines = dict(scores), dict(baselines)
    if not np.array_equal(scores["y_test"], baselines["y_test"]):
        raise ValueError(
            f"{scores_path.name} i {baselines_path.name} dotyczą różnych wiadomości; "
            "uruchom pełny `python -m src.pipeline` (bez --limit) i `python -m src.baselines`"
        )
    return scores, baselines


def agreement(
    scores_path: Path = SCORES_FILE, baselines_path: Path = BASELINES_FILE
) -> dict[str, float | int]:
    """Zwraca zgodność wersji zaszyfrowanej z jawnym NB (kryterium: 100% w obu miarach).

    - `same_prediction`: odsetek wiadomości, dla których znak `delta_enc` daje tę samą
      klasę co `MultinomialNB.predict` (niezależna od nas implementacja scikit-learn);
    - `exact_int_match`: odsetek wiadomości, dla których odszyfrowany `int` jest równy
      temu samemu obliczeniu wykonanemu jawnie na liczbach całkowitych.
    """
    scores, baselines = load_cached(scores_path, baselines_path)
    same = (scores["delta_enc"] > 0).astype(int) == baselines["plain_nb_pred"]
    return {
        "n_messages": int(len(same)),
        "same_prediction": float(same.mean()),
        "n_different_prediction": int((~same).sum()),
        "exact_int_match": float(scores["exact_int_match"].mean()),
        "n_int_mismatch": int((~scores["exact_int_match"]).sum()),
    }


def save_json(result: dict, path: Path) -> None:
    """Zapisuje mały wynik jako JSON (commitujemy go, raport cytuje liczby stąd)."""
    path.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    table = metrics_table()
    save_metrics_table(table)
    print(table.to_string(float_format="%.4f"))
    print(f"zapisano {(RESULTS_DIR / 'metrics.csv').relative_to(ROOT)} i .tex")

    agree = agreement()
    save_json(agree, RESULTS_DIR / "agreement.json")
    print(
        f"\nzgodność z jawnym NB: {agree['same_prediction']:.2%} "
        f"({agree['n_different_prediction']} różnych z {agree['n_messages']}), "
        f"exact_int_match: {agree['exact_int_match']:.2%}"
    )


if __name__ == "__main__":
    main()
