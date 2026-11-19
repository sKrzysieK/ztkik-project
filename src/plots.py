"""Wykresy do raportu, zapisywane jako PDF do `results/figures/`.

Wykresy czytają tylko wyniki zapisane wcześniej (`results/cache/`, `results/*.csv`), więc
nie trenują ani nie szyfrują niczego. Rozmiar to szerokość jednej kolumny IEEEtran
(3,5 cala), żeby w raporcie nie skalować czcionek.

Uruchomienie: `python -m src.plots`
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # bez okien: wykresy tylko do plików

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import FIGURES_DIR, RESULTS_DIR, ROOT
from src.evaluate import BASELINES_FILE, MODEL_NAMES, SCORES_FILE, load_cached
from src.model import best_params

COLUMN_WIDTH = 3.5  # cale, szerokość kolumny IEEEtran

STYLE = {
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # czcionki TrueType: tekst w PDF da się zaznaczyć i przeszukać
}

# Wersja zaszyfrowana i jawny NB mają identyczne wyniki, więc NB rysujemy przerywaną
# linią na wierzchu — obie krzywe są widoczne, choć się pokrywają.
ROC_STYLE = {
    "encrypted": {"color": "C0", "linewidth": 2.0},
    "plain_nb": {"color": "C1", "linestyle": "--", "linewidth": 1.2},
    "logreg": {"color": "C2", "linestyle": ":", "linewidth": 1.2},
}


def _save(fig: plt.Figure, path: Path) -> None:
    """Zapisuje PDF bez daty utworzenia, żeby ten sam wykres dawał ten sam plik."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)


def plot_roc(
    scores_path: Path = SCORES_FILE,
    baselines_path: Path = BASELINES_FILE,
    path: Path = FIGURES_DIR / "roc.pdf",
) -> None:
    """Rysuje krzywe ROC wersji zaszyfrowanej (`delta_enc`) i grup kontrolnych.

    Klasyfikator większościowy ma stały wynik, czyli ROC-AUC 0,5 — to przekątna.
    Wstawka powiększa lewy górny róg, gdzie widać różnice między modelami.
    """
    scores, baselines = load_cached(scores_path, baselines_path)
    y_test = scores["y_test"]
    curves = {"encrypted": scores["delta_enc"]}
    curves |= {name: baselines[f"{name}_score"] for name in ROC_STYLE if f"{name}_score" in baselines}

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.8))
        inset = ax.inset_axes([0.42, 0.12, 0.53, 0.5])
        for name, score in curves.items():
            fpr, tpr, _ = roc_curve(y_test, score)
            label = f"{MODEL_NAMES[name]} (AUC {roc_auc_score(y_test, score):.3f})"
            ax.plot(fpr, tpr, label=label, **ROC_STYLE[name])
            inset.plot(fpr, tpr, **ROC_STYLE[name])
        ax.plot([0, 1], [0, 1], color="0.5", linewidth=0.8, label=f"{MODEL_NAMES['dummy']} (AUC 0.500)")

        inset.set_xlim(0, 0.1)
        inset.set_ylim(0.8, 1.0)
        inset.tick_params(labelsize=6)
        # Sama ramka powiększanego obszaru, bez linii łączących (przecinały krzywe).
        for line in ax.indicate_inset_zoom(inset, edgecolor="0.4").connectors:
            line.set_visible(False)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), frameon=False)
        _save(fig, path)


def cv_by_k(cv_path: Path = RESULTS_DIR / "cv_results.csv", params: dict | None = None) -> pd.DataFrame:
    """Zwraca wyniki CV dla każdego k przy pozostałych hiperparametrach jak w wybranym modelu.

    Zmieniamy tylko k, więc wykres pokazuje sam wpływ liczby cech. `params=None` bierze
    parametry z `results/best_params.json`.
    """
    params = best_params() if params is None else params
    cv = pd.read_csv(cv_path)
    mask = pd.Series(True, index=cv.index)
    for name, value in params.items():
        if name != "select__k":
            # W CSV krotki (`ngram_range`) są zapisane jako tekst, np. "(1, 1)".
            mask &= cv[name].astype(str) == str(value)
    return cv[mask].sort_values("select__k").set_index("select__k")


def plot_accuracy_vs_k(
    cv_path: Path = RESULTS_DIR / "cv_results.csv",
    benchmark_path: Path = RESULTS_DIR / "benchmark.csv",
    path: Path = FIGURES_DIR / "accuracy-vs-k.pdf",
    params: dict | None = None,
) -> None:
    """Rysuje kompromis jakość–koszt: wyniki CV w funkcji k (góra) i czas na wiadomość (dół).

    Czas to suma szyfrowania, serwera i odszyfrowania z `benchmark.csv` (1 rdzeń); rośnie
    liniowo z k, bo klient szyfruje k pozycji, a serwer wykonuje k potęgowań.
    """
    cv = cv_by_k(cv_path, params)
    bench = pd.read_csv(benchmark_path)
    bench["t_total_s"] = (bench["t_enc_ms"] + bench["t_server_ms"] + bench["t_dec_ms"]) / 1000

    with plt.rc_context(STYLE):
        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(COLUMN_WIDTH, 3.4), sharex=True, height_ratios=[3, 2]
        )
        for metric, label, marker in (("accuracy", "Accuracy", "o"), ("f1", "F1 (spam)", "s")):
            top.errorbar(
                cv.index,
                cv[f"mean_test_{metric}"],
                yerr=cv[f"std_test_{metric}"],
                marker=marker,
                markersize=3,
                capsize=2,
                linewidth=1,
                label=label,
            )
        top.set_ylabel("5-fold CV score")
        top.legend(loc="lower right")

        # Inne kolory niż w górnym panelu, żeby nie sugerować związku z metrykami.
        for color, (key_bits, group) in zip(("C2", "C3", "C4"), bench.groupby("key_bits")):
            bottom.plot(
                group["k"], group["t_total_s"], color=color, marker="o", markersize=3, label=f"{key_bits}-bit key"
            )
        bottom.set_xlabel("Number of features $k$")
        bottom.set_ylabel("Time / message,\n1 core [s]")
        bottom.set_ylim(bottom=0)
        bottom.legend(loc="upper left")
        _save(fig, path)


def main() -> None:
    plot_roc()
    plot_accuracy_vs_k()
    print(f"zapisano wykresy do {FIGURES_DIR.relative_to(ROOT)}: roc.pdf, accuracy-vs-k.pdf")


if __name__ == "__main__":
    main()
