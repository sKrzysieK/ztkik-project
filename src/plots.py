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
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import FIGURES_DIR, ROOT
from src.evaluate import BASELINES_FILE, MODEL_NAMES, SCORES_FILE, load_cached

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


def main() -> None:
    plot_roc()
    print(f"zapisano {(FIGURES_DIR / 'roc.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
