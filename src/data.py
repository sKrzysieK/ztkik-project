"""Wczytanie SMS Spam Collection, deduplikacja i stratyfikowany podział 80/20."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_RAW, SEED

if TYPE_CHECKING:
    import numpy as np

# Plik pobierany przez `python data/download.py`.
RAW_FILE = DATA_RAW / "SMSSpamCollection"
LABELS = {"ham": 0, "spam": 1}
TEST_SIZE = 0.2


def load_dataset(path: Path = RAW_FILE) -> pd.DataFrame:
    """Zwraca zbiór po deduplikacji: kolumny `label` (1 = spam, 0 = ham) i `text`.

    Usuwamy dokładne duplikaty treści (403 z 5574, głównie szablonowy spam), zostawiając
    pierwsze wystąpienie. Bez tego część wiadomości testowych byłaby dosłownie w zbiorze
    treningowym i zawyżała wyniki. Żadna treść nie występuje z dwiema etykietami.
    """
    if not path.exists():
        raise FileNotFoundError(f"brak {path}; uruchom najpierw `python data/download.py`")
    # Każdy wiersz to `etykieta<TAB>treść`; cudzysłowy w SMS-ach są zwykłymi znakami.
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        quoting=csv.QUOTE_NONE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"nieznane etykiety w {path}: {sorted(unknown)}")
    df["label"] = df["label"].map(LABELS).astype("int64")
    return df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)


def load_split(path: Path = RAW_FILE) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Zwraca `texts_train, texts_test, y_train, y_test` (stratyfikowany 80/20, stały seed).

    Dzielimy po deduplikacji, więc żadna treść nie trafia jednocześnie do obu zbiorów.
    Stratyfikacja trzyma ten sam odsetek spamu (ok. 12,6%) w obu częściach.
    """
    df = load_dataset(path)
    texts_train, texts_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].to_numpy(),
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=SEED,
    )
    return texts_train, texts_test, y_train, y_test
