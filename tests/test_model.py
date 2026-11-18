import json

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.model import fit_model

# Mała siatka, żeby test trwał ułamek sekundy; pełna siatka jest w `PARAM_GRID`.
SMALL_GRID = {"select__k": [6, 12], "nb__alpha": [0.1, 1.0]}

SPAM_WORDS = ["free", "prize", "claim", "txt", "win", "cash"]
HAM_WORDS = ["ok", "home", "later", "see", "you", "tonight"]


@pytest.fixture(scope="module")
def train_data():
    # 50 wiadomości (10 spam), każda z trzech słów swojej klasy i jednego wspólnego.
    rng = np.random.default_rng(0)
    y = np.array([1 if i % 5 == 0 else 0 for i in range(50)])
    texts = [
        " ".join([*rng.choice(SPAM_WORDS if label else HAM_WORDS, size=3), "hello"])
        for label in y
    ]
    return texts, y


def test_fit_model_returns_fitted_pipeline(train_data):
    texts, y = train_data

    model = fit_model(texts, y, out_dir=None, param_grid=SMALL_GRID)

    assert isinstance(model, Pipeline)
    assert [name for name, _ in model.steps] == ["tfidf", "select", "nb"]
    assert model.predict(["free prize claim", "see you later"]).tolist() == [1, 0]


def test_fit_model_saves_cv_results(tmp_path, train_data):
    texts, y = train_data

    model = fit_model(texts, y, out_dir=tmp_path, param_grid=SMALL_GRID)

    best = json.loads((tmp_path / "best_params.json").read_text())
    assert {name: model.get_params()[name] for name in SMALL_GRID} == best["params"]
    assert 0 <= best["cv_f1"] <= 1 and 0 <= best["cv_accuracy"] <= 1

    rows = (tmp_path / "cv_results.csv").read_text().splitlines()
    assert rows[0].split(",")[:3] == ["nb__alpha", "select__k", "mean_test_accuracy"]
    assert len(rows) == 1 + 4  # nagłówek + 2 × 2 kombinacje


def test_fit_model_without_out_dir_writes_nothing(tmp_path, train_data, monkeypatch):
    texts, y = train_data
    monkeypatch.chdir(tmp_path)

    fit_model(texts, y, out_dir=None, param_grid=SMALL_GRID)

    assert list(tmp_path.iterdir()) == []
