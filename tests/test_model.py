import json

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.config import RESULTS_DIR
from src.data import RAW_FILE, load_split
from src.model import build_pipeline, features, fit_best_model, fit_model, log_odds_weights

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


@pytest.fixture(scope="module")
def fitted(train_data):
    texts, y = train_data
    # k=12 obejmuje słowa obu klas (13. słowo „hello” jest wspólne).
    return build_pipeline().set_params(select__k=12, nb__alpha=0.1).fit(texts, y)


def test_features_are_dense_float64_tfidf(fitted, train_data):
    texts, _ = train_data

    X = features(fitted, texts)

    assert isinstance(X, np.ndarray)
    assert X.shape == (len(texts), 12)
    assert X.dtype == np.float64
    assert X.min() >= 0 and X.max() <= 1


def test_log_odds_weights_reproduce_predictions(fitted, train_data):
    texts, _ = train_data
    new_texts = ["free cash tonight", "ok see you", "hello", "win win prize home"]

    b, w = log_odds_weights(fitted)

    assert isinstance(b, float)
    assert w.shape == (12,) and w.dtype == np.float64
    for batch in (texts, new_texts):
        delta = b + features(fitted, batch) @ w
        assert (delta > 0).astype(int).tolist() == fitted.predict(batch).tolist()


def test_log_odds_weights_are_spam_minus_ham(fitted):
    b, w = log_odds_weights(fitted)
    vocab = fitted[:-1].get_feature_names_out()

    assert b < 0  # 10 spamów na 50 wiadomości
    assert all(w[list(vocab).index(word)] > 0 for word in SPAM_WORDS if word in vocab)
    assert all(w[list(vocab).index(word)] < 0 for word in HAM_WORDS if word in vocab)


def test_log_odds_weights_reject_other_labels():
    model = build_pipeline().set_params(select__k=2).fit(["aa bb", "cc dd", "aa cc"], [1, 2, 1])

    with pytest.raises(ValueError):
        log_odds_weights(model)


@pytest.mark.skipif(not RAW_FILE.exists(), reason="brak danych: python data/download.py")
def test_real_test_set_predictions_match_model():
    best = json.loads((RESULTS_DIR / "best_params.json").read_text())["params"]
    best["tfidf__ngram_range"] = tuple(best["tfidf__ngram_range"])
    texts_train, texts_test, y_train, _ = load_split()
    model = build_pipeline().set_params(**best).fit(texts_train, y_train)

    b, w = log_odds_weights(model)
    X_test = features(model, texts_test)

    assert X_test.shape == (len(texts_test), best["select__k"])
    assert ((b + X_test @ w > 0).astype(int) == model.predict(texts_test)).all()


def test_fit_best_model_reproduces_fit_model(tmp_path, train_data):
    texts, y = train_data
    grid = {**SMALL_GRID, "tfidf__ngram_range": [(1, 1)]}

    tuned = fit_model(texts, y, out_dir=tmp_path, param_grid=grid)
    refitted = fit_best_model(texts, y, tmp_path / "best_params.json")

    assert refitted.get_params()["tfidf__ngram_range"] == (1, 1)
    assert np.array_equal(features(refitted, texts), features(tuned, texts))
    assert log_odds_weights(refitted)[0] == log_odds_weights(tuned)[0]
    assert np.array_equal(log_odds_weights(refitted)[1], log_odds_weights(tuned)[1])


def test_fit_best_model_without_params_points_to_tuning(tmp_path, train_data):
    texts, y = train_data

    with pytest.raises(FileNotFoundError, match="src.model"):
        fit_best_model(texts, y, tmp_path / "best_params.json")
