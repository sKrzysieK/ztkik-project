import json

import numpy as np
import pytest

from src.baselines import baseline_predictions, fit_logreg, save_baselines
from src.model import build_pipeline

SPAM = ["free prize claim now", "win cash txt now", "claim free cash"]
HAM = ["see you at home", "ok later tonight", "call you later", "home tonight ok", "see you soon"]


@pytest.fixture(scope="module")
def setup():
    texts_train = SPAM + HAM
    y_train = np.array([1] * len(SPAM) + [0] * len(HAM))
    model = build_pipeline().set_params(select__k=10, nb__alpha=0.1).fit(texts_train, y_train)
    texts_test = ["free cash prize", "see you home", "hello there"]
    return model, texts_train, y_train, texts_test


def test_plain_nb_matches_model_and_scores_are_log_odds(setup):
    model, texts_train, y_train, texts_test = setup

    y_pred, scores = baseline_predictions(model, texts_train, y_train, texts_test)["plain_nb"]

    assert y_pred.tolist() == model.predict(texts_test).tolist()
    assert ((scores > 0).astype(int) == y_pred).all()
    proba = model.predict_proba(texts_test)
    assert scores == pytest.approx(np.log(proba[:, 1]) - np.log(proba[:, 0]))


def test_dummy_always_predicts_majority_class(setup):
    model, texts_train, y_train, texts_test = setup

    y_pred, scores = baseline_predictions(model, texts_train, y_train, texts_test)["dummy"]

    assert y_pred.tolist() == [0, 0, 0]
    assert len(set(scores.tolist())) == 1


def test_save_baselines_writes_all_arrays(setup, tmp_path):
    model, texts_train, y_train, texts_test = setup
    predictions = baseline_predictions(model, texts_train, y_train, texts_test)
    path = tmp_path / "baselines.npz"

    save_baselines(np.array([1, 0, 0]), predictions, path)

    with np.load(path) as data:
        assert set(data) == {"y_test", "plain_nb_pred", "plain_nb_score", "dummy_pred", "dummy_score"}
        assert data["y_test"].tolist() == [1, 0, 0]


@pytest.fixture(scope="module")
def larger_train():
    # 50 wiadomości (10 spam), żeby 5-fold CV miało spam w każdym foldzie.
    rng = np.random.default_rng(0)
    y = np.array([1 if i % 5 == 0 else 0 for i in range(50)])
    words = {1: ["free", "prize", "claim", "txt", "win", "cash"], 0: ["ok", "home", "later", "see", "tonight"]}
    texts = [" ".join(rng.choice(words[label], size=3)) for label in y]
    return texts, y


def test_fit_logreg_uses_nb_features_and_saves_params(larger_train, tmp_path):
    texts, y = larger_train
    nb_params = {"tfidf__sublinear_tf": True, "select__k": 8, "nb__alpha": 0.1}

    logreg = fit_logreg(texts, y, nb_params, param_grid={"clf__C": [1, 10]}, out_dir=tmp_path)

    assert [name for name, _ in logreg.steps] == ["tfidf", "select", "clf"]
    assert logreg.get_params()["select__k"] == 8
    assert logreg.get_params()["tfidf__sublinear_tf"] is True
    saved = json.loads((tmp_path / "logreg_params.json").read_text())
    assert saved["params"]["clf__C"] == logreg.get_params()["clf__C"]


def test_logreg_scores_are_linear_decision_values(setup, larger_train):
    model, texts_train, y_train, texts_test = setup
    logreg = fit_logreg(*larger_train, {"select__k": 8}, param_grid={"clf__C": [10]}, out_dir=None)

    y_pred, scores = baseline_predictions(model, texts_train, y_train, texts_test, logreg)["logreg"]

    assert ((scores > 0).astype(int) == y_pred).all()
    assert y_pred.tolist()[:2] == [1, 0]
