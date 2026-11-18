import numpy as np
import pytest

from src.baselines import baseline_predictions, save_baselines
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
