import json

import numpy as np
import pytest
from sklearn.naive_bayes import BernoulliNB

from src.attacks.sparse_leak import attack_metrics, nonzero_count, presence, run, save_result
from src.data import RAW_FILE

needs_data = pytest.mark.skipif(not RAW_FILE.exists(), reason="brak danych: python data/download.py")

# Wartości jak TF-IDF: serwer widzi tylko, które pozycje są niezerowe.
X = np.array([[0.0, 0.6, 0.0, 0.8], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])


def test_presence_reveals_only_nonzero_positions():
    assert presence(X).tolist() == [[0, 1, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0]]


def test_nonzero_count_is_column():
    assert nonzero_count(X).tolist() == [[2], [0], [1]]


def test_attack_metrics_on_word_that_identifies_spam():
    # Pozycja 0 występuje tylko w spamie.
    X_train = np.array([[1, 0], [1, 1], [0, 1], [0, 0], [0, 1], [1, 0]])
    y_train = np.array([1, 1, 0, 0, 0, 1])

    metrics = attack_metrics(X_train, y_train, np.array([[1, 1], [0, 1]]), np.array([1, 0]), BernoulliNB())

    assert metrics["accuracy"] == 1.0 and metrics["f1"] == 1.0


def test_save_result_rounds_metrics(tmp_path):
    save_result({"n_test": 3, "plain_nb": {"accuracy": 2 / 3}}, tmp_path / "a.json")

    assert json.loads((tmp_path / "a.json").read_text()) == {"n_test": 3, "plain_nb": {"accuracy": 0.6667}}


@needs_data
def test_presence_alone_is_almost_as_good_as_plain_nb():
    result = run()

    majority = 1 - 131 / 1035  # zawsze ham na zbiorze testowym
    assert result["presence_bnb"]["accuracy"] > result["plain_nb"]["accuracy"] - 0.02
    assert result["count_logreg"]["accuracy"] > majority
