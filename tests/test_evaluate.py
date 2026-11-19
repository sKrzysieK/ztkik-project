import numpy as np
import pytest

from src.evaluate import (
    METRICS,
    agreement,
    classification_metrics,
    metrics_table,
    save_metrics_table,
)


def test_classification_metrics_for_spam_class():
    y_true = np.array([1, 1, 0, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    scores = np.array([2.0, -0.5, -3.0, -1.0, 0.5, 1.0])

    metrics = classification_metrics(y_true, y_pred, scores)

    assert list(metrics) == list(METRICS)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["roc_auc"] == pytest.approx(8 / 9)  # jedna zła para: (−0.5 spam, 0.5 ham)


def test_classification_metrics_for_constant_classifier():
    y_true = np.array([0, 0, 0, 1])

    metrics = classification_metrics(y_true, np.zeros(4, dtype=int), np.zeros(4))

    assert metrics == {"accuracy": 0.75, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.5}


Y_TEST = np.array([1, 0, 0, 1, 0])


@pytest.fixture
def cache(tmp_path):
    # Wersja zaszyfrowana myli się na ostatniej wiadomości, jawny NB nie.
    np.savez(
        tmp_path / "scores.npz",
        y_test=Y_TEST,
        delta_enc=np.array([2.0, -1.0, -3.0, 0.5, 0.2]),
        delta_float=np.array([2.0, -1.0, -3.0, 0.5, -0.2]),
        exact_int_match=np.ones(5, dtype=bool),
    )
    np.savez(
        tmp_path / "baselines.npz",
        y_test=Y_TEST,
        plain_nb_pred=np.array([1, 0, 0, 1, 0]),
        plain_nb_score=np.array([2.0, -1.0, -3.0, 0.5, -0.2]),
        dummy_pred=np.zeros(5, dtype=int),
        dummy_score=np.full(5, 0.4),
    )
    return tmp_path


def test_metrics_table_has_encrypted_and_baselines(cache):
    table = metrics_table(cache / "scores.npz", cache / "baselines.npz")

    assert table.index.tolist() == ["encrypted", "plain_nb", "dummy"]
    assert table.columns.tolist() == list(METRICS)
    assert table.loc["encrypted", "accuracy"] == pytest.approx(4 / 5)
    assert table.loc["encrypted", "precision"] == pytest.approx(2 / 3)
    assert table.loc["plain_nb"].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert table.loc["dummy", "accuracy"] == pytest.approx(3 / 5)
    assert table.loc["dummy", "roc_auc"] == 0.5


def test_metrics_table_rejects_scores_from_other_messages(cache):
    np.savez(cache / "scores.npz", y_test=Y_TEST[:3], delta_enc=np.zeros(3))

    with pytest.raises(ValueError, match="--limit"):
        metrics_table(cache / "scores.npz", cache / "baselines.npz")


def test_save_metrics_table_writes_csv_and_booktabs(cache, tmp_path):
    table = metrics_table(cache / "scores.npz", cache / "baselines.npz")

    save_metrics_table(table, tmp_path / "metrics.csv", tmp_path / "metrics.tex")

    csv_lines = (tmp_path / "metrics.csv").read_text().splitlines()
    assert csv_lines[0] == "model,accuracy,precision,recall,f1,roc_auc"
    assert csv_lines[1] == "encrypted,0.8000,0.6667,1.0000,0.8000,1.0000"
    tex = (tmp_path / "metrics.tex").read_text()
    assert r"\toprule" in tex and r"\midrule" in tex and r"\bottomrule" in tex
    assert r"Encrypted NB (Paillier) & 0.800 & 0.667 & 1.000 & 0.800 & 1.000 \\" in tex
    assert r"Majority class & 0.600 & 0.000 & 0.000 & 0.000 & 0.500 \\" in tex


def test_agreement_counts_different_predictions(cache):
    result = agreement(cache / "scores.npz", cache / "baselines.npz")

    assert result == {
        "n_messages": 5,
        "same_prediction": 0.8,
        "n_different_prediction": 1,
        "exact_int_match": 1.0,
        "n_int_mismatch": 0,
    }


def test_agreement_reports_int_mismatch(cache):
    with np.load(cache / "scores.npz") as scores:
        arrays = dict(scores)
    arrays["exact_int_match"] = np.array([True, True, False, True, True])
    np.savez(cache / "scores.npz", **arrays)

    result = agreement(cache / "scores.npz", cache / "baselines.npz")

    assert result["exact_int_match"] == pytest.approx(0.8)
    assert result["n_int_mismatch"] == 1
