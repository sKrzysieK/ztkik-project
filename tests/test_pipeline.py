import numpy as np
import pytest

from src.data import RAW_FILE
from src.pipeline import run

pytestmark = pytest.mark.skipif(not RAW_FILE.exists(), reason="brak danych: python data/download.py")

TEST_KEY_BITS = 512


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("cache")
    first = run(limit=4, key_bits=TEST_KEY_BITS, workers=1, out=out / "first.npz")
    second = run(limit=4, key_bits=TEST_KEY_BITS, workers=1, out=out / "second.npz")
    return (first, np.load(out / "first.npz")), (second, np.load(out / "second.npz"))


def test_scores_file_has_contract_format(two_runs):
    (summary, scores), _ = two_runs

    assert set(scores) == {"y_test", "delta_enc", "delta_float", "exact_int_match"}
    assert all(len(scores[name]) == 4 for name in scores)
    assert scores["delta_enc"].dtype == scores["delta_float"].dtype == np.float64
    assert scores["exact_int_match"].dtype == bool
    assert summary["n_messages"] == 4 and summary["key_bits"] == TEST_KEY_BITS


def test_encrypted_scores_are_exact_and_match_plain_model(two_runs):
    (summary, scores), _ = two_runs

    assert scores["exact_int_match"].all()
    assert summary["exact_int_match"] == 1.0
    assert np.abs(scores["delta_enc"] - scores["delta_float"]).max() < 1e-4
    assert ((scores["delta_enc"] > 0) == (scores["delta_float"] > 0)).all()


def test_two_runs_give_identical_scores(two_runs):
    (_, first), (_, second) = two_runs

    # Szyfrogramy są różne (nowe klucze i r), ale wyniki na liczbach całkowitych te same.
    assert np.array_equal(first["delta_enc"], second["delta_enc"])
    assert np.array_equal(first["y_test"], second["y_test"])
