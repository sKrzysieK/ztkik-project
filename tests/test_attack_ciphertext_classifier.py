import gmpy2
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.attacks.ciphertext_classifier import FEATURE_NAMES, ciphertext_features, cv_scores, run
from src.crypto_client import generate_keys
from src.data import RAW_FILE

needs_data = pytest.mark.skipif(not RAW_FILE.exists(), reason="brak danych: python data/download.py")

TEST_KEY_BITS = 512


@pytest.fixture(scope="module")
def pub():
    return generate_keys(TEST_KEY_BITS)[0]


def test_ciphertext_features_shape_and_ranges(pub):
    ciphertexts = [[pub.encrypt(m).ciphertext(be_secure=False) for m in row] for row in ([0, 5, 0], [7, 0, 1])]

    X = ciphertext_features(pub, ciphertexts)

    assert X.shape == (2, 3 * len(FEATURE_NAMES))
    low_byte, mod3, mod5, mod7, lead_zeros, jacobi = (X[:, i :: len(FEATURE_NAMES)] for i in range(6))
    assert ((0 <= low_byte) & (low_byte < 256)).all()
    assert ((mod3 < 3) & (mod5 < 5) & (mod7 < 7)).all()
    assert (lead_zeros >= 0).all()
    assert set(jacobi.ravel()) <= {-1, 1}


def test_jacobi_symbol_depends_only_on_r(pub):
    # Te same r, różne m: symbol Jacobiego się nie zmienia, więc nie zdradza m.
    for r in (3, 10, 12345):
        symbols = {gmpy2.jacobi(pub.raw_encrypt(m, r_value=r), pub.n) for m in (0, 1, 999_999)}
        assert symbols == {gmpy2.jacobi(r, pub.n)}


def test_cv_scores_reports_mean_and_std():
    y = np.array([0, 1] * 10)
    X = y.reshape(-1, 1).astype(float)  # cecha równa etykiecie

    perfect = cv_scores(X, y, LogisticRegression())
    majority = cv_scores(X, y, DummyClassifier(strategy="most_frequent"))

    assert perfect == {"mean_accuracy": 1.0, "std_accuracy": 0.0, "mean_f1": 1.0, "std_f1": 0.0}
    assert majority["mean_accuracy"] == 0.5 and majority["mean_f1"] == 0.0


@needs_data
def test_run_compares_ciphertext_attack_with_controls():
    result = run(n_messages=40, key_bits=TEST_KEY_BITS, workers=1)

    assert result["n_messages"] == 40 and result["key_bits"] == TEST_KEY_BITS
    assert set(result) >= {"ciphertext_rf", "plaintext_rf", "dummy"}
    assert result["dummy"]["mean_accuracy"] == pytest.approx(1 - result["n_spam"] / 40)
