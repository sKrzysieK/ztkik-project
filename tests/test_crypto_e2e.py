"""Test integracyjny całego protokołu: klient szyfruje, serwer liczy, klient odszyfrowuje."""

import numpy as np
import pytest

from src.crypto_client import decrypt_predict, encrypt_vectors, generate_keys
from src.crypto_server import server_score
from src.fixed_point import encode_features, encode_weights, plain_int_scores

TEST_KEY_BITS = 512


@pytest.fixture(scope="module")
def protocol_run():
    rng = np.random.default_rng(42)
    n, k, nnz = 20, 50, 8

    # Wiersze jak z TfidfVectorizer: kilka niezerowych słów, normalizacja L2.
    X = np.zeros((n, k))
    for row in X:
        row[rng.choice(k, size=nnz, replace=False)] = rng.random(nnz)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    w = rng.uniform(-6, 6, size=k)
    b = 0.3

    pub, priv = generate_keys(TEST_KEY_BITS)
    enc_X = encrypt_vectors(pub, X, workers=2)
    y_pred, scores, raw = decrypt_predict(priv, server_score(enc_X, b, w))
    return X, b, w, y_pred, scores, raw


def test_decrypted_scores_equal_plain_integer_scores(protocol_run):
    X, b, w, _, _, raw = protocol_run

    b_int, w_int = encode_weights(b, w)
    assert raw == plain_int_scores(encode_features(X), b_int, w_int)


def test_predictions_match_plaintext_float_classifier(protocol_run):
    X, b, w, y_pred, scores, _ = protocol_run
    delta_float = b + X @ w

    # Błąd kwantyzacji jest rzędu 1e-5; margines 1e-3 gwarantuje, że test nie zależy
    # od przypadkowego Δ bliskiego zera, a obie klasy dają mu sens.
    assert np.min(np.abs(delta_float)) > 1e-3
    assert set(y_pred.tolist()) == {0, 1}

    assert y_pred.tolist() == (delta_float > 0).astype(int).tolist()
    assert np.max(np.abs(scores - delta_float)) < 1e-4
