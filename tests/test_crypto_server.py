import numpy as np
import pytest

from src.crypto_client import encrypt_vectors, generate_keys
from src.crypto_server import server_score
from src.fixed_point import encode_features, encode_weights, plain_int_scores

TEST_KEY_BITS = 512


@pytest.fixture(scope="module")
def keypair():
    return generate_keys(TEST_KEY_BITS)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.random((6, 8)) * (rng.random((6, 8)) < 0.4)  # rzadkie, jak TF-IDF
    w = rng.uniform(-8, 8, size=8)  # wagi log-odds obu znaków
    b = -1.7
    return X, b, w


def test_server_score_equals_plain_integer_computation(keypair, data):
    pub, priv = keypair
    X, b, w = data

    enc_scores = server_score(encrypt_vectors(pub, X, workers=1), b, w)

    b_int, w_int = encode_weights(b, w)
    assert [priv.decrypt(s) for s in enc_scores] == plain_int_scores(encode_features(X), b_int, w_int)


def test_server_score_keeps_exponent_zero(keypair, data):
    pub, _ = keypair
    X, b, w = data

    enc_scores = server_score(encrypt_vectors(pub, X, workers=1), b, w)

    assert {s.exponent for s in enc_scores} == {0}


def test_server_score_rejects_weight_length_mismatch(keypair, data):
    pub, _ = keypair
    X, b, w = data

    with pytest.raises(ValueError):
        server_score(encrypt_vectors(pub, X, workers=1), b, w[:-1])
