import numpy as np
import pytest

from src.crypto_client import encrypt_vectors, generate_keys
from src.fixed_point import encode_features

TEST_KEY_BITS = 512


@pytest.fixture(scope="module")
def keypair():
    return generate_keys(TEST_KEY_BITS)


def test_generate_keys_has_requested_modulus_length(keypair):
    pub, _ = keypair

    assert pub.n.bit_length() == TEST_KEY_BITS


def test_generate_keys_returns_matching_pair(keypair):
    pub, priv = keypair

    assert priv.public_key == pub
    assert priv.decrypt(pub.encrypt(-12345)) == -12345


@pytest.fixture(scope="module")
def sparse_X():
    # Jak TF-IDF: głównie zera, kilka wartości z (0, 1], różne liczby cyfr po przecinku.
    X = np.zeros((4, 6), dtype=np.float32)
    X[0, [0, 3]] = [0.03, 1.0]
    X[1, 2] = 0.5
    X[3, [1, 4, 5]] = [0.1, 0.7, 0.123456]
    return X


def test_encrypt_vectors_encrypts_every_position(keypair, sparse_X):
    pub, priv = keypair

    enc_X = encrypt_vectors(pub, sparse_X, workers=1)

    assert [len(row) for row in enc_X] == [6, 6, 6, 6]
    assert [[priv.decrypt(c) for c in row] for row in enc_X] == encode_features(sparse_X)


def test_encrypt_vectors_exponent_does_not_depend_on_value(keypair, sparse_X):
    pub, _ = keypair

    enc_X = encrypt_vectors(pub, sparse_X, workers=1)

    assert {c.exponent for row in enc_X for c in row} == {0}


def test_encrypt_vectors_is_probabilistic(keypair, sparse_X):
    pub, _ = keypair

    first = encrypt_vectors(pub, sparse_X, workers=1)
    second = encrypt_vectors(pub, sparse_X, workers=1)

    for row_a, row_b in zip(first, second):
        for a, b in zip(row_a, row_b):
            assert a.ciphertext(be_secure=False) != b.ciphertext(be_secure=False)


def test_encrypt_vectors_parallel_matches_sequential(keypair):
    pub, priv = keypair
    X = np.random.default_rng(0).random((12, 5))

    sequential = encrypt_vectors(pub, X, workers=1)
    parallel = encrypt_vectors(pub, X, workers=4)

    def decrypt(enc_X):
        return [[priv.decrypt(c) for c in row] for row in enc_X]

    assert decrypt(parallel) == decrypt(sequential) == encode_features(X)
    assert {c.exponent for row in parallel for c in row} == {0}
    assert all(c.public_key == pub for row in parallel for c in row)


def test_encrypt_vectors_rejects_non_positive_workers(keypair, sparse_X):
    pub, _ = keypair

    with pytest.raises(ValueError):
        encrypt_vectors(pub, sparse_X, workers=0)
