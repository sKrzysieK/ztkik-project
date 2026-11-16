import numpy as np
import pytest

from src.config import SCALE
from src.fixed_point import decode_score, encode_features, encode_weights, plain_int_scores


def _all_python_ints(values) -> bool:
    return all(type(v) is int for v in values)


def test_encode_features_returns_python_ints():
    X = np.array([[0.0, 0.5, 1.0], [0.03, 0.1, 0.7]], dtype=np.float32)

    X_int = encode_features(X)

    assert X_int[0] == [0, SCALE // 2, SCALE]
    assert all(_all_python_ints(row) for row in X_int)


def test_encode_weights_scales_bias_by_scale_squared():
    b_int, w_int = encode_weights(np.float64(-1.5), np.array([0.25, -2.0]))

    assert b_int == -3 * SCALE**2 // 2
    assert w_int == [SCALE // 4, -2 * SCALE]
    assert type(b_int) is int
    assert _all_python_ints(w_int)


def test_plain_int_scores_matches_manual_computation():
    assert plain_int_scores([[1, 2], [0, 3]], b_int=10, w_int=[5, -1]) == [13, 7]


def test_plain_int_scores_rejects_length_mismatch():
    with pytest.raises(ValueError):
        plain_int_scores([[1, 2, 3]], b_int=0, w_int=[1, 2])


def test_decode_score():
    assert decode_score(3 * SCALE**2 // 2) == 1.5
    assert decode_score(-SCALE**2) == -1.0


def test_encode_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        encode_features(np.zeros(3))
    with pytest.raises(ValueError):
        encode_weights(0.0, np.zeros((2, 2)))


def test_quantization_error_within_theoretical_bound():
    # x' = x + e_x, w' = w + e_w, |e| <= 1 / (2 SCALE), więc
    # |x'w' - xw| <= |x| |e_w| + |w| |e_x| + |e_x| |e_w|, plus błąd zaokrąglenia b.
    rng = np.random.default_rng(0)
    X = rng.random((50, 200))
    w = rng.uniform(-8, 8, size=200)
    b = rng.uniform(-3, 3)
    half = 0.5 / SCALE

    b_int, w_int = encode_weights(b, w)
    decoded = np.array([decode_score(d) for d in plain_int_scores(encode_features(X), b_int, w_int)])

    bound = (np.abs(X) * half + np.abs(w) * half + half**2).sum(axis=1) + half / SCALE
    assert np.all(np.abs(decoded - (b + X @ w)) <= bound + 1e-9)


def test_quantization_error_small_on_tfidf_like_data():
    # SMS ma kilkanaście słów: rzadkie wiersze znormalizowane L2, jak z TfidfVectorizer.
    rng = np.random.default_rng(1)
    n, k, nnz = 100, 1000, 15
    X = np.zeros((n, k))
    for row in X:
        row[rng.choice(k, size=nnz, replace=False)] = rng.random(nnz)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    w = rng.uniform(-8, 8, size=k)
    b = -1.9

    b_int, w_int = encode_weights(b, w)
    decoded = np.array([decode_score(d) for d in plain_int_scores(encode_features(X), b_int, w_int)])

    assert np.max(np.abs(decoded - (b + X @ w))) < 1e-4
