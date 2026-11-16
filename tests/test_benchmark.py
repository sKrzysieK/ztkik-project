import numpy as np

from src.benchmark import WORDS_PER_MESSAGE, measure, synthetic_data, write_csv


def test_synthetic_data_looks_like_tfidf():
    X, b, w = synthetic_data(n_messages=4, k=50)

    assert X.shape == (4, 50)
    assert w.shape == (50,)
    assert isinstance(b, float)
    assert ((X > 0).sum(axis=1) == WORDS_PER_MESSAGE).all()
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)


def test_synthetic_data_is_reproducible():
    first, second = synthetic_data(3, 20), synthetic_data(3, 20)

    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_measure_reports_all_stages():
    result = measure(key_bits=512, k=10, n_messages=2)

    assert result["key_bits"] == 512
    assert result["k"] == 10
    assert result["ct_bytes"] == 128  # element Z_{n^2}: 2 · 512 bit
    assert all(result[key] > 0 for key in ("t_enc_ms", "t_server_ms", "t_dec_ms"))


def test_write_csv_rounds_times(tmp_path):
    path = tmp_path / "out" / "benchmark.csv"

    write_csv([{"key_bits": 512, "k": 10, "t_enc_ms": 1.23456, "ct_bytes": 128}], path)

    assert path.read_bytes() == b"key_bits,k,t_enc_ms,ct_bytes\n512,10,1.23,128\n"


def test_write_csv_append_adds_rows_without_second_header(tmp_path):
    path = tmp_path / "benchmark.csv"

    write_csv([{"key_bits": 1024, "k": 10}], path)
    write_csv([{"key_bits": 3072, "k": 10}], path, append=True)

    assert path.read_text().splitlines() == ["key_bits,k", "1024,10", "3072,10"]


def test_write_csv_append_to_missing_file_writes_header(tmp_path):
    path = tmp_path / "benchmark.csv"

    write_csv([{"key_bits": 3072, "k": 10}], path, append=True)

    assert path.read_text().splitlines() == ["key_bits,k", "3072,10"]
