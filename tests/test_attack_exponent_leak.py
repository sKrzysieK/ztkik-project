import numpy as np
import pytest

from src.attacks.exponent_leak import exponent_leak, exponents, run, save_result
from src.crypto_client import encrypt_vectors, generate_keys
from src.data import RAW_FILE

needs_data = pytest.mark.skipif(not RAW_FILE.exists(), reason="brak danych: python data/download.py")

# Wiersz 0: wartości z [1/16, 1) — przy floatach wykładnik taki sam jak zera, brak wycieku.
# Wiersz 1: 0.03 < 1/16 i 1.0 — przy floatach inne wykładniki niż zero, wyciek.
X = np.array(
    [
        [0.0, 0.5, 0.0, 0.7, 0.0],
        [0.03, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


@pytest.fixture(scope="module")
def pub():
    return generate_keys(512)[0]


def test_float_encoding_leaks_small_values_and_one(pub):
    leak = exponent_leak(pub, X, "float")

    assert leak["messages_leaking"] == pytest.approx(1 / 3)  # tylko wiersz 1
    assert leak["nonzero_revealed"] == pytest.approx(2 / 4)  # 0.03 i 1.0 z 4 niezerowych


def test_fixed_encoding_leaks_nothing(pub):
    leak = exponent_leak(pub, X, "fixed")

    assert leak == {"messages_leaking": 0.0, "nonzero_revealed": 0.0}


def test_exponents_match_real_ciphertexts(pub):
    float_ct = [[pub.encrypt(float(x)) for x in row] for row in X]

    assert exponents(pub, X, "float").tolist() == [[c.exponent for c in row] for row in float_ct]
    assert exponents(pub, X, "fixed").tolist() == [
        [c.exponent for c in row] for row in encrypt_vectors(pub, X, workers=1)
    ]


def test_unknown_encoding_is_rejected(pub):
    with pytest.raises(ValueError):
        exponent_leak(pub, X, "decimal")


def test_save_result_writes_csv_with_lf(tmp_path):
    rows = [
        {"k": 500, "encoding": "float", "messages_leaking": 0.5},
        {"k": 500, "encoding": "fixed", "messages_leaking": 0.0},
    ]
    save_result(rows, tmp_path / "leak.csv")

    assert (tmp_path / "leak.csv").read_bytes() == (
        b"k,encoding,messages_leaking\n500,float,0.5\n500,fixed,0.0\n"
    )


@needs_data
def test_real_data_leaks_only_with_float_encoding():
    rows = run()

    assert {(row["k"], row["encoding"]) for row in rows} == {
        (k, encoding) for k in (500, 1000) for encoding in ("float", "fixed")
    }
    for row in rows:
        assert row["n_messages"] == 1035
        if row["encoding"] == "float":
            assert row["messages_leaking"] > 0 and row["nonzero_revealed"] > 0
        else:
            assert row["messages_leaking"] == 0 and row["nonzero_revealed"] == 0
