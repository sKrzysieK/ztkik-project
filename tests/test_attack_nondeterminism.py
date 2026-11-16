import pytest
from phe import EncryptedNumber

from src.attacks.nondeterminism import (
    fixed_r_encrypt,
    recover_assuming_zero_most_frequent,
    recovered_fraction,
    run,
)
from src.crypto_client import generate_keys

TEST_KEY_BITS = 512

# Rzadkie wartości jak po encode_features: dużo zer, kilka niezerowych.
X_INT = [
    [0, 0, 300_000, 0, 0, 954_000],
    [0, 707_107, 0, 0, 707_107, 0],
    [0, 0, 300_000, 0, 0, 954_000],  # duplikat pierwszej wiadomości
]


@pytest.fixture(scope="module")
def keypair():
    return generate_keys(TEST_KEY_BITS)


def test_fixed_r_is_valid_but_deterministic_paillier(keypair):
    pub, priv = keypair

    ct = fixed_r_encrypt(pub, X_INT, r=12345)

    assert [[priv.decrypt(EncryptedNumber(pub, c, 0)) for c in row] for row in ct] == X_INT
    assert ct[0] == ct[2]  # ta sama wiadomość → te same szyfrogramy
    assert len({c for row in ct for c in row}) == len({m for row in X_INT for m in row})


def test_attack_recovers_everything_from_fixed_r(keypair):
    pub, _ = keypair

    recovered = recover_assuming_zero_most_frequent(pub, fixed_r_encrypt(pub, X_INT, r=12345))

    assert recovered == X_INT
    assert recovered_fraction(recovered, X_INT) == 1.0


def test_attack_recovers_nothing_from_random_r(keypair):
    pub, _ = keypair
    random_ct = [[pub.encrypt(m).ciphertext(be_secure=False) for m in row] for row in X_INT]

    recovered = recover_assuming_zero_most_frequent(pub, random_ct)

    # Jedyną „odzyskaną” pozycją może być sam szyfrogram uznany za Enc(0).
    assert sum(r is not None for row in recovered for r in row) == 1


def test_run_compares_both_variants():
    result = run(key_bits=TEST_KEY_BITS, n_messages=5, k=20, workers=1)

    assert result["distinct_ct_random"] == result["n_ciphertexts"]
    assert result["distinct_ct_fixed_r"] == result["distinct_plaintexts"]
    assert result["recovered_fixed_r"] == 1.0
    assert result["recovered_random"] < 0.05
