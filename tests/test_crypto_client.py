import pytest

from src.crypto_client import generate_keys

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
