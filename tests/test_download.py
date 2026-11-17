"""Testy pobierania bez sieci: archiwum podajemy jako lokalny plik (`file://`)."""

import hashlib
import zipfile

import pytest

from data.download import FILENAME, download

CONTENT = b"ham\tOk lar... Joking wif u oni...\nspam\tFree entry in 2 a wkly comp\n"
SHA256 = hashlib.sha256(CONTENT).hexdigest()


@pytest.fixture
def archive_url(tmp_path):
    path = tmp_path / "sms.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(FILENAME, CONTENT)
        archive.writestr("readme", b"nie wypakowujemy")
    return path.as_uri()


def test_download_extracts_only_dataset_file(tmp_path, archive_url):
    dest = tmp_path / "raw"

    path = download(dest, archive_url, SHA256)

    assert path == dest / FILENAME
    assert path.read_bytes() == CONTENT
    assert [p.name for p in dest.iterdir()] == [FILENAME]


def test_second_run_does_not_download(tmp_path, archive_url):
    dest = tmp_path / "raw"
    download(dest, archive_url, SHA256)

    # Adres, którego nie da się otworzyć: drugi przebieg nie może po niego sięgnąć.
    missing_url = (tmp_path / "brak.zip").as_uri()
    assert download(dest, missing_url, SHA256).read_bytes() == CONTENT


def test_corrupted_file_is_downloaded_again(tmp_path, archive_url):
    dest = tmp_path / "raw"
    dest.mkdir()
    (dest / FILENAME).write_bytes(b"ucieta tresc")

    assert download(dest, archive_url, SHA256).read_bytes() == CONTENT


def test_wrong_checksum_is_rejected_and_nothing_is_written(tmp_path, archive_url):
    dest = tmp_path / "raw"

    with pytest.raises(ValueError):
        download(dest, archive_url, "0" * 64)

    assert not (dest / FILENAME).exists()
