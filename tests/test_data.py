import pytest

from src.data import RAW_FILE, load_dataset

# Cudzysłów bez pary jak w prawdziwych SMS-ach; trzeci wiersz to duplikat pierwszego.
SAMPLE = (
    'ham\tOk lar... Joking wif u oni...\n'
    'spam\tFree entry in 2 a wkly comp "to win FA Cup\n'
    'ham\tOk lar... Joking wif u oni...\n'
    'ham\tU dun say so early hor...\n'
)


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "SMSSpamCollection"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


def test_load_dataset_maps_labels_and_drops_duplicates(sample_file):
    df = load_dataset(sample_file)

    assert list(df.columns) == ["label", "text"]
    assert df["label"].tolist() == [0, 1, 0]
    assert df["text"].tolist()[1] == 'Free entry in 2 a wkly comp "to win FA Cup'
    assert df.index.tolist() == [0, 1, 2]


def test_load_dataset_rejects_unknown_label(tmp_path):
    path = tmp_path / "SMSSpamCollection"
    path.write_text("ham\tok\nphishing\tclick here\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_dataset(path)


def test_load_dataset_without_file_points_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="download.py"):
        load_dataset(tmp_path / "brak")


@pytest.mark.skipif(not RAW_FILE.exists(), reason="brak danych: python data/download.py")
def test_real_dataset_after_deduplication():
    df = load_dataset()

    assert len(df) == 5171
    assert not df["text"].duplicated().any()
    assert set(df["label"]) == {0, 1}
    assert df["label"].sum() == 653  # spam
