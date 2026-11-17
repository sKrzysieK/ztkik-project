import pytest

from src.data import RAW_FILE, load_dataset, load_split, read_raw

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
    assert len(read_raw()) == 5574
    df = load_dataset()

    assert len(df) == 5171
    assert not df["text"].duplicated().any()
    assert set(df["label"]) == {0, 1}
    assert df["label"].sum() == 653  # spam


@pytest.fixture
def balanced_file(tmp_path):
    # 100 różnych wiadomości, 20% spamu, jak w małej wersji prawdziwego zbioru.
    lines = [f"{'spam' if i % 5 == 0 else 'ham'}\twiadomosc {i}\n" for i in range(100)]
    path = tmp_path / "SMSSpamCollection"
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_load_split_is_stratified_80_20(balanced_file):
    texts_train, texts_test, y_train, y_test = load_split(balanced_file)

    assert (len(texts_train), len(texts_test)) == (80, 20)
    assert (len(y_train), len(y_test)) == (80, 20)
    assert y_train.mean() == y_test.mean() == 0.2


def test_load_split_is_reproducible_and_disjoint(balanced_file):
    first, second = load_split(balanced_file), load_split(balanced_file)

    assert first[0] == second[0] and first[1] == second[1]
    assert (first[2] == second[2]).all() and (first[3] == second[3]).all()
    assert not set(first[0]) & set(first[1])


@pytest.mark.skipif(not RAW_FILE.exists(), reason="brak danych: python data/download.py")
def test_real_split_keeps_spam_ratio():
    texts_train, texts_test, y_train, y_test = load_split()

    assert (len(texts_train), len(texts_test)) == (4136, 1035)
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert not set(texts_train) & set(texts_test)
