import numpy as np
import pytest

from src.plots import plot_roc

Y_TEST = np.array([1, 0, 0, 1, 0, 1])


@pytest.fixture
def cache(tmp_path):
    scores = np.array([2.0, -1.0, -3.0, 0.5, 0.2, 1.0])
    np.savez(
        tmp_path / "scores.npz",
        y_test=Y_TEST,
        delta_enc=scores,
        delta_float=scores,
        exact_int_match=np.ones(6, dtype=bool),
    )
    np.savez(
        tmp_path / "baselines.npz",
        y_test=Y_TEST,
        plain_nb_pred=(scores > 0).astype(int),
        plain_nb_score=scores,
        dummy_pred=np.zeros(6, dtype=int),
        dummy_score=np.full(6, 0.4),
    )
    return tmp_path


def test_plot_roc_writes_pdf(cache):
    path = cache / "figures" / "roc.pdf"

    plot_roc(cache / "scores.npz", cache / "baselines.npz", path)

    assert path.read_bytes().startswith(b"%PDF")


def test_plot_roc_is_reproducible(cache):
    plot_roc(cache / "scores.npz", cache / "baselines.npz", cache / "a.pdf")
    plot_roc(cache / "scores.npz", cache / "baselines.npz", cache / "b.pdf")

    # Bez daty utworzenia ten sam wykres daje identyczny plik (czysty `git diff`).
    assert (cache / "a.pdf").read_bytes() == (cache / "b.pdf").read_bytes()


def test_plot_roc_rejects_scores_from_other_messages(cache):
    np.savez(cache / "scores.npz", y_test=Y_TEST[:3], delta_enc=np.zeros(3))

    with pytest.raises(ValueError, match="--limit"):
        plot_roc(cache / "scores.npz", cache / "baselines.npz", cache / "roc.pdf")
