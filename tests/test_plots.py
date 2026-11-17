import numpy as np
import pytest

from src.plots import cv_by_k, plot_accuracy_vs_k, plot_roc

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


PARAMS = {"nb__alpha": 0.1, "select__k": 20, "tfidf__ngram_range": (1, 1)}


@pytest.fixture
def cv_and_benchmark(tmp_path):
    # Dwie wartości alpha na każde k; wykres ma brać tylko alpha = 0.1.
    lines = ["nb__alpha,select__k,tfidf__ngram_range,mean_test_accuracy,std_test_accuracy,mean_test_f1,std_test_f1"]
    for k, acc in ((20, 0.90), (10, 0.80)):
        lines.append(f'0.1,{k},"(1, 1)",{acc},0.01,{acc - 0.1},0.02')
        lines.append(f'1.0,{k},"(1, 1)",0.5,0.01,0.4,0.02')
        lines.append(f'0.1,{k},"(1, 2)",0.6,0.01,0.5,0.02')
    (tmp_path / "cv_results.csv").write_text("\n".join(lines) + "\n")
    (tmp_path / "benchmark.csv").write_text(
        "key_bits,k,n_messages,workers,t_enc_ms,t_server_ms,t_dec_ms,ct_bytes\n"
        "1024,10,5,1,100,5,0.2,256\n"
        "1024,20,5,1,200,10,0.2,256\n"
    )
    return tmp_path


def test_cv_by_k_keeps_other_params_fixed(cv_and_benchmark):
    cv = cv_by_k(cv_and_benchmark / "cv_results.csv", PARAMS)

    assert cv.index.tolist() == [10, 20]
    assert cv["mean_test_accuracy"].tolist() == [0.80, 0.90]


def test_plot_accuracy_vs_k_writes_pdf(cv_and_benchmark):
    path = cv_and_benchmark / "accuracy-vs-k.pdf"

    plot_accuracy_vs_k(
        cv_and_benchmark / "cv_results.csv", cv_and_benchmark / "benchmark.csv", path, PARAMS
    )

    assert path.read_bytes().startswith(b"%PDF")
