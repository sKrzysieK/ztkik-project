import sys

import pytest

from src import run_all

STEPS = ("tune", "pipeline", "baselines", "evaluation", "attacks", "figures")


@pytest.fixture
def calls(monkeypatch):
    # Zamiast prawdziwych kroków (pełny przebieg to kilkanaście minut) zapisujemy kolejność.
    order = []
    for name in STEPS:
        monkeypatch.setattr(run_all, name, lambda *args, _name=name: order.append((_name, args)))
    return order


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["run_all", *argv])
    run_all.main()


def test_full_run_order(calls, monkeypatch):
    run_main(monkeypatch)

    assert [name for name, _ in calls] == ["pipeline", "baselines", "evaluation", "attacks", "figures"]
    assert calls[0][1] == (run_all.KEY_BITS, None)


def test_skip_pipeline_and_tune(calls, monkeypatch):
    run_main(monkeypatch, "--skip-pipeline", "--tune", "--key-bits", "1024", "--workers", "2")

    assert [name for name, _ in calls] == ["tune", "baselines", "evaluation", "attacks", "figures"]
    assert dict(calls)["attacks"] == (1024, 2)
