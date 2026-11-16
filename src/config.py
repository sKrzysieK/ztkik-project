"""Wspólne stałe projektu. Inne moduły importują je stąd, zamiast definiować własne."""

from pathlib import Path

SEED = 42

# Kodowanie stałoprzecinkowe: x_i i w_i mnożone przez SCALE, b przez SCALE**2,
# żeby b miało tę samą skalę co iloczyny x_i · w_i.
SCALE = 10**6

# Długość modułu n klucza Paillier: wyniki końcowe 2048, iteracje 1024, testy 512.
KEY_BITS = 2048

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
FIGURES_DIR = RESULTS_DIR / "figures"

for _dir in (RESULTS_DIR, CACHE_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
