# Detekcja spamu na danych zaszyfrowanych (Paillier)

Projekt 9 z ZTKIK: klient szyfruje cechy SMS-a (TF-IDF) kryptosystemem Pailliera, serwer
liczy na szyfrogramach wynik Naive Bayesa `Δ = b + x·w` i odsyła go zaszyfrowanego, a
klient po odszyfrowaniu zna klasę (spam ⇔ `Δ > 0`). Serwer nie widzi treści wiadomości
ani wyniku, klient nie dostaje modelu.

Autorzy: Krzysztof Śliwiński, Piotr Waluszek. Szczegóły: [`docs/plan-implementacji.md`](docs/plan-implementacji.md),
lista zadań: [`docs/zadania.md`](docs/zadania.md), raport: [`report/`](report/).

## Instalacja

Testowane na Pythonie 3.14 (macOS, Linux).

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import phe.util; print(phe.util.HAVE_GMP)"   # musi być True
```

`gmpy2` jest wymagane: bez niego `phe` szyfruje ok. 13× wolniej (pełny przebieg zamiast
ok. 10 min trwałby kilka godzin).

## Dane

```sh
.venv/bin/python data/download.py
```

Pobiera [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
z UCI do `data/raw/` i sprawdza sumę SHA-256. Drugie uruchomienie niczego nie pobiera.
Plik ma 5574 wiadomości; po usunięciu duplikatów zostaje 5171 (653 spam), z czego 20%
(1035) to zbiór testowy.

## Uruchomienie

Wszystkie wyniki jednym poleceniem (ok. 15 min na 15 rdzeniach, klucz 2048 bit):

```sh
.venv/bin/python -m src.run_all                  # pełny przebieg
.venv/bin/python -m src.run_all --skip-pipeline  # bez ponownego szyfrowania zbioru testowego
.venv/bin/python -m src.run_all --tune           # także strojenie hiperparametrów od nowa
```

Poszczególne kroki (każdy czyta wyniki poprzednich z `results/`):

| polecenie | wynik |
| --- | --- |
| `python -m src.model` | strojenie NB (5-fold CV na train): `results/best_params.json`, `cv_results.csv` |
| `python -m src.pipeline [--limit N] [--key-bits B]` | przebieg zaszyfrowany: `results/cache/scores.npz`, `pipeline_run.json` |
| `python -m src.baselines` | grupy kontrolne: `results/cache/baselines.npz`, `logreg_params.json` |
| `python -m src.evaluate` | `results/metrics.csv` / `.tex`, `agreement.json`, `quantization.json` |
| `python -m src.attacks.ciphertext_classifier` | atak 1: `results/attack_ciphertext.json` |
| `python -m src.attacks.sparse_leak` | atak 2: `results/attack_sparse.json` |
| `python -m src.attacks.nondeterminism` | atak 3 (w `run_all` zapis do `results/attack_nondeterminism.json`) |
| `python -m src.attacks.exponent_leak` | atak 4: `results/exponent_leak.csv` |
| `python -m src.plots` | `results/figures/roc.pdf`, `accuracy-vs-k.pdf` |
| `python -m src.benchmark …` | koszt na wiadomość: `results/benchmark.csv` (polecenia w docstringu) |

Do szybkich prób: `python -m src.pipeline --limit 50 --key-bits 1024` (kilka sekund).

Testy (klucze 512 bit, ok. 10 s; testy na prawdziwych danych są pomijane, jeśli
danych nie pobrano):

```sh
.venv/bin/python -m pytest
```

Raport: `cd report && latexmk main.tex` (IEEEtran; wykresy bierze z `results/figures/`).
Slajdy na obronę: `cd slides && latexmk main.tex` (Beamer, te same wykresy i tabela metryk).

## Wyniki w skrócie

Zbiór testowy: 1035 wiadomości (131 spam), k = 1000 cech, klucz 2048 bit.

| model | accuracy | F1 (spam) | ROC-AUC |
| --- | --- | --- | --- |
| NB na danych zaszyfrowanych | 0,9807 | 0,9219 | 0,9911 |
| jawny NB | 0,9807 | 0,9219 | 0,9911 |
| regresja logistyczna | 0,9797 | 0,9195 | 0,9860 |
| klasa większościowa | 0,8734 | 0 | 0,5 |

- **Zgodność:** wersja zaszyfrowana daje te same predykcje co jawny NB dla 1035/1035
  wiadomości, a odszyfrowany wynik całkowity jest równy (`==`) obliczeniu jawnemu dla
  wszystkich. Błąd kodowania stałoprzecinkowego względem NB na floatach: maks. 1,4 · 10⁻⁵
  przy najmniejszym |Δ| = 0,031, więc żadna predykcja się nie zmienia.
- **Koszt:** ok. 4 s na wiadomość na jednym rdzeniu (głównie szyfrowanie 1000 cech);
  cały zbiór testowy ok. 10 min na 15 rdzeniach. Wiadomość to 512 kB szyfrogramów.
- **Ataki:** klasyfikator na szyfrogramach (atak 1) nie jest lepszy od klasy
  większościowej. Gdyby klient wysyłał tylko niezerowe pozycje, same ich indeksy
  dawałyby serwerowi 98,4% accuracy (atak 2) — dlatego szyfrujemy pełny wektor. Przy
  kodowaniu floatów w `phe` jawny wykładnik zdradzałby niezerowe pozycje w 0,68% wiadomości
  (atak 4, k = 1000); przy kodowaniu stałoprzecinkowym wykładnik nie zdradza nic.

## Odtwarzalność

Podział danych, CV i modele mają stały seed (`SEED = 42` w `src/config.py`), więc metryki,
zgodność i błąd kwantyzacji są za każdym razem identyczne. **Szyfrogramy celowo nie są
odtwarzalne:** `phe` losuje `r` z `SystemRandom` przy każdym szyfrowaniu (przy stałym `r`
szyfrowanie byłoby deterministyczne i atak 3 odczytałby dane bez klucza). Obliczenie na liczbach
całkowitych jest jednak dokładne, więc odszyfrowane wyniki są zawsze te same. Zmieniają
się tylko zmierzone czasy.

Szyfrogramów nie zapisujemy (przy k = 1000 to ok. 530 MB) — `results/cache/` jest
w `.gitignore`, a w repo są tylko małe wyniki końcowe.

## Struktura

```
data/download.py        pobranie danych
src/config.py           wspólne stałe (SEED, SCALE, KEY_BITS, katalogi)
src/data.py             wczytanie, deduplikacja, podział 80/20
src/model.py            TF-IDF → chi² → Multinomial NB, eksport wag log-odds
src/fixed_point.py      kodowanie stałoprzecinkowe (S = 10⁶)
src/crypto_client.py    klucze, szyfrowanie (równoległe), odszyfrowanie
src/crypto_server.py    iloczyn skalarny na szyfrogramach
src/pipeline.py         przebieg end-to-end
src/baselines.py        grupy kontrolne
src/evaluate.py         metryki, zgodność, błąd kwantyzacji
src/plots.py            wykresy do raportu
src/benchmark.py        pomiar kosztu
src/attacks/            ataki 1–4
src/run_all.py          wszystko jednym poleceniem
notebooks/01-eda.ipynb  analiza danych
report/                 raport (IEEEtran)
slides/                 slajdy na obronę (Beamer)
```
