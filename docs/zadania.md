# Lista zadań — Projekt 9: Detekcja spamu na danych zaszyfrowanych (Paillier)

Zespół: **K = Krzysztof Śliwiński** (kryptografia, protokół, bezpieczeństwo), **P = Piotr Waluszek** (dane, ML, ewaluacja).
Kontekst techniczny i uzasadnienia: [`plan-implementacji.md`](plan-implementacji.md). Ten plik jest jedynym źródłem prawdy dla listy zadań i interfejsu między nami.

## Jak korzystać z tego pliku

- Zadania są pogrupowane w **rundy**. Runda 1 (K) nie zależy od niczego, co robi P. Runda 2 (P) potrzebuje rundy 1. Runda 3 (K) potrzebuje wyników z rundy 2. Gdybyśmy pracowali równolegle, wystarczy pilnować kolumny „zależy od”.
- Jedno zadanie = jedna gałąź `zadanie/<ID>-<krótki-opis>` = jeden PR, review robi druga osoba. Commity małe, z własnego konta (Contribution Matrix liczy się z logów Git).
- Zadanie odhaczasz (`[x]`) w tym pliku w tym samym PR, w którym je kończysz.
- Testy jednostkowe są częścią zadania, nie osobnym zadaniem. Klucze w testach: 512 bit (szybko).
- „(opc.)” = robimy tylko, jeśli zostanie czas; nie blokują żadnego zadania obowiązkowego (poza K22, które opisuje atak 3 tylko, jeśli K15 jest zrobione).
- Przy przekazaniu pliku dopisz krótką notatkę w sekcji [Przekazania](#przekazania).

## Kontrakt interfejsu (ustalony z góry, żeby nikt nie czekał na stuby)

```python
# src/config.py (K04) — wspólne stałe
SEED = 42
SCALE = 10**6          # x_i i w_i mnożone przez SCALE, b przez SCALE**2
KEY_BITS = 2048        # wyniki końcowe; iteracje: 1024; testy: 512
DATA_RAW, RESULTS_DIR, CACHE_DIR, FIGURES_DIR  # pathlib.Path

# src/data.py (P)
def load_dataset() -> pd.DataFrame                 # kolumny: label (1 = spam, 0 = ham), text; po deduplikacji
def load_split() -> tuple[list[str], list[str], np.ndarray, np.ndarray]
                                                   # texts_train, texts_test, y_train, y_test (stratyfikowany 80/20, SEED)

# src/model.py (P)
def fit_model(texts_train, y_train) -> Pipeline    # TfidfVectorizer -> SelectKBest(chi2) -> MultinomialNB, najlepsze z CV
def features(model, texts) -> np.ndarray           # (n, k), float64, dense
def log_odds_weights(model) -> tuple[float, np.ndarray]   # b: float, w: (k,) float64; spam <=> b + x·w > 0

# src/fixed_point.py (K)
def encode_features(X) -> list[list[int]]          # round(x * SCALE), Pythonowe int
def encode_weights(b, w) -> tuple[int, list[int]]  # round(b * SCALE**2), round(w * SCALE)
def plain_int_scores(X_int, b_int, w_int) -> list[int]    # to samo co serwer, ale jawnie
def decode_score(delta_int: int) -> float          # delta_int / SCALE**2

# src/crypto_client.py (K)
def generate_keys(bits=KEY_BITS) -> tuple[PaillierPublicKey, PaillierPrivateKey]
def encrypt_vectors(pub, X: np.ndarray, workers: int | None = None) -> list[list[EncryptedNumber]]
def decrypt_predict(priv, enc_scores) -> tuple[np.ndarray, np.ndarray, list[int]]
                                                   # y_pred (0/1), scores (float, do ROC-AUC), surowe int (do testu ==)

# src/crypto_server.py (K)
def server_score(enc_X, b: float, w: np.ndarray) -> list[EncryptedNumber]   # Enc(Δ) per wiadomość

# results/cache/scores.npz (P09/P10) — wejście dla evaluate.py
#   y_test, delta_enc (float64), delta_float (float64, NB bez kwantyzacji), exact_int_match (bool)
```

Zasady na granicy z `phe`: do `encrypt` / mnożenia trafiają wyłącznie Pythonowe `int` (nie `np.int64` / `np.float32` — `TypeError`). Dzięki temu wykładnik (`exponent`) każdego szyfrogramu wynosi 0.

---

## Runda 1 — Krzysztof (setup, kryptografia, szkielet raportu)

### Setup

- [ ] **K01 · Repo na GitHubie** — zależy od: —
  - Zrobić: utworzyć repo, dodać Piotra jako współpracownika, wypchnąć obecny stan (`.gitignore`, `docs/`, karta projektu).
  - Gotowe gdy: Piotr klonuje repo i może pushować gałęzie.
- [x] **K02 · Szkielet katalogów** — zależy od: K01
  - Zrobić: katalogi z sekcji 6 planu (`src/`, `src/attacks/`, `tests/`, `data/`, `notebooks/`, `results/`, `report/`), `src/__init__.py`, puste moduły z kontraktu z docstringiem i `raise NotImplementedError`.
  - Gotowe gdy: `python -c "import src.data, src.model, src.crypto_client"` działa.
- [x] **K03 · `requirements.txt`** — zależy od: K01
  - Zrobić: przypięte wersje `phe`, `gmpy2`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `pytest`.
  - Gotowe gdy: instalacja w czystym `.venv` przechodzi, a szyfrowanie kluczem 2048 bit trwa kilka ms (≈ 4 ms; ≈ 60 ms oznacza, że `phe` nie używa `gmpy2`).
- [x] **K04 · `src/config.py`** — zależy od: K02
  - Zrobić: stałe z kontraktu; katalogi wyników tworzone przy imporcie, jeśli nie istnieją.
  - Gotowe gdy: inne moduły importują stałe z `src.config`, a nie definiują własnych.

### Kryptografia

- [x] **K05 · Kodowanie stałoprzecinkowe** — `src/fixed_point.py` — zależy od: K04
  - Zrobić: cztery funkcje z kontraktu.
  - Gotowe gdy: testy: wynik to Pythonowe `int`; `decode_score(plain_int_scores(...))` różni się od `b + X·w` (float) o mniej niż 1e-4 na losowych danych; ujemne `b` i `w` działają.
- [x] **K06 · Generowanie kluczy** — `generate_keys` w `src/crypto_client.py` — zależy od: K04
  - Gotowe gdy: test: klucz ma żądaną długość `n`.
- [x] **K07 · Szyfrowanie wektorów (1 proces)** — `encrypt_vectors` z `workers=1` — zależy od: K05, K06
  - Zrobić: `encode_features` → `pub.encrypt(int)` dla **każdej** pozycji, także zer.
  - Gotowe gdy: testy: wynik ma kształt `n × k`; każdy szyfrogram ma `exponent == 0`; dwa szyfrowania tego samego wektora dają różne `ciphertext()`.
- [ ] **K08 · Szyfrowanie równoległe** — `encrypt_vectors` z `workers > 1` — zależy od: K07
  - Zrobić: `multiprocessing.Pool` po wierszach; na macOS jest `spawn`, więc funkcja robocza na poziomie modułu, a uruchomienie pod `if __name__ == "__main__"`.
  - Gotowe gdy: test: odszyfrowane wyniki identyczne dla `workers=1` i `workers=4`; pomiar przyspieszenia wpisany w opis PR.
- [ ] **K09 · Obliczenie serwera** — `server_score` w `src/crypto_server.py` — zależy od: K05, K07
  - Zrobić: `encode_weights`, potem `Σ Enc(x_i) * w_int_i + b_int`. Moduł nie importuje niczego z klucza prywatnego.
  - Gotowe gdy: test: wynik ma `exponent == 0`.
- [ ] **K10 · Odszyfrowanie i predykcja** — `decrypt_predict` — zależy od: K09
  - Gotowe gdy: test: `y_pred == (scores > 0)`; zwracane surowe `int` gotowe do porównania `==`.
- [ ] **K11 · Test integracyjny `enc == plain`** — `tests/test_crypto_e2e.py` — zależy od: K10
  - Zrobić: losowe `X` (20 × 50, rzadkie, wartości jak TF-IDF), losowe `b`, `w`; pełny przebieg K07 → K09 → K10.
  - Gotowe gdy: surowe int z `decrypt_predict` są **równe** `plain_int_scores`, a predykcje są równe znakowi `b + X·w` (float). `pytest` przechodzi w < 30 s.

### Benchmark i ataki (część niezależna od danych)

- [ ] **K12 · Funkcja benchmarku** — `src/benchmark.py` — zależy od: K10
  - Zrobić: `measure(key_bits, k, n_messages)` na syntetycznych danych → czas szyfrowania, serwera i odszyfrowania na wiadomość, rozmiar szyfrogramu w bajtach.
  - Gotowe gdy: zwraca słownik; `python -m src.benchmark --key-bits 1024 --k 100 --n 5` wypisuje wynik.
- [ ] **K13 · Przebiegi benchmarku** — zależy od: K12
  - Zrobić: klucze 1024 i 2048 bit × k ∈ {500, 750, 1000}, uśrednienie z kilku wiadomości.
  - Gotowe gdy: `results/benchmark.csv` z kolumnami `key_bits, k, t_enc_ms, t_server_ms, t_dec_ms, ct_bytes` (CSV commitujemy, to mały wynik końcowy).
- [ ] **K14 · Benchmark 3072 bit (opc.)** — zależy od: K13
  - Gotowe gdy: dopisane wiersze 3072 w `results/benchmark.csv`.
- [ ] **K15 · Atak 3: niedeterminizm szyfrowania (opc.)** — `src/attacks/nondeterminism.py` — zależy od: K07
  - Zrobić: ta sama wiadomość szyfrowana N razy → wszystkie szyfrogramy różne; dla kontrastu „deterministyczny” wariant (stałe `r`) → identyczne szyfrogramy, więc analiza częstości jest możliwa.
  - Gotowe gdy: skrypt wypisuje liczbę unikalnych szyfrogramów w obu wariantach.
- [ ] **K16 · Atak 4: funkcja wycieku przez `exponent`** — `src/attacks/exponent_leak.py` — zależy od: K07
  - Zrobić: `exponent_leak(pub, X, encoding)` dla `encoding ∈ {"float", "fixed"}` → odsetek wiadomości, w których serwer odczytuje z wykładnika co najmniej jedną pozycję niezerową, oraz odsetek odczytanych pozycji. Rozwijać na danych syntetycznych.
  - Gotowe gdy: test: dla `"fixed"` wyciek = 0; dla `"float"` wyciek > 0 na danych zawierających np. 0.03 lub 1.0.

### Raport (część niezależna od wyników)

- [ ] **K17 · Szablon IEEEtran** — `report/` — zależy od: K02
  - Zrobić: `IEEEtran.cls` (`tlmgr install ieeetran` albo plik w `report/`), `main.tex` (conference), **każda sekcja w osobnym pliku** `report/sections/*.tex` (mniej konfliktów przy równoległej pracy), `\graphicspath{{../results/figures/}}`, `references.bib` z trzema źródłami z karty projektu.
  - Gotowe gdy: `latexmk -pdf main.tex` buduje PDF z pustymi sekcjami bez błędów.
- [ ] **K18 · Sekcja II. Background** — `sections/background.tex` — zależy od: K17
  - Zrobić: Paillier (generowanie kluczy, szyfrowanie, własności homomorficzne, DCRA, IND-CPA), Multinomial NB i sprowadzenie do log-odds.
- [ ] **K19 · Sekcja III. Threat Model** — `sections/threat-model.tex` — zależy od: K17
  - Zrobić: na podstawie sekcji 3 planu: profil przeciwnika, interfejs (CPA), Kerckhoffs, poza zakresem, kryteria sukcesu (sekcja 1 planu).
- [ ] **K20 · Sekcja IV. System Design** — `sections/system-design.tex` — zależy od: K11, K17
  - Zrobić: protokół (tabela kroków z sekcji 2 planu), kodowanie stałoprzecinkowe i uzasadnienie (wykładnik, dokładność `==`), szyfrowanie pełnego wektora.

---

## Runda 2 — Piotr (dane, model, przebieg, ewaluacja, ataki 1–2)

### Dane i model

- [ ] **P01 · Pobieranie danych** — `data/download.py` — zależy od: K04
  - Zrobić: pobranie SMS Spam Collection z UCI do `data/raw/` (idempotentnie: drugi przebieg nic nie pobiera).
  - Gotowe gdy: `python data/download.py` na czystym klonie daje plik z 5574 wierszami.
- [ ] **P02 · `load_dataset` z deduplikacją** — `src/data.py` — zależy od: P01
  - Gotowe gdy: test: 5171 wierszy, brak duplikatów `text`, `label` ∈ {0, 1}.
- [ ] **P03 · `load_split`** — `src/data.py` — zależy od: P02
  - Gotowe gdy: test: proporcja spamu w train i test różni się o < 1 p.p.; dwa wywołania dają identyczny podział; zbiory rozłączne.
- [ ] **P04 · EDA** — `notebooks/01-eda.ipynb` — zależy od: P02
  - Zrobić: rozkład klas, długości wiadomości, liczba duplikatów przed i po deduplikacji, najczęstsze słowa per klasa.
- [ ] **P05 · Strojenie modelu** — `fit_model` w `src/model.py` — zależy od: P03
  - Zrobić: `Pipeline` (TF-IDF → `SelectKBest(chi2)` → `MultinomialNB`) w `GridSearchCV`, 5-fold, **tylko na train**; siatka: `alpha`, k ∈ 500–1000, parametry TF-IDF.
  - Gotowe gdy: zapisane `results/best_params.json` i `results/cv_results.csv` (potrzebne do wykresu accuracy vs. k).
- [ ] **P06 · Eksport cech i wag** — `features`, `log_odds_weights` — zależy od: P05
  - Zrobić: `b = class_log_prior_[1] − class_log_prior_[0]`, `w = feature_log_prob_[1] − feature_log_prob_[0]`; `features` zwraca dense `float64`.
  - Gotowe gdy: test: `(b + X_test·w > 0)` daje **te same** predykcje co `model.predict(texts_test)`.
- [ ] **P07 · Grupy kontrolne** — `src/evaluate.py` lub `src/baselines.py` — zależy od: P06
  - Zrobić: metryki jawnego NB i `DummyClassifier(strategy="most_frequent")` na zbiorze testowym.
- [ ] **P08 · Regresja logistyczna (opc.)** — zależy od: P07
  - Zrobić: jako dodatkowa grupa kontrolna (też liniowa, więc też działałaby na Paillierze).

### Przebieg zaszyfrowany

- [ ] **P09 · Skrypt end-to-end** — `src/pipeline.py` — zależy od: P06, K11
  - Zrobić: `python -m src.pipeline [--limit N] [--key-bits B]`: podział → model → cechy → klucze → `encrypt_vectors` → `server_score` → `decrypt_predict` → zapis `results/cache/scores.npz` (format z kontraktu); `exact_int_match` liczone przez porównanie z `plain_int_scores`.
  - Gotowe gdy: `--limit 50 --key-bits 1024` przechodzi, a `exact_int_match` jest wszędzie `True` (kamień milowy: pierwszy przebieg end-to-end).
- [ ] **P10 · Pełny przebieg** — zależy od: P09, K08
  - Zrobić: cały zbiór testowy, klucz 2048 bit, `workers` = liczba rdzeni.
  - Gotowe gdy: pełny `scores.npz` istnieje; czas przebiegu wpisany w opis PR / notatkę przekazania.

### Ewaluacja (`src/evaluate.py`, czyta tylko z cache)

- [ ] **P11 · Tabela metryk** — zależy od: P07, P10
  - Zrobić: accuracy, F1 (spam), precision, recall, ROC-AUC (z `delta_enc`) dla: wersji zaszyfrowanej, jawnego NB, Dummy.
  - Gotowe gdy: `results/metrics.csv` i ta sama tabela jako `results/metrics.tex` (booktabs) do raportu.
- [ ] **P12 · Zgodność i dokładność** — zależy od: P10
  - Zrobić: odsetek identycznych predykcji enc vs. jawny NB (kryterium: 100%), odsetek `exact_int_match` (kryterium: 100%).
- [ ] **P13 · Błąd kwantyzacji** — zależy od: P10
  - Zrobić: statystyki `|delta_enc − delta_float|` (max, średnia) i liczba wiadomości, dla których kwantyzacja zmieniła znak.
- [ ] **P14 · Krzywa ROC** — zależy od: P10
  - Gotowe gdy: `results/figures/roc.pdf`.
- [ ] **P15 · Wykres accuracy vs. k vs. czas** — zależy od: P05, K13
  - Zrobić: accuracy z `cv_results.csv`, czas na wiadomość z `benchmark.csv`.
  - Gotowe gdy: `results/figures/accuracy-vs-k.pdf`.

### Ataki 1–2

- [ ] **P16 · Atak 1: klasyfikator na szyfrogramach (CPA)** — `src/attacks/ciphertext_classifier.py` — zależy od: K08, P06
  - Zrobić: „serwer” szyfruje kluczem publicznym oznaczone wiadomości (wystarczy kilkaset, dla czasu), cechy z szyfrogramów (np. najmłodsze bajty, reszty modulo małe liczby), Random Forest z CV.
  - Gotowe gdy: wynik (accuracy, F1) porównany z Dummy; oczekiwany ≈ Dummy.
- [ ] **P17 · Atak 2: wyciek przy wektorze rzadkim** — `src/attacks/sparse_leak.py` — zależy od: P06
  - Zrobić: serwer widzi tylko indeksy niezerowych pozycji → klasyfikator na binarnym wektorze obecności słów.
  - Gotowe gdy: wynik porównany z jawnym NB; oczekiwany zbliżony do jawnego modelu.

### Odtwarzalność

- [ ] **P18 · Jedno polecenie** — zależy od: P11–P17, K13
  - Zrobić: `python -m src.run_all` (pipeline → evaluate → ataki 1–2 i 3, jeśli jest → wykresy). Dwa przebiegi dają identyczne metryki (szyfrogramy różne, wyniki te same).
- [ ] **P19 · README** — zależy od: P18
  - Zrobić: instalacja, pobranie danych, uruchomienie, wyniki w skrócie, uwaga, że szyfrogramy celowo nie są odtwarzalne (`phe` losuje `r` z `SystemRandom`).

### Raport

- [ ] **P20 · Sekcja V. Experimental Setup** — `sections/setup.tex` — zależy od: K17, P05, P07
  - Zrobić: dane i deduplikacja, podział, CV, siatka hiperparametrów, grupy kontrolne, sprzęt i klucze.
- [ ] **P21 · Sekcja VI. Results** — `sections/results.tex` — zależy od: P11–P15
  - Zrobić: tabela metryk, zgodność, błąd kwantyzacji, wydajność (tabela z `benchmark.csv`, wykres accuracy vs. k).
- [ ] **P22 · Sekcja VII. Security Analysis — ataki 1–2** — `sections/security.tex` (podsekcje 1–2) — zależy od: P16, P17
- [ ] **P23 · Abstract, Index Terms, I. Introduction** — zależy od: P21, K18–K20
- [ ] **P24 · Slajdy na obronę** — zależy od: P21, P22, K20

---

## Runda 3 — Krzysztof (atak 4 na danych, domknięcie raportu)

- [ ] **K21 · Atak 4 na prawdziwych danych** — zależy od: K16, P06
  - Zrobić: `exponent_leak` na `X_test` dla k = 500 i k = najlepsze z CV, oba kodowania; dopisać atak do `src.run_all`.
  - Gotowe gdy: `results/exponent_leak.csv`; dla `"float"` wyciek > 0 (spodziewane ok. 1,6% wiadomości przy k=500), dla `"fixed"` 0. Po merge'u **zamrożenie kodu**: tag `v1.0-kod`, dalej zmieniamy tylko raport.
- [ ] **K22 · Sekcja VII. Security Analysis — ataki 3–4** — `sections/security.tex` (podsekcje 3–4) — zależy od: K21, P22, K15 (jeśli zrobione)
- [ ] **K23 · Sekcja VIII. Conclusions** — zależy od: P21, P22, K22
  - Zrobić: czy kryteria z Threat Modelu spełnione (accuracy > 94%, zgodność 100%, `==`), koszt, ograniczenia (jawny model, pasywny serwer).
- [ ] **K24 · Korekta całości** — zależy od: K23, P23
  - Gotowe gdy: PDF w limicie stron (domyślnie 6), brak `??` i ostrzeżeń o brakujących cytowaniach, wszystkie wykresy czytelne w dwóch kolumnach.
- [ ] **K25 · Próba obrony** — zależy od: P24, K24
  - Zrobić: każdy tłumaczy kod drugiej osoby i odpowiada na pytania z Threat Modelu.

---

## Podsumowanie

| Runda | Osoba | Zadania | Opcjonalne |
| --- | --- | --- | --- |
| 1 | Krzysztof | K01–K20 | K14, K15 |
| 2 | Piotr | P01–P24 | P08 |
| 3 | Krzysztof | K21–K25 | — |

Krzysztof: 25 zadań (2 opc.), Piotr: 24 zadania (1 opc.).

## Przekazania

Przy każdym przekazaniu dopisz: data, kto → komu, co skończone, co nie działa / na co uważać.

| Data | Kto → komu | Notatka |
| --- | --- | --- |
| | | |
