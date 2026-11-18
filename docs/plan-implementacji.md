# Plan implementacji — Projekt 9: Detekcja spamu na danych zaszyfrowanych (Paillier)

Zespół: 2 osoby — **A: Piotr Waluszek** (dane, ML, ewaluacja) i **B: Krzysztof Śliwiński** (kryptografia, protokół, bezpieczeństwo). W dalszej części A = Piotr, B = Krzysztof.
Źródło wymagań: `../ZTKIK_PROJEKT_2026_2027.md`.
Termin oddania planu i listy zadań: **poniedziałek 09.11.2026**. Termin oddania projektu: do potwierdzenia — harmonogram zakłada ok. 2 tygodnie realizacji.
Repozytorium: osobne repo zespołowe (ten katalog), niezależne od repo kursu.
Raport: LaTeX, szablon **IEEEtran** (format konferencyjny IEEE), nie `unireport`.

## 1. Cel i kryteria sukcesu

- Klasyfikator Multinomial Naive Bayes na cechach TF-IDF, w którym **serwer liczy wynik na zaszyfrowanych cechach** (Paillier, biblioteka `phe`), a klient tylko szyfruje i odszyfrowuje.
- Kryteria (do wpisania w Threat Model jako „ilorazowe kryterium sukcesu”):
  1. Accuracy na danych zaszyfrowanych **> 94%**.
  2. **100% zgodności predykcji** wersji zaszyfrowanej z jawną (ta sama predykcja dla każdej wiadomości) — to formalizuje „identyczną dokładność”. Część kryptograficzną sprawdzamy jeszcze ostrzej: odszyfrowany wynik musi być **dokładnie równy** wynikowi jawnemu policzonemu na tych samych liczbach całkowitych (patrz sekcja 2).
  3. Dodatkowo raportujemy F1 (klasa spam), precision/recall, ROC-AUC — zbiór jest niezbalansowany (~87% ham), więc sama accuracy jest myląca.

## 2. Idea techniczna

Paillier wspiera: `Enc(a) ⊕ Enc(b) = Enc(a+b)` oraz `Enc(a) ⊗ k = Enc(a·k)` dla jawnego `k`. To wystarcza dla NB, bo decyzja jest liniowa w cechach. Przy dwóch klasach wystarczy jeden wynik — logarytm ilorazu szans (log-odds):

```
Δ = [log P(spam) − log P(ham)] + Σ_i x_i · [log P(w_i | spam) − log P(w_i | ham)]
  = b + Σ_i x_i · w_i                spam ⇔ Δ > 0
```

| Krok | Kto | Co |
| --- | --- | --- |
| 1 | Klient | TF-IDF wiadomości → wektor `x` (stały rozmiar k) → skalowanie do liczb całkowitych → `Enc(x_i)` kluczem publicznym |
| 2 | Serwer | `Enc(Δ) = Σ Enc(x_i) ⊗ w_i ⊕ b` — jeden homomorficzny iloczyn skalarny na wiadomość |
| 3 | Klient | odszyfrowuje `Δ` kluczem prywatnym, znak → spam/ham |
| 4 | — | ewaluacja: porównanie z wersją jawną |

Uwagi projektowe:
- Model (wagi `b`, `w`) jest jawny dla serwera — chronimy treść e-maili, nie model. To zgodne z Threat Modelem z karty projektu.
- Jeden wynik zamiast dwóch (osobno dla spam i ham): o połowę mniej pracy serwera i odszyfrowań, `Δ` jest od razu wynikiem do ROC-AUC, a klient poznaje mniej o modelu.
- Szyfrujemy **pełny wektor** (także zera). Wysyłanie tylko niezerowych indeksów zdradza, jakie słowa występują — to pokażemy jako eksperyment w części ataku.
- **Kodowanie stałoprzecinkowe (wariant podstawowy):** `phe` wysyła z każdym `EncryptedNumber` jawny `exponent`. Przy domyślnym kodowaniu floatów zależy on od wartości (zmierzone: `0.0 → -14`, `0.03 → -15`, `0.1–0.7 → -14`, `1.0 → -13`), więc serwer widzi część pozycji niezerowych (dla k=500 w ok. 1,6% wiadomości testowych). Dlatego `x_i` i `w_i` skalujemy do liczb całkowitych (np. ×10⁶, wyraz wolny `b` ×10¹²) — wykładnik jest wtedy zawsze 0. Alternatywa: `pub.encrypt(x, precision=1e-6)` (wykładnik zawsze -5), ale bez dokładnej równości wyników.
- Na liczbach całkowitych obliczenie homomorficzne jest **dokładne**: odszyfrowane `Δ` musi być równe (`==`) temu samemu obliczeniu na jawnych liczbach całkowitych. Błąd kwantyzacji względem zmiennoprzecinkowego NB mierzymy osobno.
- Na granicy z `phe` rzutujemy wartości na `int`/`float` Pythona — `np.float32` i `np.int64` dają `TypeError`.
- **Koszt obliczeń:** `phe` sam korzysta z `gmpy2`, jeśli jest zainstalowane. Zmierzone lokalnie dla klucza 2048 bit: 4,4 ms na szyfrowanie z `gmpy2`, 59,8 ms bez. Zbiór testowy przy k=1000 (~1035 × 1000 szyfrowań) to ok. 76 min na jednym rdzeniu i kilka minut z `multiprocessing`; bez `gmpy2` ok. 17 h. Selekcja cech chi² z k ≈ 500–1000 (przy k=200 F1 spamu spada do ok. 0.75, patrz sekcja 4), klucz 2048 bit do wyników końcowych i 1024 bit do iteracji. Wybór k uzasadniamy wykresem accuracy vs. k vs. czas.

## 3. Threat Model (sekcja raportu)

- **Profil przeciwnika:** Honest-but-Curious Cloud Server — pasywny, wykonuje protokół poprawnie, ale zapisuje i analizuje wszystko, co widzi.
- **Interfejs przeciwnika:** szyfrogramy cech (wraz z jawnym `exponent` z `phe` — stałym dzięki kodowaniu stałoprzecinkowemu), klucz publiczny, jawny model, metadane (liczba wiadomości, czasy). Brak klucza prywatnego i wyników po odszyfrowaniu. Serwer ma klucz publiczny, więc może zaszyfrować dowolną wiadomość — to model **CPA** (chosen-plaintext), nie COA. Wymagana własność to IND-CPA, którą Paillier spełnia przy założeniu DCRA.
- **Trening:** model trenuje dostawca usługi na własnych, jawnych danych (w eksperymencie: część treningowa SMS Spam Collection). Szyfrowanie chroni wyłącznie inferencję, czyli wiadomości klientów.
- **Kerckhoffs:** kod, model, słownik i parametry TF-IDF są jawne; tajny jest wyłącznie klucz prywatny klienta.
- **Poza zakresem:** aktywny serwer (fałszowanie wyników), side-channel czasowy po stronie klienta, ochrona modelu przed klientem.

## 4. Metodyka i grupa kontrolna

- Dane: SMS Spam Collection (UCI), 5574 wiadomości, w tym 403 dokładne duplikaty (5171 unikalnych; 309 ham, głównie szablonowe odpowiedzi, i 94 spam — spam jest duplikowany proporcjonalnie dwa razy częściej). **Deduplikacja przed podziałem** — bez niej ok. 12% wiadomości testowych (22% spamu testowego) występuje dosłownie w zbiorze treningowym, co zawyża wyniki. Podział **stratyfikowany 80/20** (~1035 wiadomości testowych), stały `random_state`, zbiór testowy nietykany do końca.
- Hiperparametry (`alpha` NB, k cech, parametry TF-IDF) dobierane **5-fold CV wyłącznie na zbiorze treningowym**. TF-IDF i chi² dopasowujemy **wewnątrz** foldów (`Pipeline` w `GridSearchCV`), inaczej CV przecieka.
- Zakres k: wstępny przebieg (bez strojenia, α=0.1, dane po deduplikacji) dał accuracy 0.927 / 0.975 / 0.982 dla k = 200 / 500 / 1000 — to wyniki na zbiorze testowym, więc do doboru k używamy tylko CV. Po strojeniu (P05, 5-fold CV na train) najlepsze wyniki dla k = 200 / 500 / 1000 to accuracy 0.949 / 0.976 / 0.980 i F1 spamu 0.754 / 0.897 / 0.915: k=200 przekracza 94% accuracy, ale F1 spamu jest dużo niższe, więc zostajemy przy k ≈ 500–1000.
- Grupy kontrolne:
  1. **Jawny NB** z identycznymi parametrami — bezpośredni punkt odniesienia dla wersji zaszyfrowanej.
  2. **Klasyfikator większościowy** (`DummyClassifier`, ~87%) — pokazuje, że 94% to realny wynik, a nie efekt niezbalansowania.
  3. (opcjonalnie) Regresja logistyczna — też liniowa, więc też działa na Paillierze; porównanie jakości.
- Benchmark wydajności: czas szyfrowania / obliczeń serwera / odszyfrowania na wiadomość i rozmiar szyfrogramu dla kluczy 1024/2048/3072 bit oraz różnych k.

## 5. Analiza bezpieczeństwa i próba ataku

1. **Atak na szyfrogramy (IND-CPA w praktyce)** — A: serwer sam szyfruje oznaczone wiadomości kluczem publicznym (CPA) i trenuje klasyfikator (np. RF) na cechach wyciągniętych z szyfrogramów (np. bajty / wartości mod), żeby odgadnąć spam/ham. Oczekiwany wynik ≈ klasyfikator większościowy → szyfrowanie nic nie zdradza.
2. **Kontrprzykład — wyciek przy wektorze rzadkim** — A: jeśli klient wysyła tylko niezerowe pozycje, serwer zna zbiór słów i klasyfikuje spam prawie tak dobrze jak model jawny. Uzasadnia szyfrowanie pełnego wektora.
3. **Determinizm** — B: demonstracja, że ta sama wiadomość szyfrowana dwukrotnie daje różne szyfrogramy (losowość `r`) — vs. hipotetyczny deterministyczny wariant podatny na analizę częstości.
4. **Wyciek przez jawny wykładnik `phe`** — B: przy domyślnym kodowaniu floatów serwer odczytuje z `exponent` część pozycji niezerowych (dla k=500 w ok. 1,6% wiadomości). Pokazujemy wyciek, a potem, że kodowanie stałoprzecinkowe z sekcji 2 go usuwa.

## 6. Struktura repozytorium

```
project/
├── README.md               # jak uruchomić, wyniki w skrócie
├── requirements.txt        # phe, gmpy2, scikit-learn, pandas, numpy, matplotlib, pytest (przypięte wersje)
├── data/                   # skrypt pobierający dataset (bez commitowania danych)
├── src/
│   ├── config.py           # wspólne stałe: SEED, SCALE, KEY_BITS, ścieżki (B)
│   ├── data.py             # wczytanie, deduplikacja, podział              (A)
│   ├── model.py            # TF-IDF, chi², trening NB, wagi log-odds (b, w) (A)
│   ├── fixed_point.py      # skalowanie do int, wynik jawny na int         (B)
│   ├── crypto_client.py    # klucze, szyfrowanie, odszyfrowanie            (B)
│   ├── crypto_server.py    # homomorficzny iloczyn skalarny                (B)
│   ├── pipeline.py         # przebieg end-to-end, zapis wyników do cache   (A)
│   ├── evaluate.py         # metryki, zgodność predykcji, wykresy          (A)
│   ├── benchmark.py        # czasy, rozmiary, k vs. klucz                  (B)
│   ├── run_all.py          # jedno polecenie odtwarzające wyniki           (A)
│   └── attacks/            # eksperymenty z sekcji 5, plik na atak         (A: 1–2, B: 3–4)
├── tests/                  # pytest: enc == plain (int) na małej próbce, klucze 512 bit dla szybkości
├── notebooks/              # eksploracja (EDA) i prezentacja wyników
├── results/                # wygenerowane wykresy/tabele (ew. tylko finalne)
└── report/                 # raport IEEE: main.tex (IEEEtran, conference), references.bib, figures/
```

Raport w strukturze IEEE: Abstract, Index Terms, I. Introduction, II. Background (Paillier, NB), III. Threat Model, IV. System Design (protokół), V. Experimental Setup (dane, podział, grupy kontrolne), VI. Results (dokładność, zgodność, wydajność), VII. Security Analysis (ataki), VIII. Conclusions, References. Limit objętości do potwierdzenia — domyślnie 6 stron w formacie dwukolumnowym. Klasa `IEEEtran.cls` nie jest zainstalowana lokalnie — zainstalować (`tlmgr install ieeetran`) albo dołączyć plik `.cls` do katalogu `report/`.

Interfejs między A i B (sygnatury funkcji i format plików wynikowych) jest ustalony w [`zadania.md`](zadania.md#kontrakt-interfejsu-ustalony-z-góry-żeby-nikt-nie-czekał-na-stuby). W skrócie: A dostarcza cechy `X` (dense, k kolumn), `y` i wagi log-odds `b`, `w` jako floaty; B dostarcza szyfrowanie, obliczenie serwera i odszyfrowanie. Skalowanie `x`, `b`, `w` do int jest po stronie B.

## 7. Podział pracy i lista zadań

Lista atomowych zadań z właścicielami, zależnościami i kryteriami ukończenia jest w [`zadania.md`](zadania.md). Zadania są podzielone na trzy rundy: Krzysztof (setup, kryptografia, szkielet raportu) → Piotr (dane, model, ewaluacja, ataki 1–2) → Krzysztof (atak 4, domknięcie raportu).

Kryterium „Contribution Matrix” (15%) jest liczone z logów Git, dlatego każdy commituje własną pracę z własnego konta, małymi commitami. Każde zadanie to osobna gałąź i PR z review drugiej osoby, więc obie osoby znają cały kod (obrona jest **indywidualna**, 15%).

Wyniki muszą być odtwarzalne jednym poleceniem (`python -m src.run_all`), ze stałymi seedami. Seedy ustalają podział, model i predykcje; szyfrogramy celowo nie są odtwarzalne (`phe` losuje `r` z `SystemRandom`). Komplet szyfrogramów przy k=1000 to ok. 530 MB (512 B każdy), dlatego w `results/cache/` zapisujemy tylko wyniki `Δ`.

## 8. Ryzyka

| Ryzyko | Mitygacja |
| --- | --- |
| Zbyt długie szyfrowanie całego zbioru testowego | `gmpy2` (~13× szybciej), k ≤ 1000, `multiprocessing`, iteracje na kluczu 1024 bit, finalny przebieg raz z cache wyników |
| Błędy precyzji zmieniają predykcje | Kodowanie stałoprzecinkowe: część kryptograficzna jest dokładna (test `==`); błąd kwantyzacji względem float NB mierzony osobno, w razie potrzeby większy mnożnik |
| Wyciek przez metadane szyfrogramów (`exponent` w `phe`) | Stały wykładnik dla wszystkich wartości; atak 4 z sekcji 5 to potwierdza |
| Accuracy < 94% przy małym k | k ≈ 500–1000 (k=200: accuracy 0.949, ale F1 0.754 w CV), krzywa accuracy vs. k, dobór k z CV; NB na SMS zwykle osiąga ~97% |
| Nierówny wkład w Git | Lista zadań z jednym właścicielem każdego zadania (sekcja 7), regularne małe commity obu osób |
| Krótki termin (2 tygodnie) | Zadania „(opc.)” (regresja logistyczna, klucz 3072 bit, atak 3) robimy tylko, jeśli zostanie czas; nie blokują zadań obowiązkowych |
