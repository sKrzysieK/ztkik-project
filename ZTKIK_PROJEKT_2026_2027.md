*ZTKIK, projekt, Praktyczne Projekty Badawczo-Programistyczne*

| **Nazwa** **przedmiotu:** | Zaawansowane Techniki Kryptografii i Kryptoanalizy |
| --- | --- |
| **Poziom** **i** **kierunek** **studiów:** | Studia II stopnia (magisterskie) | Kierunek: Informatyka | Specjalność: Cyberbezpieczeństwo |
| **Forma zajęć:** | Projekt |
| **Forma** **realizacji** **projektów:** | Praca w zespołach 2–4 osobowych (Wybór 1 projektu z 12 propozycji) |
| **Wymagania** **sprzętowe:** | 100% Wykonalność Programowa (PC/Laptop + Python/Colab + Internet, BEZ fizycznego sprzętu) |
# 1.
## Słownik Akademicki Skrótów i Pojęć (Acro-Dictionary)


| **Skrót** **/** **Pojęcie** |  |
| --- | --- |
| **AI / ML** | Artificial Intelligence / Machine Learning – Sztuczna inteligencja i uczenie maszynowe. |
| **RNN / LSTM** | Recurrent Neural Network / Long Short-Term Memory – Rekurencyjne sieci neuronowe dedykowane do przetwarzania sekwencji i ciągów znaków (np. haseł). |
| **RF / DT** | Random Forest / Decision Tree – Drzewa decyzyjne i lasy losowe do szybkiej i czytelnej klasyfikacji cech. |
| **k-NN / SVM** | k-Nearest Neighbors / Support Vector Machine – Klasyczne algorytmy klasyfikacji i analizy skupień. |
| **CNN** | Convolutional Neural Network – Konwolucyjne sieci neuronowe stosowane w analizie obrazów i stegananalizie LSB. |
| **GA** | Genetic Algorithm – Algorytmy genetyczne służące do ewolucyjnego przeszukiwania przestrzeni kluczy/podstawień. |
| **NLP / TF-IDF** | Natural Language Processing / Term Frequency-Inverse Document Frequency – Przetwarzanie języka naturalnego i ekstrakcja cech tekstowych. |
| **HTTPS / TLS** | Hypertext Transfer Protocol Secure / Transport Layer Security – Szyfrowane protokoły komunikacji sieciowej. |
| **LSB** | Least Significant Bit – Najmniej znaczący bit w pikselu obrazu, używany w steganografii do ukrywania danych. |
| **PRNG / LCG** | Pseudorandom Number Generator / Linear Congruential Generator – Generatory liczb pseudolosowych. |
| **PHE / Paillier** | Partially Homomorphic Encryption / Paillier Cryptosystem – Szyfrowanie częściowo homomorficzne pozwalające na dodawanie zaszyfrowanych wartości. |
| **FL /** **FedAvg** | Federated Learning / Federated Averaging – Uczenie federacyjne pozwalające na rozproszony trening bez dzielenia się danymi. |
| **AE / MSE** | Autoencoder / Mean Squared Error – Autoenkodery głębokie do bezsuperwizyjnej detekcji anomalii (błąd rekonstrukcji MSE). |
| **COA / KPA** | Ciphertext-Only Attack / Known-Plaintext Attack – Model dostępu przeciwnika (tylko szyfrogram / znany tekst jawny). |
# 2. Rygor Metodologiczny: Formalny Model Zagrożeń (Threat Model)
Każdy projekt realizowany przez zespół studencki musi definiować formalny Model Zagrożeń (Threat Model). Podejście to gwarantuje utrzymanie akademickiego rygoru naukowego, zmuszając studentów do precyzyjnego określenia granic bezpieczeństwa badanego systemu:

**Profil Przeciwnika (Adversary** **Profile):** Określenie czy przeciwnik jest bierny (Passive Eavesdropper – obserwujący pakiet lub szyfrogram) czy aktywny (Active Attacker – modyfikujący wektory cech lub wysyłający zapytania).

**Interfejs Dostępny dla Przeciwnika:** Zdefiniowanie poziomu dostępu – COA (tylko szyfrogram), KPA (sparowany tekst jawny i szyfrogram), dostęp do logów HTTPS czy próbek malware.

**Zasada** **Kerckhoffsa:** Założenie pełnej jawności stosowanych modeli ML, architektury sieci oraz kodu źródłowego. Bezpieczeństwo zależy wyłącznie od klucza lub tajności haseł.

**Ilorazowe Kryterium Sukcesu:** Sformułowanie jednoznacznego wskaźnika sukcesu ataku (np. Accuracy > 85%, F1-Score > 0.90, ROC-AUC > 0.92, MSE > 0.35).
# 3. System Oceniania Projektów Zespołowych
W celu zapewnienia pełnej sprawiedliwości i ujednolicenia oceny końcowej, niezależnie od wybranego tematu projektu, stosuje się dwuosiowy system weryfikacji efektów uczenia się (70% ocena zespołowa, 30% ocena indywidualna):

| **Kryterium** **Oceny** | **Waga** | **Opis Zakresu Weryfikacji** |
| --- | --- | --- |
| **Poprawność Teoretyczna & Threat Model** | **15%** | Poprawność zdefiniowania problemu kryptoanalitycznego i profilu przeciwnika. |
| **Metodyka** **Badawcza** **& Grupa Kontrolna** | **15%** | Prawidłowy dobór grupy kontrolnej, hiperparametrów i podziału danych. |
| **Jakość** **Kodu Python & Git** | **15%** | Czytelność kodu, modularność, obecność pliku requirements.txt oraz historia commitów w Git. |
| **Analiza** **Bezpieczeństwa** **&** **Próba** **Ataku** | **12.5%** | Przeprowadzenie eksperymentalnej próby przełamania/klasyfikacji i wyliczenie metryk. |
| **Raport** **Zespołowy** **(IEEE/ACM)** | **12.5%** | Struktura raportu naukowego z opisem eksperymentów, wykresami i wnioskami. |
| **Indywidualna** **Obrona** **Zespołowa** | **15%** | Odpowiedzi na pytania problemowe podczas końcowej prezentacji projektu. |
| **Indydidualna** **Macierz** **Wkładu** **(Contribution Matrix)** | **15%** | Weryfikacja indywidualnego wkładu każdego członka zespołu na podstawie logów Git. |
# 4. Tematyka Projektu

## Projekt: Bezpieczna Detekcja Spamu na Danych Zaszyfrowanych Szyfrem Homomorficznym Paillier
*Nazwa* *angielska: Project 9: Privacy-Preserving Spam Detection on Data Encrypted with Paillier Cryptosystem* **Słownik skrótów: PHE (Partially Homomorphic Encryption), Paillier Cryptosystem, Naive Bayes.**

| **Opis i Cel Badawczy:** | Wdrożenie klasyfikatora Naive Bayes na wektorach cech zaszyfrowanych kryptosystemem Paillier (phe w Pythonie), prezentujące podstawy obliczeń na danych zaszyfrowanych. |
| --- | --- |
| **Formalny** **Threat Model:** | Honest-but-Curious Cloud Server. Serwer wykonuje iloczyn i sumowanie cech bez możliwości odszyfrowania treści e-maila. |
| **Scenariusz** **/** **Kontekst:** | Prywatne filtry antyspamowe w chmurze chroniące prywatność korespondencji klientów. |
| **Etapy** **Realizacji** **(Phases):** | 1. Szyfrowanie cech TF-IDF szyfrem Paillier w Pythonie; 2. Wykonanie iloczynów homomorficznych; 3. Odszyfrowanie wyniku przez klienta; 4. Ocena dokładności. |
| **Próba Ataku / Kryterium:** | Osiągnięcie idetycznej dokładności klasyfikacji spamu na danych zaszyfrowanych jak na danych czystych (Accuracy > 94%). |
| **Baza** **Danych** **& Link:** | SMS Spam Collection Dataset – https://archive.ics.uci.edu/dataset/228/sms+spam+collection |
**Literatura Naukowa (IEEE/APA):**

Paillier, P. (1999). Public-key cryptosystems based on composite degree residuosity classes. EUROCRYPT '99, Springer.

Python phe Library Documentation (2024). Python Paillier Homomorphic Encryption. https://github.com/data61/python-paillier

UCI Machine Learning Repository (2024). SMS Spam Collection Dataset.

