# Predviđanje cene akcija pomoću rekurentnih neuronskih mreža

## O projektu

Ovaj projekat predstavlja implementaciju modela za predviđanje cene akcija korišćenjem rekurentnih neuronskih mreža (RNN), konkretno LSTM (Long Short-Term Memory) arhitekture. Cilj modela je da na osnovu istorijskih podataka o ceni i obimu trgovine za prethodnih 60 dana, predvidi cenu zatvaranja (Close price) za sledeći dan.

Projekat je realizovan kao deo predmeta Masinskog Ucenja. Kompletan opis problema, metodologije i analize rezultata nalazi se u pratećem izveštaju (`izvestaj.pdf`).

**Autor:** Stefan Djurica, SV35/2021

---

## Tehnologije i biblioteke

* **Python:** 3.9+
* **Biblioteke:** `tensorflow==2.16.1`, `keras-tuner==1.4.7`, `scikit-learn==1.3.2`, `yfinance==0.2.37`, `pandas==2.1.4`, `numpy==1.26.4`, `matplotlib==3.8.2`, `seaborn==0.13.2`

---

## Postavljanje okruženja i pokretanje koda

Pratite sledeće korake kako biste uspešno pokrenuli projekat na vašem računaru.

### 1. Preduslovi

* Instaliran **Python 3.9** ili novija verzija.
* Instaliran **pip** (obično dolazi uz Python).
* Instaliran **Git** za kloniranje repozitorijuma.

### 2. Koraci za instalaciju

**a) Klonirajte repozitorijum:**
Otvorite terminal ili Command Prompt i izvršite sledeću komandu:
```bash
git clone [https://github.com/StefanDjuricaSV35-2021/AAPL_Stock_Prediction.git]
cd [AAPL_Stock_Prediction]
```

**b) Kreirajte i aktivirajte virtualno okruženje (preporučeno):**
Kreiranje virtualnog okruženja osigurava da se zavisnosti projekta neće mešati sa drugim Python projektima na vašem sistemu.

* **Na Windows-u:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

* **Na macOS / Linux-u:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
Nakon aktivacije, videćete `(venv)` na početku linije u vašem terminalu.

**c) Instalirajte potrebne biblioteke:**
Sve potrebne biblioteke sa odgovarajućim verzijama su definisane u `requirements.txt` fajlu. Instalirajte ih jednom komandom:
```bash
pip install -r requirements.txt
```
Ako instalacija ne radi iz prve probajte ponovo nakon upgrade-ovanje pip-a

### 3. Pokretanje koda

Kod se nalazi u Jupyter Notebook fajlu (`predvidjanje_akcija.ipynb`).

1.  Uverite se da je vaše virtualno okruženje (`venv`) aktivirano.
2.  Pokrenite python file predict_stocks.py pomocu komande:
```bash
python predict_stocks.py
```

Ako ne bude radilo moguce da nije updejtovan yfinance module koji se azurira pomocu sledece komande:
```bash
pip install --upgrade yfinance
```

---

## Struktura repozitorijuma

```
.
├── projekat.ipynb              # Glavni Jupyter Notebook sa izvornim kodom
├── predict_stocks.py           # Python skripta za lakse pokretanje koda
├── izvestaj.pdf                # Detaljan izveštaj o projektu
├── requirements.txt            # Lista potrebnih Python biblioteka
└── README.md                   # Ovaj fajl
```
