import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Preuzimanje podataka od 2000 godine do 2025
ticker_symbol = 'AAPL'
start_date = '2000-01-01'
end_date = '2025-01-01'

df = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

if df.empty:
    print("Preuzimanje podataka nije uspelo.")
    sys.exit()
else:
    print(f"Podaci uspešno preuzeti. Broj redova: {len(df)}")


# 1. Dan u nedelji
df['Day_Of_Week'] = df.index.dayofweek
day_names = {0:'Ponedeljak', 1:'Utorak', 2:'Sreda', 3:'Četvrtak', 4:'Petak'}
df['Day_Name'] = df['Day_Of_Week'].map(day_names)

# 2. Dnevni trend
df['Daily_Trend'] = np.where(df['Close'].diff() > 0, 'Rast', 'Pad')

# Pokretni proseci
df['SMA_20'] = ta.sma(df['Close'], length=20)
df['EMA_20'] = ta.ema(df['Close'], length=20)

# RSI (Momentum)
df['RSI_14'] = ta.rsi(df['Close'], length=14)

# MACD (Trend i momentum)
macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
df = pd.concat([df, macd], axis=1)

df = df.drop('Day_Of_Week', axis=1)

# Uklanjamo redove sa NaN vrednostima
df.dropna(inplace=True)

prediction_days = 60 

# 1. Enkodiranje kategorijskih kolona (sada možemo vratiti drop_first=True)
df_encoded = pd.get_dummies(df, columns=['Day_Name', 'Daily_Trend'], drop_first=True)

# 2. Programski sastavljamo listu obeležja
features = [
    'Close', 
    'Volume', 
    'MACD_12_26_9', 
    'MACDh_12_26_9', 
    'MACDs_12_26_9'
]

encoded_cols = [col for col in df_encoded.columns if 'Day_Name_' in col or 'Daily_Trend_' in col]

features.extend(encoded_cols)

data = df_encoded[features]

dataset = data.values
training_data_len = int(np.ceil(len(dataset) * .8))
train_data = dataset[0:training_data_len, :]
test_data = dataset[training_data_len:, :]

# --- KORAK 2: Skaliranje podataka ---

# Kreiramo skaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Skaler će naučiti min/max vrednosti za svaku od vaših 10+ kolona.
scaled_train_data = scaler.fit_transform(train_data)

# Transformišemo test podatke koristeći skaler fitovan na trening podacima.
scaled_test_data = scaler.transform(test_data)

# --- KORAK 3: Kreiranje sekvenci za TRENING skup ---

x_train = []
y_train = []

# Broj dana na osnovu kojih predviđamo ostaje isti
prediction_days = 60 

for i in range(prediction_days, len(scaled_train_data)):
    # x_train dobija 60 prethodnih redova sa svim obeležjima
    x_train.append(scaled_train_data[i-prediction_days:i, :]) # ':' uzima sve kolone
    
    # y_train dobija 61. 'Close' cenu. Ona je i dalje naša prva kolona (indeks 0).
    y_train.append(scaled_train_data[i, 0])

# Konvertovanje u numpy nizove
x_train, y_train = np.array(x_train), np.array(y_train)

# --- KORAK 4: Kreiranje sekvenci za TEST skup ---

# Kreiramo testni skup. Prvo uzimamo stvarne vrednosti za y_test.
y_test = test_data[prediction_days:, :] # Počinjemo od 61. dana test skupa

# Sada kreiramo x_test. Za svaku vrednost u y_test, treba nam 60 prethodnih dana.
x_test = []

for i in range(len(test_data) - prediction_days):
    # Izdvajamo sekvencu od 60 dana
    x_test.append(scaled_test_data[i:i+prediction_days, :])

# Konvertovanje u numpy niz
x_test = np.array(x_test)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

# --- KORAK 5: Definisanje arhitekture LSTM modela (prilagođeno za više obeležja) ---

# Određujemo broj obeležja (kolona) dinamički iz oblika x_train niza.
n_features = x_train.shape[2]

# Inicijalizacija Sequential modela
model = Sequential()

# Dodajemo ulazni sloj koji sada prihvata sekvence sa 'n_features' obeležja
model.add(Input(shape=(x_train.shape[1], n_features)))

# Ostatak arhitekture ostaje isti, sa tvojim optimizovanim hiperparametrima
model.add(LSTM(units=384, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(units=384, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(units=1))

# Definisemo optimizer koji je najbolji ispao Adam sa learning rate-om 0.001
optimised_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# --- KORAK 6: Kompajliranje modela ---
model.compile(optimizer=optimised_optimizer, loss='mean_squared_error')

# Prikaz arhitekture modela i broja parametara
model.summary()

import time
from tensorflow.keras.callbacks import EarlyStopping

# --- KORAK 7: Treniranje modela sa VALIDACIJOM i RANIM ZAUSTAVLJANJEM ---

# Definišemo pravila za rano zaustavljanje kako ne zelimo overfitovan model
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience= 10,
    restore_best_weights=True
)

# Pokrećemo treniranje sa callback-om
start_time = time.time()
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.1,
    callbacks=[early_stopping_callback]
)

end_time = time.time()
training_duration = end_time - start_time
minutes = int(training_duration // 60)
seconds = int(training_duration % 60)

print(f"\nTrening je završen (ili rano zaustavljen) za {minutes} minuta i {seconds} sekundi.")

# --- KORAK 8: Pravljenje predviđanja na test podacima ---

predictions_scaled = model.predict(x_test)

# --- KORAK 9: Vraćanje predviđanja na originalnu skalu (Ažurirano) ---

# 1. Određujemo broj obeležja na kojima je skaler učen
n_features = dataset.shape[1]

# 2. Kreiramo privremeni niz sa ispravnim brojem kolona
# Svi unosi će biti 0, osim prve kolone koju ćemo popuniti
temp_array = np.zeros((len(predictions_scaled), n_features))

# 3. Stavljamo naša predviđanja u prvu kolonu (indeks 0)
# Ovo je ispravno jer je 'Close' bila prva kolona u našem 'features' nizu
temp_array[:, 0] = predictions_scaled.flatten()

# 4. Sada radimo inverznu transformaciju na nizu ispravnih dimenzija
predictions = scaler.inverse_transform(temp_array)

# 5. Na kraju, uzimamo samo prvu kolonu koja sadrži naše de-skalirane 'Close' predikcije
predictions_unscaled = predictions[:, 0]

from sklearn.metrics import mean_squared_error

# --- KORAK 10: Evaluacija modela - računanje RMSE ---

# Preskačemo prvih 'prediction_days' (60) vrednosti jer za njih nemamo predikciju.
y_test_aligned = dataset[training_data_len + prediction_days:, 0]

print("Provera dimenzija nakon poravnanja:")
print(f"Dimenzije y_test_aligned (stvarne vrednosti): {y_test_aligned.shape}")
print(f"Dimenzije predictions_unscaled (predviđene vrednosti): {predictions_unscaled.shape}")

rmse = np.sqrt(mean_squared_error(y_test_aligned, predictions_unscaled))

print(f"\nEvaluacija modela na test podacima:")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")


# --- KORAK 11: Vizuelizacija rezultata ---
data_for_plot = df[['Close']]
train = data_for_plot[:training_data_len]

valid_start_index = training_data_len + prediction_days
valid = data_for_plot[valid_start_index:].copy()

valid['Predictions'] = predictions_unscaled
valid.rename(columns={'Close': 'Actual'}, inplace=True)

# Izračunavanje procentualne razlike za svaki dan
# Formula: ((Predviđeno - Stvarno) / Stvarno) * 100
actual_values = valid['Actual'].values.flatten()
prediction_values = valid['Predictions'].values.flatten()

valid['Percentage_Difference'] = ((prediction_values - actual_values) / actual_values) * 100

plt.figure(figsize=(16,8))
plt.title(f'Predviđanje cene akcije za {ticker_symbol}', fontsize=20)
plt.xlabel('Datum', fontsize=18)
plt.ylabel('Cena zatvaranja (USD)', fontsize=18)

plt.plot(train['Close'], label='Trening podaci')
plt.plot(valid['Actual'], color='red', label='Stvarna cena')
plt.plot(valid['Predictions'], color='green', label='Predviđena cena')

plt.legend(loc='lower right', fontsize='large')
plt.show()

# Prikaz poslednjih nekoliko predviđanja
print("\nPoslednjih nekoliko stvarnih i predviđenih cena:")
print(
    valid.tail().to_string(
        formatters={'Percentage_Difference': '{:.2f}%'.format}
    )
)