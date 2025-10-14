import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


ticker_symbol = 'AAPL'
start_date = '2000-01-01'
end_date = '2025-01-01'

df = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=True)

# --- KORAK 1: Izdvajanje CILJNIH KOLONA i podela na skupove ---
prediction_days = 60 

features = ['Close', 'Volume']
data = df[features]

dataset = data.values

training_data_len = int(np.ceil(len(dataset) * .8))

train_data = dataset[0:training_data_len, :]
test_data = dataset[training_data_len: , :]

# --- KORAK 2: Skaliranje podataka ---
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

# --- KORAK 3: Kreiranje sekvenci za TRENING skup ---
x_train = []
y_train = []

for i in range(prediction_days, len(scaled_train_data)):
    x_train.append(scaled_train_data[i-prediction_days:i, :])
    y_train.append(scaled_train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# --- KORAK 4: Kreiranje sekvenci za TEST skup ---
y_test = test_data[prediction_days:, :] # Počinjemo od 61. dana test skupa
x_test = []

for i in range(len(test_data) - prediction_days):
    x_test.append(scaled_test_data[i:i+prediction_days, :])

x_test = np.array(x_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout 

# --- KORAK 5: Definisanje arhitekture LSTM modela ---

model = Sequential()
model.add(Input(shape=(x_train.shape[1], 2)))
model.add(LSTM(units=224, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(units=224, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(units=1))

optimised_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimised_optimizer, loss='mean_squared_error')

model.summary()

import time
from tensorflow.keras.callbacks import EarlyStopping

# --- KORAK 7: Treniranje modela sa VALIDACIJOM i RANIM ZAUSTAVLJANJEM ---
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience= 6,
    restore_best_weights=True
)

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

# --- KORAK 9: Vraćanje predviđanja na originalnu skalu ---

temp_array = np.zeros((len(predictions_scaled), 2))
temp_array[:, 0] = predictions_scaled.flatten()

predictions = scaler.inverse_transform(temp_array)
predictions_unscaled = predictions[:, 0]

from sklearn.metrics import mean_squared_error

# --- KORAK 10: Evaluacija modela - računanje RMSE ---

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

print("\nPoslednjih nekoliko stvarnih i predviđenih cena:")
print(
    valid.tail().to_string(
        formatters={'Percentage_Difference': '{:.2f}%'.format}
    )
)