# aws s3 ls s3://proyecto3er-parcial-all-time-stock-price-data
# proyecto3er-parcial-all-time-stock-price-data.csv
# /home/starryboy/Starcode/Python/LSTM/codes

import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


s3 = boto3.client('s3')
bucket_name = "proyecto3er-parcial-all-time-stock-price-data"
object_key = "NFLX.csv"
local_filename = "/home/starryboy/Starcode/Python/LSTM/codes/NFLX.csv"

print("Downloading data...")
s3.download_file(bucket_name, object_key, local_filename)
print("Downloaded!")

### ACCESSO A LOS DATOS
# Leemos el csv y lo limpiamos
nflx = pd.read_csv(local_filename)
nflx = nflx.dropna(axis=0, how='any')
# Convertir la columna Date a tipo datetime
nflx['Date'] = pd.to_datetime(nflx['Date'])
# Ordernar los datos por fecha
nflx = nflx.sort_values('Date')
# Seleccionaremos la columna Close para predecir
data = nflx['Close'].values.reshape(-1, 1)
# Normalizamos los datos
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

### DESARROLLO DEL MODELO
sequence_length = 60  # 60 días para predecir el siguiente día
# Seccionamiento de los datos
X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i+sequence_length])
    y.append(data_scaled[i+sequence_length])
X = np.array(X)
y = np.array(y)
# Dividir en conjuntos de entrenamiento y prueba
split = int(0.9 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
# Definimos el modelo
model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(sequence_length, 1)))
model.add(Dense(1))
# Compilamos el modelo
model.compile(optimizer='adam', loss='mse')
# Entrenamiento del modelo
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)


### EVALUACION DEL MODELO
# Desarrollar predicciones
predictions = model.predict(X_test)
# Desnormalizar las predcciones
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)
# Evaluar el modelo
plt.figure(figsize=(14,5))
plt.plot(nflx['Date'][split+sequence_length:], y_test, color='blue', label='Real')
plt.plot(nflx['Date'][split+sequence_length:], predictions, color='red', label='Predicción')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre')
plt.title('Predicción del Precio de las Acciones de Netflix')
plt.legend()
plt.show()


