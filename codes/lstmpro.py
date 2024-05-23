import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


def download_file_from_s3(bucket_name, object_key, local_filename):
    """Descargar archivo desde S3."""
    s3 = boto3.client('s3')
    print(f"Downloading {object_key}...")
    s3.download_file(bucket_name, object_key, local_filename)
    print(f"Downloaded {object_key}!")


def load_and_preprocess_data(filename):
    """Leer y limpiar los datos."""
    data = pd.read_csv(filename)
    data = data.dropna(axis=0, how='any')
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    return data


def create_sequences(data, sequence_length):
    """Crear secuencias de datos."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


def build_and_train_model(X_train, y_train, sequence_length):
    """Construir y entrenar el modelo LSTM."""
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    return model


def evaluate_and_plot_model(model, X_test, y_test, dates, filename, output_dir):
    """Evaluar el modelo y generar gráficas."""
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    plt.figure(figsize=(14, 5))
    plt.plot(dates, y_test, color='blue', label='Real')
    plt.plot(dates, predictions, color='red', label='Predicción')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de Cierre')
    plt.title(f'Predicción del Precio de las Acciones - {filename}')
    plt.legend()

    # Guardar la gráfica como imagen
    image_filename = os.path.join(output_dir, f'{filename.split(".")[0]}_prediction.png')
    plt.savefig(image_filename)
    plt.close()

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f'{filename} - MSE: {mse}')
    print(f'{filename} - MAE: {mae}')


# Parámetros
bucket_name = "proyecto3er-parcial-all-time-stock-price-data"
files = ["NFLX.csv", "AAPL.csv", "GOOGL.csv", "AMZN.csv"]
local_path = "/home/starryboy/Starcode/Python/LSTM/codes/"
output_dir = "/home/starryboy/Starcode/Python/LSTM/codes/plots/"
sequence_length = 60

# Crear directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

for file in files:
    # Descargar archivo
    object_key = file
    local_filename = local_path + file
    download_file_from_s3(bucket_name, object_key, local_filename)

    # Cargar y preprocesar datos
    data = load_and_preprocess_data(local_filename)
    close_prices = data['Close'].values.reshape(-1, 1)

    # Normalizar datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(close_prices)

    # Crear secuencias
    X, y = create_sequences(data_scaled, sequence_length)

    # Dividir en conjuntos de entrenamiento y prueba
    split = int(0.9 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates = data['Date'][split + sequence_length:].values

    # Construir y entrenar el modelo
    model = build_and_train_model(X_train, y_train, sequence_length)

    # Evaluar y graficar el modelo
    evaluate_and_plot_model(model, X_test, y_test, dates, file, output_dir)
