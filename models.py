import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
import json


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Hàm tạo dataset theo sliding window
def create_dataset(series, window_size=6):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)


# === HUẤN LUYỆN LSTM ===
def train_lstm_model_from_csv(file_path, window_size=60, epochs=20, batch_size=32):
    return train_model(file_path, window_size, epochs, batch_size, model_type="LSTM")

# === HUẤN LUYỆN GRU ===
def train_gru_model_from_csv(file_path, window_size=60, epochs=20, batch_size=32):
    return train_model(file_path, window_size, epochs, batch_size, model_type="GRU")



# ======================= CORE FUNCTION (dùng chung cho LSTM & GRU) ======================
def train_model(file_path, window_size, epochs, batch_size, model_type="LSTM"):
    selected_symbol = os.path.splitext(os.path.basename(file_path))[0]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    error_dir = os.path.join(base_dir, "errors")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    model_ext = "model" if model_type == "LSTM" else "gru_model"
    scaler_ext = "scaler" if model_type == "LSTM" else "gru_scaler"
    meta_ext = "metadata.json" if model_type == "LSTM" else "gru_metadata.json"
    error_ext = "errors.csv" if model_type == "LSTM" else "gru_errors.csv"

    model_path = os.path.join(model_dir, f"{selected_symbol}_{model_ext}.h5")
    scaler_path = os.path.join(model_dir, f"{selected_symbol}_{scaler_ext}.pkl")
    metadata_path = os.path.join(model_dir, f"{selected_symbol}_{meta_ext}")
    error_path = os.path.join(error_dir, f"{selected_symbol}_{error_ext}")

    # Load and sort data
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    data = df['Close'].values.reshape(-1, 1)

    # Split raw data
    split_idx = int(len(data) * 0.8)
    train_raw = data[:split_idx]
    test_raw = data[split_idx:]

    # Scale
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)
    joblib.dump(scaler, scaler_path)

    # Create datasets
    def create_dataset(series, window_size):
        X, y = [], []
        for i in range(len(series) - window_size):
            X.append(series[i:i + window_size])
            y.append(series[i + window_size])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_scaled, window_size)
    X_test, y_test = create_dataset(test_scaled, window_size)

    # Reshape for RNN
    X_train = X_train.reshape(-1, window_size, 1)
    X_test = X_test.reshape(-1, window_size, 1)

    # Define model
    RNNLayer = LSTM if model_type == "LSTM" else GRU
    model = Sequential()
    model.add(RNNLayer(units=64, return_sequences=True, input_shape=(window_size, 1)))
    model.add(Dropout(0.3))
    model.add(RNNLayer(units=32))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              callbacks=[early_stop, checkpoint], verbose=1)

    # Predict and inverse transform
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_train_pred_inv = scaler.inverse_transform(y_train_pred)
    y_test_pred_inv = scaler.inverse_transform(y_test_pred)

    # Dates
    dates = df['Date'].reset_index(drop=True)
    scaled_full = scaler.transform(data)
    X_all, _ = create_dataset(scaled_full, window_size)
    total_dates = dates[window_size:]
    train_dates = total_dates[:len(y_train)]
    test_dates = total_dates[len(y_train):len(y_train) + len(y_test)]

    # Metrics
    mae_train = mean_absolute_error(y_train_inv, y_train_pred_inv)
    rmse_train = np.sqrt(mean_squared_error(y_train_inv, y_train_pred_inv))
    mape_train = np.mean(np.abs((y_train_inv - y_train_pred_inv) / np.clip(y_train_inv, 1e-6, None))) * 100

    mae_test = mean_absolute_error(y_test_inv, y_test_pred_inv)
    rmse_test = np.sqrt(mean_squared_error(y_test_inv, y_test_pred_inv))
    mape_test = np.mean(np.abs((y_test_inv - y_test_pred_inv) / np.clip(y_test_inv, 1e-6, None))) * 100

    pd.DataFrame([{
        'Model': selected_symbol,
        'MAE_Train': round(mae_train, 2),
        'RMSE_Train': round(rmse_train, 2),
        'MAPE_Train (%)': round(mape_train, 2),
        'MAE_Test': round(mae_test, 2),
        'RMSE_Test': round(rmse_test, 2),
        'MAPE_Test (%)': round(mape_test, 2)
    }]).to_csv(error_path, index=False)

    metadata = {
        "window_size": window_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "model_type": model_type
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    return {
        'model': model,
        'scaler': scaler,
        'window_size': window_size,
        'df': df,
        'train_dates': train_dates.reset_index(drop=True),
        'test_dates': test_dates.reset_index(drop=True),
        'y_train_real': y_train_inv,
        'y_train_pred': y_train_pred_inv,
        'y_test_real': y_test_inv,
        'y_test_pred': y_test_pred_inv,
        'mae_train': mae_train,
        'rmse_train': rmse_train,
        'mape_train': mape_train,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'mape_test': mape_test
    }


# ======================= DỰ ĐOÁN =======================

def predict_future(df, scaler, window_size, steps=10, selected_symbol=None):
    return _predict(df, scaler, window_size, steps, selected_symbol, model_type="LSTM")

def predict_future_gru(df, scaler, window_size, steps=10, selected_symbol=None):
    return _predict(df, scaler, window_size, steps, selected_symbol, model_type="GRU")
    

def _predict(df, scaler, window_size, steps, selected_symbol, model_type="LSTM"):
    if selected_symbol is None:
        raise ValueError("Thiếu mã chứng khoán")

    model_ext = "model" if model_type == "LSTM" else "gru_model"
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_path = os.path.join(model_dir, f"{selected_symbol}_{model_ext}.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy mô hình {model_path}")

    model = load_model(model_path)

    data = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    if len(scaled_data) < window_size:
        raise ValueError("Không đủ dữ liệu")

    last_scaled = scaled_data[-window_size:]
    input_seq = last_scaled.reshape(1, window_size, 1)

    predictions = []
    for _ in range(steps):
        pred = model.predict(input_seq, verbose=0)
        pred_price = scaler.inverse_transform(pred)[0][0]
        predictions.append(pred_price)
        input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    return predictions