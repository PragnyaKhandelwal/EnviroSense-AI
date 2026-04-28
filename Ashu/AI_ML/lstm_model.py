import os
import sys
import numpy as np
import pandas as pd
import hashlib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from dotenv import load_dotenv
from sqlalchemy import text

# Load env
load_dotenv("/home/ashu/.env")

sys.path.insert(0, "/home/shared/envirosense")
from db_utils import get_engine

engine = get_engine()

# Load data
df = pd.read_sql("SELECT * FROM ashu_features ORDER BY time", engine)
df = df[['time','pm1_avg','pm25_avg','pm10_avg']].dropna()

def create_sequences(data, seq_len=5):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][1]) 
    return np.array(X), np.array(y)

values = df[['pm1_avg','pm25_avg','pm10_avg']].values

SEQ_LEN = 5
X, y = create_sequences(values, SEQ_LEN)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(50, input_shape=(SEQ_LEN, 3)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)


preds = model.predict(X_test, verbose=0)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))


model_dir = "Ashu/AI_ML"
os.makedirs(model_dir, exist_ok=True)
model.save(f"{model_dir}/pm25_lstm_model.h5")


features = ['pm1_avg','pm25_avg','pm10_avg']
feature_hash = hashlib.md5(",".join(features).encode()).hexdigest()

train_start = df['time'].min()
train_end = df['time'].max()

with engine.connect() as conn:

    result = conn.execute(text("""
    SELECT COALESCE(MAX(version), 0)
    FROM ashu_model_registry
    WHERE model_name = 'pm25_lstm_model'
    """))

    version = result.scalar() + 1

    conn.execute(text("""
    UPDATE ashu_model_registry
    SET selected_flag = FALSE
    WHERE model_name = 'pm25_lstm_model'
    """))

    conn.execute(text("""
    INSERT INTO ashu_model_registry (
        model_name, version, algorithm,
        feature_set_hash,
        train_start, train_end,
        val_mae, val_rmse,
        selected_flag
    )
    VALUES (
        :name, :version, :algo,
        :feature_hash,
        :train_start, :train_end,
        :mae, :rmse,
        FALSE
    )
    """), {
        "name": "pm25_lstm_model",
        "version": version,
        "algo": "LSTM",
        "feature_hash": feature_hash,
        "train_start": train_start,
        "train_end": train_end,
        "mae": float(mae),
        "rmse": float(rmse)
    })

    conn.commit()