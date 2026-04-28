import pandas as pd
import joblib
import sys
import time
from sqlalchemy import text
from datetime import datetime

sys.path.insert(0, "/home/shared/envirosense")
from db_utils import get_engine

engine = get_engine()

# Load models
model_pm1 = joblib.load("Ashu/AI_ML/pm1_model.pkl")
model_pm25 = joblib.load("Ashu/AI_ML/pm25_model.pkl")
model_pm10 = joblib.load("Ashu/AI_ML/pm10_model.pkl")

# Load feature columns
feature_cols = joblib.load("Ashu/AI_ML/feature_columns.pkl")

print("Prediction Service Started...")

# Track last processed timestamp
last_processed_time = None

while True:
    try:
        query = """
        SELECT * FROM ashu_features
        ORDER BY time DESC
        LIMIT 2
        """
        df = pd.read_sql(query, engine)

        if len(df) < 2:
            print("Not enough data...")
            time.sleep(60)
            continue

        df = df.sort_values("time").reset_index(drop=True)

        current_time = df.iloc[-1]['time']

        # Skip if no new data
        if last_processed_time == current_time:
            print("No new data, skipping...")
            time.sleep(60)
            continue

        last_processed_time = current_time

        # Create lag features
        df['pm1_lag1'] = df['pm1_avg'].shift(1)
        df['pm25_lag1'] = df['pm25_avg'].shift(1)
        df['pm10_lag1'] = df['pm10_avg'].shift(1)

        df = df.dropna()

        row = df.iloc[[-1]]
        X_new = row[feature_cols]

        # Predictions
        pred_pm1 = model_pm1.predict(X_new)[0]
        pred_pm25 = model_pm25.predict(X_new)[0]
        pred_pm10 = model_pm10.predict(X_new)[0]

        actual_pm1 = row['pm1_avg'].values[0]
        actual_pm25 = row['pm25_avg'].values[0]
        actual_pm10 = row['pm10_avg'].values[0]

        res_pm1 = actual_pm1 - pred_pm1
        res_pm25 = actual_pm25 - pred_pm25
        res_pm10 = actual_pm10 - pred_pm10

        current_time = row['time'].iloc[0].to_pydatetime()

        print(f"\n{datetime.now()}")
        print(f"PM1  → Pred: {pred_pm1:.2f}, Actual: {actual_pm1:.2f}")
        print(f"PM2.5→ Pred: {pred_pm25:.2f}, Actual: {actual_pm25:.2f}")
        print(f"PM10 → Pred: {pred_pm10:.2f}, Actual: {actual_pm10:.2f}")

        #Duplicate protection (DB-level)
        check_query = """
        SELECT COUNT(*) FROM ashu_model_predictions
        WHERE time = :time
        """

        with engine.connect() as conn:

            result = conn.execute(text(check_query), {"time": current_time})
            exists = result.scalar()

            if exists > 0:
                print("Already predicted for this timestamp, skipping insert...")
                continue

            conn.execute(text("""
            INSERT INTO ashu_model_predictions
            (model_id, device_id, time, horizon, y_true, y_pred, residual, confidence)
            VALUES
            (:model_id, :device_id, :time, :horizon, :y_true, :y_pred, :residual, :confidence)
            """), [
                {
                    "model_id": "pm1_model",
                    "device_id": "device_1",
                    "time": current_time,
                    "horizon": 1,
                    "y_true": float(actual_pm1),
                    "y_pred": float(pred_pm1),
                    "residual": float(res_pm1),
                    "confidence": 0.9
                },
                {
                    "model_id": "pm25_model",
                    "device_id": "device_1",
                    "time": current_time,
                    "horizon": 1,
                    "y_true": float(actual_pm25),
                    "y_pred": float(pred_pm25),
                    "residual": float(res_pm25),
                    "confidence": 0.9
                },
                {
                    "model_id": "pm10_model",
                    "device_id": "device_1",
                    "time": current_time,
                    "horizon": 1,
                    "y_true": float(actual_pm10),
                    "y_pred": float(pred_pm10),
                    "residual": float(res_pm10),
                    "confidence": 0.9
                }
            ])

            conn.commit()

        print("Predictions stored in DB")

    except Exception as e:
        print("Error:", e)

    time.sleep(60)