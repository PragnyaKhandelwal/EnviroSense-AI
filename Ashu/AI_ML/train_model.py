import pandas as pd
import sys
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sqlalchemy import text
import hashlib

sys.path.insert(0, "/home/shared/envirosense")
from db_utils import get_engine

engine = get_engine()

query = "SELECT * FROM ashu_features ORDER BY time"

df = pd.read_sql(query, engine)


base_X = df.drop(columns=[
    'id',
    'time',
    'created_at',
    'day',
    'pm1_avg',
    'pm25_avg',
    'pm10_avg'
])

X_pm1 = base_X
X_pm25 = base_X
X_pm10 = base_X

y_pm1 = df['pm1_avg']
y_pm25 = df['pm25_avg']
y_pm10 = df['pm10_avg']

## spliting the data in 80-20 ratio for training and testing

X_train_pm1, X_test_pm1, y_train_pm1, y_test_pm1 = train_test_split(
    X_pm1, y_pm1, test_size=0.2, shuffle=False
)

X_train_pm25, X_test_pm25, y_train_pm25, y_test_pm25 = train_test_split(
    X_pm25, y_pm25, test_size=0.2, shuffle=False
)

X_train_pm10, X_test_pm10, y_train_pm10, y_test_pm10 = train_test_split(
    X_pm10, y_pm10, test_size=0.2, shuffle=False
)

print(X_train_pm1.shape)
print(X_train_pm25.shape)
print(X_train_pm10.shape)


lr_pm1 = LinearRegression()
lr_pm25 = LinearRegression()
lr_pm10 = LinearRegression()

lr_pm25.fit(X_train_pm25,y_train_pm25)
lr_pm1.fit(X_train_pm1,y_train_pm1)
lr_pm10.fit(X_train_pm10,y_train_pm10)

lr_preds_pm25 = lr_pm25.predict(X_test_pm25)
lr_preds_pm1 = lr_pm1.predict(X_test_pm1)
lr_preds_pm10 = lr_pm10.predict(X_test_pm10)

lr_mae_pm25 = mean_absolute_error(y_test_pm25, lr_preds_pm25)
lr_rmse_pm25 = np.sqrt(mean_squared_error(y_test_pm25, lr_preds_pm25))

lr_mae_pm1 = mean_absolute_error(y_test_pm1, lr_preds_pm1)
lr_rmse_pm1 = np.sqrt(mean_squared_error(y_test_pm1, lr_preds_pm1))

lr_mae_pm10 = mean_absolute_error(y_test_pm10, lr_preds_pm10)
lr_rmse_pm10 = np.sqrt(mean_squared_error(y_test_pm10, lr_preds_pm10))


rf_pm1 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_pm25 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_pm10 = RandomForestRegressor(n_estimators=100, random_state=42)

rf_pm25.fit(X_train_pm25, y_train_pm25)
rf_pm1.fit(X_train_pm1, y_train_pm1)
rf_pm10.fit(X_train_pm10, y_train_pm10)

rf_preds_pm25 = rf_pm25.predict(X_test_pm25)
rf_preds_pm1 = rf_pm1.predict(X_test_pm1)
rf_preds_pm10 = rf_pm10.predict(X_test_pm10)

rf_mae_pm25 = mean_absolute_error(y_test_pm25, rf_preds_pm25)
rf_rmse_pm25 = np.sqrt(mean_squared_error(y_test_pm25, rf_preds_pm25))

rf_mae_pm1 = mean_absolute_error(y_test_pm1, rf_preds_pm1)
rf_rmse_pm1 = np.sqrt(mean_squared_error(y_test_pm1, rf_preds_pm1))

rf_mae_pm10 = mean_absolute_error(y_test_pm10, rf_preds_pm10)
rf_rmse_pm10 = np.sqrt(mean_squared_error(y_test_pm10, rf_preds_pm10))

def select_best(lr_mae, lr_rmse, rf_mae, rf_rmse):
    if rf_rmse < lr_rmse:
        return "RandomForest", rf_mae, rf_rmse
    else:
        return "LinearRegression", lr_mae, lr_rmse


algo_pm1, mae_pm1, rmse_pm1 = select_best(lr_mae_pm1, lr_rmse_pm1, rf_mae_pm1, rf_rmse_pm1)
algo_pm25, mae_pm25, rmse_pm25 = select_best(lr_mae_pm25, lr_rmse_pm25, rf_mae_pm25, rf_rmse_pm25)
algo_pm10, mae_pm10, rmse_pm10 = select_best(lr_mae_pm10, lr_rmse_pm10, rf_mae_pm10, rf_rmse_pm10)


## Model Saving
model_dir = "Ashu/AI_ML"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(lr_pm1, f"{model_dir}/pm1_model.pkl")
joblib.dump(lr_pm25, f"{model_dir}/pm25_model.pkl")
joblib.dump(lr_pm10, f"{model_dir}/pm10_model.pkl")



## Storing in the Database
features = list(X_pm25.columns)
feature_hash = hashlib.md5(",".join(features).encode()).hexdigest()

train_start = df['time'].min()
train_end = df['time'].max()

models = [
    ("pm1_model", algo_pm1, mae_pm1, rmse_pm1),
    ("pm25_model", algo_pm25, mae_pm25, rmse_pm25),
    ("pm10_model", algo_pm10, mae_pm10, rmse_pm10)
]

with engine.connect() as conn:

    for name, algo, mae, rmse in models:
        
        result = conn.execute(text("""
        SELECT COALESCE(MAX(version), 0)
        FROM ashu_model_registry
        WHERE model_name = :name
        """), {"name": name})

        version = result.scalar() + 1

        conn.execute(text("""
        UPDATE ashu_model_registry
        SET selected_flag = FALSE
        WHERE model_name = :name
        """), {"name": name})

        # insert new model
        conn.execute(text("""
        INSERT INTO ashu_model_registry (
            model_name, version, algorithm,
            feature_set_hash,
            train_start, train_end,
            val_mae, val_rmse,
            selected_flag
        )
        VALUES (
            :name, 1, :algo,
            :feature_hash,
            :train_start, :train_end,
            :mae, :rmse,
            TRUE
        )
        """), {
            "name": name,
            "algo": algo,
            "feature_hash": feature_hash,
            "train_start": train_start,
            "train_end": train_end,
            "mae": float(mae),
            "rmse": float(rmse)
        })

    conn.commit()





