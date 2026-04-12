from pathlib import Path
import sys

# -------- SHARED PATH --------
SHARED_UTILS_DIR = Path("/home/shared/envirosense")
if str(SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS_DIR))

print("🔥 MEMBER4 PIPELINE STARTED 🔥")

import db_utils
import pandas as pd


def run_pipeline():
    print("Running pipeline...")

    with db_utils.get_engine().connect() as conn:
        df = pd.read_sql("SELECT * FROM clean_data", conn)

    df = df.sort_values("time")

    # -------- FEATURES --------
    df['pm2_5_lag1'] = df['pm2_5'].shift(1)
    df['pm2_5_roll_1h'] = df['pm2_5'].rolling(3).mean()

    df = df.dropna(subset=["pm2_5_lag1", "pm2_5_roll_1h"])

    # -------- TAKE LATEST --------
    result_df = df.tail(1)[[
        "time","device_id","pm2_5","temperature","humidity",
        "pm2_5_lag1","pm2_5_roll_1h"
    ]].copy()

    result_df["created_by"] = "member4"

    print("Latest row time:", result_df["time"].iloc[0])

    # -------- DUPLICATE CHECK --------
    last_time = result_df["time"].iloc[0]

    with db_utils.get_engine().connect() as conn:
        check = pd.read_sql(
            f"SELECT 1 FROM anushka WHERE time = '{last_time}' LIMIT 1",
            conn
        )

    if not check.empty:
        print("⚠️ Duplicate detected, skipping insert")
        return

    # -------- INSERT --------
    print("Inserting row:")
    print(result_df)

    with db_utils.get_engine().begin() as conn:
        result_df.to_sql("anushka", conn, if_exists="append", index=False)

    print("✅ Inserted successfully!")