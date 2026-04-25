import os
import time
import pandas as pd
import sys
from sqlalchemy import text
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/home/ashu/.env")

# DB connection
sys.path.insert(0, "/home/shared/envirosense")
from db_utils import get_engine

engine = get_engine()

print("Retraining service started...")

while True:
    try:
        print("\nChecking drift...")

        #  Get last 6 hours rolling error
        query = """
        SELECT *
        FROM ashu_model_rolling_error
        WHERE "window" >= NOW() - INTERVAL '6 hours'
        ORDER BY "window"
        """

        df = pd.read_sql(query, engine)

        # If no data
        if df.empty:
            print("No rolling error data found")
            time.sleep(600)
            continue

        retrain_needed = False

        # Check drift per model
        for model in df['model_id'].unique():
            sub = df[df['model_id'] == model]

            if len(sub) < 2:
                continue

            old_mae = sub.iloc[0]['mae']
            new_mae = sub.iloc[-1]['mae']

            print(f"{model}: old MAE={old_mae:.4f}, new MAE={new_mae:.4f}")

            # Drift condition
            if new_mae > old_mae * 1.5:
                print(f"Drift detected in {model}")
                retrain_needed = True

        #  Trigger retraining
        if retrain_needed:
            print("Running full training pipeline...")

            exit_code = os.system(
                "/home/ashu/.venv/EnviroSense-AI/.venv/bin/python Ashu/AI_ML/train_model.py"
            )

            if exit_code == 0:
                print("Retraining completed successfully")
            else:
                print("Retraining failed")

        else:
            print("No retraining needed")

    except Exception as e:
        print(" Error:", e)

    # Run every 10 minutes
    time.sleep(600)