from pathlib import Path
import sys
import traceback

# -------- SHARED PATH --------
SHARED_UTILS_DIR = Path("/home/shared/envirosense")
if str(SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS_DIR))

print("🔥 MEMBER4 PIPELINE STARTED 🔥")

import db_utils
import pandas as pd
from sqlalchemy import text


def get_last_processed_time():
    """Get MAX(time) from anushka_features to use as watermark."""
    try:
        with db_utils.get_engine().connect() as conn:
            result = conn.execute(text("SELECT MAX(time) FROM anushka_features"))
            last_time = result.scalar()
            if last_time is None:
                return None
            return last_time
    except Exception as e:
        print(f"❌ Error getting last processed time: {e}")
        traceback.print_exc()
        return None


def fetch_new_rows():
    """Fetch all rows from clean_data that are newer than the last processed time."""
    last_time = get_last_processed_time()
    
    try:
        with db_utils.get_engine().connect() as conn:
            if last_time is None:
                # First run: fetch recent rows from clean_data
                query_obj = text("SELECT * FROM clean_data ORDER BY time ASC")
                print("First run: fetching all clean_data rows")
                df = pd.read_sql(query_obj, conn)
            else:
                # Incremental: only fetch NEW rows since watermark
                query_obj = text("""
                    SELECT * FROM clean_data 
                    WHERE time > :last_time
                    ORDER BY time ASC
                """)
                print(f"Incremental fetch: rows newer than {last_time}")
                df = pd.read_sql(query_obj, conn, params={'last_time': last_time})
            return df
    except Exception as e:
        print(f"❌ Error fetching rows: {e}")
        traceback.print_exc()
        return pd.DataFrame()


def run_pipeline():
    print("\n▶️ Running Member4 feature pipeline...")
    
    try:
        # Fetch new rows since last watermark
        df = fetch_new_rows()
        
        if df.empty:
            print("✅ No new rows to process")
            return
        
        print(f"📊 Processing {len(df)} new rows")
        
        df = df.sort_values(["device_id", "time"])

        # -------- FEATURES --------
        df['pm2_5_lag1'] = df.groupby("device_id")['pm2_5'].shift(1)
        df['pm2_5_roll_1h'] = (
            df.groupby("device_id")['pm2_5']
            .rolling(3)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df = df.dropna(subset=["pm2_5_lag1", "pm2_5_roll_1h"])
        df = df.sort_values("time")

        if df.empty:
            print("⚠️ No rows after feature generation, skipping insert")
            return

        # -------- PREPARE ALL ROWS FOR INSERT (batch, not just tail(1)) --------
        result_df = df[[
            "time","device_id","pm2_5","temperature","humidity",
            "pm2_5_lag1","pm2_5_roll_1h"
        ]].copy()

        result_df["created_by"] = "member4"

        print(f"✅ Generated {len(result_df)} feature rows")
        print(f"   Time range: {result_df['time'].min()} → {result_df['time'].max()}")

        # -------- BATCH INSERT --------
        rows_inserted = 0
        rows_skipped = 0
        
        with db_utils.get_engine().begin() as conn:
            for _, row in result_df.iterrows():
                row_dict = row.to_dict()
                try:
                    stmt = text("""
                        INSERT INTO anushka_features (
                            time, device_id, pm2_5, temperature, humidity,
                            pm2_5_lag1, pm2_5_roll_1h, created_by
                        ) VALUES (
                            :time, :device_id, :pm2_5, :temperature, :humidity,
                            :pm2_5_lag1, :pm2_5_roll_1h, :created_by
                        )
                        ON CONFLICT (device_id, time) DO NOTHING
                    """)
                    result = conn.execute(stmt, row_dict)
                    if result.rowcount > 0:
                        rows_inserted += 1
                    else:
                        rows_skipped += 1
                except Exception as e:
                    print(f"⚠️ Error inserting row {row['time']}: {e}")
                    rows_skipped += 1

        print(f"✅ Inserted {rows_inserted} rows, skipped {rows_skipped} (duplicates)\n")

    except Exception as e:
        print(f"❌ PIPELINE ERROR: {e}")
        traceback.print_exc()

   