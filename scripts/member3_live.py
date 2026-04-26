import sys
import os
import time
import traceback
import logging
from pathlib import Path
from sqlalchemy import text
from dotenv import load_dotenv

import pandas as pd

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SHARED UTILS
SHARED_UTILS_DIR = Path('/home/shared/envirosense')
if str(SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS_DIR))

import db_utils

# HELPER
def get_last_saved_time():
    """Get MAX(time) from pratishtha_features to use as watermark."""
    try:
        with db_utils.get_engine().connect() as conn:
            result = conn.execute(text("SELECT MAX(time) FROM pratishtha_features"))
            last_time = result.scalar()
            if last_time is None:
                return None
            return last_time
    except Exception as e:
        logger.error(f"Error getting last saved time: {e}")
        return None


def fetch_new_rows():
    """Fetch all rows from sensor_data newer than last watermark."""
    last_time = get_last_saved_time()
    
    try:
        with db_utils.get_engine().connect() as conn:
            if last_time is None:
                # First run: fetch recent rows from sensor_data
                query_obj = text("SELECT * FROM sensor_data ORDER BY time DESC LIMIT 1000")
                df = pd.read_sql(query_obj, conn)
            else:
                # Incremental: only fetch NEW rows since watermark
                query_obj = text("""
                    SELECT * FROM sensor_data 
                    WHERE time > :last_time
                    ORDER BY time ASC
                """)
                df = pd.read_sql(query_obj, conn, params={'last_time': last_time})
            return df
    except Exception as e:
        logger.error(f"Error fetching rows: {e}")
        return pd.DataFrame()


# SETUP
os.makedirs("plots", exist_ok=True)

bins = [
    'bin_0_3_0_5',
    'bin_0_5_1_0',
    'bin_1_0_2_5',
    'bin_2_5_5_0',
    'bin_5_0_10_0'
]

# MAIN LOOP
logger.info("🚀 Member3 Feature Pipeline Started (60s cadence, watermark-based incremental sync)")

while True:
    try:
        logger.info("Fetching new data...")
        df = fetch_new_rows()
        
        if df.empty:
            logger.info("✅ No new data to process")
            time.sleep(60)
            continue

        logger.info(f"📊 Processing {len(df)} new rows")

        # SENSOR STATUS
        current_time = pd.Timestamp.utcnow()
        last_time = pd.to_datetime(df['time'].max(), utc=True)
        diff = (current_time - last_time).total_seconds()
        status = "LIVE" if diff <= 300 else "OFFLINE"
        logger.info(f"Sensor {status} (lag: {diff:.1f}s)")

        # Skip time conversion complexity - keep as UTC
        df['time'] = pd.to_datetime(df['time'], utc=True)

        # CLUSTERING (with error handling)
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            logger.info("Running clustering...")
            X = df[bins + ['pm2_5', 'pm10_0', 'temperature', 'humidity']].fillna(0)
            X_scaled = StandardScaler().fit_transform(X)

            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(X_scaled)

            # LABELING
            cluster_means = df.groupby('cluster')[['pm2_5', 'temperature', 'humidity']].mean()
            sorted_clusters = cluster_means.sort_values(by='pm2_5')

            labels_map = {}
            for i, c in enumerate(sorted_clusters.index):
                pm = sorted_clusters.loc[c, 'pm2_5']
                temp = sorted_clusters.loc[c, 'temperature']
                hum = sorted_clusters.loc[c, 'humidity']

                if i == 0:
                    if temp > 35:
                        labels_map[c] = "Warm but Cleaner Air"
                    elif hum > 70:
                        labels_map[c] = "Humid but Cleaner Air"
                    else:
                        labels_map[c] = "Better Air"
                elif i == 1:
                    labels_map[c] = "Moderate Pollution"
                else:
                    if temp > 35:
                        labels_map[c] = "Hot & Polluted"
                    else:
                        labels_map[c] = "Unhealthy Air"

            df['label'] = df['cluster'].map(labels_map)
            logger.info("✅ Clustering complete")

        except Exception as e:
            logger.error(f"❌ Clustering error: {e}")
            traceback.print_exc()
            # If clustering fails, use a default label
            df['cluster'] = 0
            df['label'] = "Unknown"

        # SAVE (with deduplication)
        try:
            features_df = df[['time', 'device_id', 'cluster', 'label']].copy()
            features_df = features_df.drop_duplicates(subset=['time', 'device_id'])

            if not features_df.empty:
                inserted = db_utils.save_dataframe(features_df, "pratishtha_features")
                logger.info(f"✅ Inserted {inserted} rows to pratishtha_features")
            else:
                logger.info("⚠️ No data after deduplication")

        except Exception as e:
            logger.error(f"❌ Error saving features: {e}")
            traceback.print_exc()

        # PLOTTING (optional, with error handling)
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            logger.info("Generating plots...")

            # Particle distribution
            if not df.empty and all(col in df.columns for col in bins):
                df_bins = df[bins].div(df[bins].sum(axis=1), axis=0)
                df_bins.columns = ["0.3–0.5 µm", "0.5–1.0 µm", "1.0–2.5 µm", "2.5–5.0 µm", "5.0–10 µm"]
                df_bins.plot.area(figsize=(10, 6))
                plt.title(f"Particle Distribution ({status})\nLast Update: {df['time'].max()}")
                plt.xlabel("Index")
                plt.ylabel("Proportion")
                plt.grid()
                plt.savefig("plots/live_distribution.png", dpi=150)
                plt.close()

            # Count vs mass
            if 'pm2_5_pcs' in df.columns and 'pm2_5' in df.columns:
                plt.figure()
                plt.scatter(df['pm2_5_pcs'], df['pm2_5'])
                plt.title(f"Count vs Mass ({status})\nLast Update: {df['time'].max()}")
                plt.xlabel("PM2.5 Count")
                plt.ylabel("PM2.5 Mass")
                plt.grid()
                plt.savefig("plots/live_count_mass.png", dpi=150)
                plt.close()

            # Cluster plot
            plt.figure()
            for label in df['label'].unique():
                subset = df[df['label'] == label]
                if 'pm2_5' in subset.columns and 'pm10_0' in subset.columns:
                    plt.scatter(subset['pm2_5'], subset['pm10_0'], label=label, alpha=0.7)
            plt.title(f"Air Quality ({status})\nLast Update: {df['time'].max()}")
            plt.xlabel("PM2.5")
            plt.ylabel("PM10")
            plt.legend()
            plt.grid()
            plt.savefig("plots/live_clusters.png", dpi=150)
            plt.close()

            logger.info("✅ Plots updated")

        except Exception as e:
            logger.warning(f"⚠️ Plot generation skipped: {e}")

        logger.info("✅ Cycle complete\n")

    except Exception as e:
        logger.error(f"❌ PIPELINE ERROR: {e}")
        traceback.print_exc()

    time.sleep(60)

