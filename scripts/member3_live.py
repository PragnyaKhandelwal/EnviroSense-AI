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

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SHARED UTILS
SHARED_UTILS_DIR = Path('/home/shared/envirosense')
if str(SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS_DIR))

import db_utils

# =========================
# HELPERS
# =========================
def get_last_saved_time():
    with db_utils.get_engine().connect() as conn:
        result = conn.execute(text("SELECT MAX(time) FROM pratishtha_features"))
        return result.scalar()

def fetch_new_rows():
    last_time = get_last_saved_time()

    with db_utils.get_engine().connect() as conn:
        if last_time is None:
            query = text("SELECT * FROM sensor_data ORDER BY time DESC LIMIT 1000")
            return pd.read_sql(query, conn)
        else:
            query = text("""
                SELECT * FROM sensor_data
                WHERE time > :last_time
                ORDER BY time ASC
            """)
            return pd.read_sql(query, conn, params={'last_time': last_time})

def fetch_plot_window():
    with db_utils.get_engine().connect() as conn:
        query = text("""
            SELECT *
            FROM sensor_data
            ORDER BY time DESC
            LIMIT 200
        """)
        return pd.read_sql(query, conn)

# =========================
# SETUP
# =========================
os.makedirs("plots", exist_ok=True)

bins = [
    'bin_0_3_0_5',
    'bin_0_5_1_0',
    'bin_1_0_2_5',
    'bin_2_5_5_0',
    'bin_5_0_10_0'
]

logger.info("🚀 Member3 Pipeline Started")

# =========================
# MAIN LOOP
# =========================
while True:
    try:
        # =========================
        # FETCH NEW DATA
        # =========================
        df = fetch_new_rows()

        if df.empty:
            logger.info("No new data")
        else:
            logger.info(f"Processing {len(df)} rows")
            df['time'] = pd.to_datetime(df['time'], utc=True)

            # =========================
            # WINDOW FOR CLUSTERING
            # =========================
            window_df = fetch_plot_window()

            if window_df.empty or len(window_df) < 20:
                logger.warning("Not enough data for clustering → skipping")
            else:
                window_df['time'] = pd.to_datetime(window_df['time'], utc=True)

                try:
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler

                    X = window_df[bins + ['pm2_5', 'pm10_0', 'temperature', 'humidity']].fillna(0)
                    X_scaled = StandardScaler().fit_transform(X)

                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    window_df['cluster'] = kmeans.fit_predict(X_scaled)

                    # Labeling
                    cluster_means = window_df.groupby('cluster')['pm2_5'].mean().sort_values()

                    labels_map = {}
                    for i, c in enumerate(cluster_means.index):
                        if i == 0:
                            labels_map[c] = "Better Air"
                        elif i == 1:
                            labels_map[c] = "Moderate Pollution"
                        else:
                            labels_map[c] = "Unhealthy Air"

                    window_df['label'] = window_df['cluster'].map(labels_map)

                    # Assign latest label
                    latest = window_df.sort_values('time').iloc[-1]

                    df['cluster'] = latest['cluster']
                    df['label'] = latest['label']

                    logger.info(f"Assigned label: {latest['label']}")

                    # =========================
                    # SAVE TO DB
                    # =========================
                    features_df = df[['time', 'device_id', 'cluster', 'label']]
                    features_df = features_df.drop_duplicates(subset=['time', 'device_id'])

                    if not features_df.empty:
                        inserted = db_utils.save_dataframe(features_df, "pratishtha_features")
                        logger.info(f"Inserted {inserted} rows")

                except Exception as e:
                    logger.error(f"Clustering failed: {e}")
                    traceback.print_exc()

        # =========================
        # PLOTTING (ALWAYS RUN)
        # =========================
        plot_df = fetch_plot_window()

        if not plot_df.empty:
            plot_df['time'] = pd.to_datetime(plot_df['time'], utc=True)

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Particle Distribution
            df_bins = plot_df[bins].div(plot_df[bins].sum(axis=1), axis=0)
            df_bins.plot.area(figsize=(10, 6))
            plt.title("Particle Distribution (LIVE)")
            plt.savefig("plots/live_distribution.png")
            plt.close()

            # Count vs Mass
            plt.figure()
            plt.scatter(plot_df['pm2_5_pcs'], plot_df['pm2_5'])
            plt.xlabel("PM2.5 Count")
            plt.ylabel("PM2.5 Mass")
            plt.title("Count vs Mass (LIVE)")
            plt.grid()
            plt.savefig("plots/live_count_mass.png")
            plt.close()

            # Cluster Plot
            plt.figure()
            plt.scatter(plot_df['pm2_5'], plot_df['pm10_0'])
            plt.xlabel("PM2.5")
            plt.ylabel("PM10")
            plt.title("Air Quality (LIVE)")
            plt.grid()
            plt.savefig("plots/live_clusters.png")
            plt.close()

            logger.info("Plots updated")

    except Exception as e:
        logger.error(f"ERROR: {e}")
        traceback.print_exc()

    time.sleep(60)