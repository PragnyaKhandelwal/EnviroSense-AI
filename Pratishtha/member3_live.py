import sys
import os
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

# SHARED UTILS
SHARED_UTILS_DIR = Path('/home/shared/envirosense')
if str(SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS_DIR))

import db_utils

# HELPER
def get_last_saved_time():
    with db_utils.get_engine().connect() as conn:
        result = conn.execute(text("SELECT MAX(time) FROM pratishtha_features"))
        return result.scalar()

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
while True:
    print("\nFetching latest data...")

    query = """
    SELECT *
    FROM sensor_data
    ORDER BY time DESC
    LIMIT 200
    """

    with db_utils.get_engine().connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        print("No data yet...")
        time.sleep(10)
        continue

    # SENSOR STATUS
    current_time = pd.Timestamp.utcnow()
    last_time = pd.to_datetime(df['time'].max(), utc=True)

    diff = (current_time - last_time).total_seconds()
    status = "LIVE" if diff <= 300 else "OFFLINE"

    print("Sensor LIVE" if status == "LIVE" else "Sensor OFF")

    # TIME CONVERSION
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Asia/Kolkata')

    # =========================
    # PARTICLE DISTRIBUTION
    # =========================
    df_bins = df[bins].div(df[bins].sum(axis=1), axis=0)

    df_bins.columns = [
        "0.3–0.5 µm",
        "0.5–1.0 µm",
        "1.0–2.5 µm",
        "2.5–5.0 µm",
        "5.0–10 µm"
    ]

    df_bins.plot.area(figsize=(10,6))
    plt.title(f"Particle Distribution ({status})\nLast Update: {df['time'].max()}")
    plt.xlabel("Index")
    plt.ylabel("Proportion")
    plt.grid()
    plt.savefig("plots/live_distribution.png", dpi=150)
    plt.close()

    print("Updated distribution plot")

    # =========================
    # COUNT vs MASS
    # =========================
    plt.figure()

    plt.scatter(df['pm2_5_pcs'], df['pm2_5'])
    plt.title(f"Count vs Mass ({status})\nLast Update: {df['time'].max()}")
    plt.xlabel("PM2.5 Count")
    plt.ylabel("PM2.5 Mass")
    plt.grid()

    plt.savefig("plots/live_count_mass.png", dpi=150)
    plt.close()

    print("Updated count vs mass plot")

    # =========================
    # CLUSTERING
    # =========================
    X = df[bins + ['pm2_5', 'pm10_0', 'temperature', 'humidity']].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # =========================
    # LABELING
    # =========================
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

    # =========================
    # SAVE
    # =========================
    features_df = df[['time', 'device_id', 'cluster', 'label']].copy()
    features_df['time'] = features_df['time'].dt.tz_convert('UTC')

    features_df.drop_duplicates(subset=['time', 'device_id'], inplace=True)

    last_saved_time = get_last_saved_time()

    if last_saved_time is not None:
        last_saved_time = pd.to_datetime(last_saved_time, utc=True)
        features_df = features_df[features_df['time'] > last_saved_time]

    if not features_df.empty:
        inserted = db_utils.save_dataframe(features_df, "pratishtha_features")
        print(f"Inserted {inserted} rows")
    else:
        print("No new data to save")

    # =========================
    # CLUSTER PLOT
    # =========================
    plt.figure()

    for label in df['label'].unique():
        subset = df[df['label'] == label]
        plt.scatter(
            subset['pm2_5'],
            subset['pm10_0'],
            label=label,
            alpha=0.7
        )

    plt.title(f"Air Quality ({status})\nLast Update: {df['time'].max()}")
    plt.xlabel("PM2.5")
    plt.ylabel("PM10")
    plt.legend()
    plt.grid()
    plt.savefig("plots/live_clusters.png", dpi=150)
    plt.close()

    print("Updated all plots!\n")

    time.sleep(20)

