import sys
import os
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.db_util import get_recent_data

os.makedirs("plots", exist_ok=True)

bins = [
    'bin_0_3_0_5',
    'bin_0_5_1_0',
    'bin_1_0_2_5',
    'bin_2_5_5_0',
    'bin_5_0_10_0'
]

while True:

    print("\nFetching latest data...")

    df = get_recent_data(200)

    if df.empty:
        print("No data yet...")
        time.sleep(10)
        continue

    # SENSOR STATUS CHECK
  
    current_time = pd.Timestamp.utcnow()
    last_time = pd.to_datetime(df['time'].max(), utc=True)
    diff = (current_time - last_time).total_seconds()

    if diff > 300:
        status = "OFFLINE"
        print("⚠️ Sensor OFF or delayed")
    else:
        status = "LIVE"
        print("✅ Sensor LIVE")

    #CONVERT UTC → IST
   
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Asia/Kolkata')

    # SIZE DISTRIBUTION
   
    df_bins = df[bins].div(df[bins].sum(axis=1), axis=0)

    df_bins.columns = [
        "0.3–0.5 µm",
        "0.5–1.0 µm",
        "1.0–2.5 µm",
        "2.5–5.0 µm",
        "5.0–10 µm"
    ]

    df_bins.plot.area(figsize=(10,6))

    plt.xlabel("Time Index")
    plt.ylabel("Particle Proportion (%)")
    plt.title(f"Particle Distribution ({status})\nLast Update (IST): {df['time'].max()}")

    plt.legend(title="Particle Size")
    plt.grid()

    plt.savefig("plots/live_distribution.png")
    plt.close()

    # COUNT vs MASS
    plt.scatter(df['pm2_5_pcs'], df['pm2_5'], color='blue')

    plt.xlabel("Particle Count (PM2.5)")
    plt.ylabel("Particle Mass (µg/m³)")
    plt.title(f"Count vs Mass ({status})\nLast Update (IST): {df['time'].max()}")

    plt.grid()

    plt.savefig("plots/live_count_mass.png")
    plt.close()

    #  CLUSTERING
    X = df[bins]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    # Label clusters
    cluster_means = df.groupby('cluster')[['pm2_5', 'pm10_0']].mean()
    print("\nCluster Means:")
    print(cluster_means)
    ratios = cluster_means['pm2_5'] / cluster_means['pm10_0']
    labels_map = {}

# sort clusters by ratio
    sorted_clusters = ratios.sort_values()

    labels_map[sorted_clusters.index[0]] = "Dust"
    labels_map[sorted_clusters.index[1]] = "Traffic"
    labels_map[sorted_clusters.index[2]] = "Smoke"

    df['label'] = df['cluster'].map(labels_map)

    # Plot clusters
    colors = {
        "Smoke": "red",
        "Dust": "brown",
        "Traffic": "blue"
    }

    for label in df['label'].unique():
        subset = df[df['label'] == label]

        plt.scatter(subset['pm2_5'], subset['pm10_0'],
                    label=label,
                    color=colors[label])

    plt.xlabel("PM2.5 (µg/m³)")
    plt.ylabel("PM10 (µg/m³)")
    plt.title(f"Pollution Types ({status})\nLast Update (IST): {df['time'].max()}")

    plt.legend()
    plt.grid()

    plt.savefig("plots/live_clusters.png")
    plt.close()

    # DEBUG
    print("Updated plots!")
    print("Last data time (IST):", df['time'].max())

    time.sleep(20)