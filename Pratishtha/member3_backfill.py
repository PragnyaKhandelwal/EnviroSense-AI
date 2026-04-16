import sys
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from dotenv import load_dotenv
load_dotenv()

SHARED_UTILS_DIR = Path('/home/shared/envirosense')
if str(SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS_DIR))

import db_utils   


with db_utils.get_engine().connect() as conn:
    result = conn.execute(text("SELECT MIN(time) FROM pratishtha_features"))
    min_time = result.scalar()

print(f"Earliest existing time: {min_time}")


query =text("""
SELECT *
FROM sensor_data
WHERE time < :min_time
ORDER BY time
""")

with db_utils.get_engine().connect() as conn:
    df = pd.read_sql(query, conn, params={"min_time": min_time})

print(f"Loaded {len(df)} rows for backfill")

if df.empty:
    print("No older data to backfill ")
    exit()


df['time'] = pd.to_datetime(df['time'], utc=True)

bins = [
    'bin_0_3_0_5',
    'bin_0_5_1_0',
    'bin_1_0_2_5',
    'bin_2_5_5_0',
    'bin_5_0_10_0'
]


X = df[bins + ['pm2_5', 'pm10_0', 'temperature', 'humidity']].fillna(0)
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)


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


features_df = df[['time', 'device_id', 'cluster', 'label']].copy()
features_df.drop_duplicates(subset=['time', 'device_id'], inplace=True)


print("Backfilling older data...")

features_df.to_sql(
    "pratishtha_features",
    db_utils.get_engine(),
    if_exists='append',
    index=False
)

print("Backfill completed without touching existing rows!")