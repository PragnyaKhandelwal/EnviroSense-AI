import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from dotenv import load_dotenv

load_dotenv()

# SHARED UTILS

SHARED_UTILS_DIR = Path('/home/shared/envirosense')
sys.path.insert(0, str(SHARED_UTILS_DIR))

import db_utils

print("Running Regime Assignment...")

# LOAD FEATURE TABLE
query = """
SELECT *
FROM pratishtha_features
ORDER BY time
"""

with db_utils.get_engine().connect() as conn:
    df = pd.read_sql(query, conn)

if df.empty:
    print("No data found")
    exit()
# LOAD SENSOR DATA
query2 = """
SELECT time, pm2_5, pm10_0, temperature, humidity
FROM sensor_data
"""

with db_utils.get_engine().connect() as conn:
    df2 = pd.read_sql(query2, conn)

# FIX TIME + ALIGN
df['time'] = pd.to_datetime(df['time'], utc=True).dt.floor('min')
df2['time'] = pd.to_datetime(df2['time'], utc=True).dt.floor('min')

# MERGE
df = df.merge(df2, on='time', how='left')

# remove duplicates after merge
df = df.drop_duplicates(subset=['time', 'device_id'])

# FEATURE ENGINEERING
df['pm_ratio'] = df['pm2_5'] / (df['pm10_0'] + 1)
df['temp_humidity'] = df['temperature'] * df['humidity']
df['pm_diff'] = df['pm10_0'] - df['pm2_5']

features = [
    'pm2_5',
    'pm10_0',
    'temperature',
    'humidity',
    'pm_ratio',
    'temp_humidity',
    'pm_diff'
]

X = df[features].fillna(0)
X_scaled = StandardScaler().fit_transform(X)

# CLUSTERING
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
df['regime_id'] = kmeans.fit_predict(X_scaled)

# debug
print("\nCluster distribution:")
print(df['regime_id'].value_counts())

# REGIME LABELING
cluster_stats = df.groupby('regime_id')[['pm2_5', 'temperature', 'humidity']].mean()

print("\nCluster stats:")
print(cluster_stats)

labels_map = {}

for r in cluster_stats.index:
    pm = cluster_stats.loc[r, 'pm2_5']

    if pm < 15:
        labels_map[r] = "Stable Clean"
    elif pm < 35:
        labels_map[r] = "Moderate Fluctuation"
    else:
        labels_map[r] = "Unstable / Polluted"

df['regime_label'] = df['regime_id'].map(labels_map)

# PREPARE OUTPUT
out = df[['time', 'device_id', 'regime_id', 'regime_label']].copy()

# REMOVE DUPLICATES (DB LEVEL SAFE)
with db_utils.get_engine().connect() as conn:
    existing = pd.read_sql("SELECT time, device_id FROM regime_profiles", conn)

if not existing.empty:
    existing['time'] = pd.to_datetime(existing['time'], utc=True)

    out = out.merge(existing, on=['time', 'device_id'], how='left', indicator=True)
    out = out[out['_merge'] == 'left_only'].drop(columns=['_merge'])


# SAVE
if not out.empty:
    inserted = db_utils.save_dataframe(out, "regime_profiles")
    print(f"\nInserted {inserted} rows")
else:
    print("\nNo new rows to insert")

print("Regime assignment complete")