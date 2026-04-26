import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

# shared utils
SHARED_UTILS_DIR = Path('/home/shared/envirosense')
sys.path.insert(0, str(SHARED_UTILS_DIR))

import db_utils

print("Running Stability Analysis...")

# LOAD DATA
query = """
SELECT time, device_id, regime_id
FROM regime_profiles
ORDER BY device_id, time
"""

with db_utils.get_engine().connect() as conn:
    df = pd.read_sql(query, conn)

if df.empty:
    print("No data found")
    exit()

# CLEAR OLD DATA
with db_utils.get_engine().begin() as conn:
    conn.execute(text("DELETE FROM regime_stability"))

# SORT
df = df.sort_values(['device_id', 'time'])

durations = []

# COMPUTE DURATIONS (PER DEVICE)
for device_id, group in df.groupby('device_id'):
    group = group.reset_index(drop=True)

    current = group.loc[0, 'regime_id']
    count = 1

    for i in range(1, len(group)):
        if group.loc[i, 'regime_id'] == current:
            count += 1
        else:
            durations.append((device_id, current, count))
            current = group.loc[i, 'regime_id']
            count = 1

    #  ADD LAST SEGMENT (important fix)
    durations.append((device_id, current, count))

# CREATE DF
dur_df = pd.DataFrame(durations, columns=[
    'device_id', 'regime_id', 'duration'
])

# STATS
summary = dur_df.groupby(['device_id', 'regime_id'])['duration'].agg([
    'mean', 'median'
]).reset_index()

summary.columns = [
    'device_id',
    'regime_id',
    'avg_duration',
    'median_duration'
]

# ENTROPY (PER DEVICE)
entropy_list = []

for device_id, group in dur_df.groupby('device_id'):
    probs = group['regime_id'].value_counts(normalize=True)
    entropy = -np.sum(probs * np.log2(probs))

    for regime in group['regime_id'].unique():
        entropy_list.append((device_id, regime, entropy))

entropy_df = pd.DataFrame(entropy_list, columns=[
    'device_id', 'regime_id', 'entropy'
])

# MERGE
summary = summary.merge(entropy_df, on=['device_id', 'regime_id'])

# SAVE
db_utils.save_dataframe(summary, "regime_stability")

print("\nStability computed")
print(summary)