import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

# shared utils
SHARED_UTILS_DIR = Path('/home/shared/envirosense')
sys.path.insert(0, str(SHARED_UTILS_DIR))

import db_utils

print("Running Transition Analysis...")

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
    conn.execute(text("DELETE FROM regime_transitions"))

# SORT PROPERLY
df = df.sort_values(['device_id', 'time'])

transitions = {}
# BUILD TRANSITIONS (PER DEVICE SAFE)

for device_id, group in df.groupby('device_id'):
    group = group.reset_index(drop=True)

    for i in range(len(group) - 1):
        r1 = group.loc[i, 'regime_id']
        r2 = group.loc[i + 1, 'regime_id']

        key = (r1, r2)
        transitions[key] = transitions.get(key, 0) + 1


# CREATE DATAFRAME
rows = []

for (r1, r2), count in transitions.items():
    rows.append([r1, r2, count])

trans_df = pd.DataFrame(rows, columns=[
    'from_regime', 'to_regime', 'transition_count'
])

# PROBABILITIES
trans_df['transition_prob'] = trans_df.groupby('from_regime')[
    'transition_count'
].transform(lambda x: x / x.sum())

# SAVE
db_utils.save_dataframe(trans_df, "regime_transitions")

print("\nTransitions saved")
print(trans_df)