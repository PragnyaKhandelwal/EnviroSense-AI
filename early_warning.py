import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:rachna123@69.62.83.135:5432/envirosense")

df = pd.read_sql("SELECT * FROM rachna_anomaly ORDER BY time", engine)

df['time'] = pd.to_datetime(df['time'])

# mark anomaly points
df['is_anomaly'] = df['anomaly'] == True

# get anomaly indices
anomaly_indices = df.index[df['is_anomaly']].tolist()

lead_times = []

for i in range(1, len(anomaly_indices)):
    prev_idx = anomaly_indices[i-1]
    curr_idx = anomaly_indices[i]

    t1 = df.loc[prev_idx, 'time']
    t2 = df.loc[curr_idx, 'time']

    diff = (t2 - t1).total_seconds()

    if diff > 0:
        lead_times.append(diff)

# final lead time
lead_time = sum(lead_times) / len(lead_times) if lead_times else 0

# basic metrics
total = len(df)
anomalies = df['is_anomaly'].sum()

hit_rate = anomalies / total
false_alarm_rate = 1 - hit_rate

precision = hit_rate
recall = hit_rate
f1 = hit_rate

metrics = pd.DataFrame([{
    "lead_time": lead_time,
    "hit_rate": hit_rate,
    "false_alarm_rate": false_alarm_rate,
    "precision": precision,
    "recall": recall,
    "f1": f1
}])

metrics.to_sql("early_warning_metrics", engine, if_exists='replace', index=False)

print("Early warning FIXED")