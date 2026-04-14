
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/shared/envirosense')

from db_utils import get_engine
import pandas as pd
from db_utils import get_engine

engine = get_engine()

query = "SELECT * FROM sensor_data ORDER BY time;"
df = pd.read_sql(query, engine)

print("Data Preview:\n")
print(df.head())

df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')

df['pm_diff'] = df['pm2_5'].diff()
df['anomaly'] = df['pm_diff'].abs() > 200

anomalies = df[df['anomaly'] == True]

print("\nRule-Based Anomalies:\n")
print(anomalies)

data_to_insert = anomalies[['time', 'pm2_5']].copy()
data_to_insert['anomaly'] = True

data_to_insert.to_sql('rachna_anomaly', engine, if_exists='append', index=False)
plt.figure(figsize=(10,5))

plt.plot(df['time'], df['pm2_5'], label='PM2.5')

plt.scatter(anomalies['time'], anomalies['pm2_5'],
            color='red', label='Anomalies')

plt.xlabel('Time')
plt.ylabel('PM2.5')
plt.title('Anomaly Detection')

plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()