import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 🔴 STEP 1: CONNECT DATABASE
conn = psycopg2.connect(
    host="69.62.83.135",
    database="envirosense",
    user="postgres",
    password="rachna",
    port="5432",
    sslmode="disable"
)

# 🔴 STEP 2: FETCH DATA
query = "SELECT * FROM sensor_data ORDER BY time;"
df = pd.read_sql(query, conn)

print("Data Preview:\n")
print(df.head())

# 🔴 STEP 3: PREPROCESS DATA
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')

# 🔴 STEP 4: RULE-BASED ANOMALY DETECTION
df['pm_diff'] = df['pm2_5'].diff()
df['anomaly_rule'] = df['pm_diff'].abs() > 200

rule_anomalies = df[df['anomaly_rule'] == True]

print("\nRule-Based Anomalies:\n")
print(rule_anomalies)

# 🔴 STEP 5: MACHINE LEARNING

features = ['pm2_5', 'temperature', 'humidity']
features = [col for col in features if col in df.columns]

df_ml = df[features].dropna()

model = IsolationForest(contamination=0.01, random_state=42)
df_ml['anomaly_ml'] = model.fit_predict(df_ml)

df_ml['anomaly_ml'] = df_ml['anomaly_ml'].map({1: 0, -1: 1})

df.loc[df_ml.index, 'anomaly_ml'] = df_ml['anomaly_ml']

ml_anomalies = df[df['anomaly_ml'] == 1]

print("\nML-Based Anomalies:\n")
print(ml_anomalies)

# 🔴 ✅ NEW STEP 6: Z-SCORE (ADDED PART ONLY)

df['rolling_mean'] = df['pm2_5'].rolling(window=10).mean()
df['rolling_std'] = df['pm2_5'].rolling(window=10).std()

df['z_score'] = (df['pm2_5'] - df['rolling_mean']) / df['rolling_std']

df['anomaly_zscore'] = df['z_score'].abs() > 3

z_anomalies = df[df['anomaly_zscore'] == True]

print("\nZ-Score Anomalies:\n")
print(z_anomalies)

# 🔴 STEP 7: PLOT RULE-BASED
plt.figure(figsize=(12,6))

plt.plot(df['time'], df['pm2_5'], label='PM2.5')

plt.scatter(rule_anomalies['time'], rule_anomalies['pm2_5'],
            color='red', label='Rule-Based')

plt.xlabel('Time')
plt.ylabel('PM2.5')
plt.title('Rule-Based Anomaly Detection')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# 🔴 STEP 8: PLOT ML
plt.figure(figsize=(12,6))

plt.plot(df['time'], df['pm2_5'], label='PM2.5')

plt.scatter(ml_anomalies['time'], ml_anomalies['pm2_5'],
            color='orange', label='ML')

plt.xlabel('Time')
plt.ylabel('PM2.5')
plt.title('ML-Based Anomaly Detection')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# 🔴 ✅ NEW STEP 9: PLOT Z-SCORE (ADDED)

plt.figure(figsize=(12,6))

plt.plot(df['time'], df['pm2_5'], label='PM2.5')

plt.scatter(z_anomalies['time'], z_anomalies['pm2_5'],
            color='green', label='Z-Score')

plt.xlabel('Time')
plt.ylabel('PM2.5')
plt.title('Z-Score Anomaly Detection')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()