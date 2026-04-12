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

# Filter anomalies
rule_anomalies = df[df['anomaly_rule'] == True]

print("\nRule-Based Anomalies:\n")
print(rule_anomalies)

# 🔴 STEP 5: MACHINE LEARNING (FIXED FEATURES)

# Use only columns that exist in your dataset
features = ['pm2_5', 'temperature', 'humidity']
features = [col for col in features if col in df.columns]

# Remove missing values
df_ml = df[features].dropna()

# Train model
model = IsolationForest(contamination=0.01, random_state=42)
df_ml['anomaly_ml'] = model.fit_predict(df_ml)

# Convert (-1 = anomaly → 1, normal → 0)
df_ml['anomaly_ml'] = df_ml['anomaly_ml'].map({1: 0, -1: 1})

# Merge back to original dataframe
df.loc[df_ml.index, 'anomaly_ml'] = df_ml['anomaly_ml']

# Filter ML anomalies
ml_anomalies = df[df['anomaly_ml'] == 1]

print("\nML-Based Anomalies:\n")
print(ml_anomalies)

# 🔴 STEP 6: PLOT RULE-BASED ANOMALIES
plt.figure(figsize=(12,6))

plt.plot(df['time'], df['pm2_5'], label='PM2.5')

plt.scatter(rule_anomalies['time'], rule_anomalies['pm2_5'],
            color='red', label='Rule-Based Anomalies')

plt.xlabel('Time')
plt.ylabel('PM2.5')
plt.title('Rule-Based Anomaly Detection')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# 🔴 STEP 7: PLOT ML ANOMALIES
plt.figure(figsize=(12,6))

plt.plot(df['time'], df['pm2_5'], label='PM2.5')

plt.scatter(ml_anomalies['time'], ml_anomalies['pm2_5'],
            color='orange', label='ML Anomalies')

plt.xlabel('Time')
plt.ylabel('PM2.5')
plt.title('ML-Based Anomaly Detection (Isolation Forest)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()