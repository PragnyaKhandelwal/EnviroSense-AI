import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:rachna123@69.62.83.135:5432/envirosense")

df = pd.read_sql("SELECT * FROM rachna_anomaly", engine)

df['rolling_mean'] = df['pm2_5'].rolling(5).mean()
df['rolling_std'] = df['pm2_5'].rolling(5).std()
df['trend'] = df['pm2_5'].diff()

df['lead_window'] = 5

df[['time','pm2_5','rolling_mean','rolling_std','trend','lead_window']]\
.to_sql('anomaly_precursors', engine, if_exists='replace', index=False)