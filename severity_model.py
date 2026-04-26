import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:rachna123@69.62.83.135:5432/envirosense")

df = pd.read_sql("SELECT * FROM rachna_anomaly", engine)

def classify(pm):
    if pm > 300:
        return "HIGH"
    elif pm > 150:
        return "MEDIUM"
    else:
        return "LOW"

df['severity_class'] = df['pm2_5'].apply(classify)
df['severity_score'] = df['pm2_5']

df[['time','pm2_5','severity_score','severity_class']]\
.to_sql('anomaly_severity', engine, if_exists='replace', index=False)