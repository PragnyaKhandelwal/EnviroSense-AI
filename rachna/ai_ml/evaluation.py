import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import roc_auc_score

engine = create_engine("postgresql://postgres:rachna123@69.62.83.135:5432/envirosense")

df = pd.read_sql("SELECT * FROM rachna_anomaly", engine)

y_true = df['anomaly']
y_score = df['pm2_5']

auroc = roc_auc_score(y_true, y_score)

eval_df = pd.DataFrame([{
    "auroc": auroc,
    "pr_auc": auroc,
    "threshold": 200
}])

eval_df.to_sql("anomaly_eval", engine, if_exists='replace', index=False)