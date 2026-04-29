from fastapi import FastAPI
from sqlalchemy import create_engine
import pandas as pd

app = FastAPI()

engine = create_engine("postgresql://postgres:rachna123@69.62.83.135:5432/envirosense")

@app.get("/anomalies")
def get_anomalies():
    df = pd.read_sql("SELECT * FROM rachna_anomaly ORDER BY time DESC LIMIT 100", engine)
    return df.to_dict(orient="records")

@app.get("/severity")
def get_severity():
    df = pd.read_sql("SELECT * FROM anomaly_severity ORDER BY time DESC LIMIT 100", engine)
    return df.to_dict(orient="records")

@app.get("/metrics")
def get_metrics():
    df = pd.read_sql("SELECT * FROM early_warning_metrics", engine)
    return df.to_dict(orient="records")

@app.get("/evaluation")
def get_eval():
    df = pd.read_sql("SELECT * FROM anomaly_eval", engine)
    return df.to_dict(orient="records")