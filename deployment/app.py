# app.py
import joblib
import json
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Credit Card Fraud Detection API")

# Load model artifacts once, at startup
with open('models/final_model_config.json') as f:
    config = json.load(f)

model = joblib.load('models/baseline_random_forest.pkl')
scaler = joblib.load('models/amount_hour_scaler.pkl')
threshold = config['threshold']
feature_columns = config['feature_columns']


class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds since first transaction in dataset")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/")
def root():
    return {"message": "Credit Card Fraud Detection API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": config['model_name'], "threshold": threshold}


@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.dict()

    # Replicate EXACT preprocessing pipeline from training
    hour = (data['Time'] // 3600) % 24

    row = {k: v for k, v in data.items() if k != 'Time'}
    row['Hour'] = hour

    df = pd.DataFrame([row])[feature_columns]

    df[['Amount', 'Hour']] = scaler.transform(df[['Amount', 'Hour']])

    fraud_probability = float(model.predict_proba(df)[:, 1][0])
    is_fraud = bool(fraud_probability >= threshold)

    return {
        "fraud_probability": round(fraud_probability, 4),
        "is_fraud": is_fraud,
        "threshold_used": threshold,
        "model": config['model_name']
    }