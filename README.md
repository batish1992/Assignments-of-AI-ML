# Credit Card Fraud Detection with Model Deployment

AI/ML Specialisation Capstone Project — FinTech / Risk Analytics domain.

## Problem Statement

Financial institutions need to detect fraudulent card transactions quickly while
minimizing false positives for legitimate customers. This project builds a fraud
detection model on highly imbalanced data and deploys it as a scoring API.

## Dataset

- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 492 fraudulent (~0.17%)
- **Features:** `Time`, `V1`–`V28` (PCA-anonymized), `Amount`, `Class` (target)

### Downloading the dataset
```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d ./data
```
(Requires a Kaggle account and API token — see [Kaggle API docs](https://www.kaggle.com/docs/api).)

## Project Structure

notebooks/ → Full EDA, preprocessing, modeling, and evaluation notebook
models/ → All trained model artifacts and the fitted scaler
deployment/ → FastAPI scoring service
reports/ → Model evaluation report
presentation/ → Final presentation deck


## Methodology Summary

1. **EDA** — identified severe class imbalance (~0.17% fraud), found fraud
   concentrated at low transaction amounts and clustered at specific hours,
   confirmed PCA features are mutually uncorrelated.
2. **Preprocessing** — dropped 1,081 duplicate rows (leakage prevention),
   engineered `Hour` from `Time`, applied stratified 70/15/15 train/val/test
   split, scaled `Amount`/`Hour` with `RobustScaler` (fit on train only).
3. **Modeling** — trained Logistic Regression and Random Forest (baselines),
   an MLP neural network (Keras/TensorFlow), and XGBoost, all using
   class-weighting to address imbalance. An initial SMOTE-based MLP
   experiment was found to overfit and was replaced with class weights.
4. **Threshold tuning** — optimized decision thresholds per model using
   F1-maximization on the validation set rather than the default 0.5 cutoff.
5. **Model selection** — Random Forest at threshold 0.170 selected as the
   final model (best F1 / recall / precision balance; PR-AUC 0.83 validation,
   0.80 test).
6. **Deployment** — FastAPI scoring service (`deployment/app.py`) loads the
   saved model without retraining, replicates the full preprocessing
   pipeline, and returns fraud probability + decision.

## Results (Test Set)

| Metric | Value |
|---|---|
| PR-AUC | 0.7987 |
| ROC-AUC | 0.9350 |
| Precision (Fraud) | 0.85 |
| Recall (Fraud) | 0.79 |
| F1 (Fraud) | 0.82 |
| Avg. inference latency | 39.36 ms |

## Running the API locally

```bash
cd deployment
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Then open `http://127.0.0.1:8000/docs` to test the `/predict` endpoint interactively.

### Example request
```json
POST /predict
{
  "Time": 32400.0,
  "V1": -0.557, "V2": -0.382, "V3": 1.733, "V4": -1.054, "V5": -0.988,
  "V6": 0.570, "V7": -0.014, "V8": 0.281, "V9": -1.445, "V10": 0.325,
  "V11": 1.068, "V12": -0.886, "V13": -0.741, "V14": 0.024, "V15": 0.863,
  "V16": 1.050, "V17": 0.351, "V18": -0.442, "V19": 1.207, "V20": 0.486,
  "V21": 0.516, "V22": 1.123, "V23": 0.053, "V24": -0.299, "V25": -0.054,
  "V26": 0.006, "V27": 0.080, "V28": 0.102,
  "Amount": 148.00
}
```

### Example response
```json
{
  "fraud_probability": 0.00,
  "is_fraud": false,
  "threshold_used": 0.17,
  "model": "Random Forest"
}
```

## Known Limitations

- PCA-anonymized features (`V1`–`V28`) limit business interpretability of
  individual predictions.
- Model evaluated on historical data; no real-time streaming architecture
  implemented (out of scope per problem statement).
- Minor scikit-learn version mismatch observed between training (1.6.1) and
  deployment (1.8.0) environments — did not affect prediction correctness in
  this project, but production systems should pin exact versions.
- System is intended to support human fraud analysts via risk scoring, not
  to serve as an autonomous approve/deny decision system.

## Responsible Use

This model should be used to flag transactions for human review, not to
make automated adverse decisions against customers, consistent with
responsible AI practices for fraud detection systems.