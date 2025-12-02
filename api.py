# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# Define the data schema based on the dataset features
class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., description="Transaction amount")

app = FastAPI(
    title="Fraud Detection Microservice",
    description="Real-time credit card fraud detection API.",
)

model = None
scaler = None

@app.on_event("startup")
def load_assets():
    """Loads the serialized model and scaler."""
    global model, scaler
    model_path = 'model/xgb_fraud_detector.joblib'
    scaler_path = 'model/scaler.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        # NOTE: This will raise an error if you haven't run train.py and uploaded the files.
        print("WARNING: Model assets not found. Run train.py and upload 'model/' folder.")
        # We don't raise a full error yet to allow the health check to work if deployed empty.
        return
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and Scaler loaded successfully!")

@app.post("/predict", tags=["Prediction"])
def predict_fraud(transaction: Transaction):
    """Accepts transaction features and returns a fraud prediction (0 or 1)."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Service unavailable.")

    try:
        input_data = np.array([list(transaction.dict().values())])
        scaled_data = scaler.transform(input_data)
        
        probability = model.predict_proba(scaled_data)[:, 1][0]
        prediction = int(model.predict(scaled_data)[0]) 

        return {
            "prediction": prediction,
            "probability_of_fraud": round(probability, 4),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/health", tags=["Monitoring"])
def get_health():
    """Simple health check."""
    return {"status": "ok", "model_status": "loaded" if model else "not_loaded"}
  
