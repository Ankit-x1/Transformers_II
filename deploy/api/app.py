
"""
FastAPI deployment endpoint for real-time inference.

Features:
- Load versioned model from MLflow
- REST API for predictions
- Input validation
- Latency monitoring
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import torch
import pickle
import mlflow
import time

app = FastAPI(title="Sensor Failure Prediction API")

MODEL = None
SCALER = None
DEVICE = None


class SensorInput(BaseModel):
    """Input schema validation"""
    temperature: List[float]
    vibration: List[float]
    pressure: List[float]
    rpm: List[float]


class PredictionOutput(BaseModel):
    """Output schema"""
    prediction: int  
    probability: float
    latency_ms: float


@app.on_event("startup")
async def load_model():
    """Load model on API startup"""
    global MODEL, SCALER, DEVICE
    
    # Load from MLflow
    model_uri = "models:/sensor_failure_model/production"
    MODEL = mlflow.pytorch.load_model(model_uri)
    
    # Load scaler
    with open('models/scaler.pkl', 'rb') as f:
        SCALER = pickle.load(f)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    
    print(f"Model loaded on {DEVICE}")


@app.post("/predict", response_model=PredictionOutput)
async def predict(data: SensorInput):
    """
    Predict equipment failure from sensor readings.
    
    Input: 100 timesteps of 4 sensor readings
    Output: Binary prediction (0=normal, 1=failure) with probability
    """
    start_time = time.time()
    
    if not all(len(x) == 100 for x in [data.temperature, data.vibration, data.pressure, data.rpm]):
        raise HTTPException(status_code=400, detail="All sensors must have 100 timesteps")
    
    X = np.array([data.temperature, data.vibration, data.pressure, data.rpm]).T  # [100, 4]
    X = SCALER.transform(X)
    X = torch.FloatTensor(X).unsqueeze(0).to(DEVICE)  # [1, 100, 4]
    
    with torch.no_grad():
        logits = MODEL(X)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        prob = probs[0, pred].item()
    
    latency = (time.time() - start_time) * 1000  # Convert to ms
    
    return PredictionOutput(
        prediction=pred,
        probability=prob,
        latency_ms=latency
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": MODEL is not None}
