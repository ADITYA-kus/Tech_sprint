# main.py - Merged FastAPI Backend

import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest

app = FastAPI(title="Merged Energy Analysis API")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_asset(filename):
    file_path = os.path.join(CURRENT_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Error: {filename} not found at {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Load shared assets once
try:
    model = load_asset("isolation_forest_model.pkl")
    group_averages = load_asset("group_averages.pkl")
    label_encoder = load_asset("label_encoder.pkl")
    print("✅ All AI assets loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR LOADING ASSETS: {e}")

# ==================== ANALYSIS 2 ENDPOINT ====================
class HouseholdData(BaseModel):
    energy_sum: float
    energy_std: float
    energy_max: float
    energy_mean: float
    acorn_grouped: str

@app.post("/inspect")
async def inspect_household(data: HouseholdData):
    avg = group_averages.get(data.acorn_grouped, 1.0) 
    peer_ratio = data.energy_sum / (avg + 1e-6)
    flatness = data.energy_std / (data.energy_mean + 1e-6)
    peak = data.energy_max / (data.energy_mean + 1e-6)
    
    try:
        acorn_enc = label_encoder.transform([data.acorn_grouped])[0]
    except:
        acorn_enc = 0

    features_array = [[
        data.energy_sum, 
        data.energy_std, 
        peer_ratio, 
        flatness, 
        peak, 
        acorn_enc
    ]]
    
    prediction = model.predict(features_array)[0]
    decision_score = model.decision_function(features_array)[0]
    risk_percent = max(0, min(100, (0.2 - decision_score) * 150)) 

    return {
        "status": "Suspicious" if prediction == -1 else "Normal",
        "risk_score": round(risk_percent, 2),
        "peer_ratio": round(peer_ratio, 2),
        "flatness": round(flatness, 4),
        "acorn_used": data.acorn_grouped
    }

# ==================== ANALYSIS 1 ENDPOINT ====================
class BatchAnalysisRequest(BaseModel):
    contamination: float = 0.03

@app.post("/analysis1/detect")
async def batch_anomaly_detection(request: BatchAnalysisRequest):
    """Placeholder for Analysis 1 batch processing"""
    return {
        "message": "Analysis 1 uses client-side processing",
        "contamination": request.contamination
    }