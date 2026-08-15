from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import json
from pathlib import Path
import uvicorn
from typing import Optional, Dict, Any

app = FastAPI(title="Mobile Price Predictor API")

BASE_DIR = Path(__file__).resolve().parent
EVALUATION_FILE = BASE_DIR / "model_evaluation_results.json"
MODEL_DIR = BASE_DIR / "models"

# Global cache for models
models_cache = {}
evaluation_data = {}

def load_evaluation_data() -> dict:
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)

def get_recommended_model_info(phone_type: str, eval_data: dict) -> tuple:
    recommendation = eval_data.get("recommendations", {}).get(phone_type, {})
    model_name = str(recommendation.get("recommended_model", "XGBoost Regressor"))
    model_prefix = "xgboost" if "xgboost" in model_name.lower() else "random_forest"
    model_key = f"{model_prefix}_{phone_type}"
    model_path = MODEL_DIR / f"{model_key}.pkl"
    metrics = eval_data.get("results", {}).get(model_key, {})
    return model_name, model_path, metrics

@app.on_event("startup")
def startup_event():
    global evaluation_data
    if EVALUATION_FILE.exists():
        evaluation_data = load_evaluation_data()
        
        # Pre-load recommended models
        for phone_type in ["android", "iphone"]:
            try:
                name, path, metrics = get_recommended_model_info(phone_type, evaluation_data)
                if path.exists():
                    models_cache[phone_type] = joblib.load(str(path))
                    print(f"Loaded {name} for {phone_type}")
            except Exception as e:
                print(f"Error loading model for {phone_type}: {e}")

class PredictRequest(BaseModel):
    phone_type: str = "android"
    brand: str
    model: str
    storage_gb: float = 128.0
    ram_gb: float = 6.0
    warranty_days: float = 0.0
    dual_sim: bool = False
    has_5g: bool = False
    has_esim: bool = False

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "mobile_price_predictor"}

@app.post("/predict")
def predict(request: PredictRequest):
    phone_type = request.phone_type.lower()
    if phone_type not in ["android", "iphone"]:
        phone_type = "android"
        
    if phone_type not in models_cache:
        # Attempt to load it on demand
        name, path, metrics = get_recommended_model_info(phone_type, evaluation_data)
        if path.exists():
            models_cache[phone_type] = joblib.load(str(path))
        else:
            raise HTTPException(status_code=500, detail=f"Model for {phone_type} not found.")

    model_pipeline = models_cache[phone_type]
    
    input_data = {
        "brand": request.brand,
        "model": request.model,
        "condition": "used",
        "currency": "LKR",
        "dual_sim": int(request.dual_sim),
        "has_5g": int(request.has_5g),
        "has_esim": int(request.has_esim),
        "warranty_days": float(request.warranty_days),
        "storage_gb": float(request.storage_gb),
        "ram_gb": float(request.ram_gb),
    }
    
    df = pd.DataFrame([input_data])
    
    try:
        predicted_price = float(model_pipeline.predict(df)[0])
        return {
            "predicted_price": max(0.0, predicted_price),
            "phone_type": phone_type,
            "inputs": input_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=True)
