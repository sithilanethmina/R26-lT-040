from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import uvicorn
from typing import Optional, Dict, Any

app = FastAPI(title="Mobile Price Predictor API")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_preprocessing import standardize_brand
from src.feature_engineering import (
    compute_brand_tier,
    compute_model_tier,
    compute_phone_age,
    compute_is_flagship
)

EVALUATION_FILE = BASE_DIR / "outputs" / "model_evaluation_results.json"
MODEL_DIR = BASE_DIR / "models"

# Global cache for models
models_cache = {}
evaluation_data = {}

def load_evaluation_data() -> dict:
    if not EVALUATION_FILE.exists():
        return {}
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)

def get_recommended_model_info(phone_type: str, eval_data: dict) -> tuple:
    recommendation = eval_data.get("recommendations", {}).get(phone_type, {})
    model_name = str(recommendation.get("recommended_model", "CatBoost"))
    # The actual saved models are named best_{phone_type}_model.pkl
    model_path = MODEL_DIR / f"best_{phone_type}_model.pkl"
    model_key = f"{model_name.lower().replace(' ', '_')}_{phone_type}"
    metrics = eval_data.get("results", {}).get(model_key, {})
    return model_name, model_path, metrics

@app.on_event("startup")
def startup_event():
    global evaluation_data
    evaluation_data = load_evaluation_data()
    
    # Pre-load recommended models
    for phone_type in ["android", "iphone"]:
        try:
            name, path, metrics = get_recommended_model_info(phone_type, evaluation_data)
            if path.exists():
                models_cache[phone_type] = joblib.load(str(path))
                print(f"Loaded {name} for {phone_type} from {path}")
            else:
                print(f"Warning: Model file not found at {path}")
        except Exception as e:
            print(f"Error loading model for {phone_type}: {e}")

class PredictRequest(BaseModel):
    phone_type: Optional[str] = None
    brand: str
    model: str
    storage_gb: float = 128.0
    ram_gb: float = 6.0
    warranty_days: float = 0.0
    battery_health_percent: Optional[float] = None
    dual_sim: bool = False
    has_5g: bool = False
    has_esim: bool = False
    model_tier: Optional[int] = None
    brand_tier: Optional[int] = None
    phone_age_years: Optional[float] = None
    is_flagship: Optional[int] = None

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "mobile_price_predictor"}

@app.post("/predict")
def predict(request: PredictRequest):
    # 1. Standardize brand casing (e.g. "VIVO" -> "Vivo", "APPLE" -> "Apple")
    clean_brand = standardize_brand(request.brand)
    clean_model = request.model.strip()

    # 2. Determine and validate phone_type
    req_phone_type = (request.phone_type or "").lower().strip()
    if clean_brand == "Apple" or "iphone" in clean_model.lower() or "iphone" in clean_brand.lower():
        phone_type = "iphone"
        clean_brand = "Apple"
    elif req_phone_type in ["android", "iphone"]:
        # If explicitly passed iphone but brand is Android (e.g. Vivo, Samsung), override to android
        if req_phone_type == "iphone" and clean_brand not in ["Apple", "Unknown"]:
            phone_type = "android"
        else:
            phone_type = req_phone_type
    else:
        phone_type = "android"
        
    if phone_type not in models_cache:
        # Attempt to load it on demand
        name, path, metrics = get_recommended_model_info(phone_type, evaluation_data)
        if path.exists():
            models_cache[phone_type] = joblib.load(str(path))
        else:
            raise HTTPException(status_code=500, detail=f"Model for {phone_type} not found at {path}.")

    model_pipeline = models_cache[phone_type]
    
    # 3. Compute or validate engineered features using Python modules for 100% fidelity
    row_dict = {"brand": clean_brand, "model": clean_model}
    model_tier = request.model_tier if request.model_tier is not None else compute_model_tier(row_dict)
    brand_tier = request.brand_tier if request.brand_tier is not None else compute_brand_tier(clean_brand)
    phone_age_years = request.phone_age_years if request.phone_age_years is not None else compute_phone_age(row_dict)
    is_flagship = request.is_flagship if request.is_flagship is not None else compute_is_flagship(row_dict)

    # 4. Build input DataFrame matching FEATURE_COLUMNS
    input_data = {
        "brand": clean_brand,
        "model": clean_model,
        "storage_gb": float(request.storage_gb),
        "ram_gb": float(request.ram_gb),
        "warranty_days": float(request.warranty_days) if request.warranty_days is not None else 0.0,
        "battery_health_percent": float(request.battery_health_percent) if request.battery_health_percent is not None else np.nan,
        "dual_sim": int(request.dual_sim),
        "has_5g": int(request.has_5g),
        "has_esim": int(request.has_esim),
        "model_tier": int(model_tier),
        "brand_tier": int(brand_tier),
        "phone_age_years": float(phone_age_years),
        "is_flagship": int(is_flagship),
    }
    
    df = pd.DataFrame([input_data])
    
    try:
        predicted_price = float(model_pipeline.predict(df)[0])
        predicted_price = max(0.0, predicted_price)
        
        # Compute fair market range using the model's MAE
        _, _, metrics = get_recommended_model_info(phone_type, evaluation_data)
        mae = float(metrics.get("mae", predicted_price * 0.1))
        
        # Sanitize input data to ensure JSON compliance (replace NaN with None)
        json_safe_inputs = {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in input_data.items()
        }

        return {
            "predicted_price": predicted_price,
            "fair_market_range": {
                "lower_price_lkr": max(0.0, predicted_price - mae),
                "upper_price_lkr": predicted_price + mae
            },
            "phone_type": phone_type,
            "inputs": json_safe_inputs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=True)